# SM120 MXFP4 x MXFP8 Split MegaMoE

This package splits the SM120 MXFP4-weight/MXFP8-activation MegaMoE path into
three device phases:

1. `kernel_dispatch_fc1.py`: dispatch, mixed W4A8 FC1, SwiGLU, amax, and
   MXFP8 output quantization.
2. `kernel_fc2_combine.py`: FC2 and direct token-back to the source rank.
3. `kernel_combine_reduce.py`: top-k reduction into the final BF16 output.

K1 publishes the existing HBM FC1-output ready counters with GPU-scope release
semantics. K2 consumes those counters with acquire polling and can start FC2
bundle-by-bundle before K1 finishes all down-projection columns. K2 executes a
cross-rank completion barrier after its peer stores, so K3 cannot observe a
partially written remote combine buffer.

Weights use packed E2M1 storage (`torch.float4_e2m1fn_x2`) with two logical K
elements per byte. Activations and the FC1-to-FC2 handoff use E4M3, while both
weight and activation scales use E8M0 with K32 block scaling. FC1 and FC2 emit
SM120 mixed `E2M1 x E4M3` block-scaled QMMA instructions; communication payloads
and the BF16 token-back contract are unchanged from the W8A8 split path.

## Validated Environment

The current SM120 path is validated with:

- public `nvidia-cutlass-dsl==4.6.0`;
- CUDA Toolkit 13.3;
- NCCL 2.30.7 with the device API (LSA symmetric memory and GIN); and
- Python 3.12 on RTX Pro 5000-class SM120 GPUs.

NVSHMEM is not used. P2P buffers live in NCCL symmetric windows, and
cross-NUMA EP peers are reached through NCCL GIN.

## Launch Modes

`mega_runner.py` provides two launch modes:

- `green_graph` (default): captures K1, K2, and K3 into one native CUDA Graph on
  ordinary streams, rebinds the K1 and K2 kernel nodes to disjoint driver
  Green Contexts, instantiates the graph once, and replays it from the caller
  stream. K3 remains on the primary context and depends on both K1 and K2.
- `sequential`: runs K1, host synchronization, K2, then K3. This mode is kept
  only for bring-up and debugging.

`heuristic.py` resolves explicit K1/K2/TX/RX SM counts before JIT compilation.
The benchmark runner can override them with `--k1_sms`, `--k2_sms`,
`--tx_sms`, and `--rx_sms`; all four counts must cover the physical device's
SMs. The runtime queries the Green Context minimum partition and co-schedule
alignment, then scales the measured 110-SM presets proportionally for other
SM120 devices. The FlashInfer-facing path performs no online candidate
compilation or timing.

The prefill selector uses expected routed rows per local expert,
`tokens_per_rank * EP * topk / total_experts`, as its tile key: N32 below 30
rows, N64 for 30--63 rows, and N128 from 64 rows onward. Same-NUMA EP4 decode
buckets 7/16/32/64 instead use K1 N32 plus K2 N16. Full-width production
shapes use the dual-N8 K2 implementation, while an explicit `k2_warps=8`
override keeps the single-group diagnostic baseline. On a 110-SM RTX Pro
5000, same-NUMA P2P uses K1/K2 `80/30` for N64 and `72/38` otherwise;
cross-NUMA EP uses hybrid P2P + IBGDA with K1/TX/RX/K2 `48/16/16/30`.
Other SM120 devices preserve these ratios after Green Context alignment.

`dispatch_chunk_tokens` describes only the standalone staged transport used
by the cross-NUMA hybrid path; it is not the direct-P2P dispatch granularity.
The hybrid selector uses 32- or 64-token chunks. It chooses 32 tokens for the
512--1024-token latency band and when a 64-token wide-hidden IBGDA payload
would exceed the configured chunk-byte limit; otherwise it uses 64. Hybrid
dispatch uses two channels and two slots. Hybrid combine uses 16-row chunks,
two channels, and normally two slots (four slots for the small cross-NUMA
window at 128 tokens/rank or below). IBGDA uses one RC per PE with warp RC
mapping.

The same-NUMA `p2p_direct` path does not packetize dispatch into those staged
chunks. Its persistent K1 dispatch group uses token-strided pull: one dispatch
warp pulls one routed token's E4M3 activation with one 1-D TMA transaction,
loads that token's E8M0 scale words and selected top-k weight, stores the row
in the local expert pool, and then advances to another routed token. K1 work
is published only when the selected FC1 N tile is complete, so the visible
ready-work granularity is N32, N64, or N128 according to the tile heuristic,
not a fixed 64 tokens. DP and model names are deliberately absent from the
selector; identical EP shape and topology produce the same specialization.

PCIe peer *reads* are slow on these hosts (a pulled row took 15--30 us under
load), so decode sizes use push dispatch (`MegaMoEKernelConfig.push_dispatch`,
on for `p2p_direct` worlds up to `heuristic.PUSH_DISPATCH_MAX_TOKENS` tokens
per rank). Right after the reset barrier every source K1 writes the
activation, scale and top-k weight rows of each token routed to a same-NUMA
peer into that peer's local shadow of the source's inputs (posted 8-byte peer
stores), and the owner's token-strided pull then reads its shadows instead of
the peers. The reset barrier orders a source's writes after the owner's pull
of the previous call; the dispatch barrier orders them before this call's
pull. Callers allocate the shadows: `world * stride` symmetric bytes placed
directly after the input span (activation, scales, top-k ids, top-k weights,
each page-aligned), where `stride` is that span's size, so source `s`'s
shadow starts at `activation + (s + 1) * stride` on every rank; then set
`k1.token_comm.push = PushDispatchConfig(stride, peers, peer_mask)` before
compiling K1 (`peers`: the same-NUMA ranks of the dispatch barrier).
For `p2p_split` the cross-NUMA forwarder joins the compute graph after the
K2 finalizer, so the finalizer's same-NUMA barrier no longer waits for the
slowest rank's last cross-NUMA return (`MEGA_SPLIT_LATE_FORWARDER_JOIN=0`
restores the old order).

`p2p_split` sends its inputs to network (GIN) peers in the sender kernel that
precedes K1. By default it puts the whole input span to every cross owner,
which at EP16 (12 network peers per rank) is NIC-bandwidth bound: phase
stamps show K1 waiting 35--50 us (T=16) to ~400 us (T=256) for the cross
dispatch while only `1 - (1 - 1/ep)^topk` of the tokens (32 % for top-6,
40 % for top-8 at EP16) reach a given owner. Packed dispatch
(`XnumaLayout.pack`; `--xnuma_pack` / `MEGAMOE_XNUMA_PACK=0|1`, default
`heuristic.select_xnuma_pack`: EP16 from `XNUMA_PACK_MIN_TOKENS[16]` tokens
per rank) makes each sender CTA find the tokens routed to its owner (a ballot
scan in token order), copy their activation rows into a staging region
(`pstage`) and put only those rows, followed by one block holding
`map[token] -> packed row`, the top-k weights and the scale rows by packed
row; it lands at the scale base of the owner's shadow-of-source. The owner's
token-strided pull reads the map for cross sources and addresses the packed
rows; records, the return path and same-NUMA sources are unchanged. At EP8
(4 network peers on one host) packing measured neutral to slower and stays
off. `MEGAMOE_XNUMA_FSPLIT=1|2|4` (diagnostic) sets the forwarder CTAs per
cross source; 2 (default) measured best at EP16.

Decode front path (default on; `MEGA_SPLIT_FUSED_RESET=0` /
`MEGA_SPLIT_FENCED_PUBLISH=0` restore the old forms): the graph-owned replay
reset is one fill kernel instead of a chain of fill nodes, and a K1 ready tile
is published with one fence and relaxed stores. t=0 FC1 weight prefetch
(`MegaMoEKernelConfig.k1_t0_prefetch`, K tiles; heuristic default: the whole
tile for EP <= 8 up to `PUSH_DISPATCH_MAX_TOKENS` tokens per rank; table knob
and `--k1_t0_prefetch`): the dispatch publishes each (expert, token tile)
block at its static expert-major position in the K1 ready queue instead of in
publish order, so the position a K1 scheduler warp claims at kernel entry
tells which weight tile it will most likely run first (assuming every local
expert holds one token tile); the warp prefetches that tile and its scales
into L2 while the reset barrier, push and dispatch barrier run.
`MEGA_SPLIT_K1_STATIC_ORDER=1` (diagnostic) gives the static order without
the prefetch. DSv4.1 ABBA (all three vs none): EP8 T16/32/64 -1.5/-1.3/-1.4 %,
EP4 -0.6/-0.5/-1.0 %. Diagnostics: `MEGA_SPLIT_PHASE_STAMPS=1` records per-call
`%globaltimer` phase stamps (sender, K1 barriers/push/pull, K2, forwarder,
scatter) that the runner prints after a `--pr_timing` run, and
`MEGA_SPLIT_DIAG_JITTER_US=N` injects per-call 0..N us delays in K1 before the
reset barrier and before the pull to exercise cross-rank buffer reuse.

The regular K2 path uses the compact eight-warp CTA: four compute warps, two
TMA producer warps, one scheduler warp, and one FC2-ready auxiliary warp. The
small decode heuristic selects the 12-warp dual-N8 path, where two compute
groups share A/TMA work and can run two CTAs per SM. After K1 retires, the
tail-reclaim launch reuses its SM partition to drain the same K2 queue; one
finalizer owns epoch reset and cross-rank completion. The N16/N32/N64/N128
paths use exact compact SFB selection from the fixed N128 scale staging tile.

Same-NUMA P2P may also cache one dispatched activation/scale row per
`(source token, destination rank)` when the estimated duplicate traffic is
large enough. At 8192 tokens/rank or above, the heuristic may use owner-local
BF16 combine before the final K3 reduction. On the validated 110-SM EP4
DSV4-flash shape (H4096/I4096/E256/topk6), that threshold is 4096. DSV4
(H7168/I6144/E384/topk6) retains the 8192 threshold. Through 24576 rows/rank,
these local-combine paths use one dispatch warp; Flash also keeps K2 N128
at stage3. Other shapes and explicit overrides retain their existing policy.
The owner-local reduction rounds BF16 partial sums per owner, so its numerical
contract differs slightly from direct top-k reduction. Both choices are part
of the compile spec and therefore part of the graph/JIT cache key.

## Framework API and JIT Cache

Framework integrations import `api.py`; they do not import `mega_runner.py`.
The production flow is:

1. Describe the local EP-rank problem with `MegaMoEProblemSpec`.
2. Call `select_compile_spec(...)` with topology and device SM properties.
3. Cache the resulting kernel set by `MegaMoECompileSpec.cache_key`.
4. On a miss, call `build_split_kernels(spec)` and allocate the returned
   local and symmetric workspace byte counts.

The cache key is a canonical SHA-256 digest over model dimensions, DP/TP/EP
semantics, transport/Green Context heuristic, tile/stage/warp choices, host
build options, and `Sm120JitConfig`. The cache ABI constant in `api.py` must be
bumped when generated device ABI or opaque workspace layout changes. Legacy
benchmark environment variables are parsed once by
`Sm120JitConfig.from_environment()`; production callers should construct the
dataclass explicitly.

Public CuTeDSL 4.6 CUDA-dialect executors do not expose their `CUfunction`
handles. For the current `cluster_shape_mnk=1,1,1` persistent path,
`green_graph` identifies K1 and K2 graph nodes by their distinct captured
grid volumes, which equal the two Green Context SM allocations. Other
cluster shapes are rejected until the executor exposes stable function
identity.

## Configurable DP / TP / EP

The one-node runner exposes `--data_parallel_size` and
`--tensor_parallel_size`; EP is derived as:

```text
EP = WORLD_SIZE / (DP * TP)
```

Global ranks are laid out as contiguous EP worlds:

```text
global_rank = ((dp_rank * TP + tp_rank) * EP) + ep_rank
```

The commands below use the DSV4-flash shape and 8192 input tokens per rank.
They assume an eight-GPU host whose `CUDA_VISIBLE_DEVICES` order places GPUs
0--3 on one NUMA node and GPUs 4--7 on the other. Every GPU pair within one
NUMA node must support CUDA peer access (`cuDeviceCanAccessPeer == 1`) and a
usable NCCL LSA peer mapping. Cross-NUMA pairs use NCCL GIN.

The runtime does not infer NUMA membership from ordinal ranges: it gathers
each selected GPU's PCI BDF and `/sys/bus/pci/devices/<BDF>/numa_node`, then
builds the per-EP same-NUMA/cross-NUMA mask. The ordinal ordering above is
still required for these DP/TP examples to form the intended NUMA-local EP
groups. The current selector does not independently validate every
same-NUMA pair with `cuDeviceCanAccessPeer`, so deployments with a different
topology must validate peer access before launch.

```bash
# DP1 x TP1 x EP8 (original path)
torchrun --standalone --nproc_per_node=8 \
  -m moe_sm120_mxfp4mxfp8_split.mega_runner \
  --num_tokens_per_rank 8192 \
  --num_topk 6 \
  --num_total_experts 256 \
  --hidden 4096 \
  --intermediate 4096 \
  --data_parallel_size 1 \
  --tensor_parallel_size 1 \
  --route_distribution balanced \
  --enable_static_expert_shape

# DP2 x TP1 x EP4: two complete expert replicas
torchrun --standalone --nproc_per_node=8 \
  -m moe_sm120_mxfp4mxfp8_split.mega_runner \
  --num_tokens_per_rank 8192 \
  --num_topk 6 \
  --num_total_experts 256 \
  --hidden 4096 \
  --intermediate 4096 \
  --data_parallel_size 2 \
  --tensor_parallel_size 1 \
  --route_distribution balanced \
  --enable_static_expert_shape

# DP1 x TP2 x EP4: NUMA-local dispatch/combine plus TP all-reduce
torchrun --standalone --nproc_per_node=8 \
  -m moe_sm120_mxfp4mxfp8_split.mega_runner \
  --num_tokens_per_rank 8192 \
  --num_topk 6 \
  --num_total_experts 256 \
  --hidden 4096 \
  --intermediate 4096 \
  --data_parallel_size 1 \
  --tensor_parallel_size 2 \
  --route_distribution balanced \
  --enable_static_expert_shape
```

TP partitions the configured full intermediate dimension. FC1 output
channels and the matching FC2 input channels are sharded, K3 runs locally,
then matching EP ranks sum their `(tokens, hidden)` BF16 partial output over
the TP group. With TP2 x EP4, TP pairs are `(GPU0,GPU4)`, `(GPU1,GPU5)`,
`(GPU2,GPU6)`, and `(GPU3,GPU7)`.

`--intermediate` is the full pre-TP gate+up width (4096 for the DSV4-flash
commands above), so TP2 runs a 2048-wide shard on each GPU. TP currently supports the separate
form-A K3 path; the in-kernel form-B top-k reduce is rejected until it has a
TP-aware accumulation-order reference.

The package-local process-group and NCCL memory bootstrap lives in
`runtime/parallelism.py`; public `src/` is unchanged.
