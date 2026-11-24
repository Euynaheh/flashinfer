# SM120 W4A8 Split Kernel Drop

This directory contains the SM120 MXFP4/MXFP8 split implementation at commit
`5f76519` (`ea97ff7` + `5f76519` on top of the previous drop
`500b2496c364a6029430806678ebc7c04c218ad3`).

The raw package is named `moe_sm120_mxfp4mxfp8_split`; its weight/activation
contract is MXFP4 E2M1 x MXFP8 E4M3 with E8M0 K32 scales.
The FI adapter is `flashinfer/moe_ep/shims/sm120/split_cutedsl/`.
Backends use its package exports rather than raw modules.

## Changes since the previous drop

- RTX 6000D (156 SMs, Green Context 8/8): the K1/K2 ready-queue DONE
  sentinel count is `megamoe_kernel.ready_queue_done_sentinels(total_sms)`
  (256 up to 128 SMs, unchanged kernels; 320 on 156 SMs, where the default
  EP4 T16 grid has 312 K2 consumer CTAs). `api` passes it to every kernel.
- Tuned tables are keyed by device (`device=[num_sms, min_partition,
  alignment]`, absent = the original 110/8/8 tables); new 156-SM dsv41
  `ep4_1numa` / `ep8_2numa` tables. New topologies `ep8_1domain` /
  `ep16_2x1domain`; `select_compile_spec(..., ep_split_domain_size=)` carries
  the real P2P-domain size of a folded p2p_split world (the shim passes it).
- `MEGAMOE_P2P_DOMAIN=host` (opt-in, every rank): one P2P domain per host;
  the session then sets `NCCL_P2P_LEVEL=SYS` and `NCCL_NET_DISABLE_INTRA=1`
  so NCCL maps every local pair; the shim then selects `p2p_direct`.
  Not a default: on the RTX 6000D host measured, concurrent cross-switch
  P2P stores collapse and EP8 p2p_direct over both sockets is 1.2-4.6x
  slower than `p2p_split`.
- Cross-NUMA default is `p2p_split` at every token count on any device;
  no rule selects the GIN ring any more (`select_legacy_config` rejects
  unfolded cross-NUMA peers without an explicit backend).
- Tuned-table anchors pin `k2_natural_regs` (new override) so an automatic
  decode K2 N16 tile keeps its natural registers.

## Changes in the `500b249` drop

- t=0 FC1 weight prefetch (`MegaMoEKernelConfig.k1_t0_prefetch`, K tiles;
  heuristic default: the whole tile where push dispatch is the default, i.e.
  up to 64 tokens per rank, on EP <= 8; table knob `k1_t0_prefetch`): the
  dispatch publishes each (expert, token tile) block at its static
  expert-major K1 ready-queue position, and each K1 scheduler warp prefetches
  the weight tile (and scale atoms) of its first claimed position into L2
  while the reset barrier, push and dispatch barrier run. `api` sets
  `k1.k1_t0_prefetch` and `k1.token_comm.k1_static_order`, so the shim needs
  no change. `KERNEL_CACHE_ABI=45`.
- `MEGA_SPLIT_FUSED_RESET` and `MEGA_SPLIT_FENCED_PUBLISH` now default on
  (`=0` restores the fill-node chain / a release per store); the shim's
  fused graph reset follows `fused_fill.enabled()` as before.
- Standalone measurements (host .16, DSv4.1-Flash, p50 us, 1864b29 ->
  500b249): EP4 T16/32/64/128 1175.8/1402.3/1639.4/1666.2 ->
  1167.4/1394.9/1623.7/1660.4; EP8 836.6/923.6/956.4/1077.5 ->
  823.8/911.4/943.1/1076.0.

## Changes in the previous drop (23df102)

- `heuristic.check_launch_safety` (end of `select_config_and_dispatch`, so it
  covers heuristic defaults, tuned-table entries and caller overrides) rejects
  `k2_launch_factor=2` with `k2_tail_reclaim` off for every `p2p_split`
  configuration and, on any transport, with a K2 N16 tile: that combination
  hangs at the first launch (standalone, DSv4.1-Flash T16: EP8 `p2p_split` and
  EP4 `p2p_direct` with an 8-warp K2 N16 tile; the K2 N32 variant ran). No
  shipped table entry or default is affected.

## Changes in the previous drop (1864b29)

- Packed `p2p_split` dispatch for EP16 (`XnumaLayout.pack`,
  `heuristic.select_xnuma_pack`: EP16 from 32 tokens per rank; EP8 and
  smaller worlds keep the span put): the standalone sender puts each cross
  owner only the rows routed to it plus a token -> row map, and the K1
  token-strided pull reads the map for cross sources. FlashInfer does not run
  EP16, so the shim never sets `pack` (the K1 hook reads it with `getattr`,
  and the non-pack paths compile unchanged); the sender part lives only in the
  standalone `transport/xnuma.py` (see FlashInfer adaptations).
- `transport/xnuma_layout.py` accepts forwarder splits 1, 2 or 4 and the
  forwarder handles any split (default 2, unchanged).
- Standalone EP16 measurements (two hosts, p50 us, 29e39e9 -> 7211ca9):
  DSv4.1-Flash T32/64/128/256 640.0/744.4/959.7/1475.6 ->
  621.6/684.0/850.4/1253.4; GLM5.3-Flash 441.3/544.8/696.3/1151.5 ->
  427.0/501.2/624.6/1004.5; T16 unchanged.
- `tuned_table.json` (1864b29): EP16 anchors retuned on the packed dispatch
  (DSv4.1-Flash T128/T256, GLM5.3-Flash T64/T128). FlashInfer does not run
  EP16, so the `ep16_2x2numa` entries are inert here; EP4/EP8 entries are
  unchanged.

## Changes in the previous drop (29e39e9)

- Push dispatch (`MegaMoEKernelConfig.push_dispatch`, heuristic default on
  for `p2p_direct` worlds up to 64 tokens per rank, knob `push_dispatch`):
  right after the reset barrier each source K1 writes the rows routed to a
  same-NUMA peer into that peer's local shadow of its inputs, and the owner
  pulls locally instead of with PCIe peer reads. The shim allocates the
  shadows (`world * stride` bytes directly after the staged input span) and
  sets `token_comm.push` per bucket.
- `p2p_split`: the cross-NUMA forwarder joins the compute graph after the K2
  finalizer (`MEGA_SPLIT_LATE_FORWARDER_JOIN=0` restores the old order).
- Opt-in `MEGA_SPLIT_FUSED_RESET=1` (graph-owned reset as one fill kernel,
  `runtime/fused_fill.py`; the shim honours it) and
  `MEGA_SPLIT_FENCED_PUBLISH=1`; diagnostics `MEGA_SPLIT_PHASE_STAMPS=1`
  (`phase_stamps.py`; the slots are filled but only the standalone runner
  prints them) and `MEGA_SPLIT_DIAG_JITTER_US`. `KERNEL_CACHE_ABI=44`.
- Standalone measurements (host .16, DSv4.1-Flash, p50 us, 8f3d199 ->
  29e39e9): EP4 T16/32/64/128 1187.8/1450.7/1652.8/1666.2 ->
  1176.0/1402.7/1638.9/1665.2; EP8 862.7/935.4/987.9/1083.1 ->
  833.5/923.6/956.4/1079.3.

## Changes in the previous drop (8f3d199)

- NVSHMEM is removed. P2P buffers live in NCCL symmetric windows created
  through the NCCL 2.30.7 device API (`transport/nccl/`): a small host bridge
  (`megamoe_nccl_host.cc`, built once with `g++` against the selected NCCL
  headers and cached) and prebuilt device bitcode
  (`transport/nccl/prebuilt/megamoe_nccl_device_23007.bc`) for the network
  transport kernels.
- New `p2p_split` transport for EP worlds spanning two NUMA nodes (EP8 on
  2 x 4 GPUs): same-NUMA peers keep direct P2P, and every cross-NUMA peer is
  represented by a local shadow segment (`transport/nccl/shadow.py`) that
  the cross-NUMA transport (`transport/xnuma.py`, `transport/xnuma_layout.py`)
  keeps in sync over NCCL GIN. It requires the 8-warp K2, the non-streaming
  K3, at most 64 experts per rank, and Green Context graph launch.
- Measured anchor table (`tuned_table.py`/`.json`) for DSv4.1-Flash and
  GLM5.3-Flash on `ep4_1numa`, `ep8_2numa` and `ep16_2x2numa`, plus the
  calibrated decode policy (`model_policy.py`/`.json`).
- 8f3d199 (on top of 89552a6; only `tuned_table.json` differs) retunes the
  DSv4.1-Flash EP8 decode anchors: a new T64 anchor and a retuned T128 anchor,
  both with dual K1 residency (EP8 T64 -16 %, T128 -15 % on host .16).
- Residency knobs (`k1_ctas_per_sm`, `k2_launch_factor`, `k1_stages`),
  optional weight L2 prefetch, the scalar-scale SM120 MMA emitted as direct
  PTX, the word-wide FP4 fragment shift, heuristic fixes and cheaper
  cross-NUMA sender network calls (`KERNEL_CACHE_ABI=43`).
- `src/` and `common/` now match the standalone tree, including the
  non-finite-aware host comparison and the 16-byte quantized-row check.

The standalone's two-host EP16 support is present in the topology and tuned
table code but is not exposed or tested through FlashInfer.

Not vendored: the standalone benchmark runners, the whole-invocation graph
helper used only by them, the ring/owner-plan GIN transports and the epoch
helper they use. FlashInfer runs `p2p_direct` and `p2p_split` only.

## FlashInfer adaptations

These are preserved when applying upstream kernel changes:

- Vendored import paths (`src/token_comm.py` imports
  `moe_sm120_mxfp4mxfp8_split.moe_utils`) and release-prepared comments.
- `runtime/green_context.py` injects the native Green Context graph as a
  child node when an outer CUDA stream capture is active. CUDA does not
  permit a plain `cuGraphLaunch` during capture; child-node injection keeps
  vLLM/FlashInfer CUDA Graph capture compatible while preserving the K1/K2 SM
  partitions. Under capture the `p2p_split` epoch bump and sender precede the
  child node, and the cross-NUMA scatter and K3 follow it on the caller
  stream.
- `api.select_compile_spec(..., ep_numa_split=...)` forwards the `p2p_split`
  view to the heuristic so that the `ep8_2numa` anchor entries apply.
- `transport/xnuma.py` accepts a shared control-region layout, a shared
  device epoch and per-input put regions, so one workspace serves every
  compile bucket from one reserved region. It keeps those span puts: the
  standalone packed EP16 dispatch (`XnumaLayout.pack`) is not ported, and
  `XnumaK1Config` has no `pack` field (the K1 hook treats it as off). `transport/nccl/device.bind`
  accepts the owning session instead of the process-global session.
- `transport/nccl/runtime.py` tries roots registered in `PREFERRED_ROOTS`
  first (the adapter registers FlashInfer's NCCL discovery: `NCCL_HOME`, then
  the `nvidia-nccl-cu13` wheel), then `MEGAMOE_NCCL_ROOT`, then the NCCL
  wheels; it accepts `lib64/`, and `DEFAULT_CACHE_ROOT` (FlashInfer's
  workspace) places the host bridge unless `MEGAMOE_NCCL_CACHE` is set.
  The device bitcode sources are unchanged, so its digest stamp still holds.

## Runtime requirements

Green Context execution requires CUDA toolkit 13+ and compatible
`cuda-bindings` exposing `CUdevSmResource.minSmPartitionSize` and
`smCoscheduledAlignment`. The W4A8 GPU integration tests skip unsupported
toolkit/bindings combinations, including CUDA 12.9; host-only tests remain
enabled. This test gate does not add CUDA 12.9 runtime support.

Multi-rank execution requires NCCL 2.30.7 (headers with `nccl_device.h` and
`libnccl.so.2`) and `g++`. The transport loads this NCCL privately, so it can
coexist with the NCCL that PyTorch loaded. NVSHMEM is not required.
`p2p_split` additionally requires a GIN-capable NIC per GPU (GDAKI by
default; `MEGAMOE_GIN_ALLOW_PROXY=1` accepts proxy GIN) and optionally
`MEGAMOE_GIN_HCA_MAP` to pin each CUDA device to its NIC. Single-rank
workspaces (or `MEGA_NO_DIST=1`) do not use NCCL.

## Workspace memory

Every multi-rank workspace owns one NCCL session and one symmetric arena.
NCCL windows cannot grow after peers are mapped, so allocation plans every
compile bucket up to `max_tokens_per_rank` and suballocates all of them in
one collective order. With `p2p_split` the arena additionally holds one
shadow segment per cross-NUMA peer (4 for EP8) and a control region whose
return rows take `2 * n_cross * 2 * T * topk * (16 + 2 * hidden)` bytes for
`T = max_tokens_per_rank` (DSv4.1-Flash: about 1 MB per token).

Layers with identical geometry share activation, routing, K1/K2 scratch, and
combine buffers through FlashInfer's process-level workspace pool. Each layer
keeps a separate native Green Context graph because the graph captures that
layer's weight pointers. Layer execution is sequential, so the graphs safely
reuse the same physical buffers. The backend must be warmed up collectively
before an outer CUDA Graph capture begins.

The local frontend adds 384/512/1024/2048/4096 row buckets between the
existing decode buckets and caller-provided workspace capacity. Buckets
remain part of the kernel and frontend graph cache keys; they are not
additional workspace capacities.

The same-NUMA drop includes the production dispatch rank cache and owner-local
combine policies. Their caller-stream reset, graph launch events, and finalizer
form one replay ownership protocol and must be updated together with the raw
kernel source.
