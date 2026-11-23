# SM120 W4A8 Split Kernel Drop

This directory contains the SM120 MXFP4/MXFP8 split implementation at commit
`8f3d199e36220c1f26c0501d6ea01c4a5e98daed` (previous drop:
`6aa25a07fe14afcf0c05323139551cd0533c07b0`).

The raw package is named `moe_sm120_mxfp4mxfp8_split`; its weight/activation
contract is MXFP4 E2M1 x MXFP8 E4M3 with E8M0 K32 scales.
The FI adapter is `flashinfer/moe_ep/shims/sm120/split_cutedsl/`.
Backends use its package exports rather than raw modules.

## Changes since the previous drop

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
  compile bucket from one reserved region. `transport/nccl/device.bind`
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
