"""Framework runtime for the SM120 MXFP4 x MXFP8 split MegaMoE kernels."""

from __future__ import annotations

import dataclasses
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch

from flashinfer.moe_ep.shims.sm120.split_cutedsl.comm import (
    _no_dist,
    close_session,
    compute_peer_offsets,
    create_session,
    ensure_not_capturing,
    page_bytes,
    sym_byte_view,
    sym_zeros,
)
from flashinfer.moe_ep.shims.sm120.split_cutedsl.staging import ACTIVATION_DTYPE
from flashinfer.moe_ep.shims.sm120.split_cutedsl.weights import (
    SCALE_DTYPE,
    TransformedWeights,
    ceil_div,
    round_up,
)

DECODE_GRAPH_COMPILE_BUCKETS = (
    7,
    16,
    32,
    64,
    128,
    168,
    256,
    320,
    384,
    512,
    1024,
    2048,
    4096,
)
_WARNED_DECODE_CAPACITY_FALLBACKS: set[int] = set()

COMM_BACKENDS = ("p2p_direct", "p2p_split")


def workspace_compile_buckets(workspace_capacity: int) -> tuple[int, ...]:
    """Every compile bucket a workspace of this capacity can select."""

    return tuple(
        sorted(
            {
                workspace_capacity,
                *(b for b in DECODE_GRAPH_COMPILE_BUCKETS if b <= workspace_capacity),
            }
        )
    )


def local_ep_topology(world_size: int) -> Any:
    """Topology of a single-process EP world (no peers)."""

    return SimpleNamespace(
        peers=(),
        same_numa_peer_count=world_size - 1,
        cross_numa_peer_count=0,
        force_ibgda_peer_mask=0,
    )


def discover_ep_topology(rank: int, world_size: int, control_group: Any) -> Any:
    """Gather every EP rank's PCI/NUMA identity (collective for EP > 1)."""

    if _no_dist() or world_size == 1:
        return local_ep_topology(world_size)
    from moe_sm120_mxfp4mxfp8_split.runtime.gpu_topology import (
        discover_gpu_topology,
        gather_ep_transport_topology,
    )

    topology = gather_ep_transport_topology(
        discover_gpu_topology(torch.cuda.current_device(), rank),
        ep_rank=rank,
        ep_control_group=control_group,
    )
    if len(topology.peers) != world_size:
        raise RuntimeError(
            f"EP topology has {len(topology.peers)} ranks, expected {world_size}"
        )
    return topology


def select_comm_backend(config: "MegaMoESm120W4A8Config", topology: Any) -> str:
    """Transport for an EP world: ``p2p_direct`` or ``p2p_split``.

    Same-NUMA worlds use direct P2P (NCCL LSA). A world spanning NUMA nodes
    has no P2P mapping between them, so it uses ``p2p_split``: same-NUMA peers
    stay direct and every cross-NUMA peer is served through a local shadow
    segment synchronized over NCCL GIN. The optional ``comm_backend`` knob may
    only confirm the backend that the discovered topology admits.
    """

    requested = (config.knobs or {}).get("comm_backend")
    if requested is not None and requested not in COMM_BACKENDS:
        raise ValueError(
            f"unsupported SM120 W4A8 comm_backend {requested!r}; "
            f"expected one of {COMM_BACKENDS}"
        )
    if not topology.cross_numa_peer_count:
        if requested == "p2p_split":
            raise ValueError("p2p_split needs cross-NUMA EP peers")
        return "p2p_direct"
    if requested == "p2p_direct":
        raise ValueError(
            "this EP world spans NUMA nodes without P2P between them; "
            "use comm_backend='p2p_split' (the default for this topology)"
        )
    from moe_sm120_mxfp4mxfp8_split.transport.nccl.shadow import ShadowLayout
    from moe_sm120_mxfp4mxfp8_split.tuned_table import SPLIT_MAX_EXPERTS_PER_RANK

    if config.local_experts > SPLIT_MAX_EXPERTS_PER_RANK:
        raise ValueError(
            "p2p_split supports at most "
            f"{SPLIT_MAX_EXPERTS_PER_RANK} experts per rank, got "
            f"{config.local_experts}"
        )
    ShadowLayout.from_topology(topology, config.rank)
    return "p2p_split"


def select_graph_compile_bucket(
    requested_tokens_per_rank: int | None,
    workspace_capacity: int,
) -> int:
    """Map a collective graph row count to a kernel specialization.

    Keep fine-grained decode buckets and sparse medium/prefill buckets. The
    workspace capacity remains the final fallback and is never exceeded.
    """
    if workspace_capacity <= 0:
        raise ValueError("workspace_capacity must be positive")
    if requested_tokens_per_rank is None:
        return workspace_capacity
    if not 0 < requested_tokens_per_rank <= workspace_capacity:
        raise ValueError(
            f"compile token count {requested_tokens_per_rank} is outside "
            f"workspace capacity {workspace_capacity}"
        )
    selected = next(
        (
            bucket
            for bucket in DECODE_GRAPH_COMPILE_BUCKETS
            if requested_tokens_per_rank <= bucket <= workspace_capacity
        ),
        workspace_capacity,
    )
    if (
        selected == workspace_capacity
        and requested_tokens_per_rank < workspace_capacity
        and requested_tokens_per_rank > DECODE_GRAPH_COMPILE_BUCKETS[-1]
        and workspace_capacity not in _WARNED_DECODE_CAPACITY_FALLBACKS
    ):
        _WARNED_DECODE_CAPACITY_FALLBACKS.add(workspace_capacity)
        warnings.warn(
            "SM120 W4A8 MegaMoE compile request "
            f"({requested_tokens_per_rank} rows) exceeds the largest dedicated "
            f"graph bucket ({DECODE_GRAPH_COMPILE_BUCKETS[-1]}); "
            f"falling back to workspace capacity ({workspace_capacity}). "
            "Padding and the selected kernel heuristic may affect performance.",
            RuntimeWarning,
            stacklevel=2,
        )
    return selected


@dataclass(frozen=True)
class MegaMoESm120W4A8Config:
    rank: int
    world_size: int
    max_tokens_per_rank: int
    num_topk: int
    num_total_experts: int
    hidden: int
    intermediate: int
    gate_up_clamp: float | None = None
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    knobs: dict[str, Any] | None = None

    @property
    def local_experts(self) -> int:
        return self.num_total_experts // self.world_size

    def validate(self) -> None:
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("invalid EP rank/world size")
        if self.num_total_experts % self.world_size:
            raise ValueError("num_total_experts must be divisible by EP world size")
        if self.hidden % 32 or self.intermediate % 32:
            raise ValueError(
                "hidden and post-SwiGLU intermediate must be multiples of 32"
            )
        if self.num_topk <= 0 or self.num_topk > self.num_total_experts:
            raise ValueError("invalid top-k")
        if self.max_tokens_per_rank <= 0:
            raise ValueError("max_tokens_per_rank must be positive")


@dataclass
class MegaMoESm120W4A8Inputs:
    activation: torch.Tensor
    activation_scale: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    fc1_weight: torch.Tensor
    fc1_weight_scale: torch.Tensor
    fc2_weight: torch.Tensor
    fc2_weight_scale: torch.Tensor
    output: torch.Tensor


@dataclass
class _ExecutionStorage:
    local_workspace: torch.Tensor
    shared_workspace: torch.Tensor
    combine_output: torch.Tensor
    rank_combine_ready: torch.Tensor | None
    epilogue_args: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    green_trace: torch.Tensor | None = None
    xnuma: Any = None


@dataclass
class _ExecutionBuffers:
    bundle: Any
    spec: Any
    local_workspace: torch.Tensor
    shared_workspace: torch.Tensor
    combine_output: torch.Tensor
    rank_combine_ready: torch.Tensor | None
    epilogue_args: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    green_trace: torch.Tensor | None = None
    xnuma: Any = None


@dataclass
class _CompiledSplit:
    execution: _ExecutionBuffers
    k1: Any
    k2: Any
    k2_drain: Any
    k2_finalizer: Any
    k3: Any
    streams: tuple[torch.cuda.Stream, torch.cuda.Stream, torch.cuda.Stream]
    graph: Any
    runtime_k1: dict[str, Any]
    runtime_k2: dict[str, Any]
    runtime_k2_drain: dict[str, Any] | None
    runtime_k2_finalizer: dict[str, Any] | None
    graph_owns_epoch_reset: bool


@dataclass
class MegaMoESm120W4A8Workspace:
    config: MegaMoESm120W4A8Config
    x: torch.Tensor
    x_scale: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    output: torch.Tensor
    _sym_roots: list[torch.Tensor] = field(default_factory=list)
    _control_group: Any = None
    # NCCL symmetric-memory session (None for single-rank workspaces) and
    # the EP transport derived from the discovered PCI/NUMA topology.
    _comm_session: Any = None
    _topology: Any = None
    _comm_backend: str = "p2p_direct"
    # bucket -> (combine_output, shared_workspace, rank_combine_ready),
    # suballocated from the session arena in one collective order.
    _symmetric_buffers: dict[int, tuple[Any, Any, Any]] = field(default_factory=dict)
    # p2p_split: control-region layout and device epoch shared by every
    # bucket's cross-NUMA transport.
    _xnuma_region: Any = None
    _xnuma_epoch: Any = None
    _capture_streams: Any = None
    _storages: dict[int, _ExecutionStorage] = field(default_factory=dict)
    _execution_plans: dict[int, tuple[Any, Any]] = field(default_factory=dict)
    _compiled_k12: dict[int, tuple[Any, Any]] = field(default_factory=dict)
    _executions: dict[int, _ExecutionBuffers] = field(default_factory=dict)
    _compile_bucket: int = 0
    _frontends: dict[tuple[int, ...], "MegaMoESm120W4A8Frontend"] = field(
        default_factory=dict
    )
    _staged_tokens: int = 0
    _destroyed: bool = False

    def _has_prepared_frontend(
        self,
        transformed_weights: TransformedWeights,
        output: torch.Tensor,
    ) -> bool:
        key = _frontend_graph_cache_key(
            transformed_weights, output, self._compile_bucket
        )
        frontend = self._frontends.get(key)
        return frontend is not None and frontend._compiled is not None

    def destroy(self) -> None:
        if self._destroyed:
            return
        for frontend in self._frontends.values():
            frontend.release()
        self._frontends.clear()
        self._executions.clear()
        self._compiled_k12.clear()
        self._execution_plans.clear()
        self._storages.clear()
        self._symmetric_buffers.clear()
        self._sym_roots.clear()
        # Suballocations share one arena. Closing the session deregisters it
        # and destroys the EP communicator; PyTorch storage ownership controls
        # the final deallocation.
        close_session(self._comm_session)
        self._destroyed = True


class MegaMoESm120W4A8Frontend:
    """Lazy JIT + native Green Context graph session for one layer and bucket.

    ``p2p_direct`` EP worlds reach every peer through NCCL LSA. ``p2p_split``
    EP worlds (NUMA-split, e.g. EP8 on 2 x 4 GPUs) run the same kernels with
    cross-NUMA peers mapped to local shadow segments that the cross-NUMA
    transport keeps in sync over NCCL GIN.
    """

    _LOCAL_RESET_REGIONS = (
        "l1_arrival_count",
        "expert_send_count",
        "grid_sync_counter",
        "fc1_done_counter",
        "fc2_done_counter",
        "atomic_counter",
        "load_balance_counter",
        "k2_ready_queue_desc",
        "k2_ready_queue_ready",
        "k2_ready_queue_state",
        "rank_combine_tile_done",
    )

    _DISPATCH_CACHE_RESET_REGIONS = (
        "dispatch_rank_cache_state",
        "dispatch_rank_cache_debug",
    )

    # ``nvlink_barrier_counter`` is deliberately not reset by graph replay.
    # It carries the phase/sign paired with the persistent two-slot
    # ``nvlink_barrier_signal`` allocation. Resetting only the counter can make
    # a later replay observe a stale target value and leave the rank barrier
    # before every peer has entered the current invocation.

    def __init__(
        self,
        workspace: MegaMoESm120W4A8Workspace,
        *,
        compile_bucket: int,
        control_group: Any = None,
    ) -> None:
        self.workspace = workspace
        self.config = workspace.config
        self.compile_bucket = compile_bucket
        self.control_group = control_group
        self._compiled: _CompiledSplit | None = None
        self._rank_combine_epoch = 0
        if compile_bucket not in (
            *DECODE_GRAPH_COMPILE_BUCKETS,
            self.config.max_tokens_per_rank,
        ):
            raise ValueError(f"unsupported W4A8 compile bucket {compile_bucket}")

    @staticmethod
    def _to_cute(
        tensor: torch.Tensor,
        assumed_align: int = 16,
        *,
        static_layout: bool = False,
    ):
        import cutlass.torch as cutlass_torch

        result = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
        if static_layout:
            return result
        return result.mark_layout_dynamic(
            leading_dim=cutlass_torch.get_leading_dim(tensor)
        )

    def _barrier(self) -> None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier(group=self.control_group)

    @property
    def _p2p_split(self) -> bool:
        return getattr(self.workspace, "_comm_backend", "p2p_direct") == "p2p_split"

    def _heuristic_overrides(self, *, split_compatible: bool = False):
        from moe_sm120_mxfp4mxfp8_split.heuristic import MegaMoEHeuristicOverrides

        fields = {field.name for field in dataclasses.fields(MegaMoEHeuristicOverrides)}
        values = dict(self.config.knobs or {})
        unknown = sorted(set(values) - fields)
        if unknown:
            raise ValueError(f"unknown SM120 W4A8 tuning knobs: {unknown}")
        requested = values.get("comm_backend")
        if requested is not None and requested not in COMM_BACKENDS:
            raise ValueError(f"unsupported SM120 W4A8 comm_backend {requested!r}")
        # Both transports run the p2p_direct kernels; p2p_split folds the
        # cross-NUMA peers into one P2P group of shadow segments.
        values["comm_backend"] = "p2p_direct"
        if split_compatible:
            # The cross-NUMA forwarder tracks the 8-warp K2's per-expert
            # progress and scatters into the per-slot combine output.
            if values.get("rank_local_combine") or values.get("k2_warps") == 12:
                raise ValueError(
                    "p2p_split requires rank_local_combine=False and k2_warps=8"
                )
            values["rank_local_combine"] = False
            values["k2_warps"] = 8
        return MegaMoEHeuristicOverrides(**values)

    @staticmethod
    def _split_compatible(spec) -> bool:
        return not spec.kernel.rank_local_combine and spec.kernel.k2_warps == 8

    def _select_spec(self, tokens_per_rank: int):
        from moe_sm120_mxfp4mxfp8_split.api import (
            MegaMoEProblemSpec,
            SplitKernelBuildOptions,
            select_compile_spec,
        )
        from moe_sm120_mxfp4mxfp8_split.runtime.green_context import (
            query_green_context_sm_counts,
            query_sm_resource_info,
        )

        cfg = self.config
        topology = getattr(self.workspace, "_topology", None) or local_ep_topology(
            cfg.world_size
        )
        same_numa = topology.same_numa_peer_count
        cross_numa = topology.cross_numa_peer_count
        p2p_split = self._p2p_split
        if p2p_split:
            # Same folded view as the standalone runner: one P2P group.
            same_numa, cross_numa = same_numa + cross_numa, 0
        jit = None
        if (
            os.environ.get("MEGA_SPLIT_GLOBALTIMER") == "1"
            or os.environ.get("MEGA_SPLIT_K2_TILE_TRACE") == "1"
        ):
            from moe_sm120_mxfp4mxfp8_split.jit_config import Sm120JitConfig

            jit = Sm120JitConfig.from_environment()
        num_sms, minimum, alignment = query_sm_resource_info()
        problem = MegaMoEProblemSpec(
            tokens_per_rank=tokens_per_rank,
            num_topk=cfg.num_topk,
            num_total_experts=cfg.num_total_experts,
            hidden=cfg.hidden,
            intermediate=2 * cfg.intermediate,
            expert_parallel_size=cfg.world_size,
            expert_parallel_rank=cfg.rank,
            data_parallel_size=cfg.data_parallel_size,
            tensor_parallel_size=cfg.tensor_parallel_size,
            gate_up_clamp=cfg.gate_up_clamp,
        )

        def select(split_compatible: bool):
            return select_compile_spec(
                problem=problem,
                ep_same_numa_peer_count=same_numa,
                ep_cross_numa_peer_count=cross_numa,
                num_sms=num_sms,
                sm_min_partition=minimum,
                sm_partition_alignment=alignment,
                overrides=self._heuristic_overrides(split_compatible=split_compatible),
                jit=jit,
                ep_numa_split=p2p_split,
            )

        spec = select(False)
        if p2p_split and not self._split_compatible(spec):
            spec = select(True)
        actual_k1, actual_k2 = query_green_context_sm_counts(
            k1_sm_count=spec.kernel.k1_sms
        )
        if (actual_k1, actual_k2) != (spec.kernel.k1_sms, spec.kernel.k2_sms):
            raise RuntimeError(
                "Green Context partition differs from the heuristic: "
                f"actual={(actual_k1, actual_k2)}, "
                f"expected={(spec.kernel.k1_sms, spec.kernel.k2_sms)}"
            )
        # Residency knobs multiply the launch grids on unchanged physical
        # Green partitions (two K1 CTAs per K1 SM; two K2 CTAs per K2 SM).
        return dataclasses.replace(
            spec,
            build=SplitKernelBuildOptions(
                concurrent_k1_k2=True,
                k1_active_clusters=actual_k1 * spec.kernel.k1_ctas_per_sm,
                k2_active_clusters=actual_k2 * spec.kernel.k2_launch_factor,
            ),
        )

    def _runtime_tensors(
        self,
        inputs: MegaMoESm120W4A8Inputs,
        local_workspace: torch.Tensor,
        shared_workspace: torch.Tensor,
        combine_output: torch.Tensor,
        rank_combine_ready: torch.Tensor | None,
        epilogue_args: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        green_trace: torch.Tensor | None,
        stream,
    ) -> dict[str, Any]:
        from src.sym_buffer import SymBufferHost

        base, offsets = compute_peer_offsets(
            shared_workspace,
            self.config.world_size,
            getattr(self.workspace, "_comm_session", None),
        )
        fc1_alpha, fc2_alpha, fc1_norm_const = epilogue_args
        return dict(
            activation=self._to_cute(inputs.activation),
            activation_sf=self._to_cute(inputs.activation_scale),
            topk_idx=self._to_cute(inputs.topk_ids),
            topk_weights=self._to_cute(inputs.topk_weights),
            fc1_weight=self._to_cute(inputs.fc1_weight),
            fc1_weight_sf=self._to_cute(inputs.fc1_weight_scale),
            fc2_weight=self._to_cute(inputs.fc2_weight),
            fc2_weight_sf=self._to_cute(inputs.fc2_weight_scale),
            fc1_alpha=self._to_cute(fc1_alpha, assumed_align=4),
            fc2_alpha=self._to_cute(fc2_alpha, assumed_align=4),
            fc1_norm_const=self._to_cute(fc1_norm_const, assumed_align=4),
            combine_output=self._to_cute(combine_output),
            combine_ready_flags=(
                self._to_cute(rank_combine_ready, assumed_align=4)
                if rank_combine_ready is not None
                else None
            ),
            fc2_block_done_counter=None,
            local_workspace=self._to_cute(local_workspace, static_layout=True),
            shared_workspace=self._to_cute(shared_workspace),
            green_trace=(
                self._to_cute(green_trace, static_layout=True)
                if green_trace is not None
                else None
            ),
            peer_rank_ptr_mapper_host=SymBufferHost(
                base_addr=base,
                offsets=offsets,
                rank_idx=self.config.rank,
                num_max_ranks=self.config.world_size,
            ),
            stream=stream,
        )

    def _ensure_plan(self, bucket: int) -> tuple[Any, Any]:
        plan = self.workspace._execution_plans.get(bucket)
        if plan is not None:
            return plan
        from moe_sm120_mxfp4mxfp8_split.api import build_split_kernels

        spec = self._select_spec(bucket)
        if os.environ.get("FLASHINFER_MEGAMOE_LOG_COMPILE_SPEC") == "1":
            kernel = spec.kernel
            print(
                "[FlashInfer MegaMoE compile] "
                f"rank={self.config.rank} "
                f"capacity={self.config.max_tokens_per_rank} "
                f"bucket={bucket} K1={kernel.k1_tile} "
                f"K2={kernel.k2_tile}/stage{kernel.k2_stages} "
                f"bundle={kernel.ready_queue_bundle} "
                f"tail_reclaim={kernel.k2_tail_reclaim} "
                f"rank_cache={kernel.dispatch_rank_cache} "
                f"rank_combine={kernel.rank_local_combine} "
                f"dispatch_warps={spec.jit.dispatch_warp_count} "
                f"k1_ctas_per_sm={kernel.k1_ctas_per_sm} "
                f"k2_launch_factor={kernel.k2_launch_factor} "
                f"backend={'p2p_split' if self._p2p_split else kernel.comm_backend} "
                f"green={kernel.k1_sms}/{kernel.k2_sms}",
                flush=True,
            )
        bundle = build_split_kernels(spec)
        if self._p2p_split:
            from moe_sm120_mxfp4mxfp8_split.kernel_fc2_combine import (
                Sm120Fc2CombineKernel,
            )

            if type(bundle.k2) is not Sm120Fc2CombineKernel:
                raise RuntimeError(
                    "p2p_split needs the 8-warp K2 (Sm120Fc2CombineKernel); "
                    f"bucket {bucket} selected {type(bundle.k2).__name__}"
                )
            # Peer-store barriers wait only for same-NUMA ranks; cross-NUMA
            # sources are delivered by the cross-NUMA transport.
            same_numa_mask = ((1 << self.config.world_size) - 1) & ~int(
                self.workspace._topology.force_ibgda_peer_mask
            )
            for kernel in (
                bundle.k1,
                bundle.k2,
                bundle.k2_drain,
                bundle.k2_finalizer,
            ):
                if (
                    kernel is not None
                    and getattr(kernel, "token_comm", None) is not None
                ):
                    kernel.token_comm.barrier_peer_mask = same_numa_mask
        if spec.kernel.push_dispatch and self.config.world_size > 1 and not _no_dist():
            from moe_sm120_mxfp4mxfp8_split.token_comm import PushDispatchConfig

            # Sources push routed input rows into a local shadow on each
            # same-NUMA peer (allocated right after the staged inputs).
            token_comm = bundle.k1.token_comm
            mask = token_comm.barrier_peer_mask & ~(1 << self.config.rank)
            peers = tuple(r for r in range(self.config.world_size) if (mask >> r) & 1)
            if peers:
                token_comm.push = PushDispatchConfig(
                    stride=_push_stride(self.config), peers=peers, peer_mask=mask
                )
        plan = (spec, bundle)
        self.workspace._execution_plans[bucket] = plan
        return plan

    def _create_xnuma(self, bundle, combine_output: torch.Tensor):
        """Cross-NUMA transport of this bucket (shared by its layers)."""

        from moe_sm120_mxfp4mxfp8_split.transport.xnuma import XnumaTransport
        from moe_sm120_mxfp4mxfp8_split.transport.xnuma_layout import XnumaLayout

        workspace, cfg, bucket = self.workspace, self.config, self.compile_bucket
        region = workspace._xnuma_region
        layout = XnumaLayout(
            world=cfg.world_size,
            n_cross=region.n_cross,
            experts_per_rank=cfg.local_experts,
            tokens_per_rank=bucket,
            topk=cfg.num_topk,
            hidden=cfg.hidden,
        )
        inputs = (
            workspace.x[:bucket],
            workspace.x_scale[:bucket],
            workspace.topk_ids[:bucket],
            workspace.topk_weights[:bucket],
        )
        view = SimpleNamespace(
            rank=cfg.rank,
            _kernel_k2=bundle.k2,
            my_activation=inputs[0],
            my_activation_sf=inputs[1],
            my_topk_idx=inputs[2],
            my_topk_weights=inputs[3],
            combine_output=combine_output,
        )
        # The session zeroed the control region once at creation; signal and
        # commit words are cumulative across buckets and must not be cleared.
        transport = XnumaTransport(
            view,
            workspace._comm_session,
            layout,
            region=region,
            epoch=workspace._xnuma_epoch,
            input_regions=inputs,
            zero_reserved=False,
        )
        # Compile-time hooks; set before K1/K2 are compiled for this bucket.
        bundle.k1.token_comm.xnuma = transport.k1_config()
        for kernel in (bundle.k2, bundle.k2_drain):
            if kernel is not None:
                kernel.xnuma_progress_addr = transport.k2_progress_addr
        return transport

    def _ensure_execution(self) -> _ExecutionBuffers:
        execution = self.workspace._executions.get(self.compile_bucket)
        if execution is not None:
            return execution
        ensure_not_capturing("W4A8 shared execution-buffer allocation")

        spec, bundle = self._ensure_plan(self.compile_bucket)
        storage = self.workspace._storages.get(self.compile_bucket)
        if storage is None:
            local_workspace = torch.zeros(
                (bundle.local_workspace_bytes,),
                dtype=torch.uint8,
                device="cuda",
            )
            symmetric = self.workspace._symmetric_buffers.get(self.compile_bucket)
            if symmetric is None:
                if getattr(self.workspace, "_comm_session", None) is not None:
                    raise RuntimeError(
                        f"W4A8 bucket {self.compile_bucket} was not planned at "
                        "workspace allocation"
                    )
                symmetric = _allocate_bucket_buffers(
                    self.config, self.compile_bucket, spec, bundle, session=None
                )
                self.workspace._symmetric_buffers[self.compile_bucket] = symmetric
            combine_output, shared_workspace, rank_combine_ready = symmetric
            xnuma = (
                self._create_xnuma(bundle, combine_output) if self._p2p_split else None
            )
            epilogue_args = tuple(
                torch.ones(
                    (self.config.local_experts,),
                    dtype=torch.float32,
                    device="cuda",
                )
                for _ in range(3)
            )
            green_trace = None
            if (
                os.environ.get("MEGA_SPLIT_GLOBALTIMER") == "1"
                or os.environ.get("MEGA_SPLIT_K2_TILE_TRACE") == "1"
            ):
                from moe_sm120_mxfp4mxfp8_split.split_timestamp import (
                    K2_TILE_TRACE_WORDS,
                    TRACE_BASE_WORDS,
                )

                trace_words = TRACE_BASE_WORDS
                if os.environ.get("MEGA_SPLIT_K2_TILE_TRACE") == "1":
                    trace_words += K2_TILE_TRACE_WORDS
                green_trace = torch.zeros(
                    (trace_words,), dtype=torch.int64, device="cuda"
                )
            storage = _ExecutionStorage(
                local_workspace=local_workspace,
                shared_workspace=shared_workspace,
                combine_output=combine_output,
                rank_combine_ready=rank_combine_ready,
                epilogue_args=epilogue_args,
                green_trace=green_trace,
                xnuma=xnuma,
            )
            self.workspace._storages[self.compile_bucket] = storage
            if os.environ.get("FLASHINFER_MEGAMOE_LOG_COMPILE_SPEC") == "1":
                # Logical buffer bytes include NCCL arena views; CUDA allocator
                # counters alone do not account for the symmetric arena.
                storage_bytes = sum(
                    tensor.numel() * tensor.element_size()
                    for item in self.workspace._storages.values()
                    for tensor in (
                        item.local_workspace,
                        item.shared_workspace,
                        item.combine_output,
                        item.rank_combine_ready,
                        item.green_trace,
                        *item.epilogue_args,
                    )
                    if tensor is not None
                )
                print(
                    "[FlashInfer MegaMoE buffers] "
                    f"rank={self.config.rank} bucket={self.compile_bucket} "
                    f"capacity={self.config.max_tokens_per_rank} "
                    f"prepared_buckets={sorted(self.workspace._storages)} "
                    f"logical_execution_bytes={storage_bytes} "
                    f"torch_allocated={torch.cuda.memory_allocated()} "
                    f"torch_reserved={torch.cuda.memory_reserved()}",
                    flush=True,
                )
        execution = _ExecutionBuffers(
            bundle=bundle,
            spec=spec,
            local_workspace=storage.local_workspace,
            shared_workspace=storage.shared_workspace,
            combine_output=storage.combine_output,
            rank_combine_ready=storage.rank_combine_ready,
            epilogue_args=storage.epilogue_args,
            green_trace=storage.green_trace,
            xnuma=storage.xnuma,
        )
        self.workspace._executions[self.compile_bucket] = execution
        return execution

    def _ensure_compiled(self, inputs: MegaMoESm120W4A8Inputs) -> _CompiledSplit:
        if self._compiled is not None:
            return self._compiled
        ensure_not_capturing("CuTeDSL compile and Green Context graph creation")

        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        from moe_sm120_mxfp4mxfp8_split.api import (
            compile_combine_reduce,
            compile_rank_local_combine,
        )
        from moe_sm120_mxfp4mxfp8_split.runtime.green_context import (
            NativeGreenContextGraph,
        )

        profile_start = time.monotonic()
        execution = self._ensure_execution()
        bundle = execution.bundle
        spec = execution.spec
        local_workspace = execution.local_workspace
        shared_workspace = execution.shared_workspace
        combine_output = execution.combine_output
        rank_combine_ready = execution.rank_combine_ready
        epilogue_args = execution.epilogue_args
        green_trace = execution.green_trace

        # Layers sharing this workspace execute serially; reuse the capture
        # streams instead of allocating three persistent streams per layer.
        if self.workspace._capture_streams is None:
            self.workspace._capture_streams = (
                torch.cuda.Stream(priority=0),
                torch.cuda.Stream(priority=-1),
                torch.cuda.Stream(priority=0),
            )
        root_stream, k1_stream, k2_stream = self.workspace._capture_streams
        root_cuda = cuda.CUstream(root_stream.cuda_stream)
        k1_cuda = cuda.CUstream(k1_stream.cuda_stream)
        k2_cuda = cuda.CUstream(k2_stream.cuda_stream)
        base_runtime = self._runtime_tensors(
            inputs,
            local_workspace,
            shared_workspace,
            combine_output,
            rank_combine_ready,
            epilogue_args,
            green_trace,
            root_cuda,
        )
        runtime_k1 = dict(base_runtime, stream=k1_cuda)
        runtime_k2 = dict(base_runtime, stream=k2_cuda)
        # Residency knobs multiply the launch grids on unchanged physical
        # Green partitions (k1_ctas_per_sm: two K1 CTAs per K1 SM;
        # k2_launch_factor: two 8-warp K2 CTAs per K2 SM).
        k1_launch_clusters = spec.kernel.k1_sms * spec.kernel.k1_ctas_per_sm
        compile_k1 = dict(runtime_k1, max_active_clusters=k1_launch_clusters)
        k2_ctas_per_sm = 2 if spec.kernel.k2_warps == 12 else 1
        k2_launch_clusters = (
            spec.kernel.k2_sms * k2_ctas_per_sm * spec.kernel.k2_launch_factor
        )
        compile_k2 = dict(
            runtime_k2,
            max_active_clusters=k2_launch_clusters,
        )
        compiled_pair = self.workspace._compiled_k12.get(self.compile_bucket)
        if compiled_pair is None:
            compiled_pair = (
                cute.compile(bundle.k1, **compile_k1),
                cute.compile(bundle.k2, **compile_k2),
            )
            self.workspace._compiled_k12[self.compile_bucket] = compiled_pair
        compiled_k1, compiled_k2 = compiled_pair
        kernels_compiled = time.monotonic()

        runtime_drain = None
        runtime_finalizer = None
        compiled_drain = None
        compiled_finalizer = None
        k2_drain_launch_clusters = 0
        if bundle.k2_drain is not None:
            runtime_drain = dict(base_runtime, stream=k1_cuda)
            # The dual-N8 specialization is verified at two resident CTAs/SM.
            # Use both slots on the SMs released by K1 during the drain phase.
            drain_ctas_per_sm = 2 if spec.kernel.k2_warps == 12 else 1
            k2_drain_launch_clusters = spec.kernel.k1_sms * drain_ctas_per_sm
            compiled_drain = cute.compile(
                bundle.k2_drain,
                **dict(
                    runtime_drain,
                    max_active_clusters=k2_drain_launch_clusters,
                ),
            )
        k1_graph_grid_blocks = [k1_launch_clusters]
        if k2_drain_launch_clusters > 0:
            k1_graph_grid_blocks.append(k2_drain_launch_clusters)

        if bundle.k2_finalizer is not None:
            runtime_finalizer = dict(base_runtime, stream=root_cuda)
            compiled_finalizer = cute.compile(
                bundle.k2_finalizer,
                **dict(runtime_finalizer, max_active_clusters=1),
            )

        combine_bucket = combine_output[: self.compile_bucket]
        output_bucket = inputs.output[: self.compile_bucket]
        if spec.kernel.rank_local_combine:
            if rank_combine_ready is None:
                raise RuntimeError("rank-local combine ready storage is missing")
            compiled_k3, runtime_k3 = compile_rank_local_combine(
                combine_bucket,
                rank_combine_ready,
                output_bucket,
                inputs.topk_ids,
                world_size=self.config.world_size,
                num_experts_per_rank=self.config.local_experts,
                stream=root_cuda,
            )
        else:
            k3_plan = compile_combine_reduce(
                combine_bucket,
                output_bucket,
                None,
                stream=root_cuda,
            )
            (
                compiled_k3,
                combine_cute,
                output_cute,
                score_cute,
                k3_stream,
            ) = k3_plan
            runtime_k3 = dict(
                combine_cute=combine_cute,
                reduced_cute=output_cute,
                topk_score_cute=score_cute,
                stream=k3_stream,
            )
        if bundle.k2_finalizer is not None:
            # The one-CTA finalizer runs after both worker streams retire.
            k1_graph_grid_blocks.append(1)

        k1_executor = compiled_k1.to(None)
        k2_executor = compiled_k2.to(None)
        drain_executor = compiled_drain.to(None) if compiled_drain else None
        finalizer_executor = compiled_finalizer.to(None) if compiled_finalizer else None
        k3_executor = compiled_k3.to(None)
        graph_owns_epoch_reset = (
            spec.kernel.k2_tail_reclaim or finalizer_executor is not None
        )
        all_kernels_ready = time.monotonic()
        from moe_sm120_mxfp4mxfp8_split.runtime import fused_fill

        fused_reset = fused_fill.enabled()
        if graph_owns_epoch_reset and fused_reset:
            # Compile the fused reset outside the capture (the same reset
            # every replay starts with).
            with torch.cuda.stream(root_stream):
                self._reset_execution(
                    execution, reset_dispatch_rank_cache=False, fused=True
                )
            torch.cuda.synchronize()
        graph = NativeGreenContextGraph.capture(
            root_stream=root_stream,
            k1_stream=k1_stream,
            k2_stream=k2_stream,
            k1_executor=k1_executor,
            k2_executor=k2_executor,
            k2_drain_executor=drain_executor,
            k3_executor=k3_executor,
            compiled_kernels=tuple(
                k
                for k in (compiled_k1, compiled_k2, compiled_drain, compiled_k3)
                if k is not None
            ),
            compiled_role_owners={
                0: (compiled_k1, compiled_drain),
                1: (compiled_k2,),
            },
            launch_k1=lambda: k1_executor(**runtime_k1),
            launch_k2=lambda: k2_executor(**runtime_k2),
            launch_k2_drain=(
                (lambda: drain_executor(**runtime_drain))
                if drain_executor is not None
                else None
            ),
            launch_rank_barrier=None,
            launch_k2_finalizer=(
                (lambda: finalizer_executor(**runtime_finalizer))
                if finalizer_executor is not None
                else None
            ),
            # Non-streaming K3 runs after the graph, on the root stream for
            # eager replay or the caller stream under an outer capture.
            launch_k3=lambda: k3_executor(
                **dict(
                    runtime_k3,
                    stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
                )
            ),
            launch_reset=(
                (
                    lambda: self._reset_execution(
                        execution, reset_dispatch_rank_cache=False, fused=fused_reset
                    )
                )
                if graph_owns_epoch_reset
                else None
            ),
            streaming_k3=False,
            k1_sm_count=spec.kernel.k1_sms,
            k1_grid_blocks=tuple(k1_graph_grid_blocks),
            k2_grid_blocks=k2_launch_clusters,
            xnuma=execution.xnuma,
        )
        graph_captured = time.monotonic()
        if os.environ.get("FLASHINFER_MEGAMOE_LOG_COMPILE_SPEC") == "1":
            print(
                "[FlashInfer MegaMoE frontend] "
                f"rank={self.config.rank} "
                f"bucket={self.compile_bucket} "
                f"k2_launch_ctas={k2_launch_clusters} "
                f"k2_drain_launch_ctas={k2_drain_launch_clusters} "
                f"layer={len(self.workspace._frontends)} "
                f"k12_compile={kernels_compiled - profile_start:.3f}s "
                f"k3_compile={all_kernels_ready - kernels_compiled:.3f}s "
                f"graph={graph_captured - all_kernels_ready:.3f}s "
                f"total={graph_captured - profile_start:.3f}s",
                flush=True,
            )
        self._barrier()
        self._compiled = _CompiledSplit(
            execution=execution,
            k1=compiled_k1,
            k2=compiled_k2,
            k2_drain=compiled_drain,
            k2_finalizer=compiled_finalizer,
            k3=compiled_k3,
            streams=(root_stream, k1_stream, k2_stream),
            graph=graph,
            runtime_k1=runtime_k1,
            runtime_k2=runtime_k2,
            runtime_k2_drain=runtime_drain,
            runtime_k2_finalizer=runtime_finalizer,
            graph_owns_epoch_reset=graph_owns_epoch_reset,
        )
        return self._compiled

    def _reset_execution(
        self,
        execution: _ExecutionBuffers,
        *,
        reset_dispatch_rank_cache: bool = True,
        fused: bool = False,
    ) -> None:
        # fused=True (graph-owned reset, default on; MEGA_SPLIT_FUSED_RESET=0 off): one
        # fill kernel instead of a chain of per-region fill nodes in front of K1.
        fills: list[tuple[torch.Tensor, int]] = []
        kernel = execution.bundle.k1
        reset_regions = list(self._LOCAL_RESET_REGIONS)
        if reset_dispatch_rank_cache:
            reset_regions.extend(self._DISPATCH_CACHE_RESET_REGIONS)
        for name in reset_regions:
            if name not in kernel._local_offsets:
                continue
            offset = int(kernel._local_offsets[name])
            size = int(kernel._local_region_by_name[name].nbytes)
            fills.append((execution.local_workspace[offset : offset + size], 0))
        if "rank_combine_route_map" in kernel._local_offsets:
            name = "rank_combine_route_map"
            offset = int(kernel._local_offsets[name])
            size = int(kernel._local_region_by_name[name].nbytes)
            fills.append(
                (
                    execution.local_workspace[offset : offset + size].view(torch.int32),
                    -1,
                )
            )
        if execution.rank_combine_ready is not None:
            self._rank_combine_epoch += 1
            fills.append(
                (execution.rank_combine_ready[-1, 0], self._rank_combine_epoch)
            )
        # The graph finalizer owns peer-written expert_recv_count and its
        # published sum. It clears both between two rank barriers; clearing
        # either here can erase an early peer's next-epoch dispatch write.
        if execution.green_trace is not None:
            fills.append((execution.green_trace, 0))
        if fused:
            from moe_sm120_mxfp4mxfp8_split.runtime.fused_fill import launch_fused_fill

            launch_fused_fill(fills, torch.cuda.current_stream())
        else:
            for tensor, value in fills:
                tensor.fill_(value)

    def run(self, inputs: MegaMoESm120W4A8Inputs) -> torch.Tensor:
        compiled = self._ensure_compiled(inputs)
        if compiled.graph_owns_epoch_reset:
            if compiled.execution.spec.kernel.dispatch_rank_cache:
                self._reset_dispatch_rank_cache(compiled.execution)
        else:
            self._reset_execution(compiled.execution)
        compiled.graph.launch(torch.cuda.current_stream())
        return inputs.output

    def _reset_dispatch_rank_cache(self, execution: _ExecutionBuffers) -> None:
        kernel = execution.bundle.k1
        for name in self._DISPATCH_CACHE_RESET_REGIONS:
            if name not in kernel._local_offsets:
                continue
            offset = int(kernel._local_offsets[name])
            size = int(kernel._local_region_by_name[name].nbytes)
            execution.local_workspace[offset : offset + size].zero_()

    def release(self) -> None:
        compiled = self._compiled
        if compiled is None:
            return
        ensure_not_capturing("W4A8 workspace release")
        compiled.graph.close()
        self._compiled = None


def set_compile_tokens_per_rank(
    workspace: MegaMoESm120W4A8Workspace,
    compile_tokens_per_rank: int | None,
) -> None:
    workspace._compile_bucket = select_graph_compile_bucket(
        compile_tokens_per_rank,
        workspace.config.max_tokens_per_rank,
    )


def _input_shapes(config: MegaMoESm120W4A8Config) -> tuple:
    """(shape, dtype) of the staged inputs and output, in arena order."""

    capacity = config.max_tokens_per_rank
    scale_columns = round_up(ceil_div(config.hidden, 32), 4)
    return (
        ((capacity, config.hidden), ACTIVATION_DTYPE),
        ((capacity, scale_columns), SCALE_DTYPE),
        ((capacity, config.num_topk), torch.int64),
        ((capacity, config.num_topk), torch.float32),
        ((capacity, config.hidden), torch.bfloat16),
    )


def _push_stride(config: MegaMoESm120W4A8Config) -> int:
    """Arena bytes of the staged input span (activation .. top-k weights):
    the push-dispatch shadow of source ``s`` starts ``(s + 1) * stride`` after
    the activation buffer."""
    return sum(
        page_bytes(math.prod(shape) * dtype.itemsize)
        for shape, dtype in _input_shapes(config)[:4]
    )


def _bucket_buffer_bytes(config: MegaMoESm120W4A8Config, bucket: int, spec, bundle):
    sizes = [
        bucket * config.num_topk * config.hidden * 2,
        bundle.shared_workspace_bytes,
    ]
    if spec.kernel.rank_local_combine:
        sizes.append((bucket + 1) * config.world_size * 4)
    return sizes


def _allocate_bucket_buffers(
    config: MegaMoESm120W4A8Config, bucket: int, spec, bundle, *, session
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Peer-visible buffers of one compile bucket, in the planned order."""

    combine_output, _root = sym_byte_view(
        (bucket, config.num_topk, config.hidden), torch.bfloat16, session=session
    )
    shared_workspace = sym_zeros(
        (bundle.shared_workspace_bytes,), torch.uint8, session=session
    )
    rank_combine_ready = None
    if spec.kernel.rank_local_combine:
        rank_combine_ready = sym_zeros(
            (bucket + 1, config.world_size), torch.int32, session=session
        )
    return combine_output, shared_workspace, rank_combine_ready


def allocate_workspace(
    config: MegaMoESm120W4A8Config,
    *,
    control_group: Any = None,
) -> MegaMoESm120W4A8Workspace:
    """Allocate the staged inputs and, for EP > 1, the NCCL symmetric arena.

    Multi-rank workspaces discover the EP PCI/NUMA topology, select
    ``p2p_direct`` or ``p2p_split``, and plan every compile bucket up front:
    NCCL windows cannot grow after peers are mapped, and one peer-offset table
    serves every suballocation only if all ranks allocate identical sizes in
    the same order. Collective on every EP rank; not allowed during capture.
    """
    config.validate()
    ensure_not_capturing("W4A8 workspace allocation")
    multi_rank = config.world_size > 1 and not _no_dist()
    topology = (
        discover_ep_topology(config.rank, config.world_size, control_group)
        if multi_rank
        else local_ep_topology(config.world_size)
    )
    comm_backend = select_comm_backend(config, topology)
    workspace = MegaMoESm120W4A8Workspace(
        config=config,
        x=None,
        x_scale=None,
        topk_ids=None,
        topk_weights=None,
        output=None,
        _control_group=control_group,
        _topology=topology,
        _comm_backend=comm_backend,
        _compile_bucket=config.max_tokens_per_rank,
    )
    session = None
    buckets = workspace_compile_buckets(config.max_tokens_per_rank)
    plans = {}
    push_dispatch = False
    if multi_rank:
        sizes = [
            math.prod(shape) * dtype.itemsize for shape, dtype in _input_shapes(config)
        ]
        for bucket in buckets:
            frontend = MegaMoESm120W4A8Frontend(
                workspace, compile_bucket=bucket, control_group=control_group
            )
            spec, bundle = frontend._ensure_plan(bucket)
            plans[bucket] = (spec, bundle)
            sizes += _bucket_buffer_bytes(config, bucket, spec, bundle)
        push_dispatch = any(
            getattr(bundle.k1.token_comm, "push", None) is not None
            for _, bundle in plans.values()
        )
        if push_dispatch:
            sizes.append(config.world_size * _push_stride(config))
        shadow = None
        reserved_bytes = 0
        if comm_backend == "p2p_split":
            from moe_sm120_mxfp4mxfp8_split.transport.nccl.shadow import ShadowLayout
            from moe_sm120_mxfp4mxfp8_split.transport.xnuma_layout import XnumaLayout

            shadow = ShadowLayout.from_topology(topology, config.rank)
            # One control-region placement for every bucket: commit words are
            # cumulative signal-adds keyed by address.
            workspace._xnuma_region = XnumaLayout(
                world=config.world_size,
                n_cross=shadow.num_cross,
                experts_per_rank=config.local_experts,
                tokens_per_rank=max(buckets),
                topk=config.num_topk,
                hidden=config.hidden,
            )
            reserved_bytes = workspace._xnuma_region.reserved_bytes
            workspace._xnuma_epoch = torch.zeros((1,), dtype=torch.int64, device="cuda")
        session = create_session(
            rank=config.rank,
            world_size=config.world_size,
            control_group=control_group,
            topology=topology,
            p2p_bytes=sum(page_bytes(n) for n in sizes),
            shadow=shadow,
            reserved_bytes=reserved_bytes,
        )
        workspace._comm_session = session
    try:
        roots: list[torch.Tensor] = []
        views = []
        push_shadow = None
        for index, (shape, dtype) in enumerate(_input_shapes(config)):
            if index == 4 and push_dispatch:
                # Push dispatch: one shadow input span per EP rank, written by
                # that rank's K1, right after this rank's own span.
                push_shadow = sym_zeros(
                    (config.world_size * _push_stride(config),),
                    torch.uint8,
                    session=session,
                )
                roots.append(push_shadow)
            view, root = sym_byte_view(shape, dtype, session=session)
            views.append(view)
            roots.append(root)
        workspace.x, workspace.x_scale, workspace.topk_ids = views[:3]
        workspace.topk_weights, workspace.output = views[3:]
        if push_shadow is not None and (
            int(push_shadow.data_ptr()) - int(workspace.x.data_ptr())
            != _push_stride(config)
        ):
            raise RuntimeError("push-dispatch shadow must follow the staged inputs")
        workspace.topk_ids.fill_(-1)
        for bucket, (spec, bundle) in plans.items():
            workspace._symmetric_buffers[bucket] = _allocate_bucket_buffers(
                config, bucket, spec, bundle, session=session
            )
        if session is not None:
            # One offset table serves every suballocation only if all ranks
            # allocated identical sizes in identical order.
            session.verify_symmetric()
        workspace._sym_roots = roots
    except BaseException:
        close_session(session)
        raise
    return workspace


def _frontend_graph_cache_key(
    transformed_weights: TransformedWeights,
    graph_output: torch.Tensor,
    compile_bucket: int,
) -> tuple[int, ...]:
    (fc1_weight, fc1_scale), (fc2_weight, fc2_scale) = transformed_weights
    return tuple(
        tensor.data_ptr() for tensor in (fc1_weight, fc1_scale, fc2_weight, fc2_scale)
    ) + (graph_output.data_ptr(), compile_bucket)


def run_split_mega_moe(
    workspace: MegaMoESm120W4A8Workspace,
    transformed_weights: TransformedWeights,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if workspace._destroyed:
        raise RuntimeError("W4A8 workspace has been destroyed")
    (fc1_weight, fc1_scale), (fc2_weight, fc2_scale) = transformed_weights
    graph_output = workspace.output if output is None else output
    weight_key = _frontend_graph_cache_key(
        transformed_weights, graph_output, workspace._compile_bucket
    )
    frontend = workspace._frontends.get(weight_key)
    if frontend is None:
        frontend = MegaMoESm120W4A8Frontend(
            workspace,
            compile_bucket=workspace._compile_bucket,
            control_group=workspace._control_group,
        )
        workspace._frontends[weight_key] = frontend
    inputs = MegaMoESm120W4A8Inputs(
        activation=workspace.x[: workspace._compile_bucket],
        activation_scale=workspace.x_scale[: workspace._compile_bucket],
        topk_ids=workspace.topk_ids[: workspace._compile_bucket],
        topk_weights=workspace.topk_weights[: workspace._compile_bucket],
        fc1_weight=fc1_weight,
        fc1_weight_scale=fc1_scale,
        fc2_weight=fc2_weight,
        fc2_weight_scale=fc2_scale,
        output=graph_output[: workspace._compile_bucket],
    )
    return frontend.run(inputs)


__all__ = [
    "COMM_BACKENDS",
    "DECODE_GRAPH_COMPILE_BUCKETS",
    "MegaMoESm120W4A8Config",
    "MegaMoESm120W4A8Inputs",
    "MegaMoESm120W4A8Workspace",
    "allocate_workspace",
    "discover_ep_topology",
    "run_split_mega_moe",
    "select_comm_backend",
    "select_graph_compile_bucket",
    "set_compile_tokens_per_rank",
    "workspace_compile_buckets",
]
