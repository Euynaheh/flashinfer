"""Deterministic launch heuristics for the SM120 MegaMoE kernel.

This module is deliberately independent of torch, CUDA, NVSHMEM, argparse,
and process environment state.  The FlashInfer-facing call path can therefore
select one JIT configuration with a small amount of integer arithmetic; the
benchmark runner may apply explicit user overrides after selection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Optional, Tuple


Tile = Tuple[int, int, int]

# Legacy same-NUMA EP4 decode buckets validated on RTX Pro 5000. This
# layer keeps K1 N32. The independently qualified FC1 overlay below adds
# utility postprocessing; N16 remains explicit after routing holdouts. The mapping value is
# the ready-queue bundle selected by the controlled N16 sweep. Keep the
# single-group K2 until the dual-group 76-CTA replay lifecycle is proven safe
# for the full 4096-wide production shape.
_DECODE_SPLIT_K2_N16_BUNDLE_BY_BUCKET = {
    7: 1,
    16: 2,
    32: 1,
    64: 2,
}
_DECODE_ROW_BUCKETS = (7, 16, 32, 64, 128, 168, 256, 320)

# The N16 selector is production-ready for the validated decode buckets.
# Dual-N8 is enabled only in the bounded full-width EP4 window where it is
# neutral on uniform routing and wins as routing skew grows. Benchmarks may
# select the single- or dual-group path with an explicit k2_warps override.
_ENABLE_PRODUCTION_DECODE_N16 = True
_DUAL_N8_DEFAULT_BUCKETS = (7, 16, 32, 64)


def _rank_local_combine_tuning_min_tokens(
    hidden: int, intermediate: int, num_topk: int, num_total_experts: int
) -> Optional[int]:
    # Bounded same-NUMA EP4 presets measured with the cache acquire fix.
    return {
        (4096, 4096, 6, 256): 4096,
        (7168, 6144, 6, 384): 8192,
    }.get((hidden, intermediate, num_topk, num_total_experts))


def _select_decode_row_bucket(rows_per_rank: int) -> Optional[int]:
    if rows_per_rank <= 0:
        raise ValueError("rows_per_rank must be positive")
    for bucket in _DECODE_ROW_BUCKETS:
        if rows_per_rank <= bucket:
            return bucket
    return None


# Blocking IBGDA puts avoid a long late-combine quiet/drain for large payloads,
# but make tiny transfers pay the blocking round trip for every chunk.  This
# byte threshold came from the RTX Pro 5000 EP8 scan and is intentionally
# expressed in transport work rather than model names or token-only bands.
_COMBINE_BLOCKING_PUT_BYTES_PER_PEER = 3 * 1024 * 1024
_COMBINE_BLOCKING_PUT_MAX_CHUNK_BYTES = 128 * 1024
_DISPATCH_MAX_IBGDA_CHUNK_BYTES = 384 * 1024

# Same-NUMA EP4 dispatch caches one MXFP8 activation row per
# (source rank, token) and reuses it when several top-k routes land on the
# same expert owner. T16 saves too few completed remote pulls because most
# duplicates arrive while the owner route is still in flight; the 1 MiB
# traffic floor and duplicate-opportunity floor start at the measured T128 win
# for DSV4-flash without naming that model in the selector. The row floor
# prevents a very wide hidden dimension from enabling the cache before enough
# owner publications can complete to avoid in-flight fallback.
_DISPATCH_RANK_CACHE_MIN_DUPLICATE_ROWS = 240.0
_DISPATCH_RANK_CACHE_MIN_SAVED_BYTES = 1024 * 1024
_DISPATCH_RANK_CACHE_MIN_SAVED_FRACTION = 0.20

# The measured RTX Pro 5000 presets remain the tuning anchors, but their SM
# counts must scale to the physical device.  Green Context requires every
# explicitly requested group except the final remainder to respect the
# device's co-schedule alignment.
_REFERENCE_SM_COUNT = 110
_LOCAL_N64_SM_PARTITION = (80, 30)
_LOCAL_DEFAULT_SM_PARTITION = (72, 38)
_HYBRID_SM_PARTITION = (48, 16, 16, 30)


def _scale_sm_partition(
    *,
    num_sms: int,
    reference_counts: Tuple[int, ...],
    min_partition: int,
    alignment: int,
) -> Tuple[int, ...]:
    """Scale a 110-SM preset while preserving Green Context constraints."""

    if sum(reference_counts) != _REFERENCE_SM_COUNT:
        raise ValueError(
            f"reference partition must cover {_REFERENCE_SM_COUNT} SMs, "
            f"got {reference_counts}"
        )
    if num_sms < min_partition * len(reference_counts):
        raise ValueError(
            f"{num_sms} SMs cannot form {len(reference_counts)} partitions "
            f"of at least {min_partition} SMs"
        )

    aligned_minimum = (
        (min_partition + alignment - 1) // alignment * alignment
    )
    targets = tuple(
        num_sms * count / _REFERENCE_SM_COUNT
        for count in reference_counts
    )
    counts = []
    for target in targets[:-1]:
        units = int(target / alignment + 0.5)
        counts.append(max(aligned_minimum, units * alignment))

    remainder = num_sms - sum(counts)
    while remainder < min_partition:
        candidates = [
            index
            for index, count in enumerate(counts)
            if count - alignment >= aligned_minimum
        ]
        if not candidates:
            raise ValueError(
                f"cannot align partition {reference_counts} to {num_sms} SMs"
            )
        # Choose the adjustment with the smallest increase in ratio error.
        index = min(
            candidates,
            key=lambda candidate: (
                abs(counts[candidate] - alignment - targets[candidate])
                - abs(counts[candidate] - targets[candidate])
            ),
        )
        counts[index] -= alignment
        remainder += alignment

    return (*counts, remainder)

# Largest expected rows per expert at which two resident CTAs of one kernel
# per SM (k1_ctas_per_sm=2, k2_launch_factor=2) were found bit-exact.
RESIDENCY_DOUBLING_MAX_ROWS_PER_EXPERT = 32.0

# Push dispatch (``MegaMoEKernelConfig.push_dispatch``): each source writes the
# rows routed to a same-NUMA peer into a local shadow on that peer, so K1 pulls
# from local memory instead of issuing PCIe P2P reads (which take tens of us
# per row on RTX Pro 5000 hosts). Costs one input span per EP rank of
# symmetric memory; on by default up to this many tokens per rank (DSv4.1
# EP4/EP8 ABBA: -1.1 to -3.7 % at 16-64 tokens, +1.1 to +2.6 % at 128, where
# pushing every routed row costs more than the slower pull).
PUSH_DISPATCH_MAX_TOKENS = 64


@dataclass(frozen=True)
class MegaMoEHeuristicInput:
    """Runtime shape, parallel decomposition, and EP-group topology."""

    tokens_per_rank: int
    hidden: int
    intermediate: int
    num_topk: int
    num_total_experts: int
    data_parallel_size: int
    tensor_parallel_size: int
    expert_parallel_size: int
    ep_same_numa_peer_count: int
    ep_cross_numa_peer_count: int
    num_sms: int
    sm_min_partition: int
    sm_partition_alignment: int
    # p2p_split: cross-NUMA peers are served through local shadow segments
    # and are therefore counted as same-NUMA peers above.
    ep_numa_split: bool = False

    @property
    def expected_rows_per_expert(self) -> float:
        return (
            self.tokens_per_rank
            * self.expert_parallel_size
            * self.num_topk
            / self.num_total_experts
        )

    @property
    def expected_combine_bytes_per_peer(self) -> float:
        """Uniform-route estimate of BF16 FC2 bytes returned to one peer."""

        return (
            self.tokens_per_rank
            * self.num_topk
            / self.expert_parallel_size
            * self.hidden
            * 2
        )

    @property
    def probability_one_owner_is_untouched(self) -> float:
        """Uniform-route probability that top-k selects no expert on one owner."""

        experts_per_owner = self.num_total_experts // self.expert_parallel_size
        experts_outside_owner = self.num_total_experts - experts_per_owner
        if self.num_topk > experts_outside_owner:
            return 0.0
        probability = 1.0
        for index in range(self.num_topk):
            probability *= (experts_outside_owner - index) / (
                self.num_total_experts - index
            )
        return probability

    @property
    def expected_remote_route_rows(self) -> float:
        return (
            self.tokens_per_rank
            * self.num_topk
            * (self.expert_parallel_size - 1)
            / self.expert_parallel_size
        )

    @property
    def expected_duplicate_remote_owner_rows(self) -> float:
        unique_remote_owner_rows = (
            self.tokens_per_rank
            * (self.expert_parallel_size - 1)
            * (1.0 - self.probability_one_owner_is_untouched)
        )
        return max(
            self.expected_remote_route_rows - unique_remote_owner_rows,
            0.0,
        )

    @property
    def expected_dispatch_rank_cache_saved_bytes(self) -> float:
        activation_row_bytes = self.hidden
        scale_row_bytes = (self.hidden + 31) // 32
        return self.expected_duplicate_remote_owner_rows * (
            activation_row_bytes + scale_row_bytes
        )

    @property
    def expected_dispatch_rank_cache_saved_fraction(self) -> float:
        if self.expected_remote_route_rows == 0:
            return 0.0
        return (
            self.expected_duplicate_remote_owner_rows
            / self.expected_remote_route_rows
        )

    def validate(self) -> None:
        positive = {
            "tokens_per_rank": self.tokens_per_rank,
            "hidden": self.hidden,
            "intermediate": self.intermediate,
            "num_topk": self.num_topk,
            "num_total_experts": self.num_total_experts,
            "data_parallel_size": self.data_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
            "num_sms": self.num_sms,
            "sm_min_partition": self.sm_min_partition,
            "sm_partition_alignment": self.sm_partition_alignment,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"heuristic inputs must be positive, got {invalid}")
        if self.num_total_experts % self.expert_parallel_size:
            raise ValueError(
                "num_total_experts must be divisible by expert_parallel_size"
            )
        if self.num_topk > self.num_total_experts:
            raise ValueError("num_topk must not exceed num_total_experts")
        if self.ep_same_numa_peer_count < 0 or self.ep_cross_numa_peer_count < 0:
            raise ValueError("EP peer counts must be non-negative")
        if (
            self.ep_same_numa_peer_count
            + self.ep_cross_numa_peer_count
            != self.expert_parallel_size - 1
        ):
            raise ValueError(
                "same-NUMA and cross-NUMA peers must describe the complete "
                "remote EP group"
            )
        if self.num_sms < 2 * self.sm_min_partition:
            raise ValueError(
                f"{self.num_sms} SMs cannot form two Green Context "
                f"partitions of at least {self.sm_min_partition} SMs"
            )


@dataclass(frozen=True)
class MegaMoEHeuristicOverrides:
    """Benchmark-only explicit overrides; ``None`` means use the heuristic."""

    comm_backend: Optional[str] = None
    token_back_mode: Optional[str] = None
    k1_tile: Optional[Tile] = None
    k2_tile: Optional[Tile] = None
    k2_stages: Optional[int] = None
    k2_warps: Optional[int] = None
    ready_queue_bundle: Optional[int] = None
    k1_sms: Optional[int] = None
    k2_sms: Optional[int] = None
    tx_sms: Optional[int] = None
    rx_sms: Optional[int] = None
    k2_tail_reclaim: Optional[bool] = None
    dispatch_chunk_tokens: Optional[int] = None
    combine_chunk_rows: Optional[int] = None
    dispatch_channels: Optional[int] = None
    combine_channels: Optional[int] = None
    dispatch_slots: Optional[int] = None
    combine_slots: Optional[int] = None
    combine_blocking_put: Optional[bool] = None
    dispatch_local_handoff_windows: Optional[int] = None
    dispatch_remote_handoff_windows: Optional[int] = None
    ibgda_rc_per_pe: Optional[int] = None
    ibgda_rc_mapping: Optional[str] = None
    tp_k3_chunks: Optional[int] = None
    dispatch_compute_overlap: Optional[bool] = None
    dispatch_rank_cache: Optional[bool] = None
    dispatch_rank_cache_debug: Optional[bool] = None
    rank_local_combine: Optional[bool] = None
    fc1_utility_postprocess: Optional[bool] = None
    # None selects the qualified automatic overlay; "default" forces legacy.
    weight_eviction_policy: Optional[str] = None
    activation_eviction_policy: Optional[str] = None
    # Residency / pipeline knobs (None keeps the selected value).
    k1_ctas_per_sm: Optional[int] = None
    k1_stages: Optional[int] = None
    k2_launch_factor: Optional[int] = None
    weight_prefetch_ahead: Optional[int] = None
    # Push dispatch (None: on for p2p_direct up to PUSH_DISPATCH_MAX_TOKENS).
    push_dispatch: Optional[bool] = None


@dataclass(frozen=True)
class MegaMoEKernelConfig:
    """One fully resolved JIT/launch/transport configuration."""

    comm_backend: str
    kernel_comm_backend: str
    token_back_mode: str

    k1_tile: Tile
    k2_tile: Tile
    k2_stages: int
    k2_warps: int

    k1_sms: int
    k2_sms: int
    tx_sms: int
    rx_sms: int
    sm_min_partition: int
    sm_partition_alignment: int

    dispatch_chunk_tokens: int
    combine_chunk_rows: int
    dispatch_channels: int
    combine_channels: int
    dispatch_slots: int
    combine_slots: int
    combine_blocking_put: bool
    dispatch_local_handoff_windows: int
    dispatch_remote_handoff_windows: int
    ibgda_rc_per_pe: int
    ibgda_rc_mapping: str
    tp_k3_chunks: int

    dispatch_pull_mode: str
    dispatch_warps_per_tile: int
    dispatch_compute_overlap: bool
    dispatch_rank_cache: bool
    dispatch_rank_cache_debug: bool
    rank_local_combine: bool
    k1_ready_queue: bool
    k1_ready_queue_m_rotation: int
    k2_ready_queue: bool
    ready_queue_bundle: int
    k2_natural_regs: bool
    k2_min_blocks_per_sm: int
    k2_tail_reclaim: bool

    expected_rows_per_expert: float
    fc1_utility_postprocess: bool = False
    weight_eviction_policy: str = "default"
    activation_eviction_policy: str = "default"
    # K1 CTAs resident per K1 SM (2 = dual residency: K1 grid, scheduler and
    # producer counts double on the same physical SMs; needs 1 dispatch warp).
    k1_ctas_per_sm: int = 1
    # K1 TMA A/B pipeline depth; None keeps the kernel's own choice.
    k1_stages: Optional[int] = None
    # Multiplier on the K2 launch grid for 8-warp K2 (compiled for two CTAs
    # per SM but launched one per SM by default).
    k2_launch_factor: int = 1
    # K tiles ahead for the K1/K2 weight-producer L2 prefetch (0 = off).
    weight_prefetch_ahead: int = 0
    # Sources push the routed input rows into same-NUMA owners' local shadows
    # (p2p_direct kernel transport only; the caller allocates the shadows).
    push_dispatch: bool = False

    def __post_init__(self):
        if self.weight_prefetch_ahead not in (0, 4, 8):
            raise ValueError("weight_prefetch_ahead must be 0, 4 or 8")
        if self.k1_ctas_per_sm not in (1, 2):
            raise ValueError("k1_ctas_per_sm must be 1 or 2")
        if self.k1_stages is not None and self.k1_stages not in (2, 3, 4, 5):
            raise ValueError("k1_stages must be 2..5 or None")
        if self.k2_launch_factor not in (1, 2):
            raise ValueError("k2_launch_factor must be 1 or 2")
        if self.k2_launch_factor == 2 and self.k2_warps == 12:
            raise ValueError("k2_launch_factor=2 applies to 8-warp K2 only (12-warp K2 already runs two CTAs/SM)")
        if (self.k1_ctas_per_sm == 2 or self.k2_launch_factor == 2) and (
                self.expected_rows_per_expert > RESIDENCY_DOUBLING_MAX_ROWS_PER_EXPERT):
            # Two CTAs of one kernel per SM produce intermittent wrong rows at
            # higher load (dsv41 EP4 T1024: 5/20 runs; 0/16 at <= 32 rows).
            raise ValueError(
                "k1_ctas_per_sm=2 / k2_launch_factor=2 are qualified only up to "
                f"{RESIDENCY_DOUBLING_MAX_ROWS_PER_EXPERT} expected rows per expert")
        if self.k2_tail_reclaim and self.kernel_comm_backend != "p2p_direct":
            # The tail-reclaim drain stays resident on the K1 partition while
            # the GIN ring TX/RX workers poll for progress: the graph
            # deadlocks (reproduced with nvshmem_hybrid at T16 .. T24576).
            raise ValueError("k2_tail_reclaim requires the p2p_direct kernel transport")
        if self.k2_launch_factor == 2 and self.k2_tile[1] > 32:
            # Two CTAs per SM must be co-resident: the persistent K2 grid
            # synchronizes in software and deadlocks if one cannot launch.
            # N64/N128 K2 tiles do not fit twice per SM.
            raise ValueError("k2_launch_factor=2 needs a K2 tile with N <= 32 (two CTAs per SM)")
        if self.activation_eviction_policy not in ("default", "evict_last"):
            raise ValueError("activation_eviction_policy must be default or evict_last")
        if self.push_dispatch and (self.kernel_comm_backend != "p2p_direct"
                                   or self.dispatch_pull_mode != "token_strided"):
            raise ValueError("push_dispatch needs the p2p_direct kernel transport and the token-strided pull")
        if self.weight_eviction_policy not in ("default", "evict_first"):
            raise ValueError("weight_eviction_policy must be default or evict_first")

    @property
    def total_sms(self) -> int:
        return self.k1_sms + self.k2_sms + self.tx_sms + self.rx_sms

    def with_overrides(
        self, overrides: MegaMoEHeuristicOverrides
    ) -> "MegaMoEKernelConfig":
        if (
            overrides.ready_queue_bundle is not None
            and overrides.ready_queue_bundle <= 0
        ):
            raise ValueError("ready_queue_bundle must be positive")
        if overrides.k2_warps is not None and overrides.k2_warps not in (8, 12):
            raise ValueError("k2_warps must be 8 or 12")
        if overrides.activation_eviction_policy not in (None, "default", "evict_last"):
            raise ValueError("activation_eviction_policy must be default or evict_last")
        if overrides.weight_eviction_policy not in (None, "default", "evict_first"):
            raise ValueError("weight_eviction_policy must be default or evict_first")
        updates = {
            name: value
            for name, value in vars(overrides).items()
            if value is not None
        }
        if not updates:
            return self
        sm_names = ("k1_sms", "k2_sms", "tx_sms", "rx_sms")
        provided_sms = [name for name in sm_names if name in updates]
        if provided_sms and len(provided_sms) != len(sm_names):
            raise ValueError(
                "K1/K2/TX/RX SM overrides form one transport preset; "
                f"provide all four, got {provided_sms}"
            )

        deferred = {
            "comm_backend",
            "token_back_mode",
            *sm_names,
        }
        config = replace(
            self,
            **{name: value for name, value in updates.items() if name not in deferred},
        )
        if overrides.k2_tile is not None:
            token_n = config.k2_tile[1]
            config = replace(
                config,
                k2_stages=(
                    overrides.k2_stages
                    if overrides.k2_stages is not None
                    else (
                        3
                        if (
                            token_n == 64
                            or (
                                token_n == 128
                                and config.expected_rows_per_expert <= 512.0
                            )
                        )
                        else 2
                    )
                ),
                k2_natural_regs=(token_n == 32),
                k2_min_blocks_per_sm=2 if token_n == 32 else 1,
            )

        backend = overrides.comm_backend or config.comm_backend
        if backend == "p2p_direct":
            reference = (
                _LOCAL_N64_SM_PARTITION
                if config.k2_tile[1] == 64
                else _LOCAL_DEFAULT_SM_PARTITION
            )
            if backend == self.comm_backend and config.k2_tile == self.k2_tile:
                # Unrelated overrides must preserve the workload-selected
                # partition (including the compact N64 64/46 preset).
                k1_sms, k2_sms = self.k1_sms, self.k2_sms
            else:
                k1_sms, k2_sms = _scale_sm_partition(
                    num_sms=config.total_sms,
                    reference_counts=reference,
                    min_partition=config.sm_min_partition,
                    alignment=config.sm_partition_alignment,
                )
            config = replace(
                config,
                comm_backend=backend,
                kernel_comm_backend="p2p_direct",
                token_back_mode="epi_warps",
                k1_sms=k1_sms,
                k2_sms=k2_sms,
                tx_sms=0,
                rx_sms=0,
            )
        elif backend in ("nvshmem_hybrid", "nvshmem_ibgda"):
            k1_sms, tx_sms, rx_sms, k2_sms = _scale_sm_partition(
                num_sms=config.total_sms,
                reference_counts=_HYBRID_SM_PARTITION,
                min_partition=config.sm_min_partition,
                alignment=config.sm_partition_alignment,
            )
            config = replace(
                config,
                comm_backend=backend,
                kernel_comm_backend="nvshmem_ibgda",
                token_back_mode="reuse_dispatch_warps",
                k1_sms=k1_sms,
                k2_sms=k2_sms,
                tx_sms=tx_sms,
                rx_sms=rx_sms,
            )
        else:
            raise ValueError(f"unsupported comm backend {backend!r}")

        required_token_back = (
            "epi_warps" if backend == "p2p_direct" else "reuse_dispatch_warps"
        )
        if (
            overrides.token_back_mode is not None
            and overrides.token_back_mode != required_token_back
        ):
            raise ValueError(
                f"comm backend {backend!r} requires token_back_mode "
                f"{required_token_back!r}, got {overrides.token_back_mode!r}"
            )
        if provided_sms:
            config = replace(
                config,
                **{name: updates[name] for name in sm_names},
            )
        return config


def select_legacy_config(
    shape: MegaMoEHeuristicInput,
    overrides: Optional[MegaMoEHeuristicOverrides] = None,
) -> MegaMoEKernelConfig:
    """Select exactly one SM120 configuration without autotuning."""

    shape.validate()
    rows = shape.expected_rows_per_expert
    decode_bucket = _select_decode_row_bucket(shape.tokens_per_rank)
    # DP4 decode reserves 25% admission headroom at global CC1024, producing
    # at most 320 graph rows per rank. Keep that bucket on the latency-tuned
    # N32 path instead of crossing the N64 expected-rows threshold at exactly
    # 30 rows/expert for the DSV4-flash shape.
    force_decode_n32 = (
        decode_bucket == 320
        and shape.tensor_parallel_size == 1
        and shape.expert_parallel_size == 4
        and shape.ep_cross_numa_peer_count == 0
        and rows <= 32.0
    )
    # Offline RTX Pro 5000 scan: N32 wins through 28 rows/expert, N64
    # wins from 30 through 63, and N128 wins at 64 and above.
    if force_decode_n32 or rows < 30.0:
        token_n = 32
    elif rows < 64.0:
        token_n = 64
    else:
        token_n = 128

    tile = (64, token_n, 128)
    cross_numa = shape.ep_cross_numa_peer_count > 0

    if cross_numa:
        # The RTX Pro 5000 cross-NUMA scan selected 48/16/16/30. Preserve
        # those proportions on other SM120 devices and align explicit groups
        # to the device's Green Context co-schedule granularity.
        comm_backend = "nvshmem_hybrid"
        kernel_comm_backend = "nvshmem_ibgda"
        k1_sms, tx_sms, rx_sms, k2_sms = _scale_sm_partition(
            num_sms=shape.num_sms,
            reference_counts=_HYBRID_SM_PARTITION,
            min_partition=shape.sm_min_partition,
            alignment=shape.sm_partition_alignment,
        )
    else:
        comm_backend = "p2p_direct"
        kernel_comm_backend = "p2p_direct"
        # N64 needs more K1 waves on this kernel; N32/N128 are balanced by the
        # measured 72/38 split.  This depends on the selected tile, not DP.
        reference = (
            _LOCAL_N64_SM_PARTITION
            if token_n == 64
            else _LOCAL_DEFAULT_SM_PARTITION
        )
        # Medium N64 waves with compact activation/weight rows are K2-tail
        # limited. The measured 64/46 partition leaves more SMs for K2.
        # Wider rows and N128 prefill retain their previous partition.
        if (
            token_n == 64
            and shape.expert_parallel_size == 4
            and shape.tensor_parallel_size == 1
            and shape.num_sms == 110
            and shape.hidden <= 4096
            and shape.intermediate <= 4096
        ):
            reference = (64, 46)
        k1_sms, k2_sms = _scale_sm_partition(
            num_sms=shape.num_sms,
            reference_counts=reference,
            min_partition=shape.sm_min_partition,
            alignment=shape.sm_partition_alignment,
        )
        tx_sms, rx_sms = 0, 0

    # A dispatch row contains MXFP8 activations, one E8M0 scale per 32
    # elements, and FP32 top-k weights.  Keeping a cross-NUMA IBGDA operation
    # below roughly 384 KiB avoids the wide-hidden head-of-line stalls seen on
    # RTX Pro 5000.  The 512--1024-token band also benefits from a 32-row
    # first handoff even when a 64-row payload fits the byte limit; above that
    # band the extra operations outweigh the earlier first tile.
    dispatch_bytes_per_token = (
        shape.hidden
        + (shape.hidden + 31) // 32
        + shape.num_topk * 4
    )
    dispatch_chunk_tokens = (
        32
        if (
            cross_numa
            and (
                512 <= shape.tokens_per_rank <= 1024
                or (
                    shape.tokens_per_rank >= 128
                    and dispatch_bytes_per_token * 64
                    > _DISPATCH_MAX_IBGDA_CHUNK_BYTES
                )
            )
        )
        else 64
    )
    dispatch_channels = 2
    combine_channels = 2
    dispatch_slots = 2
    # The EP8 128-token shape produces roughly three combine chunks per
    # peer/channel.  Keeping only two cyclic slots makes the producer wait for
    # an in-epoch credit while its peer is still completing the reciprocal
    # channel, which can form a cross-NUMA progress cycle.  Four slots cover
    # the complete small-message window; terminal ACKs still retire the epoch
    # before the next replay can reuse it.  Larger shapes stay on the measured
    # two-slot fast path so their streaming reuse distance is unchanged.
    combine_slots = (
        4
        if cross_numa and shape.tokens_per_rank <= 128
        else 2
    )
    combine_chunk_rows = 16
    combine_blocking_put = (
        cross_numa
        and shape.expected_combine_bytes_per_peer
        >= _COMBINE_BLOCKING_PUT_BYTES_PER_PEER
        and combine_chunk_rows * shape.hidden * 2
        <= _COMBINE_BLOCKING_PUT_MAX_CHUNK_BYTES
    )
    remote_handoff_windows = 2 if cross_numa else 1
    dispatch_rank_cache = (
        not cross_numa
        and shape.expert_parallel_size == 4
        and shape.ep_same_numa_peer_count == 3
        and shape.tensor_parallel_size == 1
        and shape.expected_duplicate_remote_owner_rows
        >= _DISPATCH_RANK_CACHE_MIN_DUPLICATE_ROWS
        and shape.expected_dispatch_rank_cache_saved_bytes
        >= _DISPATCH_RANK_CACHE_MIN_SAVED_BYTES
        and shape.expected_dispatch_rank_cache_saved_fraction
        >= _DISPATCH_RANK_CACHE_MIN_SAVED_FRACTION
    )
    # The 110-SM retune moves only the validated Flash shape's crossover to
    # 4K. Other geometries keep the existing 8K numerical-contract boundary.
    combine_tuning_min_tokens = _rank_local_combine_tuning_min_tokens(
        shape.hidden, shape.intermediate, shape.num_topk, shape.num_total_experts
    )
    combine_min_tokens = (
        combine_tuning_min_tokens
        if shape.num_sms == 110 and combine_tuning_min_tokens is not None
        else 8192
    )
    rank_local_combine = (
        not cross_numa
        and shape.expert_parallel_size == 4
        and shape.ep_same_numa_peer_count == 3
        and shape.tensor_parallel_size == 1
        and shape.hidden >= 4096
        and shape.tokens_per_rank >= combine_min_tokens
    )

    # Except for the bounded small-message window above, keep production on
    # the shallow two-slot fast path. Fixed peer/channel CTA ownership and
    # independent terminal credits provide forward progress without growing
    # the reuse distance of wide DSV4 rows.

    config = MegaMoEKernelConfig(
        comm_backend=comm_backend,
        kernel_comm_backend=kernel_comm_backend,
        token_back_mode=(
            "reuse_dispatch_warps" if cross_numa else "epi_warps"
        ),
        # Near half-filled N128 waves, wide FC1 rows favor finer token
        # tiling. Keep K2 N128: reducing both tiles regressed in controls.
        k1_tile=(
            (64, 64, 128)
            if (
                not cross_numa
                and shape.expert_parallel_size == 4
                and shape.tensor_parallel_size == 1
                and shape.num_sms == 110
                and shape.hidden >= 6144
                and 64.0 <= rows < 96.0
            )
            else tile
        ),
        k2_tile=tile,
        k2_stages=(
            3
            if token_n == 64 or (token_n == 128 and rows <= 512.0)
            else 2
        ),
        k2_warps=8,
        k1_sms=k1_sms,
        k2_sms=k2_sms,
        tx_sms=tx_sms,
        rx_sms=rx_sms,
        sm_min_partition=shape.sm_min_partition,
        sm_partition_alignment=shape.sm_partition_alignment,
        dispatch_chunk_tokens=dispatch_chunk_tokens,
        combine_chunk_rows=combine_chunk_rows,
        dispatch_channels=dispatch_channels,
        combine_channels=combine_channels,
        dispatch_slots=dispatch_slots,
        combine_slots=combine_slots,
        combine_blocking_put=combine_blocking_put,
        # Same-NUMA P2P hands off after one window; cross-NUMA IBGDA after
        # two. This early epoch is independent of terminal slot ACK/credit.
        dispatch_local_handoff_windows=1,
        dispatch_remote_handoff_windows=remote_handoff_windows,
        ibgda_rc_per_pe=1,
        ibgda_rc_mapping="warp",
        tp_k3_chunks=(
            8
            if shape.tensor_parallel_size == 2
            and shape.tokens_per_rank >= 8192
            else 1
        ),
        dispatch_pull_mode="token_strided",
        dispatch_warps_per_tile=8,
        dispatch_compute_overlap=shape.tokens_per_rank >= 1024,
        dispatch_rank_cache=dispatch_rank_cache,
        dispatch_rank_cache_debug=False,
        rank_local_combine=rank_local_combine,
        k1_ready_queue=True,
        # Keep the production selector independent of DP layout.  The old
        # DP-only queue rotation was an experiment, not a portable heuristic.
        k1_ready_queue_m_rotation=0,
        k2_ready_queue=True,
        # Very short expert waves need finer-grained K2 distribution; larger
        # waves amortize queue traffic with wider hidden-tile bundles.
        ready_queue_bundle=(
            4 if rows <= 128.0 else 8 if rows <= 512.0 else 16
        ),
        k2_natural_regs=(token_n == 32),
        k2_min_blocks_per_sm=2 if token_n == 32 else 1,
        # The decode-specific selector below enables tail reclaim only for the
        # validated dual-group N16 path. Other shapes keep the regular K2
        # scheduler/workspace contract.
        k2_tail_reclaim=False,
        expected_rows_per_expert=rows,
    )
    if overrides is not None:
        config = config.with_overrides(overrides)
    if (
        config.rank_local_combine
        and config.kernel_comm_backend == "p2p_direct"
        and shape.expert_parallel_size == 4
        and shape.tensor_parallel_size == 1
        and shape.num_sms == 110
        and combine_tuning_min_tokens == 4096
        and 4096 <= shape.tokens_per_rank <= 24576
        and config.k2_tile == (64, 128, 128)
        and (
            overrides is None
            or (overrides.k2_tile is None and overrides.k2_stages is None)
        )
    ):
        config = replace(config, k2_stages=3)
    if config.dispatch_rank_cache and config.kernel_comm_backend != "p2p_direct":
        if overrides is not None and overrides.dispatch_rank_cache is True:
            raise ValueError(
                "dispatch_rank_cache requires the p2p_direct backend"
            )
        config = replace(
            config,
            dispatch_rank_cache=False,
            dispatch_rank_cache_debug=False,
        )
    if config.dispatch_rank_cache_debug and not config.dispatch_rank_cache:
        raise ValueError(
            "dispatch_rank_cache_debug requires dispatch_rank_cache"
        )
    # The decode optimization is deliberately expressed in graph rows and
    # topology, not in model names. Explicit K2 overrides denote a controlled
    # benchmark and suppress the production tuning.
    explicit_k2_override = overrides is not None and any(
        value is not None
        for value in (overrides.k2_tile, overrides.k2_stages)
    )
    explicit_tail_override = (
        overrides is not None and overrides.k2_tail_reclaim is not None
    )
    decode_bundle = (
        _DECODE_SPLIT_K2_N16_BUNDLE_BY_BUCKET.get(decode_bucket)
        if _ENABLE_PRODUCTION_DECODE_N16
        else None
    )
    if (
        not explicit_k2_override
        and config.kernel_comm_backend == "p2p_direct"
        and config.k1_tile == (64, 32, 128)
        and config.k2_tile == (64, 32, 128)
        and decode_bundle is not None
    ):
        dual_group_default = (
            decode_bucket in _DUAL_N8_DEFAULT_BUCKETS
            and shape.expert_parallel_size == 4
            and shape.tensor_parallel_size == 1
            and shape.ep_same_numa_peer_count == 3
            and shape.ep_cross_numa_peer_count == 0
            and shape.hidden >= 4096
            and shape.intermediate >= 4096
        )
        use_dual_group = (
            dual_group_default
            if overrides is None or overrides.k2_warps is None
            else overrides.k2_warps == 12
        )
        config = replace(
            config,
            k2_tile=(64, 16, 128),
            k2_stages=2,
            k2_warps=12 if use_dual_group else 8,
            ready_queue_bundle=decode_bundle,
            k2_natural_regs=True,
            k2_min_blocks_per_sm=2 if use_dual_group else 1,
            k2_tail_reclaim=(
                config.k2_tail_reclaim
                if explicit_tail_override
                else use_dual_group
            ),
        )
    if config.total_sms != shape.num_sms:
        raise ValueError(
            "resolved SM partitions must cover all device SMs: "
            f"K1={config.k1_sms}, K2={config.k2_sms}, "
            f"TX={config.tx_sms}, RX={config.rx_sms}, "
            f"total={config.total_sms}, expected={shape.num_sms}"
        )
    if config.tp_k3_chunks <= 0:
        raise ValueError("tp_k3_chunks must be positive")
    rank_local_combine_forced = (
        overrides is not None and overrides.rank_local_combine is True
    )
    if (
        config.rank_local_combine
        and not rank_local_combine_forced
        and (
            config.comm_backend != "p2p_direct"
            or config.token_back_mode != "epi_warps"
            or shape.expert_parallel_size > shape.num_topk
            or shape.tensor_parallel_size != 1
            or config.dispatch_rank_cache_debug
            or config.k2_tail_reclaim
            or config.k2_warps == 12
        )
    ):
        config = replace(config, rank_local_combine=False)
    if config.rank_local_combine:
        if config.comm_backend != "p2p_direct":
            raise ValueError("rank_local_combine requires p2p_direct")
        if config.token_back_mode != "epi_warps":
            raise ValueError("rank_local_combine requires epi_warps token-back")
        if shape.expert_parallel_size > shape.num_topk:
            raise ValueError(
                "rank_local_combine requires expert_parallel_size <= num_topk"
            )
        if shape.tensor_parallel_size != 1:
            raise ValueError("rank_local_combine is currently validated only for TP1")
        if config.dispatch_rank_cache_debug:
            raise ValueError(
                "rank_local_combine is incompatible with dispatch_rank_cache_debug"
            )
        if config.k2_tail_reclaim or config.k2_warps == 12:
            raise ValueError(
                "rank_local_combine is not compatible with experimental K2 tail paths"
            )
    if config.dispatch_local_handoff_windows <= 0:
        raise ValueError("dispatch_local_handoff_windows must be positive")
    if config.dispatch_remote_handoff_windows <= 0:
        raise ValueError("dispatch_remote_handoff_windows must be positive")
    if shape.tensor_parallel_size == 1 and config.tp_k3_chunks != 1:
        raise ValueError("TP1 requires tp_k3_chunks=1")
    return config


@lru_cache(maxsize=1)
def _builtin_policy():
    # Private immutable-in-practice calibration data; never returned to callers.
    import json
    from pathlib import Path
    return json.loads(Path(__file__).with_name("model_policy.json").read_text())


def _with_tuned_table(shape, overrides):
    """Fill unset overrides from the ``tuned_table.json`` anchor entry.

    Table knobs are exactly the runner flags the tuner measured on top of the
    built-in selection, so they enter as overrides (explicit ones win) and
    the normal selection path below runs unchanged. A caller-forced backend
    other than the entry's leaves the table out.
    """
    from . import tuned_table

    entry = tuned_table.lookup(shape)
    if entry is None:
        return overrides
    table = tuned_table.overrides_for(entry["knobs"])
    user = {k: v for k, v in vars(overrides).items() if v is not None} if overrides else {}
    if "comm_backend" in user and user["comm_backend"] != table.get("comm_backend", "p2p_direct"):
        return overrides
    if not table or set(table) <= {"comm_backend", "token_back_mode"}:
        return overrides  # the default won at this anchor: built-in selection stays in charge
    # Knobs are changes relative to the built-in selection *at the anchor*.
    # Pin that selection's geometry so every token count mapped to the anchor
    # runs the measured configuration (the built-in tile choice changes with
    # tokens and may not combine with e.g. dual K1 residency).
    # Resolve exactly what the tuner measured: the built-in selection at the
    # anchor with the entry's knobs as overrides.
    anchor = replace(shape, tokens_per_rank=int(entry["tokens"]))
    measured, _ = _select_config_and_dispatch_base(
        anchor, MegaMoEHeuristicOverrides(**{k: v for k, v in table.items()
                                             if k not in ("weight_eviction_policy", "push_dispatch")}))
    merged = {name: getattr(measured, name) for name in _PINNED_FROM_ANCHOR}
    merged.update(table)
    if shape.expected_rows_per_expert > RESIDENCY_DOUBLING_MAX_ROWS_PER_EXPERT:
        # The anchor may be qualified for residency doubling; this load is not.
        # Its K1 stage count was sized for two CTAs per SM only if the entry
        # doubled residency; otherwise it is the measured single-CTA value.
        doubled = merged.get("k1_ctas_per_sm") == 2 or merged.get("k2_launch_factor") == 2
        merged.update(k1_ctas_per_sm=1, k2_launch_factor=1)
        if doubled:
            merged["k1_stages"] = None
    merged.update(user)
    return MegaMoEHeuristicOverrides(**merged)


_PINNED_FROM_ANCHOR = (
    "k1_tile", "k2_tile", "k2_stages", "k2_warps", "ready_queue_bundle",
    "k1_sms", "k2_sms", "tx_sms", "rx_sms", "k2_tail_reclaim", "dispatch_chunk_tokens",
    "rank_local_combine", "fc1_utility_postprocess", "k1_ctas_per_sm", "k1_stages",
    "k2_launch_factor", "weight_prefetch_ahead",
)


def _select_config_and_dispatch_base(shape, overrides=None, *, use_tuned_defaults=True):
    """Resolve the calibrated tile, resource, dispatch and utility policy.

    The public selector caches the final result including the weight policy.
    Explicit overrides remain authoritative; callers can opt out of built-in
    calibration with use_tuned_defaults=False. No CUDA state is kept.
    """
    # Validate user overrides before merging: partial SM presets must not be
    # accidentally accepted by filling their missing fields from the profile.
    incumbent = select_legacy_config(shape, overrides)
    if not use_tuned_defaults or not (
        (shape.hidden, shape.intermediate, shape.num_topk, shape.num_total_experts)
        in ((4096, 4096, 6, 256), (4096, 4096, 8, 288))
        and (shape.expert_parallel_size, shape.tensor_parallel_size,
             shape.data_parallel_size, shape.ep_same_numa_peer_count,
             shape.ep_cross_numa_peer_count, shape.num_sms,
             shape.sm_min_partition, shape.sm_partition_alignment)
        == (4, 1, 1, 3, 0, 110, 8, 8)
        and incumbent.comm_backend == "p2p_direct"
        and incumbent.token_back_mode == "epi_warps"
    ):
        return incumbent, None
    if not 16 <= shape.tokens_per_rank <= 24576:
        # Boundary measurements varied only utility on the legacy preset.
        # Do not extend the calibrated tile/resource/dispatch domain with it.
        base = select_legacy_config(shape)
        if overrides is not None and any(
            value is not None and value != getattr(base, key)
            for key, value in vars(overrides).items()
            if key != "fc1_utility_postprocess"
        ):
            return incumbent, None
        if overrides is not None and overrides.fc1_utility_postprocess is not None:
            return incumbent, None
        signature = [shape.hidden, shape.intermediate, shape.num_topk,
                     shape.num_total_experts]
        for row in _builtin_policy().get("fc1_postprocess", {}).get("ranges", []):
            if (row["signature"] == signature
                    and row["tokens"][0] <= shape.tokens_per_rank <= row["tokens"][1]
                    and (not row.get("requires_rank_local_combine", False)
                         or incumbent.rank_local_combine)):
                return replace(incumbent, fc1_utility_postprocess=row["utility"]), None
        return incumbent, None
    from .model_policy import select
    base = select_legacy_config(shape)
    compact = dict(k1=base.k1_tile[1], k2=base.k2_tile[1],
        stage=base.k2_stages, sm=base.k1_sms, bundle=base.ready_queue_bundle,
        warps=base.k2_warps, tail=base.k2_tail_reclaim,
        cache=base.dispatch_rank_cache, lc=base.rank_local_combine, dispatch=1)
    # The legacy dispatch rule for these two shapes; keep calibration's input
    # identical to the frozen explicit selector, including fallback regions.
    if base.expected_rows_per_expert > 192 and not (
        base.rank_local_combine and shape.num_topk == 6
    ):
        compact['dispatch'] = 4
    config, _ = select(dict(t=shape.tokens_per_rank, h=shape.hidden,
        i=shape.intermediate, k=shape.num_topk, e=shape.num_total_experts),
        _builtin_policy(), compact)
    defaults = dict(k1_tile=(64, config['k1'], 128),
        k2_tile=(64, config['k2'], 128), k2_stages=config['stage'],
        k1_sms=config['sm'], k2_sms=110-config['sm'], tx_sms=0, rx_sms=0,
        ready_queue_bundle=config['bundle'], k2_warps=config['warps'],
        k2_tail_reclaim=config['tail'], dispatch_rank_cache=config['cache'],
        rank_local_combine=config['lc'])
    # A separately qualified FC1 layer preserves the complete transport/K2
    # preset above. Its bounded ranges are measured jointly, not inferred by
    # adding the isolated N16 and utility gains. Custom resource presets do
    # not automatically inherit this layer; explicit utility overrides still
    # remain authoritative below.
    fc1_default = None
    custom_preset = overrides is not None and any(
        getattr(overrides, key) is not None
        and getattr(overrides, key) != value
        for key, value in defaults.items() if key != 'k1_tile'
    )
    if not custom_preset:
        signature = [shape.hidden, shape.intermediate, shape.num_topk,
                     shape.num_total_experts]
        for row in _builtin_policy().get('fc1_postprocess', {}).get('ranges', []):
            if (row['signature'] == signature
                    and row['tokens'][0] <= shape.tokens_per_rank <= row['tokens'][1]
                    and (not row.get('requires_rank_local_combine', False)
                         or defaults['rank_local_combine'])):
                fc1_default = row
                break
    qualified_k1_tile = (
        (64, fc1_default['k1_n'], 128)
        if fc1_default is not None and 'k1_n' in fc1_default
        else defaults['k1_tile']
    )
    # Changing tile/warp geometry leaves the calibrated complete preset.
    # Retain legacy dependent defaults (e.g. disabling dual N8 disables tail).
    if overrides is not None and any(
        getattr(overrides, key) is not None
        and getattr(overrides, key) not in (
            (defaults[key], qualified_k1_tile) if key == 'k1_tile'
            else (defaults[key],)
        )
        for key in ('k1_tile', 'k2_tile', 'k2_warps')
    ):
        return incumbent, None
    if overrides is not None and overrides.dispatch_rank_cache_debug:
        return incumbent, None
    if fc1_default is not None:
        defaults.update(k1_tile=qualified_k1_tile,
                        fc1_utility_postprocess=fc1_default['utility'])
    defaults.update({k:v for k,v in vars(overrides or MegaMoEHeuristicOverrides()).items()
                     if v is not None})
    selected = select_legacy_config(shape, MegaMoEHeuristicOverrides(**defaults))
    return selected, config['dispatch']


def _select_weight_cache_policy(shape, config, *, activation=False):
    """Use only qualified shapes with the complete, unchanged launch preset.

    L2 eviction priority affects reuse across both weight streams. A useful
    small-token result does not imply that larger or custom presets benefit.
    """
    ep = shape.expert_parallel_size
    if not (
        ep in (4, 8)
        and shape.data_parallel_size == shape.tensor_parallel_size == 1
        and shape.num_sms == 110
        and shape.sm_min_partition == shape.sm_partition_alignment == 8
        and shape.ep_same_numa_peer_count == 3
        and shape.ep_cross_numa_peer_count == ep - 4
    ):
        return "default"
    signature = [shape.hidden, shape.intermediate, shape.num_topk,
                 shape.num_total_experts]
    for row in _builtin_policy().get("weight_cache", {}).get("ranges", []):
        if (row["ep"] == ep and row["signature"] == signature
                and row["tokens"][0] <= shape.tokens_per_rank <= row["tokens"][1]):
            reference, _ = _select_config_and_dispatch_base(shape)
            if config == reference:
                return (row.get("activation", "default") if activation else "evict_first")
            break
    return "default"


@lru_cache(maxsize=256)
def select_config_and_dispatch(shape, overrides=None, *, use_tuned_defaults=True):
    """Resolve geometry first, then independently select both TMA cache hints.

    The policy is a performance hint shared by FC1, FC2 and tail drain. It
    changes neither synchronization nor tile/transport resource selection.
    """
    if use_tuned_defaults:
        overrides = _with_tuned_table(shape, overrides)
    explicit = overrides.weight_eviction_policy if overrides is not None else None
    if explicit not in (None, "default", "evict_first"):
        raise ValueError("weight_eviction_policy must be default or evict_first")
    activation_explicit = overrides.activation_eviction_policy if overrides is not None else None
    if activation_explicit not in (None, "default", "evict_last"):
        raise ValueError("activation_eviction_policy must be default or evict_last")
    push_explicit = overrides.push_dispatch if overrides is not None else None
    geometry_overrides = (replace(overrides, weight_eviction_policy=None, activation_eviction_policy=None,
                                  push_dispatch=None)
                          if overrides is not None else None)
    config, dispatch = _select_config_and_dispatch_base(
        shape, geometry_overrides, use_tuned_defaults=use_tuned_defaults)
    # The cache overlay is independent of tile and transport selection.
    policy = (explicit if explicit is not None else
              _select_weight_cache_policy(shape, config) if use_tuned_defaults
              else "default")
    activation_policy = (activation_explicit if activation_explicit is not None else
        _select_weight_cache_policy(shape, config, activation=True)
        if use_tuned_defaults else "default")
    push = push_explicit if push_explicit is not None else _select_push_dispatch(shape, config)
    return replace(config, weight_eviction_policy=policy,
                   activation_eviction_policy=activation_policy, push_dispatch=push), dispatch


def _select_push_dispatch(shape, config) -> bool:
    """Default push dispatch: decode-size p2p_direct worlds with peers."""
    return bool(
        config.kernel_comm_backend == "p2p_direct"
        and config.dispatch_pull_mode == "token_strided"
        and shape.expert_parallel_size > 1
        and shape.ep_same_numa_peer_count > 0
        and shape.tokens_per_rank <= PUSH_DISPATCH_MAX_TOKENS
    )


def select_megamoe_config(shape, overrides=None, *, use_tuned_defaults=True):
    """Select built-in defaults with explicit overrides and legacy fallback."""
    return select_config_and_dispatch(shape, overrides,
        use_tuned_defaults=use_tuned_defaults)[0]


__all__ = [
    "MegaMoEHeuristicInput",
    "MegaMoEHeuristicOverrides",
    "MegaMoEKernelConfig",
    "select_megamoe_config",
]
