"""NCCL symmetric-memory session for MegaMoE (replaces the NVSHMEM heap).

One session per process. It creates a dedicated EP NCCL communicator through
the host bridge, allocates symmetric arenas with ``ncclMemAlloc`` (via a
private ``torch.cuda.MemPool``), registers them as NCCL windows and exposes
LSA peer offsets in the same ``(local_base, peer_offsets)`` form the kernels
already consume through ``SymBuffer``.

* The P2P arena holds every buffer peers access by direct load/store. All
  ranks allocate it in the same order, so one per-peer offset applies to every
  sub-allocation, exactly like the NVSHMEM symmetric heap.
* An optional network arena (GIN) holds buffers that cross-NUMA peers write
  with GIN puts/signals. It is registered as a second window.

Peers outside the LSA team (cross-NUMA on PCIe hosts) have no P2P mapping;
their offset is 0 and must only be reached through GIN.
"""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .runtime import host_bridge, version_string

_ALIGN = 4096
_ARENA_ALIGN = 2 * 1024 * 1024

# NCCL settings applied only while this session's communicator is created
# (the private libnccl reads them at init). Users can pre-set any of them.
_GIN_ENV_DEFAULTS = {
    "NCCL_P2P_DISABLE": "0",  # LSA needs P2P even if the caller disabled it for torch
    "NCCL_GIN_ENABLE": "1",
    "NCCL_GIN_TYPE": "3",  # GDAKI
    "NCCL_GIN_GDAKI_NIC_HANDLER": "2",
    "NCCL_GIN_GDAKI_QP_DEPTH": "1024",
    "NCCL_IB_MERGE_NICS": "0",
}
_LSA_ENV_DEFAULTS = {"NCCL_P2P_DISABLE": "0", "NCCL_GIN_ENABLE": "0"}
_FORCED = ("NCCL_P2P_DISABLE",)


def _round_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


@contextmanager
def _scoped_env(values: Dict[str, str]):
    saved = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _physical_gpu_index(device: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    entries = [e.strip() for e in visible.split(",") if e.strip()]
    if entries and device < len(entries) and entries[device].isdigit():
        return int(entries[device])
    return device


class _Arena:
    def __init__(self, tensor: torch.Tensor, window_index: int, limit: int):
        self.tensor = tensor
        self.window_index = window_index
        self.limit = limit
        self.cursor = 0
        self.sizes: List[int] = []

    @property
    def base(self) -> int:
        return int(self.tensor.data_ptr())

    def contains(self, ptr: int) -> bool:
        return self.base <= ptr < self.base + self.cursor

    def zeros(self, shape: Sequence[int], dtype: torch.dtype, what: str) -> torch.Tensor:
        nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        size = _round_up(max(nbytes, 1), _ALIGN)
        if self.cursor + size > self.limit:
            raise MemoryError(
                f"MegaMoE NCCL {what} arena exhausted: need {self.cursor + size} bytes, "
                f"capacity {self.limit}. Set MEGAMOE_SYMMETRIC_BYTES to a larger value."
            )
        span = self.tensor[self.cursor:self.cursor + size]
        self.cursor += size
        self.sizes.append(nbytes)
        span.zero_()
        return span[:nbytes].view(dtype).reshape(tuple(shape))


class NcclSession:
    """Process-wide NCCL symmetric memory + optional GIN state."""

    def __init__(self, context, p2p_bytes: int, network: bool, lsa_required_mask: Optional[int] = None,
                 shadow=None, reserved_bytes: int = 0):
        self.context = context
        self.rank = context.ep_rank
        self.world = context.ep_size
        self.network = bool(network)
        self.bridge = host_bridge()
        lib = self.bridge.lib
        V, I, U = ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong

        uid = ctypes.create_string_buffer(128)
        if self.rank == 0:
            self.bridge.check(lib.megamoe_nccl_unique_id(uid))
        import torch.distributed as dist

        obj = [bytes(uid) if self.rank == 0 else None]
        group = getattr(context, "ep_control_group", None) or context.ep_group
        dist.broadcast_object_list(obj, src=context.ep_global_ranks[0], group=group)
        ctypes.memmove(uid, obj[0], 128)
        self._group = group

        env = dict(_GIN_ENV_DEFAULTS if self.network else _LSA_ENV_DEFAULTS)
        env = {k: (v if k in _FORCED else os.environ.get(k, v)) for k, v in env.items()}
        hca_map = os.environ.get("MEGAMOE_GIN_HCA_MAP")
        if self.network and hca_map and "NCCL_IB_HCA" not in os.environ:
            hcas = [h.strip() for h in hca_map.split(",")]
            physical = _physical_gpu_index(torch.cuda.current_device())
            env["NCCL_IB_HCA"] = "=" + hcas[physical] + ":1"
        self.gin_contexts = int(os.environ.get("MEGAMOE_GIN_CONTEXTS", "40"))
        gin_queue_depth = int(os.environ.get("MEGAMOE_GIN_QUEUE_DEPTH", "1024"))
        require_gdaki = 0 if os.environ.get("MEGAMOE_GIN_ALLOW_PROXY") == "1" else 1

        handle, state, info = V(), U(), (I * 6)()
        with _scoped_env(env):
            self.bridge.check(lib.megamoe_nccl_create(
                uid, self.rank, self.world, int(self.network), self.gin_contexts, gin_queue_depth,
                require_gdaki, ctypes.byref(handle), ctypes.byref(state), info))
        self.handle = handle
        self.state_ptr = int(state.value)
        self.info = list(info)
        self._env = env

        allocator = torch.cuda.memory.CUDAPluggableAllocator(
            str(self.bridge.path), "megamoe_nccl_pool_allocate", "megamoe_nccl_pool_free")
        self._allocator = allocator
        self._pool = torch.cuda.MemPool(allocator.allocator())

        capacity = int(os.environ.get("MEGAMOE_SYMMETRIC_BYTES", p2p_bytes))
        if capacity < p2p_bytes:
            raise ValueError(f"MEGAMOE_SYMMETRIC_BYTES={capacity} is smaller than required {p2p_bytes}")
        capacity = _round_up(capacity, _ARENA_ALIGN)
        # Shadow segments (p2p_split): segment 0 = own buffers + reserved
        # control region; segment k mirrors the k-th cross-NUMA peer.
        self.shadow = shadow
        if shadow is not None:
            if not network:
                raise ValueError("shadow segments need a GIN session")
            if shadow.world != self.world or shadow.rank != self.rank:
                raise ValueError("shadow layout does not match the EP world")
            from .shadow import segment_bytes

            # Segment 0 = own buffers + reserved control region; shadows only
            # mirror the peer's buffers (no reserved region), so the reserved
            # memory is not multiplied by the number of shadows.
            self.segment_bytes = capacity
            self.segment0_bytes = segment_bytes(capacity, reserved_bytes, _ARENA_ALIGN)
            self.reserved_offset = capacity
            self.reserved_bytes = self.segment0_bytes - capacity
            self.p2p = self._new_arena(self.segment0_bytes + capacity * shadow.num_cross, "p2p")
            self.p2p.limit = capacity  # the allocator never enters the reserved region or shadows
            if lsa_required_mask is None:
                lsa_required_mask = ((1 << self.world) - 1) & ~shadow.cross_mask
        else:
            self.segment_bytes = capacity
            self.segment0_bytes = capacity
            self.reserved_offset = capacity
            self.reserved_bytes = 0
            self.p2p = self._new_arena(capacity, "p2p")
        self.net: Optional[_Arena] = None
        self._network_scope = False

        lsa_required = self._lsa_required_mask() if lsa_required_mask is None else int(lsa_required_mask)
        offsets = []
        for peer in range(self.world):
            address = U()
            self.bridge.check(lib.megamoe_nccl_peer(self.handle, self.p2p.window_index, peer, ctypes.byref(address)))
            if not address.value and (lsa_required >> peer) & 1:
                raise RuntimeError(
                    f"EP peer {peer} is not LSA-reachable from rank {self.rank}; same-NUMA peers need "
                    "NCCL P2P (check NCCL_P2P_DISABLE / topology)")
            offsets.append(address.value - self.p2p.base if address.value else 0)
        # NCCL returns an LSA alias even for the local rank; keep the NVSHMEM
        # convention (offset 0 = direct local access) that the kernels assume.
        offsets[self.rank] = 0
        if shadow is not None:
            # Cross-NUMA peers resolve to the local shadow segment of that peer.
            for peer, segment in shadow.local_segments().items():
                offsets[peer] = self.segment_offset(segment)
        self.peer_offsets: Tuple[int, ...] = tuple(offsets)
        self.closed = False

    # -- topology -------------------------------------------------------
    def _lsa_required_mask(self) -> int:
        topology = getattr(self.context, "ep_transport_topology", None)
        everyone = (1 << self.world) - 1
        if topology is None or not self.network:
            return everyone
        return everyone & ~int(getattr(topology, "force_ibgda_peer_mask", 0))

    # -- arenas ---------------------------------------------------------
    def _new_arena(self, nbytes: int, what: str) -> _Arena:
        with torch.cuda.use_mem_pool(self._pool):
            tensor = torch.empty((nbytes,), dtype=torch.uint8, device="cuda")
        tensor.zero_()
        if tensor.data_ptr() % _ALIGN:
            raise RuntimeError("ncclMemAlloc returned an unaligned arena")
        self.bridge.check(self.bridge.lib.megamoe_nccl_register(self.handle, tensor.data_ptr(), nbytes))
        index = 0 if what == "p2p" else 1
        return _Arena(tensor, index, nbytes)

    def p2p_zeros(self, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        if self._network_scope:
            return self.network_zeros(shape, dtype)
        return self.p2p.zeros(shape, dtype, "p2p")

    @contextmanager
    def network_allocations(self):
        """Route symmetric allocations made inside this scope to the network arena."""

        if self.net is None:
            raise RuntimeError("prepare_network_arena() must run before network allocations")
        previous, self._network_scope = self._network_scope, True
        try:
            yield
        finally:
            self._network_scope = previous

    def prepare_network_arena(self, region_bytes: Sequence[int]) -> None:
        if not self.network:
            raise RuntimeError("network arena requested on an LSA-only session")
        if self.net is not None:
            raise RuntimeError("network arena already allocated")
        needed = sum(_round_up(max(n, 1), _ALIGN) for n in region_bytes)
        self.net = self._new_arena(_round_up(needed, _ARENA_ALIGN), "network")

    def network_zeros(self, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        if self.net is None:
            raise RuntimeError("network arena must be prepared before network buffers")
        return self.net.zeros(shape, dtype, "network")

    def peer_offsets_for(self, tensor: torch.Tensor) -> Tuple[int, Tuple[int, ...]]:
        ptr = int(tensor.data_ptr())
        if not self.p2p.contains(ptr):
            raise ValueError("peer offsets are only defined for tensors in the P2P arena")
        return ptr, self.peer_offsets

    def stream_barrier(self, stream) -> None:
        """Stream-ordered EP barrier (graph-capturable NCCL all-reduce on ep_group)."""

        import torch.distributed as dist

        if not hasattr(self, "_barrier_token"):
            self._barrier_token = torch.zeros((1,), dtype=torch.int32, device="cuda")
        with torch.cuda.stream(stream):
            dist.all_reduce(self._barrier_token, group=self.context.ep_group)

    # -- lifecycle ------------------------------------------------------
    def verify_symmetric(self) -> None:
        """All ranks must have allocated identical layouts in the same order."""

        import torch.distributed as dist

        mine = (self.p2p.cursor, self.p2p.sizes, self.net.cursor if self.net else 0, self.net.sizes if self.net else [])
        gathered = [None] * self.world
        dist.all_gather_object(gathered, mine, group=self._group)
        if any(g != gathered[0] for g in gathered):
            raise RuntimeError(f"non-symmetric NCCL allocation across EP ranks: {gathered}")

    def segment_offset(self, segment: int) -> int:
        """Byte offset of a segment inside the P2P arena (same on every rank)."""
        return 0 if segment == 0 else self.segment0_bytes + (segment - 1) * self.segment_bytes

    def reserved_view(self, offset: int, nbytes: int) -> torch.Tensor:
        """Bytes [offset, offset+nbytes) of segment 0's reserved control region."""
        if offset < 0 or offset + nbytes > self.reserved_bytes:
            raise ValueError(f"reserved range [{offset}, {offset + nbytes}) exceeds {self.reserved_bytes} bytes")
        start = self.reserved_offset + offset
        return self.p2p.tensor[start:start + nbytes]

    def segment_base(self, segment: int) -> int:
        return self.p2p.base + self.segment_offset(segment)

    def describe(self) -> dict:
        return dict(
            nccl_version=version_string(self.bridge.version), nccl_library=str(self.bridge.nccl.library),
            host_bridge=str(self.bridge.path), network=self.network, gin_type=self.info[0],
            gin_contexts=self.info[1], gin_connections=self.info[2], rank=self.rank, world=self.world,
            p2p_capacity=self.p2p.limit, p2p_used=self.p2p.cursor,
            segment_bytes=self.segment_bytes, reserved_bytes=self.reserved_bytes,
            shadow_segments=(self.shadow.local_segments() if self.shadow is not None else {}),
            network_capacity=self.net.limit if self.net else 0, network_used=self.net.cursor if self.net else 0,
            peer_offsets=list(self.peer_offsets), env={k: v for k, v in self._env.items()})

    def close(self) -> None:
        if self.closed:
            return
        torch.cuda.synchronize()
        self.bridge.check(self.bridge.lib.megamoe_nccl_destroy(self.handle))
        self.closed = True


_SESSION: Optional[NcclSession] = None


def initialize(context, p2p_bytes: int, network: bool, lsa_required_mask: Optional[int] = None,
               shadow=None, reserved_bytes: int = 0) -> NcclSession:
    """``lsa_required_mask``: peers that must be LSA-reachable (default: every
    peer, or every non-network peer when ``network``)."""
    global _SESSION
    if _SESSION is not None and not _SESSION.closed:
        raise RuntimeError("one MegaMoE NCCL session per process")
    _SESSION = NcclSession(context, p2p_bytes, network, lsa_required_mask, shadow, reserved_bytes)
    out = os.environ.get("MEGAMOE_NCCL_RECEIPT_DIR")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        Path(out, f"nccl_session_rank{_SESSION.rank}.json").write_text(json.dumps(_SESSION.describe(), indent=1))
    return _SESSION


def current() -> Optional[NcclSession]:
    return _SESSION if _SESSION is not None and not _SESSION.closed else None


def close() -> None:
    if _SESSION is not None:
        _SESSION.close()


def preload_kernels(compiled_kernels) -> None:
    """Load every function of each compiled library before persistent launches
    (replaces nvshmem.core.library_init for kernels that use the device bridge)."""

    from cuda.bindings import driver as cuda

    def checked(result):
        if int(result[0]):
            raise RuntimeError(f"kernel preload failed: {result}")
        return result[1] if len(result) == 2 else result[1:]

    loaded = set()
    for compiled in compiled_kernels:
        if compiled is None:
            continue
        lib = int(compiled.library)
        if lib in loaded:
            continue
        loaded.add(lib)
        module = checked(cuda.cuLibraryGetModule(cuda.CUlibrary(lib)))
        count = checked(cuda.cuModuleGetFunctionCount(module))
        for fun in checked(cuda.cuModuleEnumerateFunctions(count, module)):
            checked(cuda.cuFuncLoad(fun))


__all__ = ["NcclSession", "close", "current", "initialize", "preload_kernels"]
