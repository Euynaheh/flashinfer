"""CuTeDSL wrappers for the NCCL device bridge (GIN + LSA).

Transport kernels call these instead of ``nvshmem.core.device.cute``. Semantics
match what the ring/owner-plan protocols already rely on:

* ``put_block``/``put_nbi_block``: payload copy. Peers in the object's
  ``hybrid_force_ibgda_mask`` go through GIN (thread 0 issues, CTA barriers
  around it); every other peer is an LSA whole-CTA copy into the peer window.
* ``commit_block``/``commit_nbi_block``: an 8-byte commit record published as
  a strong VA signal-add (ordered after earlier puts on the same context).
  Each record source address must have exactly one writer.
* ``quiet``: local source retirement on the CTA's context plus a system fence.
  Remote consumption is proven by the protocol's own ACKs/credits.
* ``read_signal32``/``wait_signal32``/``wait_owner_signal``: consumers of
  NIC-written VA signals must use NCCL's acquire API, not plain loads.

Each transport object is ``bind``-ed to a role, which selects a disjoint block
of eight GIN contexts (``MEGAMOE_GIN_CONTEXTS`` must cover 5 roles x 8).

The device-state pointer is folded into the kernel IR as a constant. CuTeDSL
keys its JIT cache on the IR hash, so another process (different pointer)
recompiles these small transport kernels instead of reusing a stale binary.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64

from . import session as _session

_ROLE_CONTEXT_BASE = {
    "dispatch_plan": 0,
    "dispatch_tx": 8,
    "dispatch_rx": 16,
    "return_tx": 24,
    "return_rx": 32,
}
_CONTEXTS_PER_ROLE = 8
_SLOTS_PER_ROLE = 512


def required_gin_contexts() -> int:
    return max(_ROLE_CONTEXT_BASE.values()) + _CONTEXTS_PER_ROLE


def bind(obj, role: str, session=None) -> None:
    """Attach GIN state/context selection to a transport kernel object.

    ``session`` defaults to the process session; FlashInfer workspaces pass
    the session that owns the transport's windows.
    """

    nccl = _session.current() if session is None else session
    obj.gin_enabled = bool(nccl is not None and nccl.network)
    obj.gin_state_ptr = nccl.state_ptr if nccl is not None else 0
    obj.gin_context_base = _ROLE_CONTEXT_BASE[role]
    if obj.gin_enabled and nccl.gin_contexts < required_gin_contexts():
        raise RuntimeError(f"NCCL session has {nccl.gin_contexts} GIN contexts; need {required_gin_contexts()}")


def link_libraries() -> str:
    from .runtime import device_bitcode

    return device_bitcode()


_put_commit = cute.ffi(name="megamoe_gin_put_commit",
                       params_types=[Int64, Int32, Int32, Int32, Int64, Int64, Int64, Int64, Int64])
_put = cute.ffi(name="megamoe_gin_put", params_types=[Int64, Int32, Int32, Int32, Int64, Int64, Int64, Int32, Int32])
_flush = cute.ffi(name="megamoe_gin_flush", params_types=[Int64, Int32, Int32])
_lsa_put = cute.ffi(name="megamoe_lsa_put", params_types=[Int64, Int32, Int64, Int64, Int64])
_fence = cute.ffi(name="megamoe_system_fence", params_types=[])
_read_signal = cute.ffi(name="megamoe_gin_read_signal", params_types=[Int64, Int64], return_type=Int64)
_wait_signal = cute.ffi(name="megamoe_gin_wait_signal", params_types=[Int64, Int64, Int64, Int32])


@cute.jit
def _context(obj):
    return Int32(obj.gin_context_base) + cute.arch.block_idx()[0] % Int32(_CONTEXTS_PER_ROLE)


@cute.jit
def _slot(obj):
    return (
        Int32(obj.gin_context_base // _CONTEXTS_PER_ROLE * _SLOTS_PER_ROLE)
        + cute.arch.block_idx()[0]
        + cute.arch.grid_dim()[0] * cute.arch.block_idx()[1]
    )


@cute.jit
def fence():
    _fence()


@cute.jit
def put(obj, dst, src, peer, *, blocking: cutlass.Constexpr, signal_record: cutlass.Constexpr = False):
    nbytes = Int64(cute.size(src)) * Int64(src.element_type.width // 8)
    if (Int32(obj.hybrid_force_ibgda_mask) & (Int32(1) << peer)) != Int32(0):
        cute.arch.barrier(barrier_id=0, number_of_threads=128)
        if cute.arch.thread_idx()[0] == Int32(0):
            _put(Int64(obj.gin_state_ptr), _context(obj), _slot(obj), peer, dst.iterator.toint(),
                 src.iterator.toint(), nbytes, Int32(1 if blocking else 0), Int32(1 if signal_record else 0))
        cute.arch.barrier(barrier_id=0, number_of_threads=128)
    else:
        _lsa_put(Int64(obj.gin_state_ptr), peer, dst.iterator.toint(), src.iterator.toint(), nbytes)


@cute.jit
def put_block(obj, dst, src, peer):
    put(obj, dst, src, peer, blocking=True)


@cute.jit
def put_nbi_block(obj, dst, src, peer):
    put(obj, dst, src, peer, blocking=False)


@cute.jit
def commit_block(obj, dst, src, peer):
    put(obj, dst, src, peer, blocking=True, signal_record=True)


@cute.jit
def commit_nbi_block(obj, dst, src, peer):
    put(obj, dst, src, peer, blocking=False, signal_record=True)


@cute.jit
def put_commit_nbi_block(obj, dst, src, signal_dst, signal_src, peer):
    """GIN payload put fused with the strong VA commit of the 8-byte record at
    ``signal_src`` (NCCL delivers the signal after the payload); GIN peers only."""
    nbytes = Int64(cute.size(src)) * Int64(src.element_type.width // 8)
    cute.arch.barrier(barrier_id=0, number_of_threads=128)
    if cute.arch.thread_idx()[0] == Int32(0):
        _put_commit(Int64(obj.gin_state_ptr), _context(obj), _slot(obj), peer, dst.iterator.toint(),
                    src.iterator.toint(), nbytes, signal_dst.iterator.toint(), signal_src.iterator.toint())
    cute.arch.barrier(barrier_id=0, number_of_threads=128)


@cute.jit
def quiet(obj):
    if cute.arch.thread_idx()[0] == Int32(0):
        _flush(Int64(obj.gin_state_ptr), _context(obj), _slot(obj))
    _fence()


@cute.jit
def read_signal32(obj, addr):
    value = _read_signal(Int64(obj.gin_state_ptr), addr - addr % Int64(8))
    return Int32(value >> ((addr % Int64(8)) * Int64(8)))


@cute.jit
def wait_signal32(obj, addr, expected):
    _wait_signal(Int64(obj.gin_state_ptr), addr, Int64(expected), Int32(32))


@cute.jit
def wait_owner_signal(obj, ptr, epoch):
    _wait_signal(Int64(obj.gin_state_ptr), ptr.toint(), Int64(epoch), Int32(64))


__all__ = [
    "bind", "commit_block", "commit_nbi_block", "fence", "link_libraries", "put_block", "put_commit_nbi_block",
    "put_nbi_block",
    "quiet", "read_signal32", "required_gin_contexts", "wait_owner_signal", "wait_signal32",
]
