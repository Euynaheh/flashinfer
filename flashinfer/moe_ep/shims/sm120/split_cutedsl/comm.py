"""NCCL symmetric-memory helpers for the W4A8 split kernel.

Every multi-rank workspace owns one ``NcclSession`` of the vendored transport
(``transport/nccl/session.py``): a dedicated EP communicator created through
the NCCL 2.30.7 device API, one symmetric P2P arena registered as an NCCL
window and, for ``p2p_split``, GIN state plus one shadow segment per
cross-NUMA peer. Single-rank workspaces (and ``MEGA_NO_DIST=1``) use ordinary
CUDA allocations with zero peer offsets.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import torch

_NCCL_DISCOVERY_CONFIGURED = False


def _no_dist() -> bool:
    return bool(int(os.environ.get("MEGA_NO_DIST", "0")))


def ensure_not_capturing(operation: str) -> None:
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"{operation} cannot run during CUDA Graph capture; call warmup() "
            "collectively on every EP rank before capture."
        )


def configure_nccl_discovery() -> None:
    """Register FlashInfer's NCCL discovery with the vendored transport.

    The SM120 Cake backend's discovery (``NCCL_HOME``, then the
    ``nvidia-nccl-cu13`` wheel, NCCL 2.30.7 only) is tried first;
    ``MEGAMOE_NCCL_ROOT`` and the transport's own wheel lookup remain the
    fallbacks. The host bridge is cached under FlashInfer's workspace unless
    ``MEGAMOE_NCCL_CACHE`` is set.
    """

    global _NCCL_DISCOVERY_CONFIGURED
    if _NCCL_DISCOVERY_CONFIGURED:
        return
    from moe_sm120_mxfp4mxfp8_split.transport.nccl import runtime as nccl_runtime

    try:
        from flashinfer.moe_ep.shims.sm120.cake_mxfp4.cake_jit import (
            nccl_installation,
        )

        include, library = nccl_installation()
        root = Path(include).parent
        if library.parent.parent == root:
            nccl_runtime.PREFERRED_ROOTS.append(("flashinfer", root))
    except (ImportError, OSError, RuntimeError):
        pass
    if nccl_runtime.DEFAULT_CACHE_ROOT is None:
        try:
            from flashinfer.jit.env import FLASHINFER_WORKSPACE_DIR

            nccl_runtime.DEFAULT_CACHE_ROOT = (
                Path(FLASHINFER_WORKSPACE_DIR) / "megamoe_nccl"
            )
        except ImportError:
            pass
    _NCCL_DISCOVERY_CONFIGURED = True


def create_session(
    *,
    rank: int,
    world_size: int,
    control_group: Any,
    topology: Any,
    p2p_bytes: int,
    shadow: Any = None,
    reserved_bytes: int = 0,
):
    """Create the workspace's NCCL symmetric-memory session (collective).

    ``shadow`` (a ``ShadowLayout``) enables GIN and one shadow segment per
    cross-NUMA peer for ``p2p_split``; otherwise every EP peer must be
    LSA-reachable (``p2p_direct``).
    """

    ensure_not_capturing("NCCL workspace initialization")
    import torch.distributed as dist

    configure_nccl_discovery()
    from moe_sm120_mxfp4mxfp8_split.transport.nccl.session import NcclSession

    source = dist.get_global_rank(control_group, 0) if control_group is not None else 0
    context = SimpleNamespace(
        ep_rank=rank,
        ep_size=world_size,
        ep_global_ranks=(source,),
        ep_group=control_group,
        ep_control_group=control_group,
        ep_transport_topology=topology,
    )
    signatures: list = [None] * world_size
    signature = (p2p_bytes, shadow is not None, reserved_bytes)
    dist.all_gather_object(signatures, signature, group=control_group)
    if any(value != signature for value in signatures):
        raise ValueError(
            f"SM120 W4A8 NCCL workspace plans differ across EP ranks: {signatures}"
        )
    return NcclSession(
        context,
        p2p_bytes,
        network=shadow is not None,
        shadow=shadow,
        reserved_bytes=reserved_bytes,
    )


def sym_zeros(
    shape: Tuple[int, ...], dtype: torch.dtype, *, session: Any = None
) -> torch.Tensor:
    if session is None:
        return torch.zeros(shape, dtype=dtype, device="cuda")
    ensure_not_capturing("NCCL workspace suballocation")
    tensor = session.p2p_zeros(shape, dtype)
    tensor._mega_peer_session = session
    return tensor


def sym_byte_view(
    shape: Tuple[int, ...], dtype: torch.dtype, *, session: Any = None
) -> tuple[torch.Tensor, torch.Tensor]:
    nbytes = math.prod(shape) * dtype.itemsize
    root = sym_zeros((nbytes,), torch.uint8, session=session)
    return root.view(dtype).reshape(shape), root


def page_bytes(nbytes: int) -> int:
    """Bytes one symmetric suballocation consumes (4 KiB granularity)."""

    return (max(int(nbytes), 1) + 4095) // 4096 * 4096


def compute_peer_offsets(
    symmetric_tensor: torch.Tensor,
    world_size: int,
    session: Optional[Any] = None,
) -> tuple[int, tuple[int, ...]]:
    """``(local_base, per-peer byte offsets)`` of the workspace arena."""

    if session is None:
        return int(symmetric_tensor.data_ptr()), tuple(0 for _ in range(world_size))
    if session.closed or session.world != world_size:
        raise ValueError("tensor does not belong to the active EP workspace")
    return session.peer_offsets_for(symmetric_tensor)


def close_session(session: Optional[Any]) -> None:
    """Deregister the arena and destroy the EP communicator."""

    if session is not None and not session.closed:
        session.close()


__all__ = [
    "close_session",
    "compute_peer_offsets",
    "configure_nccl_discovery",
    "create_session",
    "ensure_not_capturing",
    "page_bytes",
    "sym_byte_view",
    "sym_zeros",
]
