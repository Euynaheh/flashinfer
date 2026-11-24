"""Measured anchor table: complete launch configurations per (model, topology, device).

``tuned_table.json`` holds, for each tuned signature (hidden, intermediate,
top-k, experts), EP topology and device, the configuration that
``tools/sm120_w4a8_tuning/tune_table.py`` selected at a set of token anchors.
A token count between anchors uses the anchor nearest in log space (the
boundary is the geometric midpoint), so every token count is covered; counts
outside the anchor range use the closest end anchor.

Device: a table's optional ``device`` field is ``[num_sms, sm_min_partition,
sm_partition_alignment]`` of the Green Context SM resource it was tuned on;
tables without it are the original 110-SM RTX PRO 5000 tables ``[110, 8, 8]``.
A table applies only to its own device (SM partitions are absolute counts).

Topologies (DP1/TP1):

* ``ep4_1numa``: EP4 inside one NUMA node (p2p_direct).
* ``ep8_2numa``: EP8 as 2 x 4 GPUs on two NUMA nodes. An entry's ``backend``
  is ``p2p_split`` (LSA + cross-NUMA shadow transport) or ``hybrid``
  (LSA + GIN rings, only in the original 110-SM tables); its knobs are valid
  only for that backend.
* ``ep8_1domain``: EP8 as one P2P domain (``MEGAMOE_P2P_DOMAIN=host``: every
  pair of the host LSA-mapped, p2p_direct over all eight GPUs).
* ``ep16_2x2numa``: EP16 over two hosts of 2 x 4 GPUs (P2P only inside a
  NUMA node, every other peer through GIN); ``p2p_split`` only.
* ``ep16_2x1domain``: EP16 over two hosts that are one P2P domain each
  (``MEGAMOE_P2P_DOMAIN=host``; the other host's 8 peers through GIN);
  ``p2p_split`` only.

The table is a performance hint. Explicit overrides always win, and shapes,
topologies, devices or backends it does not cover fall back to the
rule-based selector.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

_PATH = Path(__file__).with_name("tuned_table.json")


@lru_cache(maxsize=1)
def _load() -> dict:
    if not _PATH.exists():
        return dict(schema=1, tables=[])
    data = json.loads(_PATH.read_text())
    if data.get("schema") != 1:
        raise ValueError(f"{_PATH.name}: unsupported schema {data.get('schema')}")
    return data


LEGACY_DEVICE = (110, 8, 8)  # tables without a ``device`` field

# (EP size, same-domain peers, network peers) of the real (unfolded) layouts.
_REAL_LAYOUTS = {
    (4, 3, 0): "ep4_1numa",
    (8, 3, 4): "ep8_2numa",
    (8, 7, 0): "ep8_1domain",
    (16, 3, 12): "ep16_2x2numa",
    (16, 7, 8): "ep16_2x1domain",
}
# p2p_split folds the network peers into one P2P group (the runner passes
# ep-1 same-domain / 0 network peers with ep_numa_split and the real domain
# size in ep_split_domain_size): (EP size, domain size) -> topology.
_SPLIT_LAYOUTS = {
    (8, 4): "ep8_2numa",
    (16, 4): "ep16_2x2numa",
    (16, 8): "ep16_2x1domain",
}
# Topologies whose table entries carry a backend (p2p_split / hybrid).
_SPLIT_TOPOLOGIES = frozenset(_SPLIT_LAYOUTS.values())


def device_of(shape) -> Tuple[int, int, int]:
    return (int(shape.num_sms), int(shape.sm_min_partition), int(shape.sm_partition_alignment))


def topology_of(shape) -> Optional[str]:
    """Topology key of a heuristic input, or None if no table can apply."""
    if (shape.data_parallel_size, shape.tensor_parallel_size) != (1, 1):
        return None
    ep, same, cross = (shape.expert_parallel_size, shape.ep_same_numa_peer_count,
                       shape.ep_cross_numa_peer_count)
    if getattr(shape, "ep_numa_split", False):
        if (same, cross) != (ep - 1, 0):
            return None
        return _SPLIT_LAYOUTS.get((ep, getattr(shape, "ep_split_domain_size", 4)))
    return _REAL_LAYOUTS.get((ep, same, cross))


def nearest_entry(signature: Tuple[int, int, int, int], topology: str, tokens: int,
                  device: Tuple[int, int, int] = LEGACY_DEVICE) -> Optional[dict]:
    for table in _load().get("tables", []):
        if (tuple(table["signature"]) == tuple(signature) and table["topology"] == topology
                and tuple(table.get("device", LEGACY_DEVICE)) == tuple(device)):
            entries = table["entries"]
            if not entries:
                return None
            return min(entries, key=lambda e: (abs(math.log(tokens) - math.log(e["tokens"])), e["tokens"]))
    return None


def lookup(shape) -> Optional[dict]:
    """The anchor entry for this shape and its effective backend, or None."""
    topology = topology_of(shape)
    if topology is None or shape.tokens_per_rank <= 0:
        return None
    signature = (shape.hidden, shape.intermediate, shape.num_topk, shape.num_total_experts)
    entry = nearest_entry(signature, topology, shape.tokens_per_rank, device_of(shape))
    if entry is None:
        return None
    backend = entry["knobs"].get("backend")
    if topology in _SPLIT_TOPOLOGIES:
        want = "p2p_split" if getattr(shape, "ep_numa_split", False) else "hybrid"
        if backend != want:
            return None  # knobs were tuned for the other backend
    return entry


# Default for a cross-NUMA EP world without a table anchor: p2p_split at every
# token count. The GIN-ring hybrid transport is never chosen by a rule; only an
# explicit ``comm_backend`` or a measured (110-SM) table anchor selects it.
SPLIT_MAX_EXPERTS_PER_RANK = 64  # forwarder completion mask is 64-bit


def preferred_ep8_backend(shape) -> Optional[str]:
    """Backend for a real cross-NUMA EP shape (network peers not yet folded):
    the nearest table anchor's backend (``p2p_split`` or, in the original
    110-SM EP8 tables, ``hybrid``), else ``p2p_split``. None when the shape is
    not a cross-NUMA world (already folded, or no network peers) or when
    p2p_split cannot serve it (more than 64 experts per rank)."""
    if getattr(shape, "ep_numa_split", False) or shape.ep_cross_numa_peer_count <= 0:
        return None
    topology = topology_of(shape)
    if topology is not None:
        signature = (shape.hidden, shape.intermediate, shape.num_topk, shape.num_total_experts)
        entry = nearest_entry(signature, topology, shape.tokens_per_rank, device_of(shape))
        if entry is not None and entry["knobs"].get("backend") is not None:
            return entry["knobs"]["backend"]
    if shape.num_total_experts // shape.expert_parallel_size > SPLIT_MAX_EXPERTS_PER_RANK:
        return None
    return "p2p_split"


def overrides_for(knobs: dict) -> dict:
    """Map table knobs to ``MegaMoEHeuristicOverrides`` field values."""
    out = {}
    if knobs.get("backend") == "hybrid":
        out.update(comm_backend="nvshmem_hybrid", token_back_mode="reuse_dispatch_warps")
    if knobs.get("k1_n") is not None:
        out["k1_tile"] = (64, int(knobs["k1_n"]), 128)
    if knobs.get("k2_n") is not None:
        out["k2_tile"] = (64, int(knobs["k2_n"]), 128)
    for key in ("k1_ctas_per_sm", "k2_launch_factor", "k1_stages", "weight_prefetch_ahead",
                "dispatch_chunk_tokens", "k1_t0_prefetch"):
        if knobs.get(key) is not None:
            out[key] = int(knobs[key])
    if knobs.get("k2_warps") is not None:
        out["k2_warps"] = int(knobs["k2_warps"])
    if knobs.get("k2_ready_queue_bundle") is not None:
        out["ready_queue_bundle"] = int(knobs["k2_ready_queue_bundle"])
    for key in ("k2_tail_reclaim", "fc1_utility_postprocess", "push_dispatch"):
        if knobs.get(key) is not None:
            out[key] = knobs[key] == "on"
    if knobs.get("weight_eviction_policy") is not None:
        out["weight_eviction_policy"] = knobs["weight_eviction_policy"]
    sms = knobs.get("sms")
    if sms is not None:
        if len(sms) == 2:
            sms = (sms[0], 0, 0, sms[1])
        out.update(zip(("k1_sms", "tx_sms", "rx_sms", "k2_sms"), (int(v) for v in sms)))
    return out


__all__ = ["LEGACY_DEVICE", "device_of", "lookup", "nearest_entry", "overrides_for",
           "preferred_ep8_backend", "topology_of"]
