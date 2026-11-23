"""Measured anchor table: complete launch configurations per (model, topology).

``tuned_table.json`` holds, for each tuned signature (hidden, intermediate,
top-k, experts) and EP topology, the configuration that
``tools/sm120_w4a8_tuning/tune_table.py`` selected at a set of token anchors.
A token count between anchors uses the anchor nearest in log space (the
boundary is the geometric midpoint), so every token count is covered; counts
outside the anchor range use the closest end anchor.

Topologies (110-SM SM120, DP1/TP1, 8-SM partition granularity):

* ``ep4_1numa``: EP4 inside one NUMA node (p2p_direct).
* ``ep8_2numa``: EP8 as 2 x 4 GPUs on two NUMA nodes. An entry's ``backend``
  is ``p2p_split`` (LSA + cross-NUMA shadow transport) or ``hybrid``
  (LSA + GIN rings); its knobs are valid only for that backend.
* ``ep16_2x2numa``: EP16 over two hosts of 2 x 4 GPUs (P2P only inside a
  NUMA node, every other peer through GIN); ``p2p_split`` only.

The table is a performance hint. Explicit overrides always win, and shapes,
topologies or backends it does not cover fall back to the rule-based
selector.
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


# (topology, same-domain peers, network peers) of the real NUMA-split layouts.
_SPLIT_LAYOUTS = {8: ("ep8_2numa", 3, 4), 16: ("ep16_2x2numa", 3, 12)}


def topology_of(shape) -> Optional[str]:
    """Topology key of a heuristic input, or None if no table can apply."""
    if (shape.data_parallel_size, shape.tensor_parallel_size, shape.num_sms,
            shape.sm_min_partition, shape.sm_partition_alignment) != (1, 1, 110, 8, 8):
        return None
    ep, same, cross = (shape.expert_parallel_size, shape.ep_same_numa_peer_count,
                       shape.ep_cross_numa_peer_count)
    split = getattr(shape, "ep_numa_split", False)
    if ep == 4 and (same, cross) == (3, 0) and not split:
        return "ep4_1numa"
    if ep in _SPLIT_LAYOUTS:
        name, real_same, real_cross = _SPLIT_LAYOUTS[ep]
        # p2p_split folds the network peers into one P2P group (the runner
        # passes ep-1 same-domain / 0 network peers with ep_numa_split).
        if (same, cross) == ((ep - 1, 0) if split else (real_same, real_cross)):
            return name
    return None


def nearest_entry(signature: Tuple[int, int, int, int], topology: str, tokens: int) -> Optional[dict]:
    for table in _load().get("tables", []):
        if tuple(table["signature"]) == tuple(signature) and table["topology"] == topology:
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
    entry = nearest_entry(signature, topology, shape.tokens_per_rank)
    if entry is None:
        return None
    backend = entry["knobs"].get("backend")
    if topology != "ep4_1numa":
        want = "p2p_split" if getattr(shape, "ep_numa_split", False) else "hybrid"
        if backend != want:
            return None  # knobs were tuned for the other backend
    return entry


# Rule for untabulated signatures, from the dsv41/glm53 measurements: the
# cross-NUMA shadow transport wins at decode sizes (-8..-19% at T16/T128) but
# its forwarder does not scale (2-8x slower than hybrid from T4096 on).
SPLIT_RULE_MAX_TOKENS = 128
SPLIT_MAX_EXPERTS_PER_RANK = 64  # forwarder completion mask is 64-bit


def preferred_ep8_backend(shape) -> Optional[str]:
    """``p2p_split`` / ``hybrid`` for a real NUMA-split EP shape (EP8 on 2 x 4
    GPUs or EP16 on two such hosts, network peers not yet folded): the nearest
    table anchor's backend, else the rule above; None otherwise. EP16 uses
    ``p2p_split`` only (the hybrid rings are not validated across hosts)."""
    topology = topology_of(shape)
    if topology is None or topology == "ep4_1numa" or getattr(shape, "ep_numa_split", False):
        return None
    signature = (shape.hidden, shape.intermediate, shape.num_topk, shape.num_total_experts)
    entry = nearest_entry(signature, topology, shape.tokens_per_rank)
    if entry is not None:
        return entry["knobs"].get("backend")
    experts_per_rank = shape.num_total_experts // shape.expert_parallel_size
    if topology == "ep16_2x2numa":
        return "p2p_split" if experts_per_rank <= SPLIT_MAX_EXPERTS_PER_RANK else None
    if (shape.tokens_per_rank <= SPLIT_RULE_MAX_TOKENS
            and experts_per_rank <= SPLIT_MAX_EXPERTS_PER_RANK):
        return "p2p_split"
    return "hybrid"


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
                "dispatch_chunk_tokens"):
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


__all__ = ["lookup", "nearest_entry", "overrides_for", "preferred_ep8_backend", "topology_of"]
