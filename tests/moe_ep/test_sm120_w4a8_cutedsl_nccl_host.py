"""Host-only contracts of the SM120 W4A8 NCCL transport and EP8 p2p_split."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from flashinfer.moe_ep.shims.sm120.split_cutedsl import (
    MegaMoESm120W4A8Config,
    bootstrap_paths,
    select_comm_backend,
    workspace_compile_buckets,
)

bootstrap_paths()

from moe_sm120_mxfp4mxfp8_split import tuned_table  # noqa: E402
from moe_sm120_mxfp4mxfp8_split.api import (  # noqa: E402
    MegaMoEProblemSpec,
    select_compile_spec,
)
from moe_sm120_mxfp4mxfp8_split.heuristic import MegaMoEHeuristicInput  # noqa: E402
from moe_sm120_mxfp4mxfp8_split.runtime.gpu_topology import (  # noqa: E402
    GpuTopologyEntry,
    derive_ep_transport_topology,
)
from moe_sm120_mxfp4mxfp8_split.transport.nccl import (  # noqa: E402
    runtime as nccl_runtime,
)
from moe_sm120_mxfp4mxfp8_split.transport.nccl.shadow import (  # noqa: E402
    ShadowLayout,
)
from moe_sm120_mxfp4mxfp8_split.transport.xnuma_layout import (  # noqa: E402
    XnumaLayout,
)

# (hidden, post-SwiGLU intermediate, top-k, experts) of the tuned models.
DSV41 = (5120, 2304, 6, 384)
GLM53 = (4096, 2048, 8, 288)


def _topology(numa_nodes: tuple[int, ...], rank: int):
    peers = [
        GpuTopologyEntry(
            global_rank=index,
            local_rank=index,
            pci_bus_id=f"0000:{index + 1:02x}:00.0",
            numa_node=node,
            host="host0",
        )
        for index, node in enumerate(numa_nodes)
    ]
    return derive_ep_transport_topology(peers, ep_rank=rank)


def _config(model, *, rank: int, world_size: int, knobs=None, experts=None):
    hidden, intermediate, topk, num_experts = model
    return MegaMoESm120W4A8Config(
        rank=rank,
        world_size=world_size,
        max_tokens_per_rank=128,
        num_topk=topk,
        num_total_experts=experts or num_experts,
        hidden=hidden,
        intermediate=intermediate,
        knobs=knobs,
    )


def _shape(model, tokens: int, *, ep: int, same: int, cross: int, split=False):
    hidden, intermediate, topk, experts = model
    return MegaMoEHeuristicInput(
        tokens_per_rank=tokens,
        hidden=hidden,
        intermediate=2 * intermediate,
        num_topk=topk,
        num_total_experts=experts,
        data_parallel_size=1,
        tensor_parallel_size=1,
        expert_parallel_size=ep,
        ep_same_numa_peer_count=same,
        ep_cross_numa_peer_count=cross,
        num_sms=110,
        sm_min_partition=8,
        sm_partition_alignment=8,
        ep_numa_split=split,
    )


def _spec(model, tokens: int, *, ep: int, same: int, cross: int, split: bool):
    hidden, intermediate, topk, experts = model
    return select_compile_spec(
        problem=MegaMoEProblemSpec(
            tokens_per_rank=tokens,
            num_topk=topk,
            num_total_experts=experts,
            hidden=hidden,
            intermediate=2 * intermediate,
            expert_parallel_size=ep,
            expert_parallel_rank=0,
        ),
        ep_same_numa_peer_count=same,
        ep_cross_numa_peer_count=cross,
        num_sms=110,
        sm_min_partition=8,
        sm_partition_alignment=8,
        ep_numa_split=split,
    )


TWO_BY_FOUR = (0, 0, 0, 0, 1, 1, 1, 1)


@pytest.mark.parametrize("rank", range(8))
def test_two_numa_ep8_topology_and_shadow_segments(rank: int) -> None:
    topology = _topology(TWO_BY_FOUR, rank)
    assert (topology.same_numa_peer_count, topology.cross_numa_peer_count) == (3, 4)
    cross = [peer for peer in range(8) if TWO_BY_FOUR[peer] != TWO_BY_FOUR[rank]]
    assert topology.force_ibgda_peer_mask == sum(1 << peer for peer in cross)
    shadow = ShadowLayout.from_topology(topology, rank)
    assert shadow.num_segments == 5
    assert shadow.cross_mask == topology.force_ibgda_peer_mask
    # Every source computes the segment its owner uses for it.
    for peer, segment in shadow.local_segments().items():
        assert ShadowLayout.from_topology(topology, peer).remote_segments()[rank] == (
            segment
        )


def test_single_numa_ep4_topology_has_no_network_peers() -> None:
    topology = _topology((1, 1, 1, 1), 2)
    assert (topology.same_numa_peer_count, topology.cross_numa_peer_count) == (3, 0)
    assert topology.force_ibgda_peer_mask == 0


def test_comm_backend_follows_topology() -> None:
    ep4 = _topology((0, 0, 0, 0), 1)
    assert select_comm_backend(_config(DSV41, rank=1, world_size=4), ep4) == (
        "p2p_direct"
    )
    ep8 = _topology(TWO_BY_FOUR, 5)
    for knobs in (None, {"comm_backend": "p2p_split"}):
        config = _config(GLM53, rank=5, world_size=8, knobs=knobs)
        assert select_comm_backend(config, ep8) == "p2p_split"


def test_comm_backend_rejects_unsupported_requests() -> None:
    ep4 = _topology((0, 0, 0, 0), 0)
    ep8 = _topology(TWO_BY_FOUR, 0)
    with pytest.raises(ValueError, match="cross-NUMA"):
        select_comm_backend(
            _config(DSV41, rank=0, world_size=4, knobs={"comm_backend": "p2p_split"}),
            ep4,
        )
    with pytest.raises(ValueError, match="p2p_split"):
        select_comm_backend(
            _config(DSV41, rank=0, world_size=8, knobs={"comm_backend": "p2p_direct"}),
            ep8,
        )
    with pytest.raises(ValueError, match="unsupported"):
        select_comm_backend(
            _config(
                DSV41, rank=0, world_size=8, knobs={"comm_backend": "nvshmem_hybrid"}
            ),
            ep8,
        )
    # The forwarder tracks completed experts in a 64-bit mask.
    with pytest.raises(ValueError, match="experts per rank"):
        select_comm_backend(
            _config(DSV41, rank=0, world_size=8, experts=1024),
            ep8,
        )
    # Shadow segments need the same cross-NUMA peer count on every rank.
    with pytest.raises(ValueError, match="same cross-NUMA"):
        select_comm_backend(
            _config(DSV41, rank=0, world_size=4, experts=128),
            _topology((0, 0, 0, 1), 0),
        )


@pytest.mark.parametrize("model", (DSV41, GLM53), ids=("dsv41", "glm53"))
@pytest.mark.parametrize("tokens", (16, 32, 64, 128))
def test_ep8_decode_prefers_p2p_split(model, tokens: int) -> None:
    real = _shape(model, tokens, ep=8, same=3, cross=4)
    assert tuned_table.topology_of(real) == "ep8_2numa"
    assert tuned_table.preferred_ep8_backend(real) == "p2p_split"
    folded = _shape(model, tokens, ep=8, same=7, cross=0, split=True)
    assert tuned_table.topology_of(folded) == "ep8_2numa"
    assert tuned_table.lookup(folded)["knobs"]["backend"] == "p2p_split"
    # The real (unfolded) view is the hybrid backend FlashInfer does not run.
    assert tuned_table.lookup(real) is None


@pytest.mark.parametrize("model", (DSV41, GLM53), ids=("dsv41", "glm53"))
def test_ep4_anchor_table_lookup(model) -> None:
    shape = _shape(model, 16, ep=4, same=3, cross=0)
    assert tuned_table.topology_of(shape) == "ep4_1numa"
    entry = tuned_table.lookup(shape)
    assert entry is not None and entry["tokens"] == 16
    # A folded EP8 view is not an EP4 topology.
    assert tuned_table.lookup(_shape(model, 16, ep=8, same=7, cross=0)) is None


def _expected_table_fields(entry) -> dict:
    expected = tuned_table.overrides_for(entry["knobs"])
    for key in ("comm_backend", "token_back_mode", "weight_eviction_policy"):
        expected.pop(key, None)
    return expected


@pytest.mark.parametrize("model", (DSV41, GLM53), ids=("dsv41", "glm53"))
@pytest.mark.parametrize("tokens", (16, 128))
def test_p2p_split_compile_spec_applies_ep8_anchor(model, tokens: int) -> None:
    spec = _spec(model, tokens, ep=8, same=7, cross=0, split=True)
    entry = tuned_table.lookup(_shape(model, tokens, ep=8, same=7, cross=0, split=True))
    assert entry["tokens"] == tokens
    for field, value in _expected_table_fields(entry).items():
        assert getattr(spec.kernel, field) == value, field
    assert spec.kernel.comm_backend == "p2p_direct"
    assert not spec.kernel.rank_local_combine
    assert spec.kernel.k2_warps == 8


def test_xnuma_region_covers_every_bucket() -> None:
    hidden, _intermediate, topk, experts = DSV41
    buckets = workspace_compile_buckets(512)
    assert buckets == (7, 16, 32, 64, 128, 168, 256, 320, 384, 512)
    region = XnumaLayout(
        world=8,
        n_cross=4,
        experts_per_rank=experts // 8,
        tokens_per_rank=max(buckets),
        topk=topk,
        hidden=hidden,
    )
    for bucket in buckets:
        layout = dataclasses.replace(region, tokens_per_rank=bucket)
        assert layout.reserved_bytes <= region.reserved_bytes
        assert layout.record_bytes <= region.record_bytes
    # Return rows: n_cross * fsplit * T * topk slots of 16 B + one BF16 row.
    offsets = region.offsets
    assert offsets["stage"] - offsets["ret"] == 4 * 2 * 512 * topk * (16 + 2 * hidden)


def test_workspace_buckets_respect_capacity() -> None:
    assert workspace_compile_buckets(7) == (7,)
    assert workspace_compile_buckets(100) == (7, 16, 32, 64, 100)
    assert workspace_compile_buckets(8192)[-2:] == (4096, 8192)


def _fake_nccl(root: Path, version: int = 23007, lib: str = "lib") -> Path:
    (root / "include").mkdir(parents=True)
    (root / "include" / "nccl.h").write_text(f"#define NCCL_VERSION_CODE {version}\n")
    (root / "include" / "nccl_device.h").write_text("\n")
    (root / lib).mkdir()
    (root / lib / "libnccl.so.2").write_bytes(b"")
    return root


@pytest.fixture
def nccl_discovery(monkeypatch):
    from flashinfer.moe_ep.shims.sm120.split_cutedsl import comm

    monkeypatch.setattr(comm, "_NCCL_DISCOVERY_CONFIGURED", False)
    monkeypatch.setattr(nccl_runtime, "PREFERRED_ROOTS", [])
    monkeypatch.setattr(nccl_runtime, "DEFAULT_CACHE_ROOT", None)
    monkeypatch.delenv("MEGAMOE_NCCL_ROOT", raising=False)
    nccl_runtime.find_nccl.cache_clear()
    yield comm
    nccl_runtime.find_nccl.cache_clear()


def test_flashinfer_nccl_discovery_precedes_megamoe_root(
    nccl_discovery, monkeypatch, tmp_path
) -> None:
    from flashinfer.moe_ep.shims.sm120.cake_mxfp4 import cake_jit

    preferred = _fake_nccl(tmp_path / "flashinfer_nccl", lib="lib64")
    fallback = _fake_nccl(tmp_path / "megamoe_nccl")
    monkeypatch.setattr(
        cake_jit,
        "nccl_installation",
        lambda: (preferred / "include", preferred / "lib64" / "libnccl.so.2"),
    )
    monkeypatch.setenv("MEGAMOE_NCCL_ROOT", str(fallback))
    nccl_discovery.configure_nccl_discovery()
    found = nccl_runtime.find_nccl()
    assert found.root == preferred.resolve()
    assert found.library == (preferred / "lib64" / "libnccl.so.2").resolve()
    assert found.version == 23007
    assert nccl_runtime.DEFAULT_CACHE_ROOT is not None


def test_megamoe_nccl_root_is_the_fallback(
    nccl_discovery, monkeypatch, tmp_path
) -> None:
    from flashinfer.moe_ep.shims.sm120.cake_mxfp4 import cake_jit

    def unavailable():
        raise RuntimeError("NCCL 2.30.7 development files were not found")

    fallback = _fake_nccl(tmp_path / "megamoe_nccl")
    monkeypatch.setattr(cake_jit, "nccl_installation", unavailable)
    monkeypatch.setenv("MEGAMOE_NCCL_ROOT", str(fallback))
    nccl_discovery.configure_nccl_discovery()
    assert nccl_runtime.PREFERRED_ROOTS == []
    assert nccl_runtime.find_nccl().root == fallback.resolve()


def test_old_nccl_is_rejected(nccl_discovery, monkeypatch, tmp_path) -> None:
    old = _fake_nccl(tmp_path / "old_nccl", version=22903)
    monkeypatch.setattr(nccl_runtime, "PREFERRED_ROOTS", [("test", old)])
    monkeypatch.setenv("MEGAMOE_NCCL_ROOT", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="2.30.7"):
        nccl_runtime.find_nccl()


def test_prebuilt_device_bitcode_matches_vendored_sources() -> None:
    import hashlib
    import json

    stamp = nccl_runtime.prebuilt_bitcode_path(nccl_runtime.MIN_NCCL_VERSION)
    meta = json.loads(stamp.with_suffix(".json").read_text())
    assert meta["nccl_version"] == nccl_runtime.MIN_NCCL_VERSION
    for name, digest in meta["sources"].items():
        source = nccl_runtime.HERE / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, name


def test_runtime_requirements_do_not_need_nvshmem(monkeypatch) -> None:
    from flashinfer.moe_ep import BootstrapConfig
    from flashinfer.moe_ep.core.runtime import (
        NVSHMEM,
        TORCH_DIST,
        sm120_w4a8_cutedsl_runtime_requirements,
    )

    bootstrap = BootstrapConfig(world_size=8, rank=0, auto_bootstrap=False)
    monkeypatch.delenv("MEGA_NO_DIST", raising=False)
    requirements = sm120_w4a8_cutedsl_runtime_requirements(bootstrap)
    assert requirements == frozenset({TORCH_DIST})
    assert NVSHMEM not in requirements
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    assert sm120_w4a8_cutedsl_runtime_requirements(bootstrap) == frozenset()


def test_local_topology_is_single_numa() -> None:
    from flashinfer.moe_ep.shims.sm120.split_cutedsl.runtime import (
        local_ep_topology,
    )

    topology = local_ep_topology(1)
    assert isinstance(topology, SimpleNamespace)
    assert (topology.same_numa_peer_count, topology.cross_numa_peer_count) == (0, 0)
    assert (
        select_comm_backend(_config(DSV41, rank=0, world_size=1), topology)
        == "p2p_direct"
    )


@pytest.mark.parametrize("model", (DSV41, GLM53), ids=("dsv41", "glm53"))
def test_push_dispatch_follows_decode_buckets(model) -> None:
    from moe_sm120_mxfp4mxfp8_split.heuristic import PUSH_DISPATCH_MAX_TOKENS

    for tokens in (16, 32, 64, 128):
        expected = tokens <= PUSH_DISPATCH_MAX_TOKENS
        assert (
            _spec(
                model, tokens, ep=4, same=3, cross=0, split=False
            ).kernel.push_dispatch
            == expected
        )
        split = _spec(model, tokens, ep=8, same=7, cross=0, split=True)
        assert split.kernel.push_dispatch == expected


@pytest.mark.parametrize(
    "ep,same,split", ((8, 7, True), (4, 3, False)), ids=("ep8_split", "ep4")
)
def test_launch_guard_rejects_k2_residency_without_tail_reclaim(
    ep, same, split
) -> None:
    # k2_launch_factor=2 without tail reclaim hangs at the first launch with an
    # 8-warp K2 N16 tile (standalone check_launch_safety).
    from moe_sm120_mxfp4mxfp8_split.heuristic import MegaMoEHeuristicOverrides

    hidden, intermediate, topk, experts = DSV41
    kwargs = dict(
        problem=MegaMoEProblemSpec(
            tokens_per_rank=16,
            num_topk=topk,
            num_total_experts=experts,
            hidden=hidden,
            intermediate=2 * intermediate,
            expert_parallel_size=ep,
            expert_parallel_rank=0,
        ),
        ep_same_numa_peer_count=same,
        ep_cross_numa_peer_count=0,
        num_sms=110,
        sm_min_partition=8,
        sm_partition_alignment=8,
        ep_numa_split=split,
    )
    bad = MegaMoEHeuristicOverrides(
        k2_tile=(64, 16, 128), k2_warps=8, k2_launch_factor=2, k2_tail_reclaim=False
    )
    with pytest.raises(ValueError, match="k2_tail_reclaim"):
        select_compile_spec(overrides=bad, **kwargs)
    ok = select_compile_spec(
        overrides=dataclasses.replace(bad, k2_tail_reclaim=True), **kwargs
    )
    assert (ok.kernel.k2_launch_factor, ok.kernel.k2_tail_reclaim) == (2, True)


def test_push_shadow_stride_covers_the_staged_inputs() -> None:
    from flashinfer.moe_ep.shims.sm120.split_cutedsl import runtime as shim_runtime
    from flashinfer.moe_ep.shims.sm120.split_cutedsl.comm import page_bytes

    config = _config(DSV41, rank=0, world_size=4)
    shapes = shim_runtime._input_shapes(config)
    span = sum(
        page_bytes(n * d.itemsize) for n, d in ((s[0] * s[1], d) for s, d in shapes[:4])
    )
    assert shim_runtime._push_stride(config) == span
    # The output buffer stays outside the pushed span.
    assert len(shapes) == 5
