"""Diagnostics only: per-call %globaltimer phase stamps (MEGA_SPLIT_PHASE_STAMPS=1).

A small int64 buffer per device holds one stamp per phase of the current call.
Kernels compiled while the switch is on bake its address in and update a slot
with an atomic min ("first ...") or max ("last ...") from one thread, so the cost
is a handful of global atomics per CTA. The --pr_timing loop copies the slots
after every round (outside the timed region) and resets them; the runner prints
per-rank medians relative to the stamp taken right after the start event.
Off by default; nothing is compiled in when the switch is off.
"""

import os

import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int64, dsl_user_op

ENABLED = os.environ.get("MEGA_SPLIT_PHASE_STAMPS", "0") == "1"

# (slot, name, reduction) - reduction "min" keeps the first stamp, "max" the last.
SLOTS = (
    ("t0", "max"),            # caller stream, right after the start event
    ("sender_in", "min"),
    ("sender_out", "max"),
    ("k1_entry", "min"),
    ("k1_entry_last", "max"),
    ("k1_bar0", "max"),       # CTA 0 dispatch warps: after the reset barrier
    ("k1_push", "max"),       # CTA 0: after pushing its input rows (push dispatch)
    ("k1_prep", "max"),       # CTA 0: after dispatch_prep
    ("k1_xwait", "max"),      # CTA 0: after the cross-NUMA commits arrived
    ("k1_dbar", "max"),       # CTA 0: after dispatch_barrier (counts known)
    ("k1_first_tile", "min"),
    ("k1_first_tile_last", "max"),
    ("k1_pull_done", "max"),  # last dispatch warp finished pulling
    ("k1_pull_exit_min", "min"),  # first dispatch warp out of the pull loop
    ("k1_row_min", "min"),    # first pulled row published
    ("k1_row_max", "max"),    # last pulled row published
    ("k1_exit", "max"),
    ("k2_entry", "min"),
    ("k2_first_tile", "min"),
    ("k2_last_tile", "max"),
    ("k2_exit", "max"),
    ("fwd_out", "max"),
    ("scatter_in", "min"),
    ("scatter_wait", "max"),
    ("scatter_out", "max"),
    ("t1", "max"),            # caller stream, right before the end event
    # Summed durations (ns) over pulled rows ("add"): reported as the mean per row.
    ("pull_rows", "add"),
    ("pull_choose_ns", "add"),
    ("pull_fetch_ns", "add"),     # TMA load issue .. TMA store retired
    ("pull_publish_ns", "add"),
)
SLOT = {name: i for i, (name, _) in enumerate(SLOTS)}
N_SLOTS = 32
_INT64_MAX = (1 << 63) - 1

_buffers = {}
_init = {}


def buffer() -> torch.Tensor:
    dev = torch.cuda.current_device()
    if dev not in _buffers:
        init = torch.zeros((N_SLOTS,), dtype=torch.int64)
        for i, (_, red) in enumerate(SLOTS):
            if red == "min":
                init[i] = _INT64_MAX
        _init[dev] = init.cuda()
        _buffers[dev] = _init[dev].clone()
    return _buffers[dev]


def addr(name: str) -> int:
    """Baked-in slot address (0 when the switch is off)."""
    if not ENABLED:
        return 0
    return int(buffer().data_ptr()) + 8 * SLOT[name]


def reset() -> None:
    dev = torch.cuda.current_device()
    buffer().copy_(_init[dev], non_blocking=True)


@dsl_user_op
def _atom(op: str, address: Int64, value: Int64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        f"red.relaxed.gpu.global.{op}.{'u64' if op == 'add' else 's64'} [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def stamp(name: str, now) -> None:
    """Inside a kernel (one thread): fold ``now`` into slot ``name``.

    A no-op at trace time when the switch is off."""
    if not ENABLED:
        return
    red = SLOTS[SLOT[name]][1]
    _atom(red, Int64(addr(name)), now)


_stamp_kernels = {}


def launch_host_stamp(name: str, stream: torch.cuda.Stream) -> None:
    """Enqueue a one-thread kernel writing slot ``name`` on ``stream``."""
    if not ENABLED:
        return
    key = (torch.cuda.current_device(), name)
    if key not in _stamp_kernels:
        import cutlass.cute as cute
        from cuda.bindings import driver as cuda

        from .split_timestamp import read_globaltimer

        slot_addr = addr(name)

        @cute.kernel
        def stamp_kernel():
            _atom("max", Int64(slot_addr), read_globaltimer())

        @cute.jit
        def launch(s: cuda.CUstream):
            stamp_kernel().launch(grid=[1, 1, 1], block=[32, 1, 1], stream=s)

        _stamp_kernels[key] = cute.compile(launch, cuda.CUstream(stream.cuda_stream))
    from cuda.bindings import driver as cuda

    _stamp_kernels[key](cuda.CUstream(stream.cuda_stream))


class Recorder:
    """Host side: snapshot after each timed round, report medians at the end."""

    def __init__(self):
        self.rows = []

    def snapshot(self) -> None:
        if not ENABLED:
            return
        self.rows.append(buffer().clone())
        reset()

    def report(self, rank: int, world: int, all_gather) -> None:
        if not ENABLED or not self.rows:
            return
        import statistics

        mine = torch.stack(self.rows)  # [iters, N_SLOTS]
        gathered = [torch.empty_like(mine) for _ in range(world)]
        all_gather(gathered, mine)
        if rank != 0:
            return
        skip = len(self.rows) // 5  # drop the first 20 % of rounds
        names = [n for n, red in SLOTS if red != "add"]
        n_time = len(names)
        adds = [(i, n) for i, (n, red) in enumerate(SLOTS) if red == "add"]
        if adds:
            print("---- pull per-row means (us; rows per call) ----", flush=True)
            for r, g in enumerate(gathered):
                g = g.cpu()[skip:]
                rows = g[:, adds[0][0]].double()
                ok = rows > 0
                if bool(ok.any()):
                    parts = [f"rows={float(rows[ok].median()):.0f}"] + [
                        f"{n}={float((g[ok, i].double() / rows[ok]).median()) / 1000.0:.2f}" for i, n in adds[1:]]
                    print(f"{r:4d} " + " ".join(parts), flush=True)
        print("---- phase stamps (us after t0, median over rounds; '-' = not stamped) ----", flush=True)
        print("rank " + " ".join(f"{n:>11s}" for n in names[1:]), flush=True)
        med_all = []
        for r, g in enumerate(gathered):
            g = g.cpu()[skip:]
            meds = []
            for i in range(1, n_time):
                vals = [
                    (int(row[i]) - int(row[0])) / 1000.0
                    for row in g
                    if 0 < int(row[i]) < _INT64_MAX and int(row[0]) > 0
                ]
                meds.append(statistics.median(vals) if vals else None)
            med_all.append(meds)
            print(f"{r:4d} " + " ".join(f"{m:11.1f}" if m is not None else f"{'-':>11s}" for m in meds), flush=True)
        # Cross-rank aligned view: per round subtract the earliest t0 of all ranks.
        print("---- phase stamps aligned to the earliest rank t0 (median) ----", flush=True)
        stacked = torch.stack([g.cpu()[skip:] for g in gathered])  # [world, iters, slots]
        t0min = stacked[:, :, 0].min(dim=0).values
        for r in range(world):
            meds = []
            for i in range(0, n_time):
                col = stacked[r, :, i]
                ok = (col > 0) & (col < _INT64_MAX)
                vals = ((col[ok] - t0min[ok]).double() / 1000.0).tolist()
                meds.append(statistics.median(vals) if vals else None)
            print(f"{r:4d} " + " ".join(f"{m:11.1f}" if m is not None else f"{'-':>11s}" for m in meds), flush=True)


__all__ = ["ENABLED", "SLOTS", "SLOT", "addr", "stamp", "launch_host_stamp", "Recorder", "buffer", "reset"]
