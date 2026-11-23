# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Whole-warp BF16 owner reduction with aligned 128-bit global transfers."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, Int32, Int64, dsl_user_op


@dsl_user_op
def load128(addr: Int64, *, loc=None, ip=None):
    value = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [addr.ir_value(loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l", has_side_effects=True,
        asm_dialect=0, loc=loc, ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), value, [i])) for i in range(4))


@dsl_user_op
def store128(addr: Int64, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None, [x.ir_value(loc=loc, ip=ip) for x in (addr, a, b, c, d)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )


@cute.jit
def reduce_whole_warp(route_output, peer_row, mapping, key, lane, chunk_base,
                      hidden: cutlass.Constexpr, pool_capacity: cutlass.Constexpr,
                      topk: cutlass.Constexpr, chunk_hidden: cutlass.Constexpr):
    offset_dtype: cutlass.Constexpr = (
        Int32 if pool_capacity * (hidden // 4) < (1 << 31) else Int64
    )
    offsets = cute.make_rmem_tensor((topk,), offset_dtype)
    for slot in cutlass.range_constexpr(topk):
        offsets[slot] = offset_dtype(mapping[key, slot]) * offset_dtype(hidden // 4)
    route_words = cute.recast_ptr(route_output.iterator, dtype=Int64)
    peer_words = cute.recast_ptr(peer_row.iterator, dtype=Int64)
    # Two independent fragments, explicit coalesced LDG/STG.128.
    values = cute.make_rmem_tensor((16, 2), cutlass.Float32)
    source = cute.make_rmem_tensor((16, 2, 2), cutlass.BFloat16)
    packed = cute.make_rmem_tensor((16,), cutlass.BFloat16)
    packed_words = cute.recast_tensor(packed, Int32)
    for offset in cutlass.range(0, chunk_hidden, 1024):
        values.fill(cutlass.Float32(0.0))
        for pair in cutlass.range_constexpr((topk + 1) // 2):
            source.fill(cutlass.BFloat16(0.0))
            for b in cutlass.range_constexpr(2):
                h = chunk_base + offset + b * 512 + lane * 8
                for item in cutlass.range_constexpr(2):
                    if cutlass.const_expr(pair * 2 + item < topk):
                        route_offset = offsets[pair * 2 + item]
                        if route_offset >= offset_dtype(0):
                            words = cute.recast_tensor(source[None, item, b], Int32)
                            for part in cutlass.range_constexpr(2):
                                data = load128((route_words + route_offset + offset_dtype((h + part * 256) // 4)).toint())
                                for i in cutlass.range_constexpr(4):
                                    words[part * 4 + i] = data[i]
            for b in cutlass.range_constexpr(2):
                for item in cutlass.range_constexpr(2):
                    if cutlass.const_expr(pair * 2 + item < topk):
                        for i in cutlass.range_constexpr(16):
                            values[i, b] += cutlass.Float32(source[i, item, b])
        for b in cutlass.range_constexpr(2):
            h = chunk_base + offset + b * 512 + lane * 8
            for i in cutlass.range_constexpr(16):
                packed[i] = values[i, b].to(cutlass.BFloat16)
            for part in cutlass.range_constexpr(2):
                store128((peer_words + (h + part * 256) // 4).toint(),
                         packed_words[part * 4], packed_words[part * 4 + 1],
                         packed_words[part * 4 + 2], packed_words[part * 4 + 3])
