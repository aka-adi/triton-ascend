# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
SIMD vector-register primitives for the engram_fused_weight decode chains.

Primitives build AVE dialect ops (``ave.hir.*``) through ``cpp/tla_builder.cc``
bindings; explicitly masked arithmetic also selects zero on inactive lanes:

============================  ==========================================
API                           AVE op
============================  ==========================================
``vlds``                      ``ave.hir.vload``
``vsts``                      ``ave.hir.masked_store``
``vsstb``                     ``ave.hir.store_with_stride``
``vextf``                     ``ave.hir.vextf``
``vintlv``                    ``ave.hir.vintlv``
``vbitcast``                  ``vector.bitcast``
``vmul``                      ``ave.hir.vmul`` (mask = ``ave.hir.pge <ALL>``)
``vadd`` / ``vsub``           ``ave.hir.vadd`` / ``ave.hir.vsub``
``vor`` / ``vxor``            ``ave.hir.vor`` / ``ave.hir.vxor``
``vmins``                     ``ave.hir.vmins`` (vector-scalar min)
``vmod`` / ``vdiv``           ``ave.hir.vmod`` / ``ave.hir.vdiv``
============================  ==========================================

``vadd``/``vsub``/``vor``/``vxor``/``vmins`` accept an optional ``mask``
(default all-true) and a predication ``mode`` (``MODE.ZEROING``, the
default; ``MODE.MERGING`` is rejected until the backend exposes the
``.m`` intrinsic variants).  ``vsstb`` writes a full register to UB one
32-byte block at a time with consecutive blocks ``stride`` elements
apart (compile-time stride, multiple of the 32B block size).

The two supported decode chains (see ``TileKernels``
``quant/cast_back_asc.py`` / ``engram/engram_fused_weight_asc.py``):

- BF16: ``vlds(dist='UNPK_B16')`` -> ``vextf(f32, part='part_even')`` -> ``vmul``
- FP8 E4M3FN: ``vlds(dist='UNPK4_B8', elem=fp8e4nv)`` -> ``vextf(f32)`` -> ``vmul``
- store: ``vsts(..., dist='NORM_B32')``

A vector register always spans 2048 bits, so a load of an ``N``-bit element
type yields ``2048 // N`` lanes; conversions keep the lane count.  Dist/part/
pp/pge strings are validated by :mod:`simd_semantic`.
"""

import enum

import triton.language as tl
from .core import buffer, buffer_access
from triton.language.core import builtin, constexpr, tensor, _unwrap_if_constexpr

from . import simd_semantic as ss

__all__ = [
    "MODE",
    "vlds",
    "vsts",
    "vsstb",
    "vextf",
    "vintlv",
    "vdintlv",
    "vbitcast",
    "vmul",
    "vadd",
    "vsub",
    "vdiv",
    "vor",
    "vxor",
    "vmins",
    "vmod",
    "vfma",
    "vsel",
    "vneg",
    "vsqrt",
    "vrsqrt",
    "vreduce",
    "vcmp",
    "vbroadcast",
    "vfull",
    "vsplat",
    "vtruncf",
    "vreduce_sum",
    "vreduce_max",
    "vreduce_min",
    "vci",
    "pge",
    "membar",
]


class MODE(enum.Enum):
    """Predication mode of a masked SIMD op.

    ``ZEROING`` (default): masked-off lanes of the result are set to 0.

    ``MERGING``: masked-off lanes keep the destination's previous value
    (``.m`` variants).  Not yet implemented -- the AVE ops carry no mode
    attribute and the HIVMAVEToAVEIntrin dispatch only emits ``.x``
    variants; :func:`check_mask_mode` rejects it with guidance.
    """

    ZEROING = "zeroing"
    MERGING = "merging"

def _lanes_for(tag):
    """Full-register lane count; i64/u64 span 4096 bits (2x VL_BITS)."""
    bits = _ELEM_SPECS[tag]
    if bits == 64:
        return 2 * ss.VL_BITS // 64
    return ss.lanes_for(bits)

# Precise element tags accepted by the AVE builder (cpp/tla_builder.cc
# resolveSimdElementType).  Each tag maps to exactly one MLIR element type, so
# the frontend's declared dtype and the handle's IR type stay identical.
# ``fp8e4nv`` is the Triton spelling of the E4M3FN format (1-4-3, bias 7); the
# builder resolves it to Float8E4M3FNType.
_ELEM_SPECS = {
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "bf16": 16,
    "f16": 16,
    "f32": 32,
    "fp8e4nv": 8,
}


# short dtype name spellings accepted by bitcast(), resolved to canonical
# triton dtype names.  Kept as strings (not dtype objects) so this module does
# not evaluate tl.<dtype> at import time (extension/__init__ imports simd while
# triton.language is still being initialized).
_DTYPE_ALIASES = {
    "f32": "fp32", "fp32": "fp32",
    "bf16": "bf16",
    "fp8e4nv": "fp8e4nv", "float8e4nv": "fp8e4nv",
}


def _elem_tag(dtype_):
    """Return the precise element tag for a supported SIMD element dtype.

    Foreign E4M3FN spellings are normalized to ``fp8e4nv``.  Anything not used
    by the decode chains (f16/i16/u16/i32/u32/f64/…) is rejected instead of
    being silently mapped to a wrong bit width.
    """
    name = str(dtype_)
    tag = {
        "fp8e4nv": "fp8e4nv", "float8e4nv": "fp8e4nv",
        "fp8e4m3fn": "fp8e4nv", "float8e4m3fn": "fp8e4nv",
        "bf16": "bf16", "bfloat16": "bf16",
        "fp16": "f16", "float16": "f16",
        "fp32": "f32", "float32": "f32", "f32": "f32",
        "int8": "i8", "i8": "i8", "uint8": "i8",
        "int16": "i16", "int32": "i32", "int64": "i64",
        "uint16": "i16", "uint32": "i32", "uint64": "i64",
    }.get(name)
    if tag is None:
        raise ValueError(
            f"unsupported simd element dtype {dtype_}; supported: "
            f"i8/i16/i32/i64/bf16/f16/f32/fp8e4nv")
    return tag


def _resolve_base_and_indices(ptr_or_buffer, offsets, _semantic):
    """Return a memref base and scalar index handles from a TLA buffer."""
    if isinstance(ptr_or_buffer, buffer):
        base = ptr_or_buffer.handle
    else:
        raise TypeError(
            "simd load/store expects a tla.buffer, "
            f"got {type(ptr_or_buffer).__name__}")
    indices = []
    for off in offsets:
        if isinstance(off, tensor):
            indices.append(off.handle)
        else:
            off = _semantic.to_tensor(off if isinstance(off, constexpr)
                                      else constexpr(off))
            indices.append(off.handle)
    return base, indices


def _all_mask(lanes, _semantic):
    handle = _semantic.builder.create_simd_pge("ALL", lanes)
    return tl.tensor(handle, tl.block_type(tl.int1, [lanes]))


def _check_mode(mode, what):
    """Resolve a predication mode via simd_semantic (ZEROING only)."""
    return ss.check_mask_mode(_unwrap_if_constexpr(mode), what)


def _check_mask(mask, lanes, what):
    """Validate a user-supplied predicate tensor for a lanes-wide op."""
    lanes = _unwrap_if_constexpr(lanes)
    if (not isinstance(mask, tensor) or mask.dtype != tl.int1
            or list(mask.shape) != [lanes]):
        raise ValueError(
            f"{what} mask must be an int1 tensor of {lanes} lanes, "
            f"got {getattr(mask, 'type', type(mask).__name__)}")


def _check_vector(src, what):
    if not isinstance(src, tensor) or len(src.shape) != 1:
        raise ValueError(f"{what} expects a one-dimensional vector register")
    ss.check_total_bits(src.shape[0], _ELEM_SPECS[_elem_tag(src.dtype)], what)


def _zero_inactive(handle, src, mask, _semantic):
    """Express ZEROING in SSA instead of relying on inactive .x lanes."""
    tag = _elem_tag(src.dtype)
    scalar = _semantic.builder.create_simd_scalar(tag, 0.0)
    zero = _semantic.builder.create_simd_vsplat(scalar, tag, src.shape[0])
    return _semantic.builder.create_simd_vsel(mask.handle, handle, zero)


# Element tags accepted by the binary/binary-scalar intrinsic dispatch
# (HIVMAVEToAVEIntrin BinaryRegistry / BinaryVectorScalarRegistry).  i64 has
# no dispatch slot for any binary op and fp8 none for add/sub/or/xor/mins.
_BINARY_TAGS = frozenset(["i8", "i16", "i32", "bf16", "f16", "f32"])
_VEC_SCALAR_TAGS = frozenset(["i8", "i16", "i32", "f16", "f32"])


def _binary(kind, src0, src1, mask, mode, _semantic):
    """Shared path of the masked elementwise binary ops."""
    _check_mode(mode, kind)
    _check_vector(src0, kind)
    _check_vector(src1, kind)
    if src0.type != src1.type:
        raise ValueError(f"{kind} requires same-typed operands, "
                         f"got {src0.type} and {src1.type}")
    tag = _elem_tag(src0.dtype)
    if tag not in _BINARY_TAGS:
        raise ValueError(
            f"{kind} does not support {src0.dtype} registers; supported: "
            f"i8/i16/i32/bf16/f16/f32")
    lanes = src0.shape[0]
    explicit_mask = mask is not None
    if mask is None:
        mask = _all_mask(lanes, _semantic)
    else:
        _check_mask(mask, lanes, kind)
    handle = _semantic.builder.create_simd_binary(kind, src0.handle,
                                                  src1.handle, mask.handle)
    if explicit_mask:
        handle = _zero_inactive(handle, src0, mask, _semantic)
    return tl.tensor(handle, src0.type)


@builtin
def vlds(ptr, offsets=None, *, dist="NORM", elem=None, _semantic=None) -> tensor:
    """
    Vector-register load from UB (``ave.hir.vload``).

    :param ptr: ``tla.buffer`` in UB, or a subscripted element range
        ``buf[i0, i1, ...]`` whose indices become the offsets
    :param offsets: index values (int / constexpr / tensor), one per memref
        dim; omit when ``ptr`` is a subscripted buffer
    :param dist: load distribution; defaults to ``NORM``
    :param elem: result element dtype; defaults to the buffer element type.
        Use ``tl.float8e4nv`` (E4M3FN) for the packed FP8 profile.
    :return: vector register tensor with ``2048 // elem_bits`` lanes
    """
    if isinstance(ptr, buffer_access):
        ptr, offsets = ptr.buffer, list(ptr.indices)
    elif offsets is None:
        raise TypeError("vlds needs an index: pass buf[i, ...] or explicit offsets")
    if not isinstance(ptr, buffer):
        raise TypeError("simd.vlds expects a tla.buffer")
    elem_ty = ptr.dtype if elem is None else elem
    dist = _unwrap_if_constexpr(dist)
    elem_tag = _elem_tag(elem_ty)
    lanes = _lanes_for(elem_tag)
    ss.check_dist(dist, ss.LOAD_DISTS, "load")
    base, indices = _resolve_base_and_indices(ptr, offsets, _semantic)
    handle = _semantic.builder.create_simd_vlds(base, indices, elem_tag,
                                                 lanes, dist)
    return tl.tensor(handle, tl.block_type(elem_ty, [lanes]))


@builtin
def vsts(ptr, offsets_or_val, val=None, *, dist=None, mask=None, _semantic=None):
    """
    Vector-register masked store to UB (``ave.hir.masked_store``).

    :param ptr: ``tla.buffer`` in UB, or a subscripted element range
        ``buf[i0, i1, ...]`` whose indices become the offsets
    :param offsets_or_val: index values (int / constexpr / tensor), one per
        memref dim; or the vector register tensor to store when ``ptr`` is a
        subscripted buffer
    :param val: vector register tensor to store
    :param dist: store distribution; defaults to ``NORM_B32``
    :param mask: predicate tensor; defaults to an all-true ``pge <ALL>`` mask
    """
    if isinstance(ptr, buffer_access):
        ptr, offsets = ptr.buffer, list(ptr.indices)
        val = offsets_or_val
    elif val is None:
        raise TypeError("vsts needs a value: vsts(buf[i, ...], val)")
    else:
        offsets = offsets_or_val
    if not isinstance(ptr, buffer):
        raise TypeError("simd.vsts expects a tla.buffer")
    val_lanes = val.shape[0]
    dist = _unwrap_if_constexpr(dist)
    if dist is None:
        dist = "NORM_B32"
    ss.check_dist(dist, ss.STORE_DISTS, "vsts")
    if mask is None:
        mask = _all_mask(val_lanes, _semantic)
    base, indices = _resolve_base_and_indices(ptr, offsets, _semantic)
    _semantic.builder.create_simd_vsts(base, indices, val.handle, mask.handle,
                                         dist)


@builtin
def vsstb(src, dst, stride, mask=None, update=False, _semantic=None):
    """
    Strided masked store to UB (``ave.hir.store_with_stride`` ->
    ``intr.hivm.vsstb.v{N}{dt}``, the vsstb "stride store with broadcast"
    instruction).

    The register is written one 32-byte block at a time, with consecutive
    blocks landing ``stride`` elements apart in the destination buffer;
    masked-off lanes are not written. The caller must provide storage for
    every active lane's destination address. This operation does not itself
    pack converted values or change their register layout.

    :param src: vector register tensor to store (f32/f16/bf16/fp8/i8-i32;
        i64 is not supported by the instruction)
    :param dst: two-dimensional UB buffer, or ``ub[i, j]``. A bare buffer
        starts at index [0, 0]. The current backend stride lowering requires
        two indices.
    :param stride: compile-time element distance between consecutive 32B
        blocks; must be a positive multiple of the block size
        (``256 // elem_bits`` elements).  For bank-conflict avoidance
        prefer byte strides of ``(512*N + 32)`` (bishengir stride_align.md)
    :param mask: predicate tensor; defaults to an all-true ``pge <ALL>``
    :param update: post-update pointer return; not yet implemented (the
        intrinsic's postUpdateEnable operand is hardcoded to 0 by the
        current lowering)
    """
    if isinstance(dst, buffer_access):
        dst, offsets = dst.buffer, list(dst.indices)
    elif isinstance(dst, buffer):
        offsets = [0] * len(dst.shape)
    else:
        raise TypeError(
            "vsstb needs a subscripted buffer: vsstb(val, ub[i, ...], stride)")
    if not isinstance(dst, buffer):
        raise TypeError("simd.vsstb expects a tla.buffer")
    if dst.space != "ub":
        raise ValueError(f"vsstb targets UB buffers, got space {dst.space!r}")
    if len(dst.shape) != 2 or len(offsets) != 2:
        raise ValueError(
            "vsstb currently requires a two-dimensional UB buffer and two indices")
    update = _unwrap_if_constexpr(update)
    if type(update) is not bool:
        raise ValueError("vsstb update must be a compile-time bool")
    if update:
        raise NotImplementedError(
            "vsstb update=True (post-update pointer return) is not "
            "supported yet: the lowering hardcodes postUpdateEnable=0 and "
            "the AVE op has no result value")
    _check_vector(src, "vsstb")
    if src.dtype != dst.dtype:
        raise ValueError("vsstb source and destination element dtypes must match")
    tag = _elem_tag(src.dtype)
    if tag == "i64":
        raise ValueError("vsstb does not support i64 registers")
    stride = ss.check_vsstb_stride(_unwrap_if_constexpr(stride),
                                   _ELEM_SPECS[tag])
    if mask is None:
        mask = _all_mask(src.shape[0], _semantic)
    else:
        _check_mask(mask, src.shape[0], "vsstb")
    base, indices = _resolve_base_and_indices(dst, offsets, _semantic)
    _semantic.builder.create_simd_vsstb(base, indices, stride, src.handle,
                                        mask.handle)


@builtin
def vextf(src, target_dtype, mask=None, part="part_even", _semantic=None) -> tensor:
    """
    Widening float conversion (``ave.hir.vextf``), e.g. bf16/f8e4m3fn -> f32.

    The result is a full F32 register: ``2048 // 32 == 64`` lanes.  The ``part``
    selector (``part_even``/``part_odd``) picks which half/quarter of the
    source register the intrinsic reads; it does not change the result width.

    :param src: source vector register tensor
    :param target_dtype: destination element dtype (f32 for the decode chains)
    :param mask: predicate tensor; defaults to an all-true mask of result width
    :param part: ``part_even`` / ``part_odd`` sub-register selector
    """
    part = ss.check_part(_unwrap_if_constexpr(part))
    out_tag = _elem_tag(target_dtype)
    if out_tag != "f32":
        raise ValueError(f"vextf (cast) only widens to f32, got {target_dtype}")
    out_lanes = ss.lanes_for(_ELEM_SPECS[out_tag])
    if mask is None:
        mask = _all_mask(out_lanes, _semantic)
    handle = _semantic.builder.create_simd_vcvt("vextf", src.handle,
                                                mask.handle, out_tag, part)
    return tl.tensor(handle, tl.block_type(target_dtype, [out_lanes]))


@builtin
def vintlv(src1, src2, _semantic=None):
    """
    Pairwise interleave of two vector registers (``ave.hir.vintlv``,
    ``Layout_Change::SPARSE``), returning the two result registers.
    """
    if src1.type != src2.type:
        raise ValueError("interleave requires same-typed sources, "
                         f"got {src1.type} and {src2.type}")
    res1, res2 = _semantic.builder.create_simd_vintlv(src1.handle, src2.handle)
    ty = src1.type
    return tl.tensor(res1, ty), tl.tensor(res2, ty)


@builtin
def vdintlv(src1, src2, _semantic=None):
    """
    Pairwise deinterleave of two vector registers (``ave.hir.vdintlv``,
    ``Layout_Change::SPARSE``), the inverse of :func:`vintlv`.
    """
    if src1.type != src2.type:
        raise ValueError("deinterleave requires same-typed sources, "
                         f"got {src1.type} and {src2.type}")
    res1, res2 = _semantic.builder.create_simd_vdintlv(src1.handle, src2.handle)
    ty = src1.type
    return tl.tensor(res1, ty), tl.tensor(res2, ty)


@builtin
def vbitcast(src, dtype_str, _semantic=None) -> tensor:
    """
    Bit reinterpretation of a vector register (``vector.bitcast``).

    :param dtype_str: destination element type, e.g. ``'f32'`` or ``'bf16'``
    :return: tensor with ``src_lanes * src_bits / dst_bits`` lanes
    """
    target_name = _DTYPE_ALIASES.get(dtype_str, dtype_str)
    target = tl.dtype(target_name)
    src_tag = _elem_tag(src.dtype)
    src_bits = _ELEM_SPECS[src_tag]
    out_tag = _elem_tag(target)
    out_bits = _ELEM_SPECS[out_tag]
    total_bits = src.shape[0] * src_bits
    if total_bits % out_bits != 0:
        raise ValueError(
            f"bitcast {src.shape[0]}x{src_bits} bits does not divide {out_bits}")
    out_lanes = total_bits // out_bits
    ss.check_total_bits(out_lanes, out_bits, "bitcast result")
    handle = _semantic.builder.create_simd_bitcast(src.handle, out_tag,
                                                   out_lanes)
    return tl.tensor(handle, tl.block_type(target, [out_lanes]))


@builtin
def vmul(lhs, rhs, _semantic=None) -> tensor:
    """
    Elementwise vector multiply (``ave.hir.vmul``) under an all-true
    ``ave.hir.pge <ALL>`` mask.
    """
    if lhs.type != rhs.type:
        raise ValueError("mul requires same-typed operands, "
                         f"got {lhs.type} and {rhs.type}")
    lanes = lhs.shape[0]
    mask = _all_mask(lanes, _semantic)
    handle = _semantic.builder.create_simd_binary("mul", lhs.handle,
                                                  rhs.handle, mask.handle)
    return tl.tensor(handle, lhs.type)


@builtin
def vadd(src0, src1, mask=None, mode=MODE.ZEROING, _semantic=None) -> tensor:
    """
    Elementwise vector add (``ave.hir.vadd`` -> ``hivm.vadd.{s,u}.x``).

    :param src0: left vector register tensor
    :param src1: right vector register tensor (same type as ``src0``)
    :param mask: predicate tensor; defaults to an all-true ``pge <ALL>``
    :param mode: predication mode; only ``MODE.ZEROING`` is supported
    """
    return _binary("add", src0, src1, mask, mode, _semantic)


@builtin
def vsub(src0, src1, mask=None, mode=MODE.ZEROING, _semantic=None) -> tensor:
    """
    Elementwise vector subtract (``ave.hir.vsub`` -> ``hivm.vsub.{s,u}.x``).

    :param src0: left vector register tensor
    :param src1: right vector register tensor (same type as ``src0``)
    :param mask: predicate tensor; defaults to an all-true ``pge <ALL>``
    :param mode: predication mode; only ``MODE.ZEROING`` is supported
    """
    return _binary("sub", src0, src1, mask, mode, _semantic)


@builtin
def vor(src0, src1, mask=None, mode=MODE.ZEROING, _semantic=None) -> tensor:
    """
    Elementwise bitwise OR (``ave.hir.vor`` -> ``hivm.vor.x``).

    Bits are combined without converting the element dtype. This operation
    does not guarantee zero padding of values produced by ``vtruncf``.
    """
    return _binary("or", src0, src1, mask, mode, _semantic)


@builtin
def vdiv(lhs, rhs, _semantic=None) -> tensor:
    """
    Elementwise vector divide (``ave.hir.vdiv``) under an all-true
    ``ave.hir.pge <ALL>`` mask.
    """
    if lhs.type != rhs.type:
        raise ValueError("div requires same-typed operands, "
                         f"got {lhs.type} and {rhs.type}")
    lanes = lhs.shape[0]
    mask = _all_mask(lanes, _semantic)
    handle = _semantic.builder.create_simd_binary("div", lhs.handle,
                                                  rhs.handle, mask.handle)
    return tl.tensor(handle, lhs.type)


@builtin
def vxor(src0, src1, mask=None, mode=MODE.ZEROING, _semantic=None) -> tensor:
    """
    Elementwise bitwise XOR (``ave.hir.vxor`` -> ``hivm.vxor.x``).

    :param src0: left vector register tensor
    :param src1: right vector register tensor (same type as ``src0``)
    :param mask: predicate tensor; defaults to an all-true ``pge <ALL>``
    :param mode: predication mode; only ``MODE.ZEROING`` is supported
    """
    return _binary("xor", src0, src1, mask, mode, _semantic)


@builtin
def vmins(src, scalar, mask=None, mode=MODE.ZEROING, _semantic=None) -> tensor:
    """
    Elementwise min with a scalar (``ave.hir.vmins`` ->
    ``hivm.vmins.{s,u}.x``): ``out[i] = min(src[i], scalar)``.

    :param src: vector register tensor (f16/f32 and signed/unsigned 8/16/32-bit
        integers; bf16 and fp8
        have no vec-scalar intrinsic -- splat the scalar and use
        elementwise min instead)
    :param scalar: scalar of the same dtype as ``src``'s elements; a
        Python/constexpr number or a scalar tensor
    :param mask: predicate tensor; defaults to an all-true ``pge <ALL>``
    :param mode: predication mode; only ``MODE.ZEROING`` is supported
    """
    _check_mode(mode, "vmins")
    tag = _elem_tag(src.dtype)
    if tag not in _VEC_SCALAR_TAGS:
        raise ValueError(
            f"vmins does not support {src.dtype} registers; supported: "
            f"i8/i16/i32/f16/f32 (bf16/fp8: use vsplat + elementwise min)")
    _check_vector(src, "vmins")
    lanes = src.shape[0]
    explicit_mask = mask is not None
    if isinstance(scalar, tensor):
        if scalar.shape:
            raise ValueError("vmins scalar must be a scalar tensor, not a vector")
        if scalar.dtype != src.dtype:
            raise ValueError(
                f"vmins scalar dtype {scalar.dtype} must match the vector "
                f"element dtype {src.dtype}")
        scalar_handle = scalar.handle
    else:
        scalar = _unwrap_if_constexpr(scalar)
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            raise TypeError(
                f"vmins scalar must be a number, a constexpr, or a scalar "
                f"tensor of {src.dtype}, got {type(scalar).__name__}")
        if src.dtype.is_int():
            bits = _ELEM_SPECS[tag]
            lower = 0 if src.dtype.is_int_unsigned() else -(1 << (bits - 1))
            upper = (1 << bits) - 1 if src.dtype.is_int_unsigned() else (1 << (bits - 1)) - 1
            if type(scalar) is not int or not lower <= scalar <= upper:
                raise ValueError(f"vmins scalar must be an integer representable as {src.dtype}")
        scalar_handle = _semantic.builder.create_simd_scalar(tag,
                                                             float(scalar))
    if mask is None:
        mask = _all_mask(lanes, _semantic)
    else:
        _check_mask(mask, lanes, "vmins")
    handle = _semantic.builder.create_simd_binary_scalar(
        "umins" if src.dtype.is_int_unsigned() else "mins",
        src.handle, scalar_handle, mask.handle)
    if explicit_mask:
        handle = _zero_inactive(handle, src, mask, _semantic)
    return tl.tensor(handle, src.type)


@builtin
def vmod(lhs, rhs, _semantic=None) -> tensor:
    """
    Elementwise integer modulo (``ave.hir.vmod``) under an all-true mask.
    """
    if lhs.type != rhs.type:
        raise ValueError("mod requires same-typed operands, "
                         f"got {lhs.type} and {rhs.type}")
    lanes = lhs.shape[0]
    mask = _all_mask(lanes, _semantic)
    handle = _semantic.builder.create_simd_binary("mod", lhs.handle,
                                                  rhs.handle, mask.handle)
    return tl.tensor(handle, lhs.type)


@builtin
def vfma(acc, lhs, rhs, _semantic=None) -> tensor:
    """
    Fused multiply-add (``ave.hir.vmula``): ``acc + lhs * rhs``.
    """
    if not (acc.type == lhs.type == rhs.type):
        raise ValueError("fma requires same-typed operands, "
                         f"got {acc.type}, {lhs.type}, {rhs.type}")
    lanes = acc.shape[0]
    mask = _all_mask(lanes, _semantic)
    handle = _semantic.builder.create_simd_vfma(acc.handle, lhs.handle,
                                                rhs.handle, mask.handle)
    return tl.tensor(handle, acc.type)


@builtin
def vsel(mask, true_val, false_val, _semantic=None) -> tensor:
    """
    Bitwise select (``ave.hir.vsel``): ``mask ? true_val : false_val``.
    """
    if true_val.type != false_val.type:
        raise ValueError("select requires same-typed value operands, "
                         f"got {true_val.type} and {false_val.type}")
    handle = _semantic.builder.create_simd_vsel(mask.handle, true_val.handle,
                                                false_val.handle)
    return tl.tensor(handle, true_val.type)


def _unary(name, src, _semantic):
    lanes = src.shape[0]
    mask = _all_mask(lanes, _semantic)
    handle = _semantic.builder.create_simd_unary(name, src.handle, mask.handle)
    return tl.tensor(handle, src.type)


@builtin
def vneg(src, _semantic=None) -> tensor:
    """Elementwise negate (``ave.hir.vneg``)."""
    return _unary("neg", src, _semantic)


@builtin
def vsqrt(src, _semantic=None) -> tensor:
    """Elementwise square root (``ave.hir.vsqrt``)."""
    return _unary("sqrt", src, _semantic)


@builtin
def vrsqrt(src, _semantic=None) -> tensor:
    """Elementwise reciprocal square root (``ave.hir.vrsqrt``)."""
    return _unary("rsqrt", src, _semantic)


@builtin
def vreduce(src, kind, _semantic=None) -> tensor:
    """
    Horizontal reduction (``ave.hir.reduction``) of a vector register.

    ``kind`` is ``add`` / ``min`` / ``max``; the combining result is placed at
    the first lane (AVE reduction semantics).
    """
    kind = _unwrap_if_constexpr(kind)
    if kind not in ("add", "min", "max"):
        raise ValueError(f"reduce kind must be add/min/max, got {kind!r}")
    lanes = src.shape[0]
    mask = _all_mask(lanes, _semantic)
    handle = _semantic.builder.create_simd_vreduce(src.handle, mask.handle, kind)
    return tl.tensor(handle, src.type)


@builtin
def vcmp(lhs, rhs, cmp_type, _semantic=None) -> tensor:
    """
    Elementwise compare (``ave.hir.vcmp``) returning an i1 mask.

    ``cmp_type`` is ``eq`` / ``ne`` / ``gt`` / ``ge`` / ``lt`` / ``le``.
    """
    cmp_type = _unwrap_if_constexpr(cmp_type)
    cmp_type = str(cmp_type).upper()
    if cmp_type not in ("EQ", "NE", "GT", "GE", "LT", "LE"):
        raise ValueError(
            f"cmp_type must be eq/ne/gt/ge/lt/le, got {cmp_type!r}")
    if lhs.type != rhs.type:
        raise ValueError("vcmp requires same-typed operands, "
                         f"got {lhs.type} and {rhs.type}")
    lanes = lhs.shape[0]
    mask = _all_mask(lanes, _semantic)
    handle = _semantic.builder.create_simd_vcmp(cmp_type, lhs.handle,
                                                rhs.handle, mask.handle)
    return tl.tensor(handle, tl.block_type(tl.int1, [lanes]))


@builtin
def vbroadcast(src, low=True, _semantic=None) -> tensor:
    """
    Broadcast a vector register's half (``ave.hir.vector_broadcast``).

    ``low`` selects the low half to broadcast to the full register; used to
    splat a reduced value (e.g. the lane-0 sum) across all lanes.
    """
    lanes = src.shape[0]
    mask = _all_mask(lanes, _semantic)
    handle = _semantic.builder.create_simd_vbroadcast(src.handle, mask.handle,
                                                      bool(low))
    return tl.tensor(handle, src.type)


@builtin
def vfull(value, dtype, lanes, _semantic=None) -> tensor:
    """
    Constant vector register: broadcast a scalar ``value`` to ``lanes`` lanes.

    Used for the -inf sentinel in topk, zero vectors, etc.
    """
    value = _unwrap_if_constexpr(value)
    lanes = _unwrap_if_constexpr(lanes)
    tag = _elem_tag(dtype)
    scalar = _semantic.builder.create_simd_scalar(tag, float(value))
    handle = _semantic.builder.create_simd_vsplat(scalar, tag, lanes)
    return tl.tensor(handle, tl.block_type(dtype, [lanes]))


@builtin
def vtruncf(src, target_dtype, mask=None, part="part_even", _semantic=None) -> tensor:
    """
    Narrowing float conversion (``ave.hir.vtruncf``), e.g. f32 -> bf16.

    Keeps the lane count; only the element type narrows.

    :param src: source vector register tensor (f32)
    :param target_dtype: destination element dtype (bf16/f16)
    :param mask: predicate tensor; defaults to an all-true mask
    :param part: ``part_even`` / ``part_odd`` sub-register selector
    """
    part = ss.check_part(_unwrap_if_constexpr(part))
    out_tag = _elem_tag(target_dtype)
    if out_tag not in ("bf16", "f16", "fp8e4nv"):
        raise ValueError(f"vtruncf only narrows to bf16/f16/fp8e4nv, got {target_dtype}")
    src_tag = _elem_tag(src.dtype)
    src_bits = _ELEM_SPECS[src_tag]
    out_bits = _ELEM_SPECS[out_tag]
    out_lanes = src.shape[0] * src_bits // out_bits
    if mask is None:
        mask = _all_mask(src.shape[0], _semantic)
    handle = _semantic.builder.create_simd_vcvt("truncf", src.handle,
                                                mask.handle, out_tag, part)
    return tl.tensor(handle, tl.block_type(target_dtype, [out_lanes]))


@builtin
def vsplat(value, dtype, _semantic=None) -> tensor:
    """
    Constant vector register: broadcast a scalar ``value`` to a full register
    of ``dtype`` lanes (``ave.hir.scalar_broadcast``).

    Thin wrapper over :func:`vfull` that derives the lane count from ``dtype``
    (2048 // elem_bits; i64 spans 4096 bits).

    :param value: scalar value (Python int/float or constexpr)
    :param dtype: element dtype; lane count is derived from it
    """
    tag = _elem_tag(dtype)
    return vfull(value, dtype, _lanes_for(tag), _semantic=_semantic)


@builtin
def vreduce_sum(src, _semantic=None) -> tensor:
    """
    Horizontal reduction to sum (``ave.hir.reduction <add>``).

    Alias of ``vreduce(src, 'add')``; the combining result is placed at lane 0.
    """
    return vreduce(src, "add", _semantic=_semantic)


def _reduce_across(regs, kind, _semantic):
    """
    Reduce a single register or a tuple/list of registers to the elementwise
    max/min, then horizontally reduce and broadcast lane 0 to the full
    register (``ave.hir.vmax``/``vmin`` + ``reduction`` + ``vector_broadcast``).

    Mirrors the TileKernels ``S.vdupv(S.vcmax(...))`` chain: the result is the
    global max/min (over all registers and lanes) replicated across every lane,
    so it can be fed directly into :func:`vcmp` with the original registers.
    """
    if isinstance(regs, (list, tuple)):
        regs = list(regs)
    elif hasattr(regs, "values") and hasattr(getattr(regs, "type", None), "fields"):
        # triton visit_List materializes [a, b] as tl.tuple; unwrap it.
        regs = list(regs.values)
    else:
        regs = [regs]
    if not regs:
        raise ValueError(f"{kind}-across requires at least one register")
    acc = regs[0]
    for r in regs[1:]:
        if acc.type != r.type:
            raise ValueError(
                f"{kind}-across requires same-typed registers, "
                f"got {acc.type} and {r.type}")
        lanes = acc.shape[0]
        mask = _all_mask(lanes, _semantic)
        kind_op = ("max" if kind == "max" else "min")
        handle = _semantic.builder.create_simd_binary(kind_op, acc.handle,
                                                      r.handle, mask.handle)
        acc = tl.tensor(handle, acc.type)
    reduced = vreduce(acc, kind, _semantic=_semantic)
    return vbroadcast(reduced, low=True, _semantic=_semantic)


@builtin
def vreduce_max(regs, _semantic=None) -> tensor:
    """
    Reduce a register or list of registers to the global max across all lanes
    and registers, broadcast to every lane.

    Combines elementwise ``ave.hir.vmax`` folds across registers, a horizontal
    ``ave.hir.reduction <max>``, and ``ave.hir.vector_broadcast`` (low half).
    """
    return _reduce_across(regs, "max", _semantic)


@builtin
def vreduce_min(regs, _semantic=None) -> tensor:
    """
    Reduce a register or list of registers to the global min across all lanes
    and registers, broadcast to every lane.  Symmetric to
    :func:`vreduce_max` with ``ave.hir.vmin``.
    """
    return _reduce_across(regs, "min", _semantic)


@builtin
def vci(base, _semantic=None) -> tensor:
    """
    Lane-index register: lane ``i`` holds ``base + i`` (``ave.hir.vci``).

    The AVE VCI lowering supports i8/i16/i32/f16/f32 element widths; this
    interface fixes the element type to i32 with 64 lanes (2048 bits).  The
    ``base`` is either a Python int / constexpr (lowered to an ``arith.constant
    i32``) or a scalar i32 tensor (e.g. ``tl.program_id(0) * 64``).

    :param base: starting index for lane 0
    """
    if isinstance(base, tensor):
        base_handle = base.handle
    else:
        base = _unwrap_if_constexpr(base)
        base_handle = _semantic.builder.create_simd_scalar(
            "i32", float(base))
    lanes = _lanes_for("i32")
    handle = _semantic.builder.create_simd_vci(base_handle, "i32", lanes,
                                               False)
    return tl.tensor(handle, tl.block_type(tl.int32, [lanes]))


@builtin
def pge(n, lanes, _semantic=None) -> tensor:
    """
    Prefix predicate: the first ``n`` lanes are true, the rest false
    (``ave.hir.pge <VLn>``).

    Used for group-boundary selects (e.g. cast_back's 32-channel scale groups
    over a 64-lane register: ``pge(32, 64)``).

    :param n: number of leading true lanes; must match a supported VL pattern
        (1, 2, 3, 4, 8, 16, 32, 64, 128)
    :param lanes: mask register lane count (2048 bits of i1)
    """
    n = _unwrap_if_constexpr(n)
    lanes = _unwrap_if_constexpr(lanes)
    pattern = ss.check_pge_pattern(f"VL{n}")
    handle = _semantic.builder.create_simd_pge(pattern, lanes)
    return tl.tensor(handle, tl.block_type(tl.int1, [lanes]))


@builtin
def membar(scope, _semantic=None):
    """
    Memory barrier inside a VF (``ave.hir.membar``).

    ``scope`` names the AVE membar kind, e.g. ``VST_VLD`` (vector-store ->
    vector-load, used before reading values written by masked stores) or
    ``VV_ALL``.  Accepted scopes: VV_ALL, VST_VLD, VLD_VST, VST_VST, VS_ALL,
    VST_LD, VLD_ST, VST_ST, SV_ALL, ST_VLD, LD_VST, ST_VST, SS_ALL, ST_LD,
    LD_ST, ST_ST.
    """
    scope = str(_unwrap_if_constexpr(scope)).upper()
    _semantic.builder.create_simd_vmembar(scope)
