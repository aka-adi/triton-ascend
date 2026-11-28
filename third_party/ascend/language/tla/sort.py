"""Descending, non-stable FP32 score/index sorting in UB."""
from triton.language.core import builtin
from .core import buffer


def _ub_vector(value, dtype):
    """Return the element count of one contiguous rank-one UB buffer.

    ``dtype`` is the canonical dtype name (``"fp32"`` or ``"uint32"``).  Shared
    by the low-level sort primitives in :mod:`tla.simd`; not a public API.
    """
    if not isinstance(value, buffer):
        raise TypeError("expected a tla.buffer")
    if value.space != "ub":
        raise ValueError("expected a UB buffer")
    if len(value.shape) != 1 or value.strides != (1,):
        raise ValueError("expected a contiguous one-dimensional buffer")
    if value.dtype.name != dtype:
        raise TypeError(f"expected a {dtype} buffer")
    return value.shape[0]


def _check_scores(scores, indices, dst):
    """Validate the ``simd.vbs`` operand shapes and return the score count.

    ``scores`` holds N FP32 scores, ``indices`` N uint32 payloads and ``dst``
    2*N interleaved uint32 words.  N is a multiple of 32 in [32, 4096]; unlike
    ``sort_pairs`` it does not have to be a power of two.
    """
    width = _ub_vector(scores, "fp32")
    if not 32 <= width <= 4096 or width % 32:
        raise ValueError("score width must be a multiple of 32 in [32, 4096]")
    if _ub_vector(indices, "uint32") != width or _ub_vector(dst, "uint32") != 2 * width:
        raise ValueError("expected one index and two destination words per score")
    return width


def _check_sort_pairs(scores, indices, dst, scratch):
    """Validate ``sort_pairs`` operands and return the score count.

    Builds on :func:`_ub_vector` so buffer, address-space, rank, stride and
    dtype checks live in one place.  On top of that this adds the contract
    unique to ``sort_pairs``: a power-of-two score count, 32-byte-aligned
    static offsets and non-overlapping storage.
    """
    width = _ub_vector(scores, "fp32")
    for name, value, size in (("indices", indices, width),
                              ("dst", dst, 2 * width),
                              ("scratch", scratch, 2 * width)):
        if _ub_vector(value, "uint32") != size:
            raise ValueError(f"tla.sort_pairs {name} requires {size} uint32 words")
    if width < 32 or width > 4096 or width & (width - 1):
        raise ValueError("tla.sort_pairs requires power-of-two N in [32, 4096]")
    values = (scores, indices, dst, scratch)
    for name, value in zip(("scores", "indices", "dst", "scratch"), values):
        if type(value.offset) is not int or value.offset < 0 or value.offset % 8:
            raise ValueError(
                f"tla.sort_pairs {name} requires a static nonnegative 32-byte-aligned offset; "
                "subview offsets are not static yet and are not supported")
    for i, value in enumerate(values):
        if any(value.handle is prior.handle for prior in values[:i]):
            raise ValueError("tla.sort_pairs requires non-overlapping buffers")
    return width


@builtin
def sort_pairs(scores, indices, dst, scratch, _semantic=None):
    """Sort FP32 scores descending and preserve each opaque uint32 index payload.

    ``scores`` and ``indices`` contain N elements; ``dst`` and ``scratch``
    contain 2*N uint32 words. N is a power of two in [32, 4096]. Output words
    alternate FP32 score *bits* and the original uint32 payload. All buffers
    must be contiguous UB buffers with a 32-byte-aligned base and offset,
    and their accessed ranges must not overlap. Only statically known offsets
    are accepted; ``tla.subview`` results carry no static offset and are
    currently rejected (whole allocations at offset 0 are supported). The
    caller must guarantee base alignment and non-aliasing not provable from
    the views.

    Invoke outside register SIMD/SIMT scopes, normally in
    ``with tla.scope.vector():``. NaN scores are unsupported (a runtime data
    precondition, not a frontend check). Infinities are allowed. Equal scores,
    including +0/-0, may reorder: this is not a stable sort. Index high bits
    are payload bits, not keys, and must be preserved unchanged. uint32
    buffers lower to signless i32 in MLIR, so no bit-level conversion happens.

    Returns None. Writes ``dst`` and uses/clobbers ``scratch``; does not
    modify ``scores`` or ``indices``. No public partial-sort primitives are
    exposed by this interface.
    """
    _check_sort_pairs(scores, indices, dst, scratch)
    _semantic.builder.create_tla_sort_pairs(scores.handle, indices.handle, dst.handle, scratch.handle)
