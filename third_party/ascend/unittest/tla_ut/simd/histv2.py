"""TLA 3-in-1 test: dhistv2/chistv2 vector histogram updates.

Stages: ir (offline HIVM), compile (kernel artifacts), run (numerics).

Coverage
  * both ops x both bin windows (0 counts values [0,127], 1 counts [128,255]);
  * five mask cases in one launch, all accumulating onto a non-zero initial
    accumulator: a data-driven mask, the omitted mask (all-true default), an
    all-false mask (the accumulator must be returned unchanged), a half prefix
    mask and an explicitly built all-true mask;
  * the intended two-call usage, where bin=0 and bin=1 fill two accumulators
    from one source register and together give the full 256-bin histogram;
  * four 256-value chunks folded into one accumulator inside a single kernel,
    which also exercises the documented ``acc = op(acc, ...)`` chaining instead
    of relying on an in-place update of the operand;
  * frontend validation errors: bin range and type, acc/src dtype and lane
    count, and mask lane count.

Every histogram covers at most ``CHUNKS * SRC_LANES`` = 1024 values per counter,
well below the 65535 limit of the u16 counter, so the expected values below are
exact. The behaviour when a counter would overflow (u16 saturation versus wrap)
is defined by the instruction, not by this frontend, and is not covered here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

from triton.language.core import builtin, _unwrap_if_constexpr
from triton.language.extra.tla.core import vector, vector_type
from triton.language.extra.tla.simd import _elem_tag


@builtin
def _vfull(value, dtype, lanes, _semantic=None):
    """Lane-explicit constant broadcast, local to this test module.

    ``main-dev`` replaced the public ``tla.simd.vfull`` with ``vdup``, which
    always materialises a *full* register (``2048 // bitwidth`` lanes).  The
    negative kernels below deliberately need an exact dtype *and* a lane count
    that a full register cannot express (``vlds`` has the same limitation), so
    they go through this shim over the still-current ``create_simd_vsplat``
    binding instead of the shipped API.
    """
    value = _unwrap_if_constexpr(value)
    lanes = _unwrap_if_constexpr(lanes)
    tag = _elem_tag(dtype)
    scalar = _semantic.builder.create_simd_scalar(tag, float(value))
    handle = _semantic.builder.create_simd_vsplat(scalar, tag, lanes)
    return vector(handle, vector_type(lanes, dtype))

SRC_LANES = 256
BINS = 128
DIST = "NORM_B16"
CHUNKS = 4
PREFIX = 128

# histv2_kernel writes one row per mask case; the reference builds them in the
# same order.
ROW_DATA = 0  # mask derived from the MK input buffer
ROW_DEFAULT = 1  # mask omitted -> all-true default
ROW_NONE = 2  # all-false mask
ROW_PREFIX = 3  # first PREFIX lanes active
ROW_ALL = 4  # explicitly built all-true mask
ROWS = 5

KINDS = (("d", "dhistv2", "ave.hir.dhistv2"), ("c", "chistv2", "ave.hir.chistv2"))
BIN_VALUES = (0, 1)

MATRIX_SIGNATURE = {"SRC": "*u8", "MK": "*i8", "ACC": "*u16", "O": "*u16"}
NEG_SIGNATURE = {"O": "*u16"}


@tla.jit
def histv2_kernel(SRC, MK, ACC, O, N: tl.constexpr, PREFIX: tl.constexpr, PATTERN: tl.constexpr,
                  BIN: tl.constexpr, KIND: tl.constexpr, DIST: tl.constexpr):
    src = tla.alloc((N, ), tl.uint8, space="ub")
    mk = tla.alloc((N, ), tl.int8, space="ub")
    acc = tla.alloc((128, ), tl.uint16, space="ub")
    o = tla.alloc((5, 128), tl.uint16, space="ub")
    tla.load_gm_to_ub(tla.view(SRC, (N, )), src)
    tla.load_gm_to_ub(tla.view(MK, (N, )), mk)
    tla.load_gm_to_ub(tla.view(ACC, (128, )), acc)
    with tla.scope.simd():
        s = tla.simd.vlds(src[0])
        a = tla.simd.vlds(acc[0])
        # Build the predicate cases from 256-lane operands so each mask matches
        # the source register; comparing s with itself gives all-true for EQ and
        # all-false for NE.
        data_mask = tla.simd.vcmps(tla.simd.vlds(mk[0]), 0, op=tla.simd.CMP.NE)
        none = tla.simd.vcmp(s, s, op=tla.simd.CMP.NE)
        all_true = tla.simd.vcmp(s, s, op=tla.simd.CMP.EQ)
        prefix = tla.simd.pset(2048 // N, PATTERN)
        row0 = tla.simd.dhistv2(a, s, data_mask, bin=BIN) if KIND == "d" else tla.simd.chistv2(a, s, data_mask, bin=BIN)
        row1 = tla.simd.dhistv2(a, s, bin=BIN) if KIND == "d" else tla.simd.chistv2(a, s, bin=BIN)
        row2 = tla.simd.dhistv2(a, s, none, bin=BIN) if KIND == "d" else tla.simd.chistv2(a, s, none, bin=BIN)
        row3 = tla.simd.dhistv2(a, s, prefix, bin=BIN) if KIND == "d" else tla.simd.chistv2(a, s, prefix, bin=BIN)
        row4 = tla.simd.dhistv2(a, s, all_true, bin=BIN) if KIND == "d" else tla.simd.chistv2(a, s, all_true, bin=BIN)
        tla.simd.vsts(o[0, 0], row0, dist=DIST)
        tla.simd.vsts(o[1, 0], row1, dist=DIST)
        tla.simd.vsts(o[2, 0], row2, dist=DIST)
        tla.simd.vsts(o[3, 0], row3, dist=DIST)
        tla.simd.vsts(o[4, 0], row4, dist=DIST)
    tla.store_ub_to_gm(o, tla.view(O, (5, 128)))


@tla.jit
def histv2_full_kernel(SRC, ACC, O, N: tl.constexpr, DIST: tl.constexpr):
    """Two calls on one source register: bin=0 and bin=1 give all 256 bins."""
    src = tla.alloc((N, ), tl.uint8, space="ub")
    low_acc = tla.alloc((128, ), tl.uint16, space="ub")
    high_acc = tla.alloc((128, ), tl.uint16, space="ub")
    o = tla.alloc((256, ), tl.uint16, space="ub")
    tla.load_gm_to_ub(tla.view(SRC, (N, )), src)
    tla.load_gm_to_ub(tla.view(ACC, (128, )), low_acc)
    tla.load_gm_to_ub(tla.view(ACC, (128, )), high_acc)
    with tla.scope.simd():
        s = tla.simd.vlds(src[0])
        low = tla.simd.dhistv2(tla.simd.vlds(low_acc[0]), s, bin=0)
        high = tla.simd.dhistv2(tla.simd.vlds(high_acc[0]), s, bin=1)
        tla.simd.vsts(o[0], low, dist=DIST)
        tla.simd.vsts(o[128], high, dist=DIST)
    tla.store_ub_to_gm(o, tla.view(O, (256, )))


@tla.jit
def histv2_chunks_kernel(SRC, ACC, O, N: tl.constexpr, BIN: tl.constexpr, KIND: tl.constexpr, DIST: tl.constexpr):
    """Fold CHUNKS source registers into one accumulated register."""
    src = tla.alloc((4, N), tl.uint8, space="ub")
    acc = tla.alloc((128, ), tl.uint16, space="ub")
    o = tla.alloc((128, ), tl.uint16, space="ub")
    tla.load_gm_to_ub(tla.view(SRC, (4, N)), src)
    tla.load_gm_to_ub(tla.view(ACC, (128, )), acc)
    with tla.scope.simd():
        s0 = tla.simd.vlds(src[0, 0])
        s1 = tla.simd.vlds(src[1, 0])
        s2 = tla.simd.vlds(src[2, 0])
        s3 = tla.simd.vlds(src[3, 0])
        a0 = tla.simd.vlds(acc[0])
        a1 = tla.simd.dhistv2(a0, s0, bin=BIN) if KIND == "d" else tla.simd.chistv2(a0, s0, bin=BIN)
        a2 = tla.simd.dhistv2(a1, s1, bin=BIN) if KIND == "d" else tla.simd.chistv2(a1, s1, bin=BIN)
        a3 = tla.simd.dhistv2(a2, s2, bin=BIN) if KIND == "d" else tla.simd.chistv2(a2, s2, bin=BIN)
        a4 = tla.simd.dhistv2(a3, s3, bin=BIN) if KIND == "d" else tla.simd.chistv2(a3, s3, bin=BIN)
        tla.simd.vsts(o[0], a4, dist=DIST)
    tla.store_ub_to_gm(o, tla.view(O, (128, )))


# Negative kernels build their operands with the local _vfull shim so that the
# dtype and lane count are exact: vlds (and main-dev's vdup) always return a
# whole register, so they cannot produce the narrowed operands these cases need.
@tla.jit
def bad_acc_dtype_kernel(O, N: tl.constexpr):
    o = tla.alloc((128, ), tl.uint16, space="ub")
    with tla.scope.simd():
        res = tla.simd.dhistv2(_vfull(0, tl.int32, 128), _vfull(0, tl.uint8, N))
        tla.simd.vsts(o[0], res)
    tla.store_ub_to_gm(o, tla.view(O, (128, )))


@tla.jit
def bad_src_dtype_kernel(O, N: tl.constexpr):
    o = tla.alloc((128, ), tl.uint16, space="ub")
    with tla.scope.simd():
        res = tla.simd.dhistv2(_vfull(0, tl.uint16, 128), _vfull(0, tl.uint16, N))
        tla.simd.vsts(o[0], res)
    tla.store_ub_to_gm(o, tla.view(O, (128, )))


@tla.jit
def bad_acc_lanes_kernel(O, N: tl.constexpr):
    o = tla.alloc((128, ), tl.uint16, space="ub")
    with tla.scope.simd():
        res = tla.simd.dhistv2(_vfull(0, tl.uint16, 64), _vfull(0, tl.uint8, N))
        tla.simd.vsts(o[0], res)
    tla.store_ub_to_gm(o, tla.view(O, (128, )))


@tla.jit
def bad_src_lanes_kernel(O, N: tl.constexpr):
    o = tla.alloc((128, ), tl.uint16, space="ub")
    with tla.scope.simd():
        res = tla.simd.dhistv2(_vfull(0, tl.uint16, 128), _vfull(0, tl.uint8, 128))
        tla.simd.vsts(o[0], res)
    tla.store_ub_to_gm(o, tla.view(O, (128, )))


@tla.jit
def bad_mask_lanes_kernel(O, N: tl.constexpr):
    o = tla.alloc((128, ), tl.uint16, space="ub")
    with tla.scope.simd():
        narrow = tla.simd.pset(32, "VL64")
        res = tla.simd.dhistv2(_vfull(0, tl.uint16, 128), _vfull(0, tl.uint8, N), narrow)
        tla.simd.vsts(o[0], res)
    tla.store_ub_to_gm(o, tla.view(O, (128, )))


def _matrix_constexprs(kind, bin_value):
    return {"N": SRC_LANES, "PREFIX": PREFIX, "PATTERN": f"VL{PREFIX}", "BIN": bin_value, "KIND": kind, "DIST": DIST}


def _matrix_module(kind, bin_value, output_dir=None):
    return compile_offline(histv2_kernel, MATRIX_SIGNATURE, _matrix_constexprs(kind, bin_value), output_dir=output_dir)


def test_ir():
    for kind, op_name, _ in KINDS:
        for bin_value in BIN_VALUES:
            module, ctx = _matrix_module(kind, bin_value)
            text = str(module)
            assert text.count(f"ave.hir.{op_name}") == ROWS, f"{op_name} must appear once per mask row"
            other = "chistv2" if op_name == "dhistv2" else "dhistv2"
            assert f"ave.hir.{other}" not in text, f"bin={bin_value} must not emit {other}"
            # The histogram must stay a single native op: no compare/select
            # composition is allowed to appear in place of it.
            assert "ave.hir.vsel" not in text, "histogram must not fall back to compare/select"
            lowered = str(lower_offline(module, ctx))
            check_lowered_ir(lowered, loads=3, stores=1, buffers=None, extra=(f"ave.hir.{op_name}", ))
    print("IR PASS: histv2 five mask cases per (op, bin)")


def _mask_tensors():
    """Predicates for the five rows of histv2_kernel, in row order."""
    alternating = torch.arange(SRC_LANES) % 2 == 0
    return (alternating,  # data-driven mask from the MK buffer
            torch.ones(SRC_LANES, dtype=torch.bool),  # default all-true
            torch.zeros(SRC_LANES, dtype=torch.bool),  # all-false
            torch.arange(SRC_LANES) < PREFIX,  # prefix
            torch.ones(SRC_LANES, dtype=torch.bool),  # explicit all-true
            )


def _reference(kind, bin_value, src, acc, mask):
    values = src.to(torch.int64)[mask]
    low = 128 * bin_value
    thresholds = torch.arange(low, low + BINS, dtype=torch.int64)
    if kind == "d":
        counts = (values[:, None] == thresholds[None, :]).sum(dim=0)
    else:
        counts = (values[:, None] <= thresholds[None, :]).sum(dim=0)
    return counts.to(torch.int64) + acc.to(torch.int64)


def _initial_acc():
    return (torch.arange(BINS, dtype=torch.int64) % 7).to(torch.uint16)


def _source(seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (SRC_LANES, ), dtype=torch.uint8, generator=gen)


def _launch_matrix(kind, bin_value, src, acc):
    mk = (_mask_tensors()[ROW_DATA].to(torch.int8))
    out = torch.empty((ROWS, BINS), dtype=torch.uint16, device="npu")
    histv2_kernel[(1, )](src.npu(), mk.npu(), acc.npu(), out, **_matrix_constexprs(kind, bin_value))
    torch.npu.synchronize()
    return out


def _launch_full(src):
    acc = torch.zeros(BINS, dtype=torch.uint16)
    out = torch.empty(256, dtype=torch.uint16, device="npu")
    histv2_full_kernel[(1, )](src.npu(), acc.npu(), out, N=SRC_LANES, DIST=DIST)
    torch.npu.synchronize()
    return out


def _launch_chunks(kind, bin_value, chunks, acc):
    out = torch.empty(BINS, dtype=torch.uint16, device="npu")
    histv2_chunks_kernel[(1, )](chunks.reshape(-1).npu(), acc.npu(), out, N=SRC_LANES, BIN=bin_value, KIND=kind,
                                DIST=DIST)
    torch.npu.synchronize()
    return out


def test_compile():
    # Every (op, bin) pair must survive the full backend pipeline, not just the
    # default dhistv2/bin=0 configuration.
    src = _source()
    acc = _initial_acc()
    for kind, op_name, _ in KINDS:
        for bin_value in BIN_VALUES:
            _launch_matrix(kind, bin_value, src, acc)
            print("COMPILE PASS: histv2", op_name, "bin", bin_value)
    _launch_full(src)
    print("COMPILE PASS: histv2 full 256-bin two-call kernel")
    chunks = torch.stack([_source(seed=100 + i) for i in range(CHUNKS)])
    for kind, op_name, _ in KINDS:
        _launch_chunks(kind, bin_value=0, chunks=chunks, acc=acc)
        print("COMPILE PASS: histv2 chunks", op_name)
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: histv2 generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    src = _source(seed=3)
    acc = _initial_acc()
    masks = _mask_tensors()
    for kind, op_name, _ in KINDS:
        for bin_value in BIN_VALUES:
            out = _launch_matrix(kind, bin_value, src, acc)
            expected = torch.stack([_reference(kind, bin_value, src, acc, mask) for mask in masks])
            torch.testing.assert_close(out.cpu().to(torch.int64), expected, rtol=0, atol=0)
            print("RUN PASS: histv2", op_name, "bin", bin_value, "five masks")
            # The all-false row must give the accumulator back untouched.
            torch.testing.assert_close(out[ROW_NONE].cpu().to(torch.int64), acc.to(torch.int64), rtol=0, atol=0)
    # bin=0 and bin=1 from the same register must reconstruct all 256 bins.
    out = _launch_full(src)
    expected_full = torch.bincount(src.to(torch.int64), minlength=256).to(torch.int64)
    torch.testing.assert_close(out.cpu().to(torch.int64), expected_full, rtol=0, atol=0)
    print("RUN PASS: histv2 full 256-bin histogram from two calls")
    # Four chunks folded into one accumulator in a single kernel.
    chunks = torch.stack([_source(seed=100 + i) for i in range(CHUNKS)])
    flat = chunks.reshape(-1)
    for kind, op_name, _ in KINDS:
        for bin_value in BIN_VALUES:
            out = _launch_chunks(kind, bin_value, chunks, acc)
            expected = _reference(kind, bin_value, flat, acc, torch.ones(flat.numel(), dtype=torch.bool))
            torch.testing.assert_close(out.cpu().to(torch.int64), expected, rtol=0, atol=0)
            print("RUN PASS: histv2 chunks", op_name, "bin", bin_value, CHUNKS, "chunks")


def _expect_error(fn, needle):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - the frontend raises several kinds
        message = str(exc)
        assert needle in message, f"expected {needle!r} in error, got: {message}"
        return
    raise AssertionError(f"expected an error containing {needle!r}")


def test_validation():
    for kind in ("d", "c"):
        for bin_value in (2, -1):
            _expect_error(
                lambda kind=kind, bin_value=bin_value: compile_offline(histv2_kernel, MATRIX_SIGNATURE,
                                                                       _matrix_constexprs(kind, bin_value)),
                "histogram bin must be 0 or 1")
    for bad_bin in (True, 1.0):
        _expect_error(
            lambda bad_bin=bad_bin: compile_offline(histv2_kernel, MATRIX_SIGNATURE, _matrix_constexprs("d", bad_bin)),
            "histogram bin must be a compile-time integer 0 or 1")
    _expect_error(lambda: compile_offline(bad_acc_dtype_kernel, NEG_SIGNATURE, {"N": SRC_LANES}),
                  "acc must be a 16-bit integer vector")
    _expect_error(lambda: compile_offline(bad_src_dtype_kernel, NEG_SIGNATURE, {"N": SRC_LANES}),
                  "src must be an 8-bit integer vector")
    _expect_error(lambda: compile_offline(bad_acc_lanes_kernel, NEG_SIGNATURE, {"N": SRC_LANES}),
                  "acc must be a full 128-lane register")
    _expect_error(lambda: compile_offline(bad_src_lanes_kernel, NEG_SIGNATURE, {"N": SRC_LANES}),
                  "src must be a full 256-lane register")
    _expect_error(lambda: compile_offline(bad_mask_lanes_kernel, NEG_SIGNATURE, {"N": SRC_LANES}),
                  "mask must be vector<256xint1>")
    print("VALIDATION PASS: histv2 bin/dtype/lane-count/mask errors")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
        test_validation()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: histv2", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
