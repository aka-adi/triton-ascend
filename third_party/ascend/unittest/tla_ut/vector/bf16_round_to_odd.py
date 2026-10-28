"""TLA 3-in-1 test: TileKernels FP32-pair to BF16 round-to-odd.

Source: TileKernels/tile_kernels/quant/per_block_cast_asc.py,
f32_pair_to_bf16(x0, x1, round_odd=True), lines 44-50.
Deinterleave uint16 halves, then high | min(low, 1). The output lane
order is x0 followed by x1. No vcvt or floating-point arithmetic is used.

The source uses this helper inside its quantization operator. This test
ports the helper, not the complete scale/FP4/FP8 quantization operator.
Strided output is a TLA test adaptation; it is not a TileKernels vsstb call.

Stages: ir (offline HIVM), compile (kernel artifacts), run (exact bits).
Inputs are regenerated randomly on every run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline,
                                 check_lowered_ir, check_compile_artifacts)

SHAPE = dict(lanes=64, rows=8, cols=32, block=16)
SIGNATURE = {"X": "*fp32", "O": "*bf16"}


@triton.jit
def bf16_round_to_odd(X, O, N: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    src = tla.alloc_buffer((2, N), tl.float32, space="ub")
    dst = tla.alloc_buffer((ROWS, COLS), tl.bfloat16, space="ub")
    tla.copy(tla.view(X, (2, N)), src)
    tla.copy(tla.view(O, (ROWS, COLS)), dst)
    with tla.scope.simd():
        x0 = tla.simd.vlds(src[0, 0])
        x1 = tla.simd.vlds(src[1, 0])
        low, high = tla.simd.vdintlv(tla.simd.vbitcast(x0, "uint16"),
                                    tla.simd.vbitcast(x1, "uint16"))
        sticky = tla.simd.vmins(low, 1)
        packed = tla.simd.vor(high, sticky)
        result = tla.simd.vbitcast(packed, "bf16")
        # Each row receives a 32B block; its second half keeps the sentinel.
        tla.simd.vsstb(result, dst, COLS)
    tla.copy(dst, tla.view(O, (ROWS, COLS)))


def constexprs():
    return {"N": SHAPE["lanes"], "ROWS": SHAPE["rows"], "COLS": SHAPE["cols"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(bf16_round_to_odd, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    for name in ("vdintlv", "vumins", "vor", "store_with_stride"):
        assert "ave.hir." + name in text, name
    assert "ave.hir.vtruncf" not in text, "round-to-odd is a bit-level algorithm"
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=2, stores=1, buffers=2,
                     extra=("ave.hir.vdintlv", "ave.hir.vumins", "ave.hir.vor",
                            "ave.hir.store_with_stride"))
    print("IR PASS: TileKernels BF16 round-to-odd and strided writeback")


def _inputs():
    # Random 32-bit patterns: any exponent, sign, zero, subnormal or NaN.
    count = 2 * SHAPE["lanes"]
    bits = torch.randint(0, 1 << 32, (count,), dtype=torch.int64)
    signed = torch.where(bits >= 0x80000000, bits - 0x100000000, bits).to(torch.int32)
    return signed.view(torch.float32).reshape(2, SHAPE["lanes"]), bits


def _reference(bits):
    # Independent scalar integer definition: truncate, then force the last
    # retained bit odd iff any discarded bit is nonzero (no carry/round-even).
    rounded = [(int(value) >> 16) | int((int(value) & 0xffff) != 0) for value in bits]
    rounded = torch.tensor(rounded, dtype=torch.int32)
    signed = torch.where(rounded >= 0x8000, rounded - 0x10000, rounded).to(torch.int16)
    ref = torch.full((SHAPE["rows"], SHAPE["cols"]), 0x1234, dtype=torch.int16)
    ref[:, :SHAPE["block"]] = signed.reshape(SHAPE["rows"], SHAPE["block"])
    return ref


def _launch():
    x, bits = _inputs()
    initial = torch.full((SHAPE["rows"], SHAPE["cols"]), 0x1234, dtype=torch.int16)
    out = initial.view(torch.bfloat16).to("npu")
    bf16_round_to_odd[(1,)](x.to("npu"), out, enable_tla_mode=True, **constexprs())
    torch.npu.synchronize()
    return out, _reference(bits)


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: bf16_round_to_odd generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    out, ref = _launch()
    torch.testing.assert_close(out.cpu().view(torch.int16), ref, rtol=0, atol=0)
    print("RUN PASS: bf16_round_to_odd TileKernels round-to-odd exact bits and untouched sentinels")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: bf16_round_to_odd", "ir" if do_ir else "",
          "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
