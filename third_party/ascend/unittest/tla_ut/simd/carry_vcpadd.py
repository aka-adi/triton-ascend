"""Composed vsubc/vaddcs/vsubcs/vcpadd tests."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline, get_stages, lower_offline

MASK32 = 0xFFFFFFFF


@tla.jit
def vsubc_kernel(X, Y, M, D, B, T: tl.constexpr):
    x = tla.alloc_buffer((64, ), T, space="ub")
    y = tla.alloc_buffer((64, ), T, space="ub")
    m = tla.alloc_buffer((64, ), tl.int32, space="ub")
    d = tla.alloc_buffer((64, ), T, space="ub")
    b = tla.alloc_buffer((64, ), tl.int32, space="ub")
    tla.copy(tla.view(X, (64, )), x)
    tla.copy(tla.view(Y, (64, )), y)
    tla.copy(tla.view(M, (64, )), m)
    with tla.scope.simd():
        mask = tla.simd.vcmps(tla.simd.vlds(m[0]), 0, op=tla.simd.CMP.NE)
        diff, borrow = tla.simd.vsubc(tla.simd.vlds(x[0]), tla.simd.vlds(y[0]), mask)
        tla.simd.vsts(d[0], diff)
        bits = tla.simd.vsel(borrow, tla.simd.vdup(1, tl.int32), tla.simd.vdup(0, tl.int32))
        tla.simd.vsts(b[0], bits)
    tla.copy(d, tla.view(D, (64, )))
    tla.copy(b, tla.view(B, (64, )))


@tla.jit
def vaddc_kernel(X, Y, M, D, B, T: tl.constexpr):
    x = tla.alloc_buffer((64, ), T, space="ub")
    y = tla.alloc_buffer((64, ), T, space="ub")
    m = tla.alloc_buffer((64, ), tl.int32, space="ub")
    d = tla.alloc_buffer((64, ), T, space="ub")
    b = tla.alloc_buffer((64, ), tl.int32, space="ub")
    tla.copy(tla.view(X, (64, )), x)
    tla.copy(tla.view(Y, (64, )), y)
    tla.copy(tla.view(M, (64, )), m)
    with tla.scope.simd():
        mask = tla.simd.vcmps(tla.simd.vlds(m[0]), 0, op=tla.simd.CMP.NE)
        total, carry_out = tla.simd.vaddc(tla.simd.vlds(x[0]), tla.simd.vlds(y[0]), mask)
        tla.simd.vsts(d[0], total)
        bits = tla.simd.vsel(carry_out, tla.simd.vdup(1, tl.int32), tla.simd.vdup(0, tl.int32))
        tla.simd.vsts(b[0], bits)
    tla.copy(d, tla.view(D, (64, )))
    tla.copy(b, tla.view(B, (64, )))


@tla.jit
def vaddcs_kernel(X, Y, C, M, D, B, T: tl.constexpr):
    x = tla.alloc_buffer((64, ), T, space="ub")
    y = tla.alloc_buffer((64, ), T, space="ub")
    c = tla.alloc_buffer((64, ), tl.int32, space="ub")
    m = tla.alloc_buffer((64, ), tl.int32, space="ub")
    d = tla.alloc_buffer((64, ), T, space="ub")
    b = tla.alloc_buffer((64, ), tl.int32, space="ub")
    tla.copy(tla.view(X, (64, )), x)
    tla.copy(tla.view(Y, (64, )), y)
    tla.copy(tla.view(C, (64, )), c)
    tla.copy(tla.view(M, (64, )), m)
    with tla.scope.simd():
        mask = tla.simd.vcmps(tla.simd.vlds(m[0]), 0, op=tla.simd.CMP.NE)
        carry = tla.simd.vcmps(tla.simd.vlds(c[0]), 0, op=tla.simd.CMP.NE)
        total, carry_out = tla.simd.vaddcs(tla.simd.vlds(x[0]), tla.simd.vlds(y[0]), carry, mask)
        tla.simd.vsts(d[0], total)
        bits = tla.simd.vsel(carry_out, tla.simd.vdup(1, tl.int32), tla.simd.vdup(0, tl.int32))
        tla.simd.vsts(b[0], bits)
    tla.copy(d, tla.view(D, (64, )))
    tla.copy(b, tla.view(B, (64, )))


@tla.jit
def vsubcs_kernel(X, Y, C, M, D, B, T: tl.constexpr):
    x = tla.alloc_buffer((64, ), T, space="ub")
    y = tla.alloc_buffer((64, ), T, space="ub")
    c = tla.alloc_buffer((64, ), tl.int32, space="ub")
    m = tla.alloc_buffer((64, ), tl.int32, space="ub")
    d = tla.alloc_buffer((64, ), T, space="ub")
    b = tla.alloc_buffer((64, ), tl.int32, space="ub")
    tla.copy(tla.view(X, (64, )), x)
    tla.copy(tla.view(Y, (64, )), y)
    tla.copy(tla.view(C, (64, )), c)
    tla.copy(tla.view(M, (64, )), m)
    with tla.scope.simd():
        mask = tla.simd.vcmps(tla.simd.vlds(m[0]), 0, op=tla.simd.CMP.NE)
        borrow = tla.simd.vcmps(tla.simd.vlds(c[0]), 0, op=tla.simd.CMP.NE)
        diff, borrow_out = tla.simd.vsubcs(tla.simd.vlds(x[0]), tla.simd.vlds(y[0]), borrow, mask)
        tla.simd.vsts(d[0], diff)
        bits = tla.simd.vsel(borrow_out, tla.simd.vdup(1, tl.int32), tla.simd.vdup(0, tl.int32))
        tla.simd.vsts(b[0], bits)
    tla.copy(d, tla.view(D, (64, )))
    tla.copy(b, tla.view(B, (64, )))


@tla.jit
def vcpadd_kernel(X, M, O, T: tl.constexpr, I: tl.constexpr, N: tl.constexpr):
    x = tla.alloc_buffer((N, ), T, space="ub")
    m = tla.alloc_buffer((N, ), I, space="ub")
    o = tla.alloc_buffer((N, ), T, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    tla.copy(tla.view(M, (N, )), m)
    with tla.scope.simd():
        mask = tla.simd.vcmps(tla.simd.vlds(m[0]), 0, op=tla.simd.CMP.NE)
        result = tla.simd.vcpadd(tla.simd.vlds(x[0]), mask)
        tla.simd.vsts(o[0], result)
    tla.copy(o, tla.view(O, (N, )))


def test_ir():
    for dtype, short in [(tl.int32, "i32"), (tl.uint32, "u32")]:
        for fn, sig in [(vaddc_kernel, {"X": "*" + short, "Y": "*" + short, "M": "*i32", "D": "*" + short, "B":
                                        "*i32"}),
                        (vsubc_kernel, {"X": "*" + short, "Y": "*" + short, "M": "*i32", "D": "*" + short, "B":
                                        "*i32"}),
                        (vaddcs_kernel,
                         {"X": "*" + short, "Y": "*" + short, "C": "*i32", "M": "*i32", "D": "*" + short, "B": "*i32"}),
                        (vsubcs_kernel,
                         {"X": "*" + short, "Y": "*" + short, "C": "*i32", "M": "*i32", "D": "*" + short, "B":
                          "*i32"})]:
            mod, ctx = compile_offline(fn, sig, {"T": dtype})
            lower_offline(mod, ctx)
    for dtype, short, mshort, mtype, n in [(tl.float32, "fp32", "i32", tl.int32, 64),
                                           (tl.float16, "fp16", "i16", tl.int16, 128)]:
        mod, ctx = compile_offline(vcpadd_kernel, {"X": "*" + short, "M": "*" + mshort, "O": "*" + short},
                                   {"T": dtype, "I": mtype, "N": n})
        text = str(lower_offline(mod, ctx))
        assert 'llvm.call_intrinsic "llvm.hivm.vcpadd.x"' in text
        assert "ave.hir.vdintlv" not in text
    print("IR PASS: carry/vcpadd")


def _u32(x):
    return x.to(torch.int64) & MASK32


def _ref_carry(x, y, carry, mask, op):
    ux, uy = _u32(x), _u32(y)
    active = mask.bool()
    cin = carry.bool()
    if op == "subc":
        diff = (ux - uy) & MASK32
        out = (ux < uy) & active
    elif op == "addcs":
        partial = (ux + uy) & MASK32
        c0 = (ux + uy) > MASK32
        total = (partial + cin.to(torch.int64)) & MASK32
        c1 = (partial == MASK32) & cin
        out = (c0 | c1) & active
        diff = total
    else:
        partial = (ux - uy) & MASK32
        b0 = ux < uy
        diff = (partial - cin.to(torch.int64)) & MASK32
        b1 = partial < cin.to(torch.int64)
        out = (b0 | b1) & active
    diff = torch.where(active, diff, torch.zeros_like(diff))
    return diff.to(torch.int32), out.to(torch.int32)


def _run_carry(fn, x, y, carry, mask):
    d = torch.empty(64, dtype=torch.int32, device="npu")
    b = torch.empty(64, dtype=torch.int32, device="npu")
    args = [x.npu(), y.npu()
            ] + ([carry.to(torch.int32).npu()] if carry is not None else []) + [mask.to(torch.int32).npu(), d, b]
    fn[(1, )](*args, T=tl.int32)
    torch.npu.synchronize()
    return d.cpu(), b.cpu()


def test_run():
    torch.manual_seed(11)
    edge = torch.tensor([0, 1, -1, -2, 0x7FFFFFFF, -0x80000000, 0x7FFFFFFE, -0x7FFFFFFF], dtype=torch.int32)
    x = torch.cat([edge, torch.randint(-(1 << 31), 1 << 31, (64 - edge.numel(), ), dtype=torch.int32)])
    y = torch.cat([edge.flip(0), torch.randint(-(1 << 31), 1 << 31, (64 - edge.numel(), ), dtype=torch.int32)])
    for mask in [torch.ones(64, dtype=torch.int32), (torch.arange(64) % 3 != 0).to(torch.int32)]:
        for carry in [
                torch.zeros(64, dtype=torch.int32),
                torch.ones(64, dtype=torch.int32), (torch.arange(64) % 2).to(torch.int32)
        ]:
            d, b = _run_carry(vaddc_kernel, x, y, None, mask)
            rd, rb = _ref_carry(x, y, torch.zeros(64), mask, "addcs")
            torch.testing.assert_close(d, rd, rtol=0, atol=0)
            torch.testing.assert_close(b, rb, rtol=0, atol=0)
            d, b = _run_carry(vsubc_kernel, x, y, None, mask)
            rd, rb = _ref_carry(x, y, torch.zeros(64), mask, "subc")
            torch.testing.assert_close(d, rd, rtol=0, atol=0)
            torch.testing.assert_close(b, rb, rtol=0, atol=0)
            d, b = _run_carry(vaddcs_kernel, x, y, carry, mask)
            rd, rb = _ref_carry(x, y, carry, mask, "addcs")
            torch.testing.assert_close(d, rd, rtol=0, atol=0)
            torch.testing.assert_close(b, rb, rtol=0, atol=0)
            d, b = _run_carry(vsubcs_kernel, x, y, carry, mask)
            rd, rb = _ref_carry(x, y, carry, mask, "subcs")
            torch.testing.assert_close(d, rd, rtol=0, atol=0)
            torch.testing.assert_close(b, rb, rtol=0, atol=0)
    print("RUN PASS: vsubc/vaddcs/vsubcs")

    for td, md, dtype, mtype, n in [(torch.float32, torch.int32, tl.float32, tl.int32, 64),
                                    (torch.float16, torch.int16, tl.float16, tl.int16, 128)]:
        x = torch.arange(n, dtype=td)
        for mask in [torch.ones(n, dtype=md), (torch.arange(n) % 3 != 0).to(md)]:
            out = torch.empty(n, dtype=td, device="npu")
            vcpadd_kernel[(1, )](x.npu(), mask.npu(), out, T=dtype, I=mtype, N=n)
            torch.npu.synchronize()
            gated = torch.where(mask.bool(), x, torch.zeros_like(x))
            ref = torch.zeros(n, dtype=td)
            ref[:n // 2] = gated[0::2] + gated[1::2]
            torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
    print("RUN PASS: vcpadd")


if __name__ == "__main__":
    do_ir, _, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_run:
        test_run()
