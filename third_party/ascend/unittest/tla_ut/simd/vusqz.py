"""Tests for native llvm.hivm.vusqz.m exclusive prefix counts."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline, get_stages, lower_offline


@tla.jit
def vusqz_kernel(M, O, D: tl.constexpr, N: tl.constexpr, DIST: tl.constexpr):
    m = tla.alloc_buffer((N, ), D, space="ub")
    o = tla.alloc_buffer((N, ), D, space="ub")
    tla.copy(tla.view(M, (N, )), m)
    with tla.scope.simd():
        mask = tla.simd.vcmps(tla.simd.vlds(m[0]), 0, op=tla.simd.CMP.NE)
        result = tla.simd.vusqz(mask, D)
        tla.simd.vsts(o[0], result, dist=DIST)
    tla.copy(o, tla.view(O, (N, )))


CASES = [
    (torch.uint8, tl.uint8, 256, "NORM_B8"),
    (torch.uint16, tl.uint16, 128, "NORM_B16"),
    (torch.uint32, tl.uint32, 64, "NORM_B32"),
]


def test_ir():
    for _, dtype, n, dist in CASES:
        mod, ctx = compile_offline(
            vusqz_kernel, {"M": "*" + dtype.name.replace("uint", "u"), "O": "*" + dtype.name.replace("uint", "u")},
            {"D": dtype, "N": n, "DIST": dist})
        text = str(lower_offline(mod, ctx))
        assert 'llvm.call_intrinsic "llvm.hivm.vusqz.m"' in text
        assert "ave.hir.vslide" not in text
    print("IR PASS: vusqz")


def test_run():
    torch.manual_seed(17)
    for td, dtype, n, dist in CASES:
        for name, mask in [
            ("zero", torch.zeros(n)),
            ("one", torch.ones(n)),
            ("alternating", torch.arange(n) % 2 == 0),
            ("sparse", torch.arange(n) % 3 != 0),
        ]:
            out = torch.empty(n, dtype=td, device="npu")
            vusqz_kernel[(1, )](mask.to(td).npu(), out, D=dtype, N=n, DIST=dist)
            torch.npu.synchronize()
            ref = mask.to(torch.int64).cumsum(0) - mask.to(torch.int64)
            torch.testing.assert_close(out.cpu().to(torch.int64), ref, rtol=0, atol=0)
            print("RUN PASS: vusqz", td, n, name)
    print("RUN PASS: vusqz all")


if __name__ == "__main__":
    do_ir, _, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_run:
        test_run()
