"""Composed dhistv2/chistv2 tests."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline, get_stages, lower_offline


@tla.jit
def hist_kernel(X, M, A, O, BIN: tl.constexpr, CUM: tl.constexpr):
    x = tla.alloc_buffer((256, ), tl.uint8, space="ub")
    m = tla.alloc_buffer((256, ), tl.uint8, space="ub")
    a = tla.alloc_buffer((128, ), tl.uint16, space="ub")
    o = tla.alloc_buffer((128, ), tl.uint16, space="ub")
    tla.copy(tla.view(X, (256, )), x)
    tla.copy(tla.view(M, (256, )), m)
    tla.copy(tla.view(A, (128, )), a)
    with tla.scope.simd():
        mask = tla.simd.vcmps(tla.simd.vlds(m[0]), 0, op=tla.simd.CMP.NE)
        src = tla.simd.vlds(x[0])
        acc = tla.simd.vlds(a[0])
        if CUM:
            result = tla.simd.chistv2(acc, src, mask, BIN)
        else:
            result = tla.simd.dhistv2(acc, src, mask, BIN)
        tla.simd.vsts(o[0], result, dist="NORM_B16")
    tla.copy(o, tla.view(O, (128, )))


def test_ir():
    for cumulative in (False, True):
        mod, ctx = compile_offline(hist_kernel, {"X": "*u8", "M": "*u8", "A": "*u16", "O": "*u16"},
                                   {"BIN": 1, "CUM": cumulative})
        text = str(lower_offline(mod, ctx))
        assert "ave.hir.reduction" in text
    print("IR PASS: histogram")


def test_run():
    torch.manual_seed(42)
    for cumulative in (False, True):
        for bin in (0, 1):
            for name, values, mask in [
                ("256x128", torch.full((256, ), 128), torch.ones(256)),
                ("empty", torch.arange(256), torch.zeros(256)),
                ("sparse", torch.arange(256), torch.arange(256) % 3 != 0),
            ]:
                acc = torch.arange(128, dtype=torch.int64) + 17
                out = torch.empty(128, dtype=torch.uint16, device="npu")
                hist_kernel[(1, )](values.to(torch.uint8).npu(), mask.to(torch.uint8).npu(), acc.to(torch.uint16).npu(),
                                   out, bin, cumulative)
                torch.npu.synchronize()
                bins = torch.arange(128) + 128 * bin
                hit = values[:, None] <= bins if cumulative else values[:, None] == bins
                ref = acc + (hit & mask.bool()[:, None]).sum(0)
                torch.testing.assert_close(out.cpu().to(torch.int64), ref, rtol=0, atol=0)
                print("RUN PASS:", "cum" if cumulative else "freq", "bin", bin, name)
    print("RUN PASS: histogram")


if __name__ == "__main__":
    do_ir, _, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_run:
        test_run()
