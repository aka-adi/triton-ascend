"""Masked SIMD arithmetic and strided-store frontend/intrinsic contracts."""
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline, lower_offline


@triton.jit
def simd_contracts(X, O, limit, BAD: tl.constexpr):
    src = tla.alloc_buffer((1, 64), tl.float32, space="ub")
    dst = tla.alloc_buffer((8, 16), tl.float32, space="ub")
    tla.copy(tla.view(X, (1, 64)), src)
    with tla.scope.simd():
        v = tla.simd.vlds(src[0, 0])
        m = tla.simd.pge(32, 64)
        if BAD == 1:
            v = tla.simd.vadd(v, v, mode=tla.simd.MODE.MERGING)
        elif BAD == 2:
            v = tla.simd.vmins(v, v)
        elif BAD == 3:
            tla.simd.vsstb(v, dst, 9)
        elif BAD == 4:
            tla.simd.vsstb(v, dst, 16, update=True)
        elif BAD == 5:
            v = tla.simd.vxor(v, v, tla.simd.pge(32, 32))
        v = tla.simd.vmins(v, limit)
        v = tla.simd.vadd(v, v, m)
        v = tla.simd.vsub(v, tla.simd.vsplat(0.5, tl.float32))
        bits = tla.simd.vbitcast(v, "int32")
        bits = tla.simd.vor(bits, tla.simd.vsplat(1, tl.int32))
        bits = tla.simd.vxor(bits, tla.simd.vsplat(3, tl.int32))
        v = tla.simd.vbitcast(bits, "fp32")
        tla.simd.vsstb(v, dst, 16)
    tla.copy(dst, tla.view(O, (8, 16)))


def _compile(bad=0):
    return compile_offline(simd_contracts, {"X": "*fp32", "O": "*fp32", "limit": "fp32"},
                           {"BAD": bad})


def test_ir(output_dir=None):
    module, ctx = _compile()
    text = str(module)
    assert "ave.hir.vsel" in text  # explicit ZEROING
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    for name in ("vmins", "vadd", "vsub", "vor", "vxor", "store_with_stride"):
        assert "ave.hir." + name in text, name
    assert "hivm.vector_function" in text
    print("IR PASS: masked SIMD arithmetic and strided store")


def test_negative():
    for bad, expected in enumerate(("MERGING", "scalar must be a scalar tensor",
                                    "32B block", "update=True", "mask must"), 1):
        try:
            _compile(bad)
        except Exception as exc:
            assert expected in str(exc), str(exc)
        else:
            raise AssertionError(f"invalid case {bad} accepted")
    print("NEGATIVE PASS: unsupported modes, scalar, stride and predicate contracts")


def test_backend():
    opt = os.environ.get("BISHENGIR_OPT")
    if not opt:
        return
    module, ctx = _compile()
    module = lower_offline(module, ctx)
    result = subprocess.run(
        [opt, "--outline-scope=outline-marked-scopes-only=true",
         "--hacc-append-device-spec=target=Ascend910_9589",
         "--ave-normalize-ops", "--remove-vector-layout-attr",
         "--convert-hivmave-to-ave-intrin"],
        input=str(module), text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    for name in ("vmins.s.x", "vadd.s.x", "vsub.s.x", "vor.x", "vxor.x", "vsstb.v64f32"):
        assert "intr.hivm." + name in result.stdout, name
    print("BACKEND PASS: explicit AVE to RegBase intrinsics")


if __name__ == "__main__":
    test_ir()
    test_negative()
    test_backend()
