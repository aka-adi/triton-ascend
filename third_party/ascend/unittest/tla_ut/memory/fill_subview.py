"""Fill preserves its pre-sort restriction on all subview-derived buffers.

Real frontend tests only. The previous positive device probe is retained in
artifacts/sort-pairs/fill-regression; full-allocation template linking is outside
this regression's scope.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline


@tla.jit
def rejected(K, MODE: tl.constexpr):
    with tla.scope.vector():
        full = tla.alloc((128,), tl.int32)
        if MODE == 0:
            dst = tla.subview(full, (0,), (32,))
        elif MODE == 1:
            dst = tla.subview(full, (8,), (32,))
        elif MODE == 2:
            dst = tla.subview(full, (K,), (32,))
        elif MODE == 3:
            first = tla.subview(full, (0,), (64,))
            dst = tla.subview(first, (0,), (32,))
        else:
            first = tla.subview(full, (0,), (32,))
            dst = tla.reshape(first, (4, 8))
        tla.fill(dst, -3)


@tla.jit
def accepted(RESHAPE: tl.constexpr):
    with tla.scope.vector():
        full = tla.alloc((128,), tl.int32)
        if RESHAPE:
            dst = tla.reshape(full, (4, 32))
        else:
            dst = full
        tla.fill(dst, -3)


def main():
    for mode in range(5):
        try:
            compile_offline(rejected, {"K": "i32"}, {"MODE": mode})
        except Exception as error:
            expected = "subview-derived" if mode in (0, 3, 4) else "offset 0"
            assert expected in str(error), str(error)
        else:
            raise AssertionError(f"fill accepted subview mode {mode}")
        print("VALIDATION PASS: fill rejects subview", mode)
    for reshape in (False, True):
        module, _ = compile_offline(accepted, {}, {"RESHAPE": reshape})
        assert "hivm.hir.vbrc" in str(module)
        print("IR PASS: fill accepts non-subview allocation", reshape)
    print("ALL PASS: fill subview restriction (frontend only)")


if __name__ == "__main__":
    main()
