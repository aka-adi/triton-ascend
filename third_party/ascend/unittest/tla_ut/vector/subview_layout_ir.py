"""Real frontend IR regression for static and dynamic subview layouts.

Run directly with a built TLA extension; no device allocation or mocks are used.
This checks frontend MLIR, not full NPUIR compilation or device numerics.
"""
import re
import sys
from pathlib import Path

import triton.language as tl
import triton.language.extra.tla as tla

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.tla_test_utils import compile_offline


@tla.jit
def subview_layout(K, MODE: tl.constexpr):
    with tla.scope.vector():
        root = tla.alloc((128,), tl.int32)
        if MODE == "zero":
            result = tla.subview(root, (0,), (32,))
        elif MODE == "static":
            result = tla.subview(root, (8,), (32,))
        elif MODE == "nested":
            first = tla.subview(root, (8,), (80,))
            result = tla.subview(first, (8,), (32,))
        elif MODE == "strided":
            first = tla.subview(root, (8,), (48,), (2,))
            result = tla.subview(first, (3,), (8,), (3,))
        else:
            result = tla.subview(root, (K,), (32,))


def test_subview_layout_ir():
    for mode in ("zero", "static", "nested", "strided", "dynamic"):
        module, context = compile_offline(subview_layout, {"K": "i32"}, {"MODE": mode})
        text = str(module)
        lines = [line for line in text.splitlines() if " = memref.subview " in line]
        assert len(lines) == (2 if mode in ("nested", "strided") else 1), text
        last = lines[-1]
        # Match result types separately: a source type must not satisfy this check.
        result_type = last.split(" to ", 1)[1].split(" loc(", 1)[0]
        if mode == "zero":
            assert "[0] [32] [1]" in last, last
            assert re.fullmatch(r"memref<32xi32, (?:strided<\[1\]>, )?#hivm.address_space<ub>>", result_type), last
        elif mode == "static":
            assert "[8] [32] [1]" in last, last
            assert "strided<[1], offset: 8>" in result_type, last
        elif mode == "nested":
            assert "strided<[1], offset: 8>" in lines[0].split(" to ", 1)[1], text
            assert "[8] [32] [1]" in last, last
            assert "strided<[1], offset: 16>" in result_type, last
        elif mode == "strided":
            assert "[8] [48] [2]" in lines[0], text
            assert "strided<[2], offset: 8>" in lines[0].split(" to ", 1)[1], text
            assert "[3] [8] [3]" in last, last
            assert "strided<[6], offset: 14>" in result_type, last
        else:
            match = re.search(r"memref.subview %[^\[]+\[(%[^\]]+)\] \[32\] \[1\]", last)
            assert match, last
            assert "strided<[1], offset: ?>" in result_type, last
            # Runtime offset is the index-cast SSA result, not an integer attribute.
            assert re.search(re.escape(match.group(1)) + r" = arith.index_cast %\S+ : i32 to index", text), text
        print("IR PASS: subview layout", mode)


if __name__ == "__main__":
    test_subview_layout_ir()
    print("ALL PASS: real subview layout IR")
