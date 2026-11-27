"""TLA 3-in-1 test: low-level ``vbs`` / ``vmrgsort4`` UB pair primitives.

Stages: ir (offline frontend IR), compile (kernel artifacts), run (numerics).

Coverage
  * N=256: one ``vbs`` plus the two ``vmrgsort4`` merges (run sizes 32 and
    128) needed for a full descending sort, and one ``vbs`` plus a single
    four-way merge (run size 32) that leaves two sorted 128-pair runs;
  * N=64: one ``vbs`` plus one two-way merge (run size 32), and the same
    ``vbs`` without any merge, which leaves four sorted 32-pair blocks;
  * N=96: ``vbs`` alone, proving the primitive accepts non-power-of-two score
    counts that ``sort_pairs`` rejects;
  * frontend validation errors: score/index/destination dtypes, rank,
    destination size, score-count range and modulus, merge buffer length,
    the 8192-word limit, run-size type, range, modulus and divisibility, and
    incomplete two-/four-way merge groups.

The IR stage asserts the AVE ops in both the frontend module and the
ttadapter-boundary IR.  The compile/run stages need a rebuilt TLA extension
and an NPU; they were not executed when the primitives were first exposed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (check_compile_artifacts, check_lowered_ir, compile_offline, get_stages,
                                  lower_offline)

SIGNATURE = {"Scores": "*fp32", "Indices": "*u32", "Out": "*u32"}
NEG_SIGNATURE = {"O": "*u32"}

# (N, STAGES): STAGES = (N.bit_length() - 5) // 2 fully sorts N pairs.
FULL_CASES = ((256, 2), (64, 1))


@tla.jit
def sort_kernel(Scores, Indices, Out, N: tl.constexpr, STAGES: tl.constexpr):
    """``vbs`` then STAGES ping-pong merges; the result must end in ``Out``."""
    with tla.scope.vector():
        scores = tla.alloc((N, ), tl.float32)
        indices = tla.alloc((N, ), tl.uint32)
        pairs = tla.alloc((2 * N, ), tl.uint32)
        scratch = tla.alloc((2 * N, ), tl.uint32)
        tla.load_gm_to_ub(tla.view(Scores, (N, )), scores)
        tla.load_gm_to_ub(tla.view(Indices, (N, )), indices)
        # Ping-pong so that the final result always lands in ``pairs``.
        if STAGES % 2 == 0:
            tla.simd.vbs(scores, indices, pairs)
        else:
            tla.simd.vbs(scores, indices, scratch)
        for stage in tl.static_range(STAGES):
            if (STAGES - stage) % 2 == 0:
                tla.simd.vmrgsort4(pairs, scratch, 32 * 4**stage)
            else:
                tla.simd.vmrgsort4(scratch, pairs, 32 * 4**stage)
        tla.store_ub_to_gm(pairs, tla.view(Out, (2 * N, )))


@tla.jit
def bad_vbs(O, N: tl.constexpr, MODE: tl.constexpr):
    """One invalid ``vbs`` operand set per MODE; the call must be rejected."""
    with tla.scope.vector():
        if MODE == "scores_dtype":
            scores = tla.alloc((N, ), tl.int32)
            indices = tla.alloc((N, ), tl.uint32)
            dst = tla.alloc((2 * N, ), tl.uint32)
        elif MODE == "indices_dtype":
            scores = tla.alloc((N, ), tl.float32)
            indices = tla.alloc((N, ), tl.int32)
            dst = tla.alloc((2 * N, ), tl.uint32)
        elif MODE == "dst_dtype":
            scores = tla.alloc((N, ), tl.float32)
            indices = tla.alloc((N, ), tl.uint32)
            dst = tla.alloc((2 * N, ), tl.int32)
        elif MODE == "rank":
            scores = tla.alloc((8, N // 8), tl.float32)
            indices = tla.alloc((N, ), tl.uint32)
            dst = tla.alloc((2 * N, ), tl.uint32)
        elif MODE == "dst_size":
            scores = tla.alloc((N, ), tl.float32)
            indices = tla.alloc((N, ), tl.uint32)
            dst = tla.alloc((N, ), tl.uint32)
        else:
            scores = tla.alloc((N, ), tl.float32)
            indices = tla.alloc((N, ), tl.uint32)
            dst = tla.alloc((2 * N, ), tl.uint32)
        tla.simd.vbs(scores, indices, dst)
        tla.store_ub_to_gm(dst, tla.view(O, (2 * N, )))


@tla.jit
def bad_vmrgsort4(O, WORDS: tl.constexpr, RUN: tl.constexpr, MODE: tl.constexpr):
    """One invalid ``vmrgsort4`` operand set per MODE; the call must be rejected."""
    with tla.scope.vector():
        if MODE == "src_dtype":
            src = tla.alloc((WORDS, ), tl.int32)
            dst = tla.alloc((WORDS, ), tl.uint32)
        elif MODE == "dst_dtype":
            src = tla.alloc((WORDS, ), tl.uint32)
            dst = tla.alloc((WORDS, ), tl.int32)
        elif MODE == "length":
            src = tla.alloc((WORDS, ), tl.uint32)
            dst = tla.alloc((WORDS + 2, ), tl.uint32)
        else:
            src = tla.alloc((WORDS, ), tl.uint32)
            dst = tla.alloc((WORDS, ), tl.uint32)
        tla.simd.vmrgsort4(src, dst, RUN)
        tla.store_ub_to_gm(dst, tla.view(O, (WORDS, )))


def test_ir():
    for n, stages, merges, runs in ((256, 2, 2, (32, 128)), (256, 1, 1, (32, )), (64, 1, 1, (32, )), (64, 0, 0, ()),
                                    (96, 0, 0, ())):
        module, ctx = compile_offline(sort_kernel, SIGNATURE, {"N": n, "STAGES": stages})
        text = str(module)
        assert text.count("ave.hir.vbs") == 1, f"N={n} must emit exactly one vbs"
        assert text.count("ave.hir.vmrgsort4") == merges, f"N={n} must emit {merges} merges"
        for run in runs:
            assert f"run_size = {run} :" in text, f"N={n} must merge with run_size {run}"
        # The sort ops must sit outside a register VF at the frontend boundary.
        assert "hivm.vector_function" not in text, "vbs/vmrgsort4 must stay outside a register vector function"
        lowered = str(lower_offline(module, ctx))
        assert lowered.count("ave.hir.vbs") == 1, f"N={n} lost vbs at the ttadapter boundary"
        assert lowered.count("ave.hir.vmrgsort4") == merges, f"N={n} lost merges at the ttadapter boundary"
        check_lowered_ir(lowered, loads=2, stores=1, extra=("ave.hir.vbs", ))
    print("IR PASS: vbs/vmrgsort4 N=256/64/96 at frontend and ttadapter boundary")


def _expect_error(fn, needle):
    try:
        fn()
    except Exception as error:  # noqa: BLE001 - the frontend raises several kinds
        assert needle in str(error), f"expected {needle!r} in {type(error).__name__}: {error}"
        return
    raise AssertionError(f"expected an error containing {needle!r}")


def test_validation():
    for mode, needle in (
        ("scores_dtype", "expected a fp32 buffer"),
        ("indices_dtype", "expected a uint32 buffer"),
        ("dst_dtype", "expected a uint32 buffer"),
        ("rank", "contiguous one-dimensional buffer"),
        ("dst_size", "two destination words per score"),
    ):
        _expect_error(lambda mode=mode: compile_offline(bad_vbs, NEG_SIGNATURE, {"N": 256, "MODE": mode}), needle)
    for n in (24, 48, 4128):
        _expect_error(lambda n=n: compile_offline(bad_vbs, NEG_SIGNATURE, {"N": n, "MODE": "width"}),
                      "multiple of 32 in [32, 4096]")

    for mode, needle in (
        ("src_dtype", "expected a uint32 buffer"),
        ("dst_dtype", "expected a uint32 buffer"),
        ("length", "same number of words"),
    ):
        _expect_error(
            lambda mode=mode: compile_offline(bad_vmrgsort4, NEG_SIGNATURE, {"WORDS": 512, "RUN": 32, "MODE": mode}),
            needle)
    _expect_error(lambda: compile_offline(bad_vmrgsort4, NEG_SIGNATURE, {"WORDS": 8256, "RUN": 32, "MODE": "ok"}),
                  "at most 8192 words")
    # run_size must be a compile-time int in [32, words // 4], a multiple of 32
    # that divides the word count in half.
    for run in (16, 33, 256):
        _expect_error(
            lambda run=run: compile_offline(bad_vmrgsort4, NEG_SIGNATURE, {"WORDS": 512, "RUN": run, "MODE": "ok"}),
            "multiple of 32 in [32, words // 4]")
    _expect_error(lambda: compile_offline(bad_vmrgsort4, NEG_SIGNATURE, {"WORDS": 512, "RUN": "bad", "MODE": "ok"}),
                  "compile-time int")
    # 384 words / (2 * 32) = 6 runs, which is neither two-way nor a four-way multiple.
    _expect_error(lambda: compile_offline(bad_vmrgsort4, NEG_SIGNATURE, {"WORDS": 384, "RUN": 32, "MODE": "ok"}),
                  "complete two-way or four-way merge groups")
    print("VALIDATION PASS: vbs/vmrgsort4 dtype, shape, range, run-size and group errors")


def _launch(n, stages):
    import torch_npu  # noqa: F401: register the NPU backend only for device stages

    gen = torch.Generator().manual_seed(20261126)
    scores = torch.randn(n, generator=gen)
    payload = torch.randperm(n, dtype=torch.int32, generator=gen)
    out = torch.empty(2 * n, dtype=torch.int32, device="npu")
    sort_kernel[(1, )](scores.npu(), triton.reinterpret(payload.npu(), tl.uint32), triton.reinterpret(out, tl.uint32), N=n,
                       STAGES=stages)
    torch.npu.synchronize()
    return scores, payload, out


def test_compile():
    for n, stages in FULL_CASES:
        _launch(n, stages)
        print("COMPILE PASS: vbs/vmrgsort4", n, "pairs,", stages, "merge stages")
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vbs/vmrgsort4 generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    for n, stages in FULL_CASES:
        scores, payload, out = _launch(n, stages)
        pairs = out.cpu().reshape(-1, 2)
        values = pairs[:, 0].contiguous().view(torch.float32)
        indices = pairs[:, 1]
        assert bool((values[:-1] >= values[1:]).all()), "not sorted descending"
        assert sorted(indices.tolist()) == list(range(n)), "uint32 payload not preserved"
        torch.testing.assert_close(values, scores.sort(descending=True).values)
        assert set(indices.tolist()) == set(payload.tolist())
        print("RUN PASS: vbs/vmrgsort4", n, "pairs fully sorted, payload preserved")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
        test_validation()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vbs/vmrgsort4", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
