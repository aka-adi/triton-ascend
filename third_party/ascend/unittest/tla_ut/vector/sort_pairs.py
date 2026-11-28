"""Full UB sort_pairs: exact payload preservation, unstable ties."""
from collections import Counter
import sys
from pathlib import Path
import numpy as np
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.tla as tla
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.tla_test_utils import compile_offline, lower_offline, get_stages, check_kernel_binary

SIZES = (32,64,128,256,512,1024,2048,4096)
PATTERNS = ('random','ascending','descending','equal','duplicates','negative','extremes','infinity','signed_zero')

@tla.jit
def sort_kernel(S, I, O, W, SO, IO, N: tl.constexpr):
    with tla.scope.vector():
        scores = tla.alloc((N,), tl.float32)
        indices = tla.alloc((N,), tl.uint32)
        dst = tla.alloc((2*N,), tl.uint32)
        scratch = tla.alloc((2*N,), tl.uint32)
        tla.load_gm_to_ub(tla.view(S, (N,)), scores)
        tla.load_gm_to_ub(tla.view(I, (N,)), indices)
        tla.sort_pairs(scores, indices, dst, scratch)
        tla.store_ub_to_gm(dst, tla.view(O, (2*N,)))
        tla.store_ub_to_gm(scratch, tla.view(W, (2*N,)))
        tla.store_ub_to_gm(scores, tla.view(SO, (N,)))
        tla.store_ub_to_gm(indices, tla.view(IO, (N,)))


def scores_for(n, pattern, seed):
    rng = np.random.default_rng(seed)
    if pattern == 'random': x = rng.normal(size=n)
    elif pattern == 'ascending': x = np.arange(n)
    elif pattern == 'descending': x = np.arange(n)[::-1]
    elif pattern == 'equal': x = np.full(n, 7)
    elif pattern == 'duplicates': x = rng.integers(-4, 5, n)
    elif pattern == 'negative': x = -1-rng.random(n)*100
    elif pattern == 'extremes': x = np.resize(np.array([np.finfo(np.float32).max,-np.finfo(np.float32).max,np.finfo(np.float32).tiny,-np.finfo(np.float32).tiny],dtype=np.float32),n)
    elif pattern == 'infinity': x = np.resize([-np.inf, np.inf, 1, -1], n)
    elif pattern == 'signed_zero': x = np.resize(np.array([0.,-0.,1.,-1.],dtype=np.float32),n)
    else: raise AssertionError(pattern)
    return np.array(x, dtype=np.float32)


def check_pairs(scores, payload, actual):
    pairs = actual.reshape(-1,2)
    values = pairs[:,0].copy().view(np.float32)
    assert not np.isnan(values).any()
    assert np.all(values[:-1] >= values[1:]), 'not descending'
    expected = Counter(zip(scores.view(np.int32).tolist(),payload.tolist()))
    assert Counter(map(tuple,pairs.tolist())) == expected, 'score/index bits changed'


def launch(n, pattern, seed, compile_only=False):
    scores = scores_for(n,pattern,seed)
    # Alternating high bits and distinct opaque payloads; never compare as keys.
    ids = (np.arange(n,dtype=np.uint32)*np.uint32(2654435761)+np.uint32(seed)).view(np.int32)
    s_dev=torch.from_numpy(scores.copy()).npu()
    i_dev=torch.from_numpy(ids.copy()).npu()
    out=torch.empty(2*n,dtype=torch.int32).npu()
    workspace=torch.empty(2*n,dtype=torch.int32).npu()
    s_after=torch.empty_like(s_dev); i_after=torch.empty_like(i_dev)
    # uint32 UB buffers; torch keeps the same 32-bit words as signed int32.
    u32=lambda t: triton.reinterpret(t, tl.uint32)
    args=(s_dev,u32(i_dev),u32(out),u32(workspace),s_after,u32(i_after))
    if compile_only:
        return sort_kernel.warmup(*args,N=n,grid=(1,))
    sort_kernel[(1,)](*args,N=n)
    torch.npu.synchronize()
    actual=out.cpu().numpy()
    check_pairs(scores,ids,actual)
    np.testing.assert_array_equal(s_after.cpu().numpy().view(np.int32),scores.view(np.int32))
    np.testing.assert_array_equal(i_after.cpu().numpy(),ids)
    np.testing.assert_array_equal(s_dev.cpu().numpy().view(np.int32),scores.view(np.int32))
    np.testing.assert_array_equal(i_dev.cpu().numpy(),ids)


@tla.jit
def invalid_simd_scope():
    with tla.scope.simd():
        s = tla.alloc((32,), tl.float32)
        i = tla.alloc((32,), tl.uint32)
        d = tla.alloc((64,), tl.uint32)
        w = tla.alloc((64,), tl.uint32)
        tla.sort_pairs(s, i, d, w)

@tla.jit
def invalid_cube_scope():
    with tla.scope.cube():
        s = tla.alloc((32,), tl.float32)
        i = tla.alloc((32,), tl.uint32)
        d = tla.alloc((64,), tl.uint32)
        w = tla.alloc((64,), tl.uint32)
        tla.sort_pairs(s, i, d, w)


def test_scope_rejections():
    for kernel, message in ((invalid_simd_scope, 'outside SIMD/SIMT'),
                            (invalid_cube_scope, 'vector core')):
        try:
            compile_offline(kernel, {}, {})
        except Exception as error:
            assert message in str(error), str(error)
        else:
            raise AssertionError('sort_pairs accepted an invalid scope')
    print('VALIDATION PASS: sort_pairs scope restrictions')


def test_ir():
    signature={'S':'*fp32','I':'*u32','O':'*u32','W':'*u32','SO':'*fp32','IO':'*u32'}
    for n in SIZES:
        mod,ctx=compile_offline(sort_kernel,signature,{'N':n})
        text=str(mod)
        assert 'hivm.hir.vsort_pairs' in text
        assert 'hivm.vector_function' not in text
        lower_offline(mod,ctx)
    print('IR PASS: sort_pairs 8 sizes at offset 0')


def main():
    ir,compile_,run=get_stages()
    if ir:
        test_ir()
        test_scope_rejections()
    for n in SIZES:
        if compile_:
            check_kernel_binary(launch(n,'random',0,True))
            print('COMPILE PASS:',n)
        if run:
            # Reuse each specialization with changing data, including repeated random launches.
            for seed,pattern in enumerate(PATTERNS+('random',)):
                launch(n,pattern,seed)
            print('RUN PASS: sort_pairs',n,'offset 0, 10 inputs, exact pairs')
    print('ALL PASS: sort_pairs')

if __name__=='__main__': main()
