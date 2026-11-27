"""Host-only sort_pairs contract; does not substitute for device tests."""
import importlib
from pathlib import Path
import sys
import types
import pytest
pytest.importorskip('triton')
import triton.language as tl
pkg = types.ModuleType('sort_tla_src')
pkg.__path__ = [str(Path(__file__).resolve().parents[3] / 'language' / 'tla')]
sys.modules.setdefault('sort_tla_src', pkg)
sort = importlib.import_module('sort_tla_src.sort')
core = importlib.import_module('sort_tla_src.core')

class Recorder:
    def __init__(self): self.calls = []
    def create_tla_sort_pairs(self, *args): self.calls.append(args)

def buf(n, dtype, **kw):
    return core.buffer(object(), core.buffer_type((n,), dtype, **kw))

def inputs(n=32, offset=0):
    return [buf(n, tl.float32, space='ub', offset=offset),
            buf(n, tl.uint32, space='ub', offset=offset),
            buf(2*n, tl.uint32, space='ub', offset=offset),
            buf(2*n, tl.uint32, space='ub', offset=offset)]

@pytest.mark.parametrize('n', [32,64,128,256,512,1024,2048,4096])
@pytest.mark.parametrize('offset', [0,8,16])
def test_legal(n, offset):
    args = inputs(n, offset)
    sem = types.SimpleNamespace(builder=Recorder())
    assert sort.sort_pairs(*args, _semantic=sem) is None
    assert sem.builder.calls == [tuple(x.handle for x in args)]

@pytest.mark.parametrize('replacement,exc', [
    (object(), TypeError),
    (buf(32,tl.float32,space='gm'), ValueError),
    (buf(32,tl.float16,space='ub'), TypeError),
    (buf(32,tl.float32,space='ub',strides=(2,)), ValueError),
    (buf(32,tl.float32,space='ub',offset=None), ValueError),
    (buf(32,tl.float32,space='ub',offset=1), ValueError),
    (buf(32,tl.float32,space='ub',offset=-8), ValueError),
    (core.buffer(object(),core.buffer_type((1,32),tl.float32,'ub')), ValueError),
])
def test_invalid_scores(replacement,exc):
    args=inputs();args[0]=replacement;sem=types.SimpleNamespace(builder=Recorder())
    with pytest.raises(exc): sort.sort_pairs(*args,_semantic=sem)
    assert not sem.builder.calls

@pytest.mark.parametrize('n',[16,48,8192])
def test_bad_size(n):
    with pytest.raises(ValueError,match='power-of-two'): sort._check_sort_pairs(*inputs(n))

@pytest.mark.parametrize('slot',[1,2,3])
def test_bad_payload_dtype_and_shape(slot):
    args=inputs();args[slot]=buf(args[slot].shape[0],tl.int32,space='ub')
    with pytest.raises(TypeError): sort._check_sort_pairs(*args)
    args=inputs();args[slot]=buf(16,tl.uint32,space='ub')
    with pytest.raises(ValueError): sort._check_sort_pairs(*args)

def test_same_handle():
    args=inputs();args[3]=args[2]
    with pytest.raises(ValueError,match='non-overlapping'): sort._check_sort_pairs(*args)
