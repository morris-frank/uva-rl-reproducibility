from .memory import Memory

def test_memory():
    mem = Memory(2)
    mem.push(1)
    assert len(mem) == 1
    assert mem.sample(1) == [1]
