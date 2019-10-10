from .utils import get_get_epsilon

def test_get_get_epsilon():
    it_at_min = 3
    min_epsilon = .01
    get_eps = get_get_epsilon(it_at_min, min_epsilon)
    assert get_eps(0) == 1
    assert get_eps(3) == .01
    assert get_eps(5) == .01
