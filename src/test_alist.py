from alist import AList

def test_alist():
    lst = AList([1,2,3])
    lst[3] = 4
    assert len(lst) == 4
