import random

class Memory(object):
    '''a limited-capacity sampleable memory'''

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._mem = []

    def __len__(self):
        return len(self._mem)

    def push(self, element: object):
        """
        Add an element to the memory
        :param element: the new element
        """
        if len(self._mem) > self.capacity:
            del self._mem[0]
        self._mem.append(element)

    def sample(self, n: int):
        """
        Get samples from the memory
        :param n: number of samples to get
        :return:
        """
        return random.sample(self._mem, n)

def get_get_epsilon(it_at_min, min_epsilon):
    def get_epsilon(it):
        if it >= it_at_min:
            return min_epsilon
        else:
            return -((1-min_epsilon)/it_at_min)*it + 1
    return get_epsilon
