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
