class AList(list):
    '''a list that lets you append by setting to the next index'''
    def __setitem__(self, key, value):
        if key == len(self):
            self.append(None)
        return super().__setitem__(key, value)
