from collections import deque

import numpy as np


# Simple class for maintaining running averages
# of numpy arrays
class RunningMean:
    def __init__(self,
                 max_size=10,
                 max_prunes=100):
        """Basic ctor"""
        self.max_size = max_size
        self.max_prunes = max_prunes
        self.prune_ctr = 0
        self.cache = deque()
        self.sum = None
        self.curr_mean = None

    def __len__(self):
        """Length operator"""
        return len(self.cache)

    def update_mean(self, new_value):
        """Update running mean"""
        self.cache.append(new_value)
        if self.sum is None:
            self.sum = np.copy(new_value)
        else:
            self.sum = np.add(self.sum, new_value)
        if self.prune():
            self.prune_ctr += 1
            if 0 < self.max_prunes < self.prune_ctr:
                self.prune_ctr = 0
                self.recalc()
        return self.build_mean()

    def clear(self):
        """Clear everything"""
        self.prune_ctr = 0
        self.cache.clear()
        self.sum = None
        self.curr_mean = None

    def recalc(self):
        """"Recalc running sum from values
        (addresses accumulated precision loss)"""
        self.sum = None
        for value in self.cache:
            if self.sum is None:
                self.sum = np.copy(value)
            else:
                self.sum = np.add(self.sum, value)

    def prune(self):
        """Discard old values"""
        result = False
        while len(self.cache) > self.max_size:
            result = True
            self.sum = np.subtract(self.sum, self.cache.popleft())
        return result

    def build_mean(self):
        """Recalc current mean"""
        self.curr_mean = (self.sum / float(len(self.cache)))
        return self.get_mean()

    def get_mean(self):
        """Gets current mean"""
        return self.curr_mean
