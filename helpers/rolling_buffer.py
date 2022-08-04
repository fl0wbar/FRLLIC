import numpy as np
from common.assertions import assert_exc

import pytorch_ext as pe


class BufferSizeMismatch(Exception):
    pass


class RollingBufferHistogram(object):
    """
    Buffer that sets shape to be number of entries of first v it receives.
    Has function `plot` to plot histogram of data.
    """
    def __init__(self, buffer_size, name=None):
        self._name = name or 'RollingBufferHistogram'
        self._buffer = None
        self._buffer_size = buffer_size
        self._idx = 0
        self._filled_idx = 0

    def add(self, v):
        v = pe.tensor_to_np(v)
        num_values = np.prod(v.shape)
        if self._buffer is None:
            print(f'Creating {v.dtype} buffer for {self._name}: {self._buffer_size}x{num_values}')
            self._buffer = np.zeros((self._buffer_size, num_values), dtype=v.dtype)
        assert_exc(self._buffer.shape[1] == num_values, (self._buffer.shape, v.shape, num_values), BufferSizeMismatch)
        self._buffer[self._idx, :] = v.flatten()
        self._idx = (self._idx + 1) % self._buffer_size
        self._filled_idx = min(self._filled_idx + 1, self._buffer_size)

    def get_buffer(self):
        return self._buffer[:self._filled_idx, :]

    def plot(self, bins='auto', most_mass=0):
        counts, bins = np.histogram(self._buffer[:self._filled_idx], bins)
        counts = counts / self._filled_idx  # normalize it by number of entries
        idx_min, idx_max = _most_mass_indices(counts, most_mass)
        return bins[idx_min:idx_max], counts[idx_min:idx_max]


def _most_mass_indices(a, mass=0):
    total_mass = np.sum(a)
    threshold = total_mass * mass
    indices, = (a > threshold).nonzero()  # non zero returns a tuple of len a.ndim -> unpack!
    return indices[0], indices[-1]


