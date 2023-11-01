from typing import *

import numpy as np


# to format params dict into yaml-like string
is_pos = lambda x: 1 if x > 0 else 0
# this doesn't work without the lambda, yes it is ugly
format_params = lambda d, t: "\n" * is_pos(t) + \
    "\n".join(
        [": ".join(["   " * t + k, format_params(v, t + 1)]) for k, v in d.items()]
    ) if isinstance(d, dict) else str(d)


class DummyMovie:
    """Really really hacky"""
    def __init__(self, image: np.ndarray, shape, ndim, size):
        self.image = image
        self.shape = shape
        self.ndim = ndim
        self.size = size

    def __getitem__(self, index: Union[int, slice]):
        if isinstance(index, tuple):
            for s in index:
                if isinstance(s, int):
                    # assumption
                    index = s
                    break

                if (s.start is None) and (s.stop is None) and (s.step is None):
                    continue
                else:
                    # assume that this is the dimension that user has asked for, and we return the image using
                    # slice size from this dimension
                    index = s

        if isinstance(index, (slice, range)):
            start, stop, step = index.start, index.stop, index.step

            if start is None:
                start = 0

            if stop is None:
                # assumption, again this is very hacky
                stop = max(self.shape)

            if step is None:
                step = 1

            r = range(start, stop, step)

            n_frames = len(r)

            return np.array([self.image] * n_frames)

        if isinstance(index, int):
            return self.image

        else:
            raise TypeError(f"DummyMovie only accept int or slice indexing, you have passed: {index}")
