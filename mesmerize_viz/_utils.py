from itertools import chain
from functools import wraps
from typing import *

import numpy as np
from mesmerize_core.arrays._base import LazyArray


def validate_data_options():
    def dec(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if "data" in kwargs:
                data = kwargs["data"]
            else:
                if len(args) > 0:
                    data = args[0]
                else:
                    # assume the extension func will take care of it
                    # the default data arg is None is nothing is passed
                    return func(self, *args, **kwargs)


            # flatten
            if any([isinstance(d, (list, tuple)) for d in data]):
                data = list(chain.from_iterable(data))

            valid_options = list(self._data_mapping.keys())

            for d in data:
                if d not in valid_options:
                    raise KeyError(f"Invalid data option: \"{d}\", valid options are:"
                                   f"\n{valid_options}")
            return func(self, *args, **kwargs)

        return wrapper

    return dec


class ZeroArray(LazyArray):
    """
    This array is used as placeholders to allow mixing data of different ndims in the ImageWidget.
    For example this allows having mean, max etc. projections in the same ImageWidget as the
    input or mcorr movie. It also allows having LineStacks or Heatmap in the same ImageWidget.
    """
    def __init__(self, ndim, n_frames):
        self._shape = [1] * ndim
        self._shape[0] = n_frames
        self.rval = np.zeros(shape=self.shape, dtype=np.int8)

    @property
    def dtype(self) -> str:
        return "int8"

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self._shape)

    @property
    def n_frames(self) -> int:
        return self.shape[0]

    @property
    def min(self) -> float:
        return 0.0

    @property
    def max(self) -> float:
        return 0.0

    def _compute_at_indices(self, indices: Union[int, slice]) -> np.ndarray:
        return np.zeros(shape=(1, 1, 1), dtype=np.int8)
