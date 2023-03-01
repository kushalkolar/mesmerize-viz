from itertools import chain
from functools import wraps
from typing import *


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
