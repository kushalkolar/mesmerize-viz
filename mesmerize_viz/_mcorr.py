from collections import OrderedDict
from typing import *
from functools import partial

import numpy as np
import pandas as pd
from mesmerize_core import MCorrExtensions
from mesmerize_core.caiman_extensions._utils import validate as validate_algo
from fastplotlib import ImageWidget

from ._utils import validate_data_options


projs = [
    "mean",
    "max",
    "std",
]


@pd.api.extensions.register_series_accessor("mcorr")
class MCorrExtensionsViz(MCorrExtensions):
    @property
    def _data_mapping(self) -> Dict[str, callable]:
        projections = {k: partial(self._series.caiman.get_projection, k) for k in projs}
        m = {
            "input": self._series.caiman.get_input_movie,
            "mcorr": self.get_output,
            "corr": self._series.caiman.get_corr_image,
            **projections
        }
        return m

    @validate_algo("mcorr")
    @validate_data_options()
    def viz(
            self,
            data: List[str] = None,
            input_movie_kwargs: dict = None,
            image_widget_kwargs: dict = None,
    ):
        """
        Visualize motion correction output.

        Parameters
        ----------
        data
        input_movie_kwargs
        image_widget_kwargs

        Returns
        -------

        """
        if data is None:
            data = ["input", "mcorr"]

        if input_movie_kwargs is None:
            input_movie_kwargs = dict()

        if image_widget_kwargs is None:
            image_widget_kwargs = dict()

        data_arrays = list()

        for d in data:
            func = self._data_mapping[d]

            if d == "input":
                a = func(**input_movie_kwargs)
            else:
                a = func()

            data_arrays.append(a)

        # default kwargs unless user has specified
        default_iw_kwargs = {
            "window_funcs": {"t": (np.mean, 11)},
            "vmin_vmax_sliders": True,
            "cmap": "gnuplot2"
        }

        image_widget_kwargs = {
            **default_iw_kwargs,
            **image_widget_kwargs
        }

        iw = ImageWidget(
            data=data_arrays,
            names=data,
            **image_widget_kwargs
        )

        iw.show()
        return iw
