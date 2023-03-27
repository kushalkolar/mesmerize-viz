from collections import OrderedDict
from typing import *
from functools import partial

import numpy as np
import pandas as pd
from mesmerize_core import MCorrExtensions
from mesmerize_core.caiman_extensions._utils import validate as validate_algo
from fastplotlib import ImageWidget

from ._utils import validate_data_options, ZeroArray


projs = [
    "mean",
    "max",
    "std",
]


# these are directly manged by the image widget since they are [t, x, y]
standard_mappings = ["input", "mcorr"]


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

    @property
    def _zero_array(self):
        mcorr = self.get_output()
        return ZeroArray(ndim=mcorr.ndim, n_frames=mcorr.shape[0])

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
        data: list of str, default ["input", "mcorr"]
            list of data to plot, can also be a list of lists.

        input_movie_kwargs: dict, optional
            kwargs passed to get_input_movie()

        image_widget_kwargs: dict, optional
            kwargs passed to ImageWidget

        Returns
        -------
        ImageWidget
            fastplotlib.ImageWidget visualization
        """

        if data is None:
            # default viz
            data = ["input", "mcorr"]

        if input_movie_kwargs is None:
            input_movie_kwargs = dict()

        if image_widget_kwargs is None:
            image_widget_kwargs = dict()

        data_arrays = list()
        # data arrays directly passed to image widget
        data_arrays_iw = list()

        for d in data:
            if d in standard_mappings:
                func = self._data_mapping[d]

                if d == "input":
                    a = func(**input_movie_kwargs)
                else:
                    a = func()

                data_arrays_iw.append(a)

            else:
                # make a placeholder array to keep imagewidget happy
                # hacky but this is the best way for now
                zero_array = self._zero_array
                data_arrays_iw.append(zero_array)

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
            data=data_arrays_iw,
            names=data,
            **image_widget_kwargs
        )

        for a, n in zip(data_arrays_iw, data):
            if isinstance(a, ZeroArray):
                # rename the existing graphic
                iw.plot[n].graphics[0].name = "zero-array-ignore"
                # get the real data
                func = self._data_mapping[n]
                real_data = func()
                # create graphic with the real data, this will not be managed by ImageWidget
                iw.plot[n].add_image(real_data, name="img", cmap="gnuplot2")

        iw.show()
        return iw
