from typing import *
from functools import partial

import numpy as np
import pandas as pd
from fastplotlib import ImageWidget
from fastplotlib.utils import quick_min_max
from mesmerize_core import MCorrExtensions

from ._utils import ZeroArray


class ImageWidgetWrapper:
    """Wraps Image Widget in a way that allows updating the data"""
    def __init__(
            self,
            data: List[str],
            data_mapping: dict,
            standard_mappings: List[str],
            reset_timepoint_on_change: bool = False,
            input_movie_kwargs: dict = None,
            image_widget_kwargs: dict = None,
    ):
        """
        Visualize motion correction output.

        Parameters
        ----------
        data: list of str, default ["input", "mcorr"]
            list of data to plot, can also be a list of lists.

        reset_timepoint_on_change: bool, default False
            reset the timepoint in the ImageWidget when changing items/rows

        input_movie_kwargs: dict, optional
            kwargs passed to get_input_movie()

        image_widget_kwargs: dict, optional
            kwargs passed to ImageWidget

        Returns
        -------
        ImageWidget
            fastplotlib.ImageWidget visualization
        """

        if input_movie_kwargs is None:
            input_movie_kwargs = dict()

        if image_widget_kwargs is None:
            image_widget_kwargs = dict()

        # the ones which are [t, x, y] images that ImageWidget can manage by itself
        self.standard_mappings = standard_mappings

        # data arrays directly passed to image widget
        data_arrays_iw = self._parse_data(
            data=data,
            data_mapping=data_mapping,
            input_movie_kwargs=input_movie_kwargs
        )

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

        self.image_widget = ImageWidget(
            data=data_arrays_iw,
            names=data,
            **image_widget_kwargs
        )

        for a, n in zip(data_arrays_iw, data):
            if isinstance(a, ZeroArray):
                # rename the existing graphic
                self.image_widget.plot[n].graphics[0].name = "zero-array-ignore"
                # get the real data
                func = data_mapping[n]
                real_data = func()
                # create graphic with the real data, this will not be managed by ImageWidget
                self.image_widget.plot[n].add_image(real_data, name="image", cmap="gnuplot2")

        self.reset_timepoint_on_change = reset_timepoint_on_change

    def _parse_data(
            self,
            data: List[str],
            data_mapping: dict,
            input_movie_kwargs: dict,
    ) -> List[Union[np.ndarray, ZeroArray]]:
        """
        Parse data string keys into actual arrays using the data_mapping for the current row.
        Returns list of arrays that ImageWidget can display and manage.
        """
        # data arrays directly passed to image widget
        data_arrays_iw = list()

        for d in data:
            if d in self.standard_mappings:
                func = data_mapping[d]

                if d == "input":
                    a = func(**input_movie_kwargs)
                else:
                    a = func()

                data_arrays_iw.append(a)

            else:
                # make a placeholder array to keep imagewidget happy
                # hacky but this is the best way for now
                zero_array = ZeroArray(ndim=data_arrays_iw[0].ndim)
                data_arrays_iw.append(zero_array)

        return data_arrays_iw

    def change_data(self, data: List[str], data_mapping: dict, input_movie_kwargs):
        data_arrays_iw = self._parse_data(
            data=data,
            data_mapping=data_mapping,
            input_movie_kwargs=input_movie_kwargs
        )

        if not len(data) == len(data_arrays_iw):
            raise ValueError("len(data) != len(data_arrays)")

        for i, (name, array) in enumerate(zip(data, data_arrays_iw)):
            # skip the ones which ImageWidget does not manage
            if name not in self.standard_mappings:
                pass

            else:
                # update the ones which ImageWidget manages
                self.image_widget._data[i] = array

                # I think it's useful to NOT reset the vmin vmax
                # if necessary the user can call ImageWidget.reset_vmin_vmax()
                # min_max = quick_min_max(array)
                #
                # self.image_widget.plot[name]["image"].vmin = min_max[0]
                # self.image_widget.plot[name]["image"].vmax = min_max[1]
                #
                # # set vmin vmax slider stuff
                # data_range = np.ptp(min_max)
                # data_range_30p = np.ptp(min_max) * 0.3
                #
                # self.image_widget.vmin_vmax_sliders[i].value = min_max
                # self.image_widget.vmin_vmax_sliders[i].min = min_max[0] - data_range_30p
                # self.image_widget.vmin_vmax_sliders[i].max = min_max[1] + data_range_30p
                # self.image_widget.vmin_vmax_sliders[i].step = data_range / 150

        # update the ones which ImageWidget does not manage
        self._set_non_standard_arrays(
            data=data,
            data_arrays_iw=data_arrays_iw,
            data_mapping=data_mapping
        )

        if self.reset_timepoint_on_change:
            # set index {t: 0}
            self.image_widget.current_index = {"t": 0}
        else:
            # forces graphic data to update in all subplots
            self.image_widget.current_index = self.image_widget.current_index

    def _set_non_standard_arrays(self, data, data_arrays_iw, data_mapping):
        for a, n in zip(data_arrays_iw, data):
            if isinstance(a, ZeroArray):
                # get the real data
                func = data_mapping[n]
                real_data = func()
                # change the graphic data
                self.image_widget.plot[n]["image"].data = real_data

                min_max = quick_min_max(real_data)
                self.image_widget.plot[n]["image"].vmin = min_max[0]
                self.image_widget.plot[n]["image"].vmax = min_max[1]
