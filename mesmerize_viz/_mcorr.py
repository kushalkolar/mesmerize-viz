from collections import OrderedDict
from typing import *
from functools import partial
from warnings import warn

import numpy as np
import pandas as pd
from mesmerize_core import MCorrExtensions
from mesmerize_core.caiman_extensions._utils import validate as validate_algo
from fastplotlib import ImageWidget

from ipydatagrid import DataGrid
from ipywidgets import Textarea, VBox

from ._utils import validate_data_options, ZeroArray
from ._common import ImageWidgetWrapper

projs = [
    "mean",
    "max",
    "std",
]

# these are directly manged by the image widget since they are [t, x, y]
standard_mappings = ["input", "mcorr"]


def get_mcorr_data_mapping(series: pd.Series) -> dict:
    projections = {k: partial(series.caiman.get_projection, k) for k in projs}
    m = {
        "input": series.caiman.get_input_movie,
        "mcorr": series.mcorr.get_output,
        "corr": series.caiman.get_corr_image,
        **projections
    }
    return m


# to format params dict into yaml-like string
is_pos = lambda x: 1 if x > 0 else 0
# this doesn't work without the lambda, yes it is ugly
format_params = lambda d, t: "\n" * is_pos(t) + \
    "\n".join(
        [": ".join(["   " * t + k, format_params(v, t + 1)]) for k, v in d.items()]
    ) if isinstance(d, dict) else str(d)


@pd.api.extensions.register_dataframe_accessor("mcorr")
class MCorrDataFrameVizExtension:
    def __init__(self, df):
        self._dataframe = df

    def viz(
            self,
            data: List[str] = None,
            start_index: int = 0,
            input_movie_kwargs=None,
            image_widget_kwargs=None,
            data_grid_kwargs: dict = None,
    ):
        """
        Visualize motion correction output.

        Parameters
        ----------
        data: list of str, default ["input", "mcorr"]
            list of data to plot

        start_index: int, default 0
            start index item used to set the initial data in the ImageWidget

        input_movie_kwargs: dict, optional
            kwargs passed to get_input_movie()

        image_widget_kwargs: dict, optional
            kwargs passed to ImageWidget

        data_grid_kwargs: dict, optional
            kwargs passed to DataGrid()

        Returns
        -------
        ImageWidget
            fastplotlib.ImageWidget visualization
        """
        if data is None:
            # default viz
            data = ["input", "mcorr"]

        if data_grid_kwargs is None:
            data_grid_kwargs = dict()

        self.grid = DataGrid(
            self._dataframe,
            selection_mode="row",
            layout={"height": "200px"},
            **data_grid_kwargs
        )
        self.params_text_area = Textarea(description="params:")
        # data options this can't be changed once an image widget has been made
        self._data = data

        if input_movie_kwargs is None:
            input_movie_kwargs = dict()

        if image_widget_kwargs is None:
            image_widget_kwargs = dict()

        self.input_movie_kwargs = input_movie_kwargs
        self.image_widget_kwargs = image_widget_kwargs

        self.image_widget: ImageWidget = None
        self._image_widget_wrapper: ImageWidgetWrapper = None
        self.current_row: int = start_index
        self._make_image_widget(index=start_index)
        self.params_text_area.value = format_params(self._dataframe.iloc[start_index].params, 0)

        self.grid.observe(self._row_changed, names="selections")

        self.widget = VBox(
            [
                self.grid,
                self.params_text_area,
                self.image_widget.show()
            ]
        )

        return self.widget

    def _make_image_widget(self, index):
        self._image_widget_wrapper = ImageWidgetWrapper(
            data=self._data,
            data_mapping=get_mcorr_data_mapping(self._dataframe.iloc[index]),
            standard_mappings=standard_mappings,
            input_movie_kwargs=self.input_movie_kwargs,
            image_widget_kwargs=self.image_widget_kwargs
        )

        self.image_widget = self._image_widget_wrapper.image_widget

    def _get_selection_row(self) -> Union[int, None]:
        r1 = self.grid.selections[0]["r1"]
        r2 = self.grid.selections[0]["r2"]

        if r1 != r2:
            warn("Only single row selection is currently allowed")
            return

        return r1

    def _row_changed(self, *args):
        index = self._get_selection_row()
        if index is None:
            return

        if self.current_row == index:
            return

        if self.image_widget is None:
            self._make_image_widget(index)
            return

        self._image_widget_wrapper.change_data(
            data=self._data,
            data_mapping=get_mcorr_data_mapping(self._dataframe.iloc[index]),
            input_movie_kwargs=self.input_movie_kwargs
        )

        self.params_text_area.value = format_params(self._dataframe.iloc[index].params, 0)

        self.current_row = index

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
