from collections import OrderedDict
from typing import *
from functools import partial
import math
import itertools
from warnings import warn

import numpy as np
import pandas as pd
from mesmerize_core.arrays._base import LazyArray
from mesmerize_core.utils import quick_min_max
from mesmerize_core.caiman_extensions._utils import validate as validate_algo
from fastplotlib import ImageWidget, GridPlot, graphics
from fastplotlib.graphics.line_slider import LineSlider

from ipydatagrid import DataGrid
from ipywidgets import Textarea, VBox, HBox, Layout

from ._utils import validate_data_options, ZeroArray
from ._common import ImageWidgetWrapper


data_options = [
    "input",
    "temporal",
    "temporal-stack",
    "contours",
    "rcm",
    "rcb",
    "residuals",
    "corr",
    "pnr",
]


for option in ["rcm", "rcb"]:
    for proj in ["mean", "min", "max", "std"]:
        data_options.append(f"{option}-{proj}")


projs = [
    "mean",
    "max",
    "std",
]


image_widget_managed = [
    "input",
    "contours",
    "rcm",
    "rcb",
    "residuals"
]

data_options += projs


class ExtensionCallWrapper:
    def __init__(self, extension_func: callable, kwargs: dict = None):
        """
        Basically like ``functools.partial`` but supports kwargs.

        Parameters
        ----------
        extension_func: callable
            extension function reference

        kwargs:
            kwargs to pass to the extension function when it is called
        """

        if kwargs is None:
            self.kwargs = dict()
        else:
            self.kwargs = kwargs

        self.func = extension_func

    def __call__(self, *args, **kwargs):
        self.func(**self.kwargs)


def get_data_mapping(series: pd.Series, data_kwargs: dict = None, other_data_loaders: dict = None) -> dict:
    """
    Returns dict that maps data option str to a callable that can return the corresponding data array.

    For example, ``{"input": series.get_input_movie}`` maps "input" -> series.get_input_movie

    Parameters
    ----------
    series: pd.Series
        row/item to get mapping from

    data_kwargs: dict, optional
        optional kwargs for each of the extension functions

    other_data_loaders: dict
        {"data_option": callable}, example {"behavior": LazyVideo}

    Returns
    -------
    dict
        {data label: callable}
    """

    default_extension_kwargs = {k: dict() for k in data_options + list(other_data_loaders.keys())}

    ext_kwargs = {
        **default_extension_kwargs,
        **data_kwargs
    }

    projections = {k: partial(series.caiman.get_projection, k) for k in projs}

    other_data_loaders_mapping = dict()

    # make ExtensionCallWrapers for other data loaders
    for option in list(other_data_loaders.keys()):
        other_data_loaders_mapping[option] = ExtensionCallWrapper(other_data_loaders[option], ext_kwargs[option])

    m = {
        "input": ExtensionCallWrapper(series.caiman.get_input_movie, ext_kwargs["input"]),
        "rcm": ExtensionCallWrapper(series.cnmf.get_rcm, ext_kwargs["rcm"]),
        "rcb": ExtensionCallWrapper(series.cnmf.get_rcb, ext_kwargs["rcb"]),
        "residuals": ExtensionCallWrapper(series.cnmf.get_residuals, ext_kwargs["residuals"]),
        "temporal": ExtensionCallWrapper(series.cnmf.get_temporal, ext_kwargs["temporal"]),
        "temporal-stack": ExtensionCallWrapper(series.cnmf.get_temporal, ext_kwargs["temporal"]),
        "corr": ExtensionCallWrapper(series.caiman.get_corr_image, ext_kwargs["corr"]),
        "empty": ZeroArray,
        **projections,
        **other_data_loaders_mapping
    }

    return m


# TODO: maybe this can be used so that ImageWidget can be used for both behavior and calcium
# TODO: but then we need an option to set window_funcs separately for each subplot
class TimeArray(LazyArray):
    """
    Wrapper for array-like that takes units of millisecond for slicing
    """
    def __init__(self, array: Union[np.ndarray, LazyArray], timestamps = None, framerate = None):
        """
        Arrays which can be sliced using timepoints in units of millisecond.
        Supports slicing with start and stop timepoints, does not support slice steps.

        i.e. You can do this: time_array[30], time_array[30:], time_array[:50], time_array[30:50].
        You cannot do this: time_array[::10], time_array[0::10], time_array[0:50:10]

        Parameters
        ----------
        array: array-like
            data array, must have shape attribute and first dimension must be frame index

        timestamps: np.ndarray, 1 dimensional
            timestamps in units of millisecond, you must provide either timestamps or framerate.
            MUST be in order such that t_(n +1) > t_n for all n.

        framerate: float
            framerate, in units of Hz (per second). You must provide either timestamps or framerate
        """
        self._array = array

        if timestamps is None and framerate is None:
            raise ValueError("Must provide timestamps or framerate")

        if timestamps is None:
            # total duration in milliseconds = n_frames / framerate
            n_frames = self.shape[0]
            stop_time_ms = (n_frames / framerate) * 1000
            timestamps = np.linspace(
                start=0,
                stop=stop_time_ms,
                num=n_frames,
                endpoint=False
            )

        if timestamps.size != self._array.shape[0]:
            raise ValueError("timestamps.size != array.shape[0]")

        self.timestamps = timestamps

    def _get_closest_index(self, timepoint: float):
        """
        from: https://stackoverflow.com/a/26026189/4697983

        This is very fast, 10 microseconds even for a

        Parameters
        ----------
        timepoint: float
            timepoint in milliseconds

        Returns
        -------
        int
            index for the closest timestamp, which also corresponds to the frame index of the data array
        """
        value = timepoint
        array = self.timestamps

        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
            return idx - 1
        else:
            return idx

    # override __getitem__ since it will not work with LazyArray base implementation since:
    # 1. base implementation requires the slice indices to be less than shape[0]
    # 2. base implementation does not consider slicing with float values
    def __getitem__(self, indices: Union[slice, int, float]) -> np.ndarray:
        if isinstance(indices, slice):
            if indices.step is not None:
                raise IndexError(
                    "TimeArray slicing does not support step, only start and stop. See docstring."
                )

            if indices.start is None:
                start = 0
            else:
                start = self._get_closest_index(indices.start)

            if indices.stop is None:
                stop = self.n_frames
            else:
                stop = self._get_closest_index(indices.stop)

            s = slice(start, stop)
            return self._array[s]

        # single index
        index = self._get_closest_index(indices)
        return self._array[index]

    def _compute_at_indices(self, indices: Union[int, slice]) -> np.ndarray:
        """not implemented here"""
        pass

    @property
    def n_frames(self) -> int:
        return self.shape[0]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._array.shape

    @property
    def dtype(self) -> str:
        return str(self._array.dtype)

    @property
    def min(self) -> float:
        if isinstance(self._array, LazyArray):
            return self._array.min
        else:
            return quick_min_max(self._array)[0]

    @property
    def max(self) -> float:
        if isinstance(self._array, LazyArray):
            return self._array.max
        else:
            return quick_min_max(self._array)[1]


class GridPlotWrapper:
    """Wraps GridPlot in a way that allows updating the data"""

    def __init__(
            self,
            data: List[str],
            data_mapping: Dict[str, ExtensionCallWrapper],
            reset_timepoint_on_change: bool,
            gridplot_kwargs: dict,
    ):
        """
        Visualize motion correction output.

        Parameters
        ----------
        data: list of str
            list of data to plot, example ["temporal", "temporal-stack"]

        data_mapping: dict
            maps {"data_option": callable}

        reset_timepoint_on_change: bool, default False
            reset the timepoint when changing items/rows

        image_widget_kwargs: dict, optional
            kwargs passed to ImageWidget

        Returns
        -------
        GridPlot
            fastplotlib.GridPlot visualization
        """

        self.line_sliders: List[LineSlider] = list()

        data_arrays  = self._parse_data(
            data=data,
            data_mapping=data_mapping,
        )

        _gridplot_kwargs = {"shape": (1, len(data))}
        _gridplot_kwargs.update(gridplot_kwargs)

        self.gridplot = GridPlot(**_gridplot_kwargs)

        self.temporal_graphics: List[graphics.LineCollection] = list()
        self.temporal_stack_graphics: List[graphics.LineStack] = list()

        # TODO: need to figure out how to properly garbage collect graphic when changing to different batch item
        # TODO: because length of LineCollection graphics will vary
        for i, subplot in enumerate(gridplot):
            # skip
            if d[i] == "empty":
                continue

            elif d[i] == "temporal":
                t = subplot.add_line_collection(
                    data_arrays[i], name="lines"
                )
                subplot.name = d[i]
                self.temporal_graphics.append(t)

            elif d[i] == "temporal-stack":
                pass

    def _parse_data(self, data, data_mapping) -> List[np.ndarray]:
        data_arrays = list()

        for d in data:
            if d == "empty":
                data_arrays.append(None)
            else:
                func = data_mapping[d]
                a = func()
                data_arrays.append(a)

        return data_arrays


    def change_data(self):
        pass

    def set_timepoint(self):
        pass


# TODO: This use a GridPlot that's manually managed because the timescale of
class CNMFVizContainer:
    """Widget that contains the DataGrid, params text box fastplotlib GridPlot, etc"""

    def __init__(
        self,
            dataframe: pd.DataFrame,
            data: List[str] = None,
            start_index: int = 0,
            reset_timepoint_on_change: bool = False,
            calcium_framerate: float = None,
            other_data_loaders: Dict[str, callable] = None,
            data_kwargs: dict = None,
            image_widget_kwargs: dict = None,
            plot_widget_kwargs: List[dict] = None,
            data_grid_kwargs: dict = None,
    ):
        """
        Visualize CNMF output and other data columns such as behavior video (optional)

        Parameters
        ----------
        dataframe: pd.DataFrame

        data: list of str
            data options, such as "input", "temporal", "contours", etc.

        start_index

        reset_timepoint_on_change

        calcium_framerate

        other_data_loaders: Dict[str, callable]
            if loading non-calcium related data arrays, provide dict of callables for opening them.
            Example, if you provide ``data = ["contours", "temporal", "behavior"]``, and the "behavior"
            column contains videos, you could provide `other_data_loads = {"behavior": LazyVideo}

        data_kwargs: dict
            kwargs passed to corresponding extension function to load data.
            example: ``{"temporal": {"component_ixs": "good"}}``

        plot_widget_kwargs: List[dict]
            kwargs passed to GridPlots or ImageWidget, useful when ``data`` is a list of lists.

        image_widget_kwargs

        data_grid_kwargs
        """

        if data is None:
            data = [["temporal"], ["contours", "rcm", "rcb", "residuals"]]

        if other_data_loaders is None:
            other_data_loaders = dict()

        self._other_data_loaders = other_data_loaders

        self.reset_timepoint_on_change = reset_timepoint_on_change

        # make sure data options are valid
        for d in list(itertools.chain(*data)):
            if d not in data_options or d not in dataframe.columns or d != "empty":
                raise ValueError(
                    f"`data` options are: {data_options} or a DataFrame column name: {dataframe.columns}\n"
                    f"You have passed: {d}"
                )

            if d in dataframe.columns:
                if d not in other_data_loaders.keys():
                    raise ValueError(
                        f"You have provided the non-CNMF related data option: {d}.\n"
                        f"If you provide a non-cnmf related data option you must also provide a "
                        f"data loader callable for it to `other_data_loaders`"
                    )

        if data_grid_kwargs is None:
            data_grid_kwargs = dict()

        self._dataframe = dataframe

        default_widths = {
            "algo": 50,
            'item_name': 200,
            'input_movie_path': 120,
            'algo_duration': 80,
            'comments': 120,
            'uuid': 60
        }

        columns = dataframe.columns
        # these add clutter
        hide_columns = [
            "params",
            "outputs",
            "added_time",
            "ran_time",

        ]

        df_show = self._dataframe[[c for c in columns if c not in hide_columns]]

        self.datagrid = DataGrid(
            df_show,  # show only a subset
            selection_mode="cell",
            layout={"height": "250px", "width": "750px"},
            base_row_size=24,
            index_name="index",
            column_widths=default_widths,
            **data_grid_kwargs
        )

        self.params_text_area = Textarea()
        self.params_text_area.layout = Layout(
            height="250px",
            max_height="250px",
            width="360px",
            max_width="500px"
        )

        # data options is private since this can't be changed once an image widget has been made
        self._data = data

        if data_kwargs is None:
            data_kwargs = dict()

        if image_widget_kwargs is None:
            image_widget_kwargs = dict()

        self.data_kwargs = data_kwargs
        self.image_widget_kwargs = image_widget_kwargs

        self.current_row: int = start_index

        self.iw_managed = image_widget_managed.copy()
        self.iw_managed += dataframe.columns.tolist()

        # TODO: check if data is list of lists
        # TODO: if all elements in a sublist are non-image widget managed just make it a gridplot

        self.gridplots: List[GridPlot] = list()
        self._image_widget_wrappers: List[ImageWidgetWrapper] = list()

        self.plots: List[Union[GridPlot, ImageWidgetWrapper]] = list()

        self._temporal_graphics: List[graphics.LineCollection] = list()
        self._contour_graphics: List[graphics.LineCollection] = list()

        if plot_widget_kwargs is None:
            plot_widget_kwargs = [dict() for i in range(len(data))]

        # list of lists
        if all(isinstance(d, list) for d in data):
            for sub_list in data:
                if all(option not in self.iw_managed for option in sub_list):
                    gp = GridPlot(
                        shape=(1, len(sub_list)),
                        **plot_widget_kwargs
                    )

                    self.plots.append(gp)
                else: # managed by imagewidget
                    iw = self._make_image_widget(sub_list, index=start_index)

    def _make_gridplot(self, sub_data, index) -> GridPlot:
        pass


    def _make_image_widget(self, sub_data: List[str], index: int) -> ImageWidgetWrapper:
        self._image_widget_wrapper = ImageWidgetWrapper(
            data=sub_data,
            data_mapping=get_data_mapping(self._dataframe.iloc[index], self.data_kwargs, self._other_data_loaders),
            image_widget_managed_data=self.iw_managed,
            reset_timepoint_on_change=self.reset_timepoint_on_change,
            input_movie_kwargs=dict(),
            image_widget_kwargs=self.image_widget_kwargs
        )
