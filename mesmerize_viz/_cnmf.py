from collections import OrderedDict
from typing import *
from functools import partial
import math
from warnings import warn

import numpy as np
import pandas as pd
from mesmerize_core.arrays._base import LazyArray
from mesmerize_core.utils import quick_min_max
from mesmerize_core.caiman_extensions._utils import validate as validate_algo
from fastplotlib import ImageWidget

from ipydatagrid import DataGrid
from ipywidgets import Textarea, VBox, HBox, Layout

from ._utils import validate_data_options, ZeroArray
from ._common import ImageWidgetWrapper


data_options = [
    "input",
    "temporal",
    "contours",
    "rcm",
    "rcb",
    "residuals"
]


projs = [
    "mean",
    "max",
    "std",
]


default_extension_kwargs = {k: dict() for k in ["input", "rcm", "rcb", "temporal"]}


def get_data_mapping(series: pd.Series, extension_kwargs: dict = None) -> dict:
    """
    Returns dict that maps data option str to a callable that can return the corresponding data array.

    For example, ``{"input": series.get_input_movie}`` maps "input" -> series.get_input_movie

    Parameters
    ----------
    series: pd.Series
        row/item to get mcorr mapping

    Returns
    -------
    dict
        {data label: callable}
    """


    projections = {k: partial(series.caiman.get_projection, k) for k in projs}
    m = {
        "input": series.caiman.get_input_movie,
        "rcm": series.cnmf.get_output,
        "corr": series.caiman.get_corr_image,
        **projections
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
            other_framerates: Dict[str, float] = None,
            input_movie_kwargs: dict = None,
            image_widget_kwargs: dict = None,
            data_grid_kwargs: dict = None,
    ):
        """
        Visualize CNMF output and behavior (optional)
        """
