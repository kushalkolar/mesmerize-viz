from typing import *
from functools import partial
import math
import itertools
from warnings import warn

import ipywidgets
import numpy as np
import pandas as pd

from mesmerize_core.arrays._base import LazyArray
from mesmerize_core.utils import quick_min_max

from fastplotlib import ImageWidget, GridPlot, graphics
from fastplotlib.graphics.selectors import LinearSelector, Synchronizer
from fastplotlib.utils import calculate_gridshape

from ipydatagrid import DataGrid
from ipywidgets import Textarea, VBox, HBox, Layout, Checkbox

from ._utils import ZeroArray, format_params


# basic data options
VALID_DATA_OPTIONS = [
    "input",
    "contours",
    "rcm",
    "rcb",
    "residuals",
    "corr",
    "pnr",
    "empty"
]


TEMPORAL_OPTIONS = [
    "temporal",
    "temporal-stack",
    "heatmap",
]

VALID_DATA_OPTIONS += TEMPORAL_OPTIONS

# RCM and RCB projections
rcm_rcb_proj_options = list()

for option in ["rcm", "rcb"]:
    for proj in ["mean", "min", "max", "std"]:
        rcm_rcb_proj_options.append(f"{option}-{proj}")

VALID_DATA_OPTIONS += rcm_rcb_proj_options


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

VALID_DATA_OPTIONS += projs


class ExtensionCallWrapper:
    def __init__(self, extension_func: callable, kwargs: dict = None, attr: str = None):
        """
        Basically like ``functools.partial`` but supports kwargs.

        Parameters
        ----------
        extension_func: callable
            extension function reference

        kwargs: dict
            kwargs to pass to the extension function when it is called

        attr: str, optional
            return an attribute of the callable's output instead of the return value of the callable.
            Example: if using rcm, can set ``attr="max_image"`` to return the max proj of the RCM.
        """

        if kwargs is None:
            self.kwargs = dict()
        else:
            self.kwargs = kwargs

        self.func = extension_func
        self.attr = attr

    def __call__(self, *args, **kwargs):
        rval = self.func(**self.kwargs)

        if self.attr is not None:
            return getattr(rval, self.attr)

        return rval


def get_cnmf_data_mapping(series: pd.Series, data_kwargs: dict = None, other_data_loaders: dict = None) -> dict:
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
    if data_kwargs is None:
        data_kwargs = dict()

    if other_data_loaders is None:
        other_data_loaders = dict()

    default_extension_kwargs = {k: dict() for k in VALID_DATA_OPTIONS + list(other_data_loaders.keys())}

    ext_kwargs = {
        **default_extension_kwargs,
        **data_kwargs
    }

    projections = {k: partial(series.caiman.get_projection, k) for k in projs}

    other_data_loaders_mapping = dict()

    # make ExtensionCallWrapers for other data loaders
    for option in list(other_data_loaders.keys()):
        other_data_loaders_mapping[option] = ExtensionCallWrapper(other_data_loaders[option], ext_kwargs[option])

    rcm_rcb_projs = dict()
    for proj in ["mean", "min", "max", "std"]:
        rcm_rcb_projs[f"rcm-{proj}"] = ExtensionCallWrapper(
            series.cnmf.get_rcm,
            ext_kwargs["rcm"],
            attr=f"{proj}_image"
        )

    temporal_mappings = {
        k: ExtensionCallWrapper(series.cnmf.get_temporal, ext_kwargs[k]) for k in TEMPORAL_OPTIONS
    }

    m = {
        "input": ExtensionCallWrapper(series.caiman.get_input_movie, ext_kwargs["input"]),
        "rcm": ExtensionCallWrapper(series.cnmf.get_rcm, ext_kwargs["rcm"]),
        "rcb": ExtensionCallWrapper(series.cnmf.get_rcb, ext_kwargs["rcb"]),
        "residuals": ExtensionCallWrapper(series.cnmf.get_residuals, ext_kwargs["residuals"]),
        "corr": ExtensionCallWrapper(series.caiman.get_corr_image, ext_kwargs["corr"]),
        "contours": ExtensionCallWrapper(series.cnmf.get_contours, ext_kwargs["contours"]),
        "empty": ZeroArray,
        **temporal_mappings,
        **projections,
        **rcm_rcb_projs,
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
            data: Union[List[str], List[List[str]]],
            data_mapping: Dict[str, ExtensionCallWrapper],
            reset_timepoint_on_change: bool = False,
            data_graphic_kwargs: dict = None,
            # slider_ipywidget: ipywidgets.IntSlider = None,
            gridplot_kwargs: dict = None,
            cmap: str = "gnuplot2",
            component_colors: str = "random"
    ):
        """
        Visualize motion correction output.

        Parameters
        ----------
        data: list of str or list of list of str
            list of data to plot, examples: ["input", "temporal-stack"], [["temporal"], ["rcm", "rcb"]]

        data_mapping: dict
            maps {"data_option": callable}

        reset_timepoint_on_change: bool, default False
            reset the timepoint in the ImageWidget when changing items/rows

        data_graphic_kwargs: dict
            passed add_<graphic> for corresponding graphic

        slider_ipywidget: ipywidgets.IntSlider
            time slider from ImageWidget

        gridplot_kwargs: dict, optional
            kwargs passed to GridPlot

        """

        self._data = data

        if data_graphic_kwargs is None:
            data_graphic_kwargs = dict()

        self.data_graphic_kwargs = data_graphic_kwargs

        if gridplot_kwargs is None:
            gridplot_kwargs = dict()

        self._cmap = cmap

        self.component_colors = component_colors

        # self._slider_ipywidget = slider_ipywidget

        self.reset_timepoint_on_change = reset_timepoint_on_change

        self.gridplots: List[GridPlot] = list()

        # gridplot for each sublist
        for sub_data in self._data:
            _gridplot_kwargs = {"shape": calculate_gridshape(len(sub_data))}
            _gridplot_kwargs.update(gridplot_kwargs)
            self.gridplots.append(GridPlot(**_gridplot_kwargs))

        self.temporal_graphics: List[graphics.LineCollection] = list()
        self.temporal_stack_graphics: List[graphics.LineStack] = list()
        self.heatmap_graphics: List[graphics.HeatmapGraphic] = list()
        self.image_graphics: List[graphics.ImageGraphic] = list()
        self.contour_graphics: List[graphics.LineCollection] = list()

        self._managed_graphics: List[list] = [
            self.temporal_graphics,
            self.temporal_stack_graphics,
            self.image_graphics,
            self.contour_graphics
        ]

        # to store only image data in a 1:1 mapping to the graphics list
        self.image_graphic_arrays: List[np.ndarray] = list()

        self.linear_selectors: List[LinearSelector] = list()

        self._current_frame_index: int = 0

        self.change_data(data_mapping)

    def _parse_data(self, data_options, data_mapping) -> List[List[np.ndarray]]:
        """
        Returns nested list of array-like
        """
        data_arrays = list()

        for d in data_options:
            if isinstance(d, list):
                data_arrays.append(self._parse_data(d, data_mapping))

            elif d == "empty":
                data_arrays.append(None)

            else:
                func = data_mapping[d]
                a = func()
                data_arrays.append(a)

        return data_arrays

    @property
    def cmap(self) -> str:
        return self._cmap

    @cmap.setter
    def cmap(self, cmap: str):
        for g in self.image_graphics:
            g.cmap = cmap

    # @property
    # def component_colors(self) -> Any:
    #     pass
    #
    # @component_colors.setter
    # def component_colors(self, colors: Any):
    #     for collection in self.contour_graphics:
    #         for g in collection.graphics:
    #

    def change_data(self, data_mapping: Dict[str, callable]):
        for l in self._managed_graphics:
            l.clear()

        self.image_graphic_arrays.clear()

        # clear existing subplots
        for gp in self.gridplots:
            gp.clear()

        # new data arrays
        data_arrays = self._parse_data(data_options=self._data, data_mapping=data_mapping)

        # rval is (contours, centeres of masses)
        contours = data_mapping["contours"]()[0]

        if self.component_colors == "random":
            n_components = len(contours)
            component_colors = np.random.rand(n_components, 4).astype(np.float32)
            component_colors[:, -1] = 1
        else:
            component_colors = self.component_colors

        # change data for all gridplots
        for sub_data, sub_data_arrays, gridplot in zip(self._data, data_arrays, self.gridplots):
            self._change_data_gridplot(sub_data, sub_data_arrays, gridplot, contours, component_colors)

        # connect events
        self._connect_events()

    def _change_data_gridplot(
            self,
            data: List[str],
            data_arrays: List[np.ndarray],
            gridplot: GridPlot,
            contours,
            component_colors
    ):

        if self.reset_timepoint_on_change:
            self._current_frame_index = 0

        for data_option, data_array, subplot in zip(data, data_arrays, gridplot):
            if data_option in self.data_graphic_kwargs.keys():
                graphic_kwargs = self.data_graphic_kwargs[data_option]
            else:
                graphic_kwargs = dict()
            # skip
            if data_option == "empty":
                continue

            elif data_option == "temporal":
                current_graphic = subplot.add_line_collection(
                    data_array,
                    colors=component_colors,
                    name="components",
                    **graphic_kwargs
                )
                current_graphic[:].present.add_event_handler(subplot.auto_scale)
                self.temporal_graphics.append(current_graphic)

                # otherwise the plot has nothing in it which causes issues
                subplot.add_line(np.random.rand(data_array.shape[1]), colors=(0, 0, 0, 0), name="pseudo-line")

            elif data_option == "temporal-stack":
                current_graphic = subplot.add_line_stack(
                    data_array,
                    colors=component_colors,
                    name="components",
                    **graphic_kwargs
                )
                self.temporal_stack_graphics.append(current_graphic)

            elif data_option == "heatmap":
                current_graphic = subplot.add_heatmap(
                    data_array,
                    colors=component_colors,
                    name="components",
                    **graphic_kwargs
                )
                self.heatmap_graphics.append(current_graphic)

            else:
                img_graphic = subplot.add_image(
                    data_array[self._current_frame_index],
                    cmap=self.cmap,
                    name="image",
                    **graphic_kwargs
                )

                self.image_graphics.append(img_graphic)
                self.image_graphic_arrays.append(data_array)

                contour_graphic = subplot.add_line_collection(
                    contours,
                    colors=component_colors,
                    name="contours"
                )

                self.contour_graphics.append(contour_graphic)

            subplot.name = data_option

            if data_option in TEMPORAL_OPTIONS:
                self.linear_selectors.append(current_graphic.add_linear_selector())
                subplot.camera.maintain_aspect = False

        if len(self.linear_selectors) > 0:
            self._synchronizer = Synchronizer(
                *self.linear_selectors, key_bind=None
            )

        for ls in self.linear_selectors:
            ls.selection.add_event_handler(self.set_frame_index)

    def _euclidean(self, source, target, event, new_data):
        """maps click events to contour"""
        # calculate coms of line collection
        indices = np.array(event.pick_info["index"])

        coms = list()

        for contour in target.graphics:
            coors = contour.data()[~np.isnan(contour.data()).any(axis=1)]
            com = coors.mean(axis=0)
            coms.append(com)

        # euclidean distance to find closest index of com
        indices = np.append(indices, [0])

        ix = int(np.linalg.norm((coms - indices), axis=1).argsort()[0])

        target._set_feature(feature="colors", new_data=new_data, indices=ix)

        return None

    def _connect_events(self):
        for image_graphic, contour_graphic in zip(self.image_graphics, self.contour_graphics):
            image_graphic.link(
                "click",
                target=contour_graphic,
                feature="colors",
                new_data="w",
                callback=self._euclidean
            )

            contour_graphic.link("colors", target=contour_graphic, feature="thickness", new_data=5)

            for temporal_graphic in self.temporal_graphics:
                contour_graphic.link("colors", target=temporal_graphic, feature="present", new_data=True)

    def set_frame_index(self, ev):
        # 0 because this will return the same number repeated * n_components
        index = ev.pick_info["selected_index"][0]
        for image_graphic, full_array in zip(self.image_graphics, self.image_graphic_arrays):
            # txy data
            if full_array.ndim > 2:
                image_graphic.data = full_array[index]

        self._current_frame_index = index


# TODO: This use a GridPlot that's manually managed because the timescales of calcium ad behavior won't match
class CNMFVizContainer:
    """Widget that contains the DataGrid, params text box fastplotlib GridPlot, etc"""

    def __init__(
        self,
            dataframe: pd.DataFrame,
            data: List[str] = None,
            start_index: int = 0,
            reset_timepoint_on_change: bool = False,
            data_graphic_kwargs: dict = None,
            gridplot_kwargs: dict = None,
            cmap: str = "gnuplot2",
            component_colors: str = "random",
            calcium_framerate: float = None,
            other_data_loaders: Dict[str, callable] = None,
            data_kwargs: dict = None,
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

        gridplot_kwargs: List[dict]
            kwargs passed to GridPlot

        data_grid_kwargs
        """

        if data is None:
            data = [["temporal"], ["input", "rcm", "rcb", "residuals"]]

        if other_data_loaders is None:
            other_data_loaders = dict()

        # simple list of str, single gridplot
        if all(isinstance(option, str) for option in data):
            data = [data]

        if not all(isinstance(option, list) for option in data):
            raise TypeError(
                "Must pass list of str or nested list of str"
            )

        # make sure data options are valid
        for d in list(itertools.chain(*data)):
            if (d not in VALID_DATA_OPTIONS) and (d not in dataframe.columns):
                raise ValueError(
                    f"`data` options are: {VALID_DATA_OPTIONS} or a DataFrame column name: {dataframe.columns}\n"
                    f"You have passed: {d}"
                )

            if d in dataframe.columns:
                if d not in other_data_loaders.keys():
                    raise ValueError(
                        f"You have provided the non-CNMF related data option: {d}.\n"
                        f"If you provide a non-cnmf related data option you must also provide a "
                        f"data loader callable for it to `other_data_loaders`"
                    )

        self._other_data_loaders = other_data_loaders

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

        self.data_kwargs = data_kwargs

        self.current_row: int = start_index

        self._make_gridplot(
            start_index=start_index,
            reset_timepoint_on_change=reset_timepoint_on_change,
            data_graphic_kwargs=data_graphic_kwargs,
            gridplot_kwargs=gridplot_kwargs,
            cmap=cmap,
            component_colors=component_colors,
        )

        self._set_params_text_area(index=start_index)

        # set initial selected row
        self.datagrid.select(
            row1=start_index,
            column1=0,
            row2=start_index,
            column2=len(df_show.columns),
            clear_mode="all"
        )

        # callback when row changed
        self.datagrid.observe(self._row_changed, names="selections")

    def _make_gridplot(
            self,
            start_index: int,
            reset_timepoint_on_change: bool,
            data_graphic_kwargs: dict,
            gridplot_kwargs: dict,
            cmap: str,
            component_colors: str,
    ):

        data_mapping = get_cnmf_data_mapping(
            self._dataframe.iloc[start_index],
            self.data_kwargs
        )

        self._gridplot_wrapper = GridPlotWrapper(
            data=self._data,
            data_mapping=data_mapping,
            reset_timepoint_on_change=reset_timepoint_on_change,
            data_graphic_kwargs=data_graphic_kwargs,
            gridplot_kwargs=gridplot_kwargs,
            cmap=cmap,
            component_colors=component_colors

        )

        self.gridplots = self._gridplot_wrapper.gridplots

    def show(self):
        """Show the widget"""

        self.show_all_checkbox = Checkbox(value=True, description="Show all components")

        widget = VBox(
            [
                HBox([self.datagrid, self.params_text_area]),
                self.show_all_checkbox,
                VBox([gp.show() for gp in self.gridplots])
            ]
        )

        self.show_all_checkbox.observe(self._toggle_show_all, "value")

        return widget

    def _toggle_show_all(self, change):
        for line_collection in self._gridplot_wrapper.temporal_graphics:
            line_collection[:].present = change["new"]

    def close(self):
        """Close the widget"""
        for gp in self.gridplots:
            gp.close()

    def _get_selection_row(self) -> Union[int, None]:
        r1 = self.datagrid.selections[0]["r1"]
        r2 = self.datagrid.selections[0]["r2"]

        if r1 != r2:
            warn("Only single row selection is currently allowed")
            return

        # get corresponding dataframe index from currently visible dataframe
        # since filtering etc. is possible
        index = self.datagrid.get_visible_data().index[r1]

        return index

    def _row_changed(self, *args):
        index = self._get_selection_row()
        if index is None:
            return

        if self.current_row == index:
            return

        try:
            data_mapping = get_cnmf_data_mapping(
                self._dataframe.iloc[index],
                self.data_kwargs
            )
            self._gridplot_wrapper.change_data(data_mapping)
        except Exception as e:
            self.params_text_area.value = f"{type(e).__name__}\n" \
                                          f"{str(e)}\n\n" \
                                          f"See jupyter log for details"
            raise e

        self._set_params_text_area(index)

        self.current_row = index

    def _set_params_text_area(self, index):
        row = self._dataframe.iloc[index]
        # try and get the param diffs
        try:
            param_diffs = self._dataframe.caiman.get_params_diffs(
                algo=row["algo"],
                item_name=row["item_name"]
            ).iloc[index]

            diffs_dict = {"diffs": param_diffs}
            diffs = f"{format_params(diffs_dict, 0)}\n\n"
        except:
            diffs = ""

        # diffs and full params
        self.params_text_area.value = diffs + format_params(self._dataframe.iloc[index].params, 0)


@pd.api.extensions.register_dataframe_accessor("cnmf")
class CNMFDataFrameVizExtension:
    def __init__(self, df):
        self._dataframe = df

    def viz(
            self,
            data: List[str] = None,
            start_index: int = 0,
            reset_timepoint_on_change: bool = False,
            data_graphic_kwargs: dict = None,
            gridplot_kwargs: dict = None,
            cmap: str = "gnuplot2",
            component_colors: str = "random",
            calcium_framerate: float = None,
            other_data_loaders: Dict[str, callable] = None,
            data_kwargs: dict = None,
            data_grid_kwargs: dict = None,
    ):
        """
        Visualize motion correction output.

        Parameters
        ----------
        data: list of str or list of list of str
            default [["temporal"], ["input", "rcm", "rcb", "residuals"]]
            list of data to plot, valid options are:

            +-------------+-------------------------------------+
            | data option | description                         |
            +=============+=====================================+
            | input       | input movie                         |
            | mcorr       | motion corrected movie              |
            | mean        | mean projection                     |
            | max         | max projection                      |
            | std         | standard deviation projection       |
            | corr        | correlation image, if computed      |
            | pnr         | peak-noise-ratio image, if computed |
            +-------------+-------------------------------------+

        start_index: int, default 0
            start index item used to set the initial data in the ImageWidget

        reset_timepoint_on_change: bool, default False
            reset the timepoint in the ImageWidget when changing items/rows

        input_movie_kwargs: dict, optional
            kwargs passed to get_input_movie()

        image_widget_kwargs: dict, optional
            kwargs passed to ImageWidget

        data_grid_kwargs: dict, optional
            kwargs passed to DataGrid()

        Returns
        -------
        McorrVizContainer
            widget that contains the DataGrid, params text box and ImageWidget
        """
        container = CNMFVizContainer(
            dataframe=self._dataframe,
            data=data,
            start_index=start_index,
            reset_timepoint_on_change=reset_timepoint_on_change,
            data_graphic_kwargs=data_graphic_kwargs,
            gridplot_kwargs=gridplot_kwargs,
            cmap=cmap,
            component_colors=component_colors,
            calcium_framerate=calcium_framerate,
            other_data_loaders=other_data_loaders,
            data_kwargs=data_kwargs,
            data_grid_kwargs=data_grid_kwargs,
        )

        return container
