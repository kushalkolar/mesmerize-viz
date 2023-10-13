from functools import partial
from itertools import product
from typing import Union, List, Dict

import numpy as np
import pandas as pd

from ipywidgets import IntSlider, BoundedIntText, jslink

from fastplotlib import GridPlot, graphics
from fastplotlib.graphics.selectors import LinearSelector, Synchronizer, LinearRegionSelector
from fastplotlib.utils import calculate_gridshape


# basic data options
VALID_DATA_OPTIONS = [
    "contours",
    "empty"
]


IMAGE_OPTIONS = [
    "input",
    "rcm",
    "rcb",
    "residuals",
    "corr",
    "pnr",
]

rcm_rcb_proj_options = list()
# RCM and RCB projections
for option in ["rcm", "rcb"]:
    for proj in ["mean", "min", "max", "std"]:
        rcm_rcb_proj_options.append(f"{option}-{proj}")

IMAGE_OPTIONS += rcm_rcb_proj_options

TEMPORAL_OPTIONS = [
    "temporal",
    "temporal-stack",
    "heatmap",
]

projs = [
    "mean",
    "max",
    "std",
]

IMAGE_OPTIONS += projs

VALID_DATA_OPTIONS += IMAGE_OPTIONS
VALID_DATA_OPTIONS += TEMPORAL_OPTIONS


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

    default_extension_kwargs["contours"] = {"swap_dim": False}

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
        "empty": None,
        **temporal_mappings,
        **projections,
        **rcm_rcb_projs,
        **other_data_loaders_mapping
    }

    return m


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

        self.component_slider = IntSlider(min=0, max=1, value=0, step=1, description="component index:")
        self.component_int_box = BoundedIntText(min=0, max=1, value=0, step=1)
        for trait in ["value", "max"]:
            jslink((self.component_slider, trait), (self.component_int_box, trait))

        self.component_int_box.observe(
            lambda change: self.set_component_index(change["new"]), "value"
        )

        # gridplot for each sublist
        for sub_data in self._data:
            _gridplot_kwargs = {"shape": calculate_gridshape(len(sub_data))}
            _gridplot_kwargs.update(gridplot_kwargs)
            self.gridplots.append(GridPlot(**_gridplot_kwargs))

        self.temporal_graphics: List[graphics.LineGraphic] = list()
        self.temporal_stack_graphics: List[graphics.LineStack] = list()
        self.heatmap_graphics: List[graphics.HeatmapGraphic] = list()
        self.image_graphics: List[graphics.ImageGraphic] = list()
        self.contour_graphics: List[graphics.LineCollection] = list()

        self.heatmap_selectors: List[LinearSelector] = list()

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

        self._current_temporal_components: np.ndarray = None

        self.change_data(data_mapping)

    def set_component_index(self, index: int):
        # TODO: more elegant way than skip_heatmap
        for g in self.contour_graphics:
            g.set_feature(feature="colors", new_data="w", indices=index)

        for g in self.temporal_graphics:
            g.data = self._current_temporal_components[index]

        for s in self.heatmap_selectors:
            # TODO: Very hacky for now, ignores if the slider is currently being moved, prevents weird slider movement
            if s._move_info is None:
                s.selection = index

        self.component_int_box.value = index

    def _heatmap_set_component_index(self, ev):
        index = ev.pick_info["selected_index"]

        self.set_component_index(index)

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
        """
        Changes the data shown in the gridplot.

        Clears all the gridplots, makes and adds new graphics

        Parameters
        ----------
        data_mapping

        Returns
        -------

        """
        for l in self._managed_graphics:
            l.clear()

        self.heatmap_selectors.clear()
        self.linear_selectors.clear()

        self.image_graphic_arrays.clear()

        # clear out old array that stores temporal components
        self._current_temporal_components = None

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

        self.component_slider.value = 0
        self.component_slider.max = len(contours) - 1

        # change data for all gridplots
        for sub_data, sub_data_arrays, gridplot in zip(self._data, data_arrays, self.gridplots):
            self._change_data_gridplot(sub_data, sub_data_arrays, gridplot, contours, component_colors)

        # connect events
        self._connect_events()

        # sync sliders if multiple are present
        if len(self.linear_selectors) > 0:
            self._synchronizer = Synchronizer(*self.linear_selectors, key_bind=None)

        for ls in self.linear_selectors:
            ls.selection.add_event_handler(self.set_frame_index)

        for hs in self.heatmap_selectors:
            hs.selection.add_event_handler(self._heatmap_set_component_index)

    def _change_data_gridplot(
            self,
            data: List[str],
            data_arrays: List[np.ndarray],
            gridplot: GridPlot,
            contours,
            component_colors
    ):
        """
        Changes data in a single gridplot.

        Create the corresponding graphics.

        Parameters
        ----------
        data
        data_arrays
        gridplot
        contours
        component_colors

        Returns
        -------

        """

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
                # Only few one line at a time
                current_graphic = subplot.add_line(
                    data_array[0],
                    colors="w",
                    name="line",
                    **graphic_kwargs
                )

                current_graphic.data.add_event_handler(subplot.auto_scale)
                self.temporal_graphics.append(current_graphic)

                if self._current_temporal_components is None:
                    self._current_temporal_components = data_array

                # otherwise the plot has nothing in it which causes issues
                # subplot.add_line(np.random.rand(data_array.shape[1]), colors=(0, 0, 0, 0), name="pseudo-line")

                # scale according to temporal dims
                subplot.camera.maintain_aspect = False

            elif data_option == "temporal-stack":
                current_graphic = subplot.add_line_stack(
                    data_array,
                    colors=component_colors,
                    name="lines",
                    **graphic_kwargs
                )
                self.temporal_stack_graphics.append(current_graphic)

                # scale according to temporal dims
                subplot.camera.maintain_aspect = False

            elif data_option == "heatmap":
                current_graphic = subplot.add_heatmap(
                    data_array,
                    name="heatmap",
                    **graphic_kwargs
                )
                self.heatmap_graphics.append(current_graphic)

                # scale according to temporal dims
                subplot.camera.maintain_aspect = False

                selector = current_graphic.add_linear_selector(
                    axis="y",
                    color=(1, 1, 1, 0.5),
                    thickness=5,
                )

                self.heatmap_selectors.append(selector)

            else:
                # else it is an image
                if data_array.ndim == 3:
                    frame = data_array[self._current_frame_index]
                else:
                    frame = data_array
                img_graphic = subplot.add_image(
                    frame,
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

        self.set_component_index(ix)

        self.component_int_box.value = ix

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

            # for temporal_graphic in self.temporal_graphics:
            #     contour_graphic.link("colors", target=temporal_graphic, feature="present", new_data=True)

            for cg, tsg in product(self.contour_graphics, self.temporal_stack_graphics):
                cg.link("colors", target=contour_graphic, feature="colors", new_data="w", bidirectional=True)

    def set_frame_index(self, ev):
        # 0 because this will return the same number repeated * n_components
        index = ev.pick_info["selected_index"]
        for image_graphic, full_array in zip(self.image_graphics, self.image_graphic_arrays):
            # txy data
            if full_array.ndim > 2:
                image_graphic.data = full_array[index]

        self._current_frame_index = index
