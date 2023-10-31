from functools import partial
from typing import *
from warnings import warn


import numpy as np
import pandas as pd
from ipydatagrid import DataGrid
from ipywidgets import Textarea, Layout, HBox, VBox
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
import fastplotlib as fpl


from ._utils import DummyMovie


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

PROJS = [
    "mean",
    "max",
    "std",
]

IMAGE_OPTIONS += PROJS


class ExtensionCallWrapper:
    def __init__(
            self,
            extension_func: callable,
            kwargs: dict = None,
            attr: str = None,
            post_process_func: callable = None,
    ):
        """
        Basically a very fancy ``functools.partial``.

        In addition to behaving like ``functools.partial``, it supports:
            - kwargs
            - returning attributes of the return value from the callable
            - postprocessing the return value

        Parameters
        ----------
        extension_func: callable
            extension function reference

        kwargs: dict
            kwargs to pass to the extension function when it is called

        attr: str, optionalself, extension_func: callable, kwargs: dict = None, attr: str = None
            return an attribute of the callable's output instead of the return value of the callable.
            Example: if using rcm, can set ``attr="max_image"`` to return the max proj of the RCM.

        post_process_func: callable
            A function to postprocess before returning, such as zscore, etc.
        """

        if kwargs is None:
            self.kwargs = dict()
        else:
            self.kwargs = kwargs

        self.func = extension_func
        self.attr = attr
        self.post_process_func = post_process_func

    def __call__(self, *args, **kwargs):
        rval = self.func(**self.kwargs)

        if self.attr is not None:
            return getattr(rval, self.attr)

        if self.post_process_func is not None:
            return self.post_process_func(rval)

        return rval


def get_cnmf_data_mapping(
        series: pd.Series,
        input_movie_kwargs: dict,
        temporal_kwargs: dict,
):
    projections = {k: partial(series.caiman.get_projection, k) for k in PROJS}

    rcm_rcb_projs = dict()
    for proj in ["mean", "min", "max", "std"]:
        rcm_rcb_projs[f"rcm-{proj}"] = ExtensionCallWrapper(
            series.cnmf.get_rcm,
            attr=f"{proj}_image"
        )

    zscore_func = TimeSeriesScalerMeanVariance().fit_transform
    norm_func = TimeSeriesScalerMinMax().fit_transform

    temporal_mappings = {
        "temporal": ExtensionCallWrapper(series.cnmf.get_temporal, temporal_kwargs),
        "zscore": ExtensionCallWrapper(series.cnmf.get_temporal, temporal_kwargs, post_process_func=zscore_func),
        "norm": ExtensionCallWrapper(series.cnmf.get_temporal, temporal_kwargs, post_process_func=norm_func),
        "dfof": partial(series.cnmf.get_detrend_dfof),
        "dfof-zscore": ExtensionCallWrapper(series.cnmf.get_detrend_dfof, post_process_func=zscore_func),
        "dfof-norm": ExtensionCallWrapper(series.cnmf.get_detrend_dfof, post_process_func=zscore_func)
    }

    mapping = {
        "input": ExtensionCallWrapper(series.caiman.get_input_movie, input_movie_kwargs),
        "rcm": series.cnmf.get_rcm,
        "rcb": series.cnmf.get_rcb,
        "residuals": series.cnmf.get_residuals,
        "corr": series.caiman.get_corr_image,
        "pnr": series.caiman.get_pnr_image,
        "contours": ExtensionCallWrapper(series.cnmf.get_contours, {"swap_dim": False}),
        **projections,
        **rcm_rcb_projs,
        **temporal_mappings,
    }

    return mapping


class CNMFVizContainer:
    """Widget that contains the DataGrid, params text box fastplotlib GridPlot, etc"""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        start_index: int = None,
        temporal_data_option: str = None,
        image_data_options: list[str] = None,
        temporal_kwargs: dict = None,
        reset_timepoint_on_change: bool = False,
        input_movie_kwargs: dict = None,
        image_widget_kwargs=None,
        data_grid_kwargs: dict = None,
    ):
        """
        Visualize CNMF output and other data columns such as behavior video (optional).

        Note: If using dfof temporal_data_option, you must have already run dfof.

        Parameters
        ----------
        dataframe: pd.DataFrame

        start_index: int

        temporal_data_option: optional, str
            if not provided or ``None`: uses cnmf.get_temporal()

            if zscore: uses zscore of cnmf.get_temporal()

            if norm: uses 0-1 normalized output of cnmf.get_temporal()

            if dfof: uses cnmf.get_dfof()

            if dfof-zscore: uses cnmf.get_dfof() and then zscores

            if dfof-norm: uses cnmf.get_dfof() and then 0-1 normalizes

        reset_timepoint_on_change: bool

        temporal_postprocess: optional, list of str or callable

        heatmap_postprocess: str, None, callable
            if str: one of "norm", "dfof", "zscore"
            Or a callable to postprocess using your own function

        temporal_kwargs: dict
            kwargs passed to cnmf.get_temporal(), example: {"add_residuals" : True}.
            Ignored if temporal_data_option contains "dfof"

        input_movie_kwargs: dict
            kwargs passed to caiman.get_input()

        data_grid_kwargs
        """

        self._dataframe = dataframe

        valid_temporal_options = [
            "temporal",
            "zscore",
            "norm",
            "dfof",
            "dfof-zscore",
            "dfof-norm"
        ]

        if temporal_data_option is None:
            temporal_data_option = "temporal"

        if temporal_data_option not in valid_temporal_options:
            raise ValueError(
                f"You have passed the following invalid temporal option: {temporal_data_option}\n"
                f"Valid options are:\n"
                f"{valid_temporal_options}"
            )

        if image_data_options is None:
            image_data_options = [
                "input",
                "rcm",
                "rcb",
                "residuals"
            ]

        for option in image_data_options:
            if option not in IMAGE_OPTIONS:
                raise ValueError(
                    f"Invalid image option passed, valid image options are:\n"
                    f"{IMAGE_OPTIONS}"
                )

        self.image_data_options = image_data_options

        self.temporal_data_option = temporal_data_option
        self.temporal_kwargs = temporal_kwargs

        if self.temporal_kwargs is None:
            self.temporal_kwargs = dict()

        # for now we will force all components, accepted and rejected, to be shown
        if "component_indices" in self.temporal_kwargs.keys():
            raise ValueError(
                "The kwarg `component_indices` is not allowed here."
            )

        self.reset_timepoint_on_change = reset_timepoint_on_change
        self.input_movie_kwargs = input_movie_kwargs

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

        if data_grid_kwargs is None:
            data_grid_kwargs = dict()

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
            max_width="500px",
            disabled=True,
        )

        if image_widget_kwargs is None:
            image_widget_kwargs = dict()

        default_image_widget_kwargs = {
            "cmap": "gnuplot2"
        }

        self.image_widget_kwargs = {
            **default_image_widget_kwargs,
            **image_widget_kwargs
        }

        if start_index is None:
            start_index = dataframe[dataframe.algo == "cnmf"].iloc[0].name

        self.current_row: int = start_index

        self.datagrid.select(
            row1=start_index,
            column1=0,
            row2=start_index,
            column2=len(df_show.columns),
            clear_mode="all"
        )

        # callback when row changed
        self.datagrid.observe(self._row_changed, names="selections")

        self._plot_temporal = fpl.Plot()
        self._plot_temporal.camera.maintain_aspect = False
        self._plot_heatmap = fpl.Plot()
        self._plot_heatmap.camera.maintain_aspect = False

        self._image_widget: fpl.ImageWidget = None

        self._synchronizer = fpl.Synchronizer(key_bind=None)

        data_arrays = self._get_row_data(index=start_index)
        self._set_data(data_arrays)

    def _get_selected_row(self) -> Union[int, None]:
        r1 = self.datagrid.selections[0]["r1"]
        r2 = self.datagrid.selections[0]["r2"]

        if r1 != r2:
            warn("Only single row selection is currently allowed")
            return

        # get corresponding dataframe index from currently visible dataframe
        # since filtering etc. is possible
        index = self.datagrid.get_visible_data().index[r1]

        return index

    def _get_row_data(self, index: int) -> Dict[str, np.ndarray]:
        data_mapping = get_cnmf_data_mapping(
            series=self._dataframe.iloc[index],
            input_movie_kwargs=self.input_movie_kwargs,
            temporal_kwargs=self.temporal_kwargs
        )

        temporal = data_mapping[self.temporal_data_option]()

        rcm = data_mapping["rcm"]()

        shape = rcm.shape
        ndim = rcm.ndim
        size = rcm.shape[0] * rcm.shape[1] * rcm.shape[2]

        images = list()
        for option in self.image_data_options:
            array = data_mapping[option]()

            if array.ndim == 2:  # for 2D images, to make ImageWidget happy
                array = DummyMovie(array, shape=shape, ndim=ndim, size=size)

            images.append(array)

        data_arrays = {
            "temporal": temporal,
            "images": images,
        }

        return data_arrays

    def _row_changed(self, *args):
        index = self._get_selected_row()
        if index is None:
            return

        if self.current_row == index:
            return

        try:
            data_arrays = self._get_row_data(index)

        except Exception as e:
            self.params_text_area.value = f"{type(e).__name__}\n" \
                                          f"{str(e)}\n\n" \
                                          f"See jupyter log for details"
            raise e

        else:
            # no exceptions, set plots
            self._set_data(data_arrays)

    def _set_data(self, data_arrays: Dict[str, np.ndarray]):
        # self._contour_graphics.clear()

        self._plot_temporal.clear()
        self._plot_heatmap.clear()

        self._synchronizer.clear()

        if self._image_widget is None:
            self._image_widget = fpl.ImageWidget(
                data=data_arrays["images"],
                **self.image_widget_kwargs
            )

            # need to start it here so that we can access the toolbar to link events with the slider
            self._image_widget.show()

        else:
            self._image_widget.set_data(data_arrays["images"])

        temporal = data_arrays["temporal"]

        self._plot_temporal.add_line(temporal[0], name="line")
        self._plot_heatmap.add_heatmap(temporal, name="heatmap")

        self._linear_selector_temporal: fpl.LinearSelector = self._plot_temporal["line"].add_linear_selector()
        self._linear_selector_temporal.selection.add_event_handler(self._set_frame_index_from_linear_selector)

        self._linear_selector_heatmap: fpl.LinearSelector = self._plot_heatmap["heatmap"].add_linear_selector()

        # sync the linear selectors
        self._synchronizer.add(self._linear_selector_temporal)
        self._synchronizer.add(self._linear_selector_heatmap)

        # absolute garbage monkey patch which I will fix once we make ImageWidget emit its own events
        if hasattr(self._image_widget.sliders["t"], "qslider"):
            self._image_widget.sliders["t"].qslider.valueChanged.connect(self._set_linear_selector_index_from_image_widget)
        else:
            # ipywidget
            self._image_widget.sliders["t"].observe(self._set_linear_selector_index_from_image_widget, "value")

    def _set_frame_index_from_linear_selector(self, ev):
        # TODO: hacky mess, need to make ImageWidget emit events
        ix = ev.pick_info["selected_index"]
        self._image_widget.sliders["t"].value = ix

    def _set_linear_selector_index_from_image_widget(self, ev):
        if isinstance(ev, dict):
            # ipywidget
            ix = ev["new"]

        # else it's directly from Qt slider
        else:
            ix = ev

        self._linear_selector_temporal.selection = ix

    def show(self, sidecar: bool = False):
        """
        Show the widget

        Parameters
        ----------
        sidecar

        Returns
        -------

        """

        datagrid_params = HBox([self.datagrid, self.params_text_area])

        temporals = VBox([self._plot_temporal.show(), self._plot_heatmap.show()])

        plots = HBox([temporals, self._image_widget.widget])

        return VBox([datagrid_params, plots])
