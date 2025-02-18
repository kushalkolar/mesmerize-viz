from collections import OrderedDict
from functools import partial
from typing import *
from warnings import warn

import numpy as np
import pandas as pd
from ipydatagrid import DataGrid
from ipywidgets import Button, Tab, Text, Textarea, Layout, HBox, VBox, Checkbox, FloatSlider, BoundedFloatText, \
    IntSlider, BoundedIntText, RadioButtons, Dropdown, jslink
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
import fastplotlib as fpl
from fastplotlib.utils import get_cmap
from caiman.source_extraction.cnmf import CNMF
from IPython.display import display
from sidecar import Sidecar

from mesmerize_core.caiman_extensions.cnmf import cnmf_cache
from mesmerize_core import CNMFExtensions
from ._store_model import TimeStore, NeuronStore

from ._utils import DummyMovie, format_params

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
        "cnmf_obj": series.cnmf.get_output,
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


class EvalController:
    def __init__(self):
        self._float_metrics = [
            "min_SNR",
            "SNR_lowest",
            "rval_thr",
            "rval_lowest",
            "min_cnn_thr",
            "cnn_lowest",

        ]

        # caiman is really annoying with this
        # maps eval metric to the estimates attrs
        self._metric_array_mapping = {
            "min_SNR": "SNR_comp",
            "SNR_lowest": "SNR_comp",
            "rval_thr": "r_values",
            "rval_lowest": "r_values",
            "min_cnn_thr": "cnn_preds",
            "cnn_lowest": "cnn_preds",
        }

        self._widgets = OrderedDict()

        param_entries = list()

        for metric in self._float_metrics:
            # start with very high and very low values so that min >! max ever with real data 
            slider = FloatSlider(value=0, min=-99_999, max=99_999, step=0.01, description=metric)
            spinbox = BoundedFloatText(
                value=0, min=-99_999, max=99_999, step=0.01,
                description_tooltip=metric, layout=Layout(width="70px"), readout_format='.2f',
            )

            slider.observe(self._call_handlers, "value")
            spinbox.observe(self._call_handlers, "value")

            jslink((slider, "value"), (spinbox, "value"))

            param_entries.append(HBox([spinbox, slider]))

            # keep this so it's easier to modify in set_limits
            self._widgets[metric] = {"slider": slider, "spinbox": spinbox}

        self.use_cnn_checkbox = Checkbox(
            value=True,
            description="use_cnn",
            description_tooltip="use CNN classifier"
        )

        self._handlers = list()

        # limits must be set first before it's usable
        self._block_handlers = True

        self.button_save_eval = Button(
            description="Save Eval to disk",
        )

        self.button_reset_eval = Button(
            description="Reset Eval Params",
            description_tooltip="Reset eval from disk"
        )

        buttons = HBox([self.button_save_eval, self.button_reset_eval])

        self.widget = VBox([*param_entries, self.use_cnn_checkbox, buttons])

    def set_limits(self, cnmf_obj: CNMF):
        self._block_handlers = True
        for metric in self._float_metrics:
            metric_array = getattr(cnmf_obj.estimates, self._metric_array_mapping[metric])
            if len(metric_array) == 0:  # for example, cnn_preds is an empty array for CNMFE
                for ui_element in ["slider", "spinbox"]:
                    self._widgets[metric][ui_element].disabled = True
            else:
                for ui_element in ["slider", "spinbox"]:
                    self._widgets[metric][ui_element].disabled = False
                    # allow 100 steps
                    self._widgets[metric][ui_element].step = np.ptp(metric_array) / 100
                    self._widgets[metric][ui_element].min = metric_array.min()
                    self._widgets[metric][ui_element].max = metric_array.max()
                    self._widgets[metric][ui_element].value = cnmf_obj.params.get_group("quality")[metric]

        if len(cnmf_obj.estimates.cnn_preds) == 0:
            # cnn_preds is not present, force use_cnn = False, disable the UI
            self.use_cnn_checkbox.value = False
            self.use_cnn_checkbox.disabled = True
        else:
            # cnn_preds is a real array, allow it to be settable
            self.use_cnn_checkbox.disabled = False
            self.use_cnn_checkbox.value = cnmf_obj.params.get_group("quality")["use_cnn"]

        self._block_handlers = False

    def get_data(self):
        data = dict()
        for metric in self._float_metrics:
            data[metric] = self._widgets[metric]["spinbox"].value

        data["use_cnn"] = self.use_cnn_checkbox.value

        return data

    def add_handler(self, func: callable):
        """Handlers must accept a dict argument, the dict has the eval params"""
        self._handlers.append(func)

    def _call_handlers(self, obj):
        if self._block_handlers:
            return

        data = self.get_data()
        for handler in self._handlers:
            handler(data)

    def remove_handler(self, func: callable):
        self._handlers.remove(func)

    def clear_handlers(self):
        self._handlers.clear()


class CNMFVizContainer:
    """Widget that contains the DataGrid, params text box fastplotlib GridPlot, etc"""

    @property
    def component_index(self) -> int:
        """Current component index"""
        return self._component_index

    @property
    def plot_temporal(self) -> fpl.Figure:
        """Figure with the single temporal trace"""
        return self._plot_temporal

    @property
    def plot_heatmap(self) -> fpl.Figure:
        """Figure with the heatmap"""
        return self._plot_heatmap

    @property
    def image_widget(self) -> fpl.ImageWidget:
        """ImageWidget"""
        return self._image_widget

    @property
    def cnmf_obj(self) -> CNMF:
        """Current CNMF object displayed in the viewer, use with care"""
        return self._cnmf_obj

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
        Visualize CNMF output and other data columns, such as behavior video (optional).

        Note: If using `dfof` for `temporal_data_option`, you must have already run dfof.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input data to be visualized.

        start_index : int
            Start index for the initial data displayed in the visualization.

        temporal_data_option : str, optional
            Defines the temporal data option to use:

            +------------------+--------------------------------------------------------------------------+
            | Option           | Description                                                              |
            +==================+==========================================================================+
            | None             | Default. Uses `cnmf.get_temporal()`                                      |
            +------------------+--------------------------------------------------------------------------+
            | zscore           | Uses the z-score of `cnmf.get_temporal()`                                |
            +------------------+--------------------------------------------------------------------------+
            | norm             | Uses 0-1 normalized output of `cnmf.get_temporal()`                      |
            +------------------+--------------------------------------------------------------------------+
            | dfof             | Uses `cnmf.get_dfof()`                                                  |
            +------------------+--------------------------------------------------------------------------+
            | dfof-zscore      | Uses `cnmf.get_dfof()` and applies z-scoring                             |
            +------------------+--------------------------------------------------------------------------+
            | dfof-norm        | Uses `cnmf.get_dfof()` and applies 0-1 normalization                     |
            +------------------+--------------------------------------------------------------------------+

        reset_timepoint_on_change : bool, default False
            If `True`, resets the timepoint in the ImageWidget when changing items/rows.

        temporal_kwargs : dict, optional
            Additional keyword arguments passed to `cnmf.get_temporal()`. Ignored if `temporal_data_option` contains "dfof".

            +-------------------+---------------------------------------------------------------+
            | Argument          | Description                                                   |
            +===================+===============================================================+
            | component_indices | str or np.ndarray, optional                                   |
            |                   | Specifies which components to include:                        |
            |                   | - `None` or "all": includes all components                    |
            |                   | - "good": uses good components (Estimates.idx_components)     |
            |                   | - "bad": uses bad components (Estimates.idx_components_bad)   |
            |                   | - np.ndarray: uses the specified indices                      |
            +-------------------+---------------------------------------------------------------+
            | add_background    | bool, default: False                                          |
            |                   | If `True`, adds the temporal background (cnmf.estimates.f)    |
            +-------------------+---------------------------------------------------------------+
            | add_residuals     | bool, default: False                                          |
            |                   | If `True`, adds residuals (cnmf.estimates.YrA)                |
            +-------------------+---------------------------------------------------------------+
            | return_copy       | bool                                                          |
            |                   | If `True`, returns a copy of the cached value.                |
            |                   | If `False`, returns the cached object directly (not           |
            |                   | recommended, may cause unexpected behavior).                  |
            +-------------------+---------------------------------------------------------------+

        input_movie_kwargs: dict, optional
            Arguments passed to mesmerize_core.caiman.get_input_movie. Valid data options are:

            +----------+---------------------------------------------------------------+
            | Argument | Description                                                   |
            +==========+===============================================================+
            | reader   | callable                                                      |
            |          | Function to read the input movie path and return an array-like|
            +----------+---------------------------------------------------------------+
            | **kwargs | passed to reader function                                     |
            +----------+---------------------------------------------------------------+

            `mesmerize_core.get_input_movie() <https://mesmerize-core.readthedocs.io/en/latest/api/common.html#mesmerize_core.CaimanSeriesExtensions.get_input_movie>`_

        image_widget_kwargs : dict, optional
            Arguments passed to ImageWidget. Valid options include:

            +-------------------+-----------------------------------------------------+
            | Argument          | Description                                         |
            +===================+=====================================================+
            | data              | array-like or list of array-like data to display    |
            +-------------------+-----------------------------------------------------+
            | window_funcs      | dictionary of window functions for "t" or "z"       |
            |                   | dimensions in format {"dim": (func, window_size)}   |
            |                   | e.g., {"t": (np.mean, 11)}, {"z": (np.max, 3)}      |
            +-------------------+-----------------------------------------------------+
            | frame_apply       | function(s) applied to each data array to generate  |
            |                   | final 2D image, can be callable or {idx: callable}  |
            +-------------------+-----------------------------------------------------+
            | figure_shape      | tuple for custom shape of the figure,               |
            |                   | defaults to auto-estimation                         |
            +-------------------+-----------------------------------------------------+
            | figure_kwargs     | dictionary of arguments passed to GridPlot          |
            +-------------------+-----------------------------------------------------+
            | names             | optional list of names for subplots                 |
            +-------------------+-----------------------------------------------------+
            | histogram_widget  | boolean to create a histogram LUT widget per subplot|
            +-------------------+-----------------------------------------------------+
            | rgb               | boolean or list indicating if arrays are RGB(A)     |
            +-------------------+-----------------------------------------------------+
            | graphic_kwargs    | arguments passed to each ImageGraphic in subplots   |
            +-------------------+-----------------------------------------------------+

            Example: `image_widget_kwargs={"cmap": "viridis"}`

        data_grid_kwargs : dict, optional
            Arguments passed to DataGrid. Valid options include:

            +-------------------------+-----------------------------------------------------------+
            | Argument                | Description                                               |
            +=========================+===========================================================+
            | base_row_size           | int, default: 20                                          |
            |                         | Default row height                                        |
            +-------------------------+-----------------------------------------------------------+
            | base_column_size        | int, default: 64                                          |
            |                         | Default column width                                      |
            +-------------------------+-----------------------------------------------------------+
            | base_row_header_size    | int, default: 64                                          |
            |                         | Default row header width                                  |
            +-------------------------+-----------------------------------------------------------+
            | base_column_header_size | int, default: 20                                          |
            |                         | Default column header height                              |
            +-------------------------+-----------------------------------------------------------+
            | header_visibility       | {'all', 'row', 'column', 'none'}, default: 'all'          |
            |                         | Controls header visibility mode                           |
            +-------------------------+-----------------------------------------------------------+
            | dataframe               | pandas.DataFrame                                          |
            |                         | Data displayed in the DataGrid                            |
            +-------------------------+-----------------------------------------------------------+
            | renderers               | dict                                                      |
            |                         | Custom renderers for cells by column name                 |
            +-------------------------+-----------------------------------------------------------+
            | default_renderer        | CellRenderer, default: TextRenderer                       |
            |                         | Default renderer for cell rendering                       |
            +-------------------------+-----------------------------------------------------------+
            | header_renderer         | CellRenderer, default: TextRenderer                       |
            |                         | Renderer for header cells                                 |
            +-------------------------+-----------------------------------------------------------+
            | corner_renderer         | CellRenderer, default: TextRenderer                       |
            |                         | Renderer for corner header cells                          |
            +-------------------------+-----------------------------------------------------------+
            | selection_mode          | {'row', 'column', 'cell', 'none'}, default: 'none'        |
            |                         | Mode for cell selection by user or programmatically       |
            +-------------------------+-----------------------------------------------------------+
            | selections              | list of dict                                              |
            |                         | Rectangular regions defined by row/column start & end     |
            +-------------------------+-----------------------------------------------------------+
            | editable                | bool, default: False                                      |
            |                         | Enables direct cell editing                               |
            +-------------------------+-----------------------------------------------------------+
            | column_widths           | dict of {str: int}, default: {}                           |
            |                         | Custom widths for columns by name                         |
            +-------------------------+-----------------------------------------------------------+
            | auto_fit_columns        | bool, default: False                                      |
            |                         | Automatically adjust column widths                        |
            +-------------------------+-----------------------------------------------------------+
            | auto_fit_params         | dict                                                      |
            |                         | Parameters for column auto-fitting                        |
            +-------------------------+-----------------------------------------------------------+
            | grid_style              | dict of {propertyName: str | VegaExpr | dict}             |
            |                         | Global styling properties for the grid                    |
            +-------------------------+-----------------------------------------------------------+
            | index_name              | str, default: "key"                                       |
            |                         | Name of the index column (set once at construction)       |
            +-------------------------+-----------------------------------------------------------+
            | horizontal_stripes      | bool, default: False                                      |
            |                         | Themed coloring for alternate grid rows                   |
            +-------------------------+-----------------------------------------------------------+
            | vertical_stripes        | bool, default: False                                      |
            |                         | Themed coloring for alternate grid columns                |
            +-------------------------+-----------------------------------------------------------+
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

        self._image_data_options = image_data_options

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
            "cmap": "gnuplot2",
            "figure_kwargs": {"size": (720, 602)},
        }

        self.image_widget_kwargs = {
            **default_image_widget_kwargs,
            **image_widget_kwargs
        }

        # figure out start index by searching for cnmf or cnmfe items
        if start_index is None:
            start_index = dataframe[(dataframe.algo == "cnmf") | (dataframe.algo == "cnmfe")].iloc[0].name

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

        self._time_store = TimeStore()
        self._neuron_store = NeuronStore()

        # ipywidgets for selecting components
        self.component_int_box = BoundedIntText(min=0, max=1, value=0, step=1, layout=Layout(width="100px"))
        self._neuron_store.subscribe(self.component_int_box)

        self._component_metrics_text = Text(
            value="",
            placeholder="component metrics",
            description='metrics:',
            disabled=True,
            layout=Layout(width="350px")
        )

        self._button_center_component = Button(description="center on component",)
        self._button_center_component.on_click(self._center_on_component)

        # checkbox to zoom into components when selected
        self.checkbox_zoom_components = Checkbox(
            value=True,
            description="auto-zoom component",
            description_tooltip="If checked, zoom into selected component"
        )
        self.checkbox_zoom_components.observe(self._neuron_store._set_auto_zoom, "value")
        # zoom factor
        self.zoom_components_scale = FloatSlider(
            min=0.25,
            max=3,
            value=1,
            step=0.25,
            description="zoom scale",
            description_tooltip="zoom scale as a factor of component width/height"
        )
        self.zoom_components_scale.observe(self._neuron_store._set_zoom_scale, "value")
        # organize these widgets to be shown at the top
        self._top_widget = VBox([
            HBox([self.datagrid, self.params_text_area]),
            HBox([self.component_int_box, self._component_metrics_text]),
            HBox([self._button_center_component, self.checkbox_zoom_components, self.zoom_components_scale])
        ])

        self._dropdown_contour_colors = Dropdown(
            options=["random", "accepted", "rejected", "snr_comps", "snr_comps_log", "r_values", "cnn_preds"],
            value="random",
            description='contour colors:',
        )

        self._dropdown_contour_colors.observe(self._ipywidget_set_component_colors, "value")

        self._radio_visible_components = RadioButtons(
            options=["all", "accepted", "rejected", "none"],
            description_tooltip="contours to make visible",
            description="visible contours"
        )

        self._radio_visible_components.observe(self._ipywidget_set_component_colors, "value")

        self._spinbox_alpha_invisible_contours = FloatSlider(
            value=0.0,
            min=0.0,
            max=1.0,
            step=0.1,
            description="invisible alpha:",
            description_tooltip="transparency of contours set to be invisible",
            disabled=False
        )

        self._spinbox_alpha_invisible_contours.observe(self._ipywidget_set_component_colors, "value")

        self._box_contour_controls = VBox([
            self._dropdown_contour_colors,
            HBox([self._radio_visible_components, self._spinbox_alpha_invisible_contours])
        ])

        self._eval_controller = EvalController()
        self._eval_controller.add_handler(self._set_eval)
        self._eval_controller.button_save_eval.on_click(self._save_eval)
        self._eval_controller.button_reset_eval.on_click(self._reset_eval)

        self._tab_contours_eval = Tab()
        self._tab_contours_eval.children = [self._box_contour_controls, self._eval_controller.widget]
        self._tab_contours_eval.titles = ["contour colors", "eval params"]

        # plots
        self._plot_temporal = fpl.Figure(size=(500, 120))
        self._plot_temporal[0, 0].camera.maintain_aspect = False
        self._plot_heatmap = fpl.Figure(size=(500, 450))
        self._plot_heatmap[0, 0].camera.maintain_aspect = False

        self._image_widget: fpl.ImageWidget = None

        self._cnmf_obj: CNMF = None

        data_arrays = self._get_row_data(index=start_index)
        self._set_data(data_arrays)

        self._set_params_text_area(index=start_index)

        self._widget = None
        self._sidecar = None

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

        if ("norm" in self.temporal_data_option) or ("zscore" in self.temporal_data_option):
            temporal = temporal[..., 0]

        rcm = data_mapping["rcm"]()

        shape = rcm.shape
        ndim = rcm.ndim
        size = rcm.shape[0] * rcm.shape[1] * rcm.shape[2]

        images = list()
        for option in self._image_data_options:
            array = data_mapping[option]()

            if array.ndim == 2:  # for 2D images, to make ImageWidget happy
                array = DummyMovie(array, shape=shape, ndim=ndim, size=size)

            images.append(array)

        contours = data_mapping["contours"]()
        cnmf_obj = data_mapping["cnmf_obj"]()

        data_arrays = {
            "temporal": temporal,
            "images": images,
            "contours": contours,
            "cnmf_obj": cnmf_obj
        }

        return data_arrays

    def _set_params_text_area(self, index):
        row = self._dataframe.iloc[index]
        # try and get the param diffs
        try:
            param_diffs = self._dataframe.caiman.get_params_diffs(
                algo=row["algo"],
                item_name=row["item_name"]
            ).loc[index]

            diffs_dict = {"diffs": param_diffs.to_dict()}
            diffs = f"{format_params(diffs_dict, 0)}\n\n"
        except:
            diffs = ""

        # diffs and full params
        self.params_text_area.value = diffs + format_params(self._dataframe.iloc[index].params, 0)

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
            self._set_params_text_area(index)
            self.current_row = index

    def _set_data(self, data_arrays: Dict[str, np.ndarray]):

        self._plot_temporal.clear()
        self._plot_heatmap.clear()

        # make our temporal plots first, else image widget slider events could trigger linear selectors
        self._temporal_data = data_arrays["temporal"]
        self._plot_temporal[0, 0].add_line(self._temporal_data[0], name="line")
        self._neuron_store.subscribe(self._plot_temporal[0, 0], data=self._temporal_data)

        # autoscale the single temporal line plot when the data changes
        self._plot_temporal[0, 0]["line"].add_event_handler(self._plot_temporal[0, 0].auto_scale)
        self._plot_heatmap[0, 0].add_image(self._temporal_data, name="heatmap",)
        self._plot_heatmap[0, 0]['heatmap'].add_event_handler(self._plot_heatmap[0, 0].auto_scale)

        # component selectors
        self._component_linear_selector: fpl.LinearSelector = self._plot_heatmap[0, 0]['heatmap'].add_linear_selector(axis="y", thickness=5)
        self._neuron_store.subscribe(self._component_linear_selector)

        # linear selectors and events
        self._linear_selector_temporal: fpl.LinearSelector = self._plot_temporal[0, 0]["line"].add_linear_selector()
        self._linear_selector_heatmap: fpl.LinearSelector = self._plot_heatmap[0, 0]["heatmap"].add_linear_selector()

        self._time_store.subscribe(self._linear_selector_temporal,)
        self._time_store.subscribe(self._linear_selector_heatmap,)

        if self._image_widget is None:
            self._image_widget = fpl.ImageWidget(
                data=data_arrays["images"],
                names=self._image_data_options,
                **self.image_widget_kwargs
            )
            self._time_store.subscribe(self._image_widget,)

            # need to start it here so that we can access the toolbar to link events with the slider
            self._image_widget.show()

        else:
            # image widget doesn't need clear, we can just use set_data
            self._image_widget.set_data(
                data_arrays["images"],
                reset_indices=self.reset_timepoint_on_change,
                reset_vmin_vmax=True
            )

            for g in self._image_widget.managed_graphics:
                g.registered_callbacks.clear()

            for subplot in self._image_widget.figure:
                if "contours" in subplot:
                    # delete the contour graphics
                    subplot.delete_graphic(subplot["contours"])

        contours = data_arrays["contours"][0]

        n_components = len(contours)
        self._random_colors = np.random.rand(n_components, 4).astype(np.float32)
        self._random_colors[:, -1] = 1

        for subplot in self._image_widget.figure:
            subplot.add_line_collection(
                contours,
                colors=self._random_colors,
                name="contours"
            )
            self._neuron_store.subscribe(subplot)

        self.component_int_box.value = 0
        self.component_int_box.max = n_components - 1

        # current state of CNMF object
        # this can be different from the one in the dataframe if the user uses eval
        self._cnmf_obj: CNMF = data_arrays["cnmf_obj"]

        self._eval_controller.set_limits(self._cnmf_obj)

    def set_component_index(self, index):

        if self.checkbox_zoom_components.value:
            self._zoom_into_component(index)

        self.component_int_box.unobserve_all()
        self.component_int_box.value = index
        self.component_int_box.observe(
            lambda change: self.set_component_index(change["new"]), "value"
        )

        metrics = f"snr: {self._cnmf_obj.estimates.SNR_comp[index]:.02f}, " \
                  f"r_values: {self._cnmf_obj.estimates.r_values[index]:.02f}"

        if len(self._cnmf_obj.estimates.cnn_preds) > 0:
            metrics += f", cnn: {self._cnmf_obj.estimates.cnn_preds[index]:.02f} "

        self._component_metrics_text.value = metrics

    def _set_zoom_flag(self, value: bool):
        self._neuron_store._set_auto_zoom(value)

    def _set_zoom_scale(self, scale):
        self._neuron_store.zoom_scale = scale

    def _center_on_component(self, obj):
        self._neuron_store.zoom_into_component()

    def _ipywidget_set_component_colors(self, *args):
        """just a wrapper to make ipywidgets happy"""
        colors = self._dropdown_contour_colors.value
        self.set_component_colors(colors)

    def set_component_colors(
            self,
            metric: Union[str, np.ndarray],
            cmap: str = None,
    ):
        """

        Parameters
        ----------
        metric: str or np.ndarray
            str, one of: random, accepted, rejected, accepted-rejected, snr_comps, snr_comps_log,
            r_values, cnn_preds.

            Can also pass a 1D array of other metrics

            If np.ndarray, it must be of the same length as the number of components

        cmap: str
            custom cmap for the colors

        Returns
        -------

        """
        cnmf_obj = self._cnmf_obj
        n_contours = len(self._image_widget.figure[0, 0]["contours"])

        # use the random colors
        if metric == "random":
            for subplot in self._image_widget.figure:
                for i, g in enumerate(subplot["contours"].graphics):
                    g.colors = self._random_colors[i]

                # set alpha values based on all, accepted, rejected selection
                self._set_component_visibility(subplot["contours"], cnmf_obj)
            return

        if metric in ["accepted", "rejected"]:
            if cmap is None:
                cmap = "Set1"

            # make a empty array for cmap_values
            classifier = np.zeros(n_contours, dtype=int)
            # set the accepted components to 1
            classifier[cnmf_obj.estimates.idx_components] = 1

        else:
            if cmap is None:
                cmap = "spring"

            if metric == "snr_comps":
                classifier = cnmf_obj.estimates.SNR_comp

            elif metric == "snr_comps_log":
                classifier = np.log10(cnmf_obj.estimates.SNR_comp)

            elif metric == "r_values":
                classifier = cnmf_obj.estimates.r_values

            elif metric == "cnn_preds":
                classifier = cnmf_obj.estimates.cnn_preds

            elif isinstance(metric, np.ndarray):
                if not metric.size == n_contours:
                    raise ValueError(f"If using np.ndarray cor component_colors, the array size must be "
                                     f"the same as n_contours: {n_contours}, your array size is: {metric.size}")

                classifier = metric

            else:
                raise ValueError("Invalid colors value")

        for subplot in self._image_widget.figure:
            # first initialize using a quantitative cmap
            # this ensures that setting cmap_values will work
            subplot["contours"].cmap = "gray"

            if len(np.unique(classifier)) == 1:
                # TODO: patch until next fastplotlib release
                color = get_cmap(cmap)[0]  # set using first color in cmap
                subplot["contours"][:].colors = color
            else:
                # change to cmap_transform when fpl pr #663 is merged
                subplot["contours"].cmap_values = classifier
                subplot["contours"].cmap = cmap

            self._set_component_visibility(subplot["contours"], cnmf_obj)

    def _set_component_visibility(self, contours: fpl.LineCollection, cnmf_obj):
        visible = self._radio_visible_components.value
        alpha_invisible = self._spinbox_alpha_invisible_contours.value

        # choose to make all or accepted or rejected visible
        if visible == "accepted":
            contours[cnmf_obj.estimates.idx_components_bad].colors[:, -1] = alpha_invisible

        elif visible == "rejected":
            contours[cnmf_obj.estimates.idx_components].colors[:, -1] = alpha_invisible

        elif visible == "none":
            # make everything invisible
            contours[:].colors[:, -1] = 0

        else:
            # make everything visible
            contours[:].colors[:, -1] = 1

    def _set_eval(self, eval_params: dict):
        index = self._get_selected_row()
        # wonky caiman params object stuff
        self._cnmf_obj.params.quality.update(eval_params)

        self._cnmf_obj.estimates.filter_components(
            imgs=self._dataframe.iloc[index].caiman.get_input_movie(),
            params=self._cnmf_obj.params
        )

        # set the colors
        colors = self._dropdown_contour_colors.value
        self.set_component_colors(colors)

    def _reset_eval(self, obj):
        index = self._get_selected_row()

        # get CNMF object from cache
        cnmf_obj = self._dataframe.iloc[index].cnmf.get_output()
        # reset eval
        self._set_eval(cnmf_obj.params.get_group("quality"))

    def _save_eval(self, obj):
        # this overwrites the hdf5 file
        index = self._get_selected_row()

        # delete existing file
        path = self._dataframe.iloc[index].cnmf.get_output_path()
        path.unlink()

        # save to disk
        self._cnmf_obj.save(str(path))

        # clear the cache
        cnmf_cache.clear_cache()

        print("Overwrote CNMF object with new eval")

    def _manual_toggle_component(self, ev):
        if not hasattr(ev, "key"):
            return

        if ev.key == "a":
            if self.component_index in self._cnmf_obj.estimates.idx_components:
                # component already in good
                return
            # else swap it from bad and put into good

            # remove from bad
            self._cnmf_obj.estimates.idx_components_bad = np.delete(
                self._cnmf_obj.estimates.idx_components_bad,
                self._cnmf_obj.estimates.idx_components_bad == self.component_index
            )

            # put index into good
            self._cnmf_obj.estimates.idx_components = np.sort(np.concatenate([
                self._cnmf_obj.estimates.idx_components,
                [self.component_index]
            ]))

        elif ev.key == "r":
            if self.component_index in self._cnmf_obj.estimates.idx_components_bad:
                # component already in bad
                return
            # else swap it from bad and put into good

            # remove from good
            self._cnmf_obj.estimates.idx_components = np.delete(
                self._cnmf_obj.estimates.idx_components,
                self._cnmf_obj.estimates.idx_components == self.component_index
            )

            # put index into bad
            self._cnmf_obj.estimates.idx_components_bad = np.sort(np.concatenate([
                self._cnmf_obj.estimates.idx_components_bad,
                [self.component_index]
            ]))

        # set the colors
        colors = self._dropdown_contour_colors.value
        self.set_component_colors(colors)

    def show(self, sidecar: bool = False):
        """
        Show the widget

        Parameters
        ----------
        sidecar

        Returns
        -------

        """

        if self.image_widget.figure.canvas.__class__.__name__ in ["JupyterWgpuCanvas", "JupyterRenderCanvas"]:
            temporals = VBox([self._plot_temporal.show(), self._plot_heatmap.show()])
            plots = HBox([temporals, self._image_widget.show()])
            self._widget = VBox([self._top_widget, plots, self._tab_contours_eval])
            if sidecar:
                self._sidecar = Sidecar()
                with self._sidecar:
                    return display(self._widget)
            else:
                return self._widget

        elif self.image_widget.figure.canvas.__class__.__name__ in ["QWgpuCanvas", "QRenderCanvas"]:
            self.plot_temporal.show()
            self.plot_heatmap.show()
            self.image_widget.show()

            self._widget = VBox([self._top_widget, self._tab_contours_eval])

            return self._widget
        else:
            raise EnvironmentError(
                "No available output context. Make sure you're running in jupyterlab or using %gui qt"
            )

    def close(self):
        self.plot_temporal.close()
        self.plot_heatmap.close()
        self.image_widget.close()
        self._widget.close()
        if self._sidecar is not None:
            self._sidecar.close()


@pd.api.extensions.register_dataframe_accessor("cnmf")
class CNMFDataFrameVizExtension:
    def __init__(self, df):
        self._dataframe = df

    def viz(
            self,
            start_index: int = None,
            temporal_data_option: str = None,
            image_data_options: list[str] = None,
            temporal_kwargs: dict = None,
            reset_timepoint_on_change: bool = False,
            input_movie_kwargs: dict = None,
            image_widget_kwargs: dict = None,
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

            +--------------------------+-------------------------------------------------------------------------------------------------------------------------+
            | option                   | description                                                                                                             |
            +==========================+=========================================================================================================================+
            | not provided or ``None`` | uses ``cnmf.get_temporal()`` to return ``cnmf.estimates.C``, with or without residuals depending on ``temporal_kwargs`` |
            | "zscore"                 | uses zscore of ``cnmf.get_tempral()``                                                                                   |
            | "norm"                   | uses 0-1 normalized output of ``cnmf.get_temporal()``                                                                   |
            | "dfof"                   | uses ``cnmf.get_detrend_dfof()``. You must have called ``run_detrend_dfof()`` beforehand                                |
            | "dfof-zscore"            | returns zscore of dfof                                                                                                  |
            | "dfof-norm"              | returns 0-1 normalized dfof                                                                                             |
            +--------------------------+-------------------------------------------------------------------------------------------------------------------------+

        image_data_options: list of str
            default: ["input", "rcm", "rcb", "residuals"]

            Valid options:

            +------------------+-----------------------------------------+
            |      option      | description                             |
            +------------------+-----------------------------------------+
            | "input"          | input movie                             |
            | "rcm"            | reconstructed movie, A * C              |
            | "rcb"            | reconstructed background, b * f         |
            | "residuals"      | residuals, input - (A * C) - (b * f)    |
            | "rcm-mean"       | rcm mean projection image               |
            | "rcm-min"        | rcm min projection image                |
            | "rcm-max"        | rcm max projection image                |
            | "rcm-std"        | rcm standard deviation projection image |
            | "rcb-mean"       | rcb mean projection image               |
            | "rcb-min"        | rcb min projection image                |
            | "rcb-max"        | rcb max projection image                |
            | "rcb-std"        | rcb standard deviation projection image |
            | "mean"           | mean projection image                   |
            | "max"            | max projection image                    |
            | "std"            | standard deviation projection image     |
            +------------------+-----------------------------------------+

        reset_timepoint_on_change: bool, default False
            reset the timepoint in the ImageWidget when changing items/rows

        temporal_kwargs: dict
            kwargs passed to cnmf.get_temporal(), example: {"add_residuals" : True}.
            Ignored if temporal_data_option contains "dfof"

        input_movie_kwargs: dict
            kwargs passed to caiman.get_input()

        image_widget_kwargs: dict
            kwargs passed to ImageWidget

            Example: `image_widget_kwargs={"cmap": "viridis"}`

        data_grid_kwargs: dict
            kwargs passed to ipywidgets.DataGrid()
        """

        container = CNMFVizContainer(
            dataframe=self._dataframe,
            start_index=start_index,
            temporal_data_option=temporal_data_option,
            image_data_options=image_data_options,
            temporal_kwargs=temporal_kwargs,
            reset_timepoint_on_change=reset_timepoint_on_change,
            input_movie_kwargs=input_movie_kwargs,
            image_widget_kwargs=image_widget_kwargs,
            data_grid_kwargs=data_grid_kwargs,

        )

        return container
