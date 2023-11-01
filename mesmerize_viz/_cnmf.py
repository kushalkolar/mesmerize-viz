from collections import OrderedDict
from functools import partial
from typing import *
from warnings import warn


import numpy as np
import pandas as pd
from ipydatagrid import DataGrid
from ipywidgets import Button, Tab, Text, Textarea, Layout, HBox, VBox, Checkbox, FloatSlider, BoundedFloatText, IntSlider, BoundedIntText, RadioButtons, Dropdown, jslink
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
import fastplotlib as fpl
from caiman.source_extraction.cnmf import CNMF


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
            slider = FloatSlider(value=0, min=0, max=1, step=0.01, description=metric)
            spinbox = BoundedFloatText(
                value=0, min=0, max=1, step=0.01, description_tooltip=metric, layout=Layout(width="70px")
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

        self.widget = VBox([*param_entries, self.use_cnn_checkbox])

        self._handlers = list()

        # limits must be set first before it's usable
        self._block_handlers = True

        self.button_save_eval = Button(description="Save Eval")

    def set_limits(self, cnmf_obj: CNMF):
        self._block_handlers = True
        for metric in self._float_metrics:
            metric_array = getattr(cnmf_obj.estimates, self._metric_array_mapping[metric])
            for kind in ["slider", "spinbox"]:
                # allow 100 steps
                self._widgets[metric][kind].step = np.ptp(metric_array) / 100
                self._widgets[metric][kind].min = metric_array.min()
                self._widgets[metric][kind].max = metric_array.max()
                self._widgets[metric][kind].value = cnmf_obj.params.get_group("quality")[metric]

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
            "grid_plot_kwargs": {"size": (720, 602)},
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

        # ipywidgets for selecting components
        self.component_slider = IntSlider(min=0, max=1, value=0, step=1, description="component index:")
        self.component_int_box = BoundedIntText(min=0, max=1, value=0, step=1, layout=Layout(width="100px"))
        for trait in ["value", "max"]:
            jslink((self.component_slider, trait), (self.component_int_box, trait))

        self.component_int_box.observe(
            lambda change: self.set_component_index(change["new"]), "value"
        )

        self._component_metrics_text = Text(
            value="",
            placeholder="component metrics",
            description='metrics:',
            disabled=True,
            layout=Layout(width="350px")
        )

        # checkbox to zoom into components when selected
        self.checkbox_zoom_components = Checkbox(
            value=True,
            description="auto-zoom component",
            description_tooltip="If checked, zoom into selected component"
        )
        # zoom factor
        self.zoom_components_scale = FloatSlider(
            min=0.25,
            max=3,
            value=1,
            step=0.25,
            description="zoom scale",
            description_tooltip="zoom scale as a factor of component width/height"
        )
        # organize these widgets to be shown at the top
        self._top_widget = VBox([
            HBox([self.datagrid, self.params_text_area]),
            HBox([self.component_slider, self.component_int_box, self._component_metrics_text]),
            HBox([self.checkbox_zoom_components, self.zoom_components_scale])
        ])

        self._dropdown_contour_colors = Dropdown(
            options=["random", "accepted", "rejected", "snr_comps", "snr_comps_log", "r_values", "cnn_preds"],
            value="random",
            description='contour colors:',
        )

        self._dropdown_contour_colors.observe(self._ipywidget_set_component_colors, "value")

        self._radio_visible_components = RadioButtons(
            options=["all", "accepted", "rejected"],
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

        self._tab_contours_eval = Tab()
        self._tab_contours_eval.children = [self._box_contour_controls, self._eval_controller.widget]

        # plots
        self._plot_temporal = fpl.Plot(size=(500, 120))
        self._plot_temporal.camera.maintain_aspect = False
        self._plot_heatmap = fpl.Plot(size=(500, 450))
        self._plot_heatmap.camera.maintain_aspect = False

        self._image_widget: fpl.ImageWidget = None

        self._synchronizer = fpl.Synchronizer(key_bind=None)

        self._contour_graphics: List[fpl.LineCollection] = list()

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
        self._contour_graphics.clear()

        self._synchronizer.clear()

        self._plot_temporal.clear()
        self._plot_heatmap.clear()

        if self._image_widget is None:
            self._image_widget = fpl.ImageWidget(
                data=data_arrays["images"],
                names=self._image_data_options,
                **self.image_widget_kwargs
            )

            # need to start it here so that we can access the toolbar to link events with the slider
            self._image_widget.show()

        else:
            # image widget doesn't need clear, we can just use set_data
            self._image_widget.set_data(data_arrays["images"])
            for subplot in self._image_widget.gridplot:
                if "contours" in subplot:
                    # delete the contour graphics
                    subplot.delete_graphic(subplot["contours"])

        self._temporal_data = data_arrays["temporal"]

        # make temporal graphics
        self._plot_temporal.add_line(self._temporal_data[0], name="line")
        # autoscale the single temporal line plot when the data changes
        self._plot_temporal["line"].data.add_event_handler(self._plot_temporal.auto_scale)
        self._plot_heatmap.add_heatmap(self._temporal_data, name="heatmap")

        self._component_linear_selector: fpl.LinearSelector = self._plot_heatmap["heatmap"].add_linear_selector(axis="y", thickness=5)
        self._component_linear_selector.selection.add_event_handler(self.set_component_index)

        # linear selectors and events
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

        contours = data_arrays["contours"][0]

        n_components = len(contours)
        self._random_colors = np.random.rand(n_components, 4).astype(np.float32)
        self._random_colors[:, -1] = 1

        for subplot in self._image_widget.gridplot:
            contour_graphic = subplot.add_line_collection(
                contours,
                colors=self._random_colors,
                name="contours"
            )
            self._contour_graphics.append(contour_graphic)

            image_graphic = subplot["image_widget_managed"]

            image_graphic.link(
                "click",
                target=contour_graphic,
                feature="colors",
                new_data="w",
                callback=self._euclidean
            )

            contour_graphic.link("colors", target=contour_graphic, feature="thickness", new_data=2)

        self.component_int_box.value = 0
        self.component_slider.value = 0
        self.component_int_box.max = n_components - 1
        self.component_slider.max = n_components - 1

        # current state of CNMF object
        # this can be different from the one in the dataframe if the user uses eval
        self._cnmf_obj: CNMF = data_arrays["cnmf_obj"]

        self._eval_controller.set_limits(self._cnmf_obj)

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

    def set_component_index(self, index):
        if hasattr(index, "pick_info"):
            # came from heatmap component selector
            if index.pick_info["pygfx_event"] is None:
                # this means that the selector was not triggered by the user but that it moved due to another event
                # so then we don't set_component_index because then infinite recursion
                return
            index = index.pick_info["selected_index"]

        for g in self._contour_graphics:
            g.set_feature(feature="colors", new_data="w", indices=index)

        self._plot_temporal["line"].data = self._temporal_data[index]

        if self._component_linear_selector._move_info is None:
            # TODO: Very hacky for now, ignores if the slider is currently being moved by the user
            # prevents weird slider movement
            self._component_linear_selector.selection = index

        self._zoom_into_component(index)

        self.component_int_box.unobserve_all()
        self.component_int_box.value = index
        self.component_int_box.observe(
            lambda change: self.set_component_index(change["new"]), "value"
        )

        metrics = (f"snr: {self._cnmf_obj.estimates.SNR_comp[index]:.02f}, "
                   f"r_values: {self._cnmf_obj.estimates.r_values[index]:.02f}, "
                   f"cnn: {self._cnmf_obj.estimates.cnn_preds[index]:.02f} ")

        self._component_metrics_text.value = metrics

    def _zoom_into_component(self, index: int):
        if not self.checkbox_zoom_components.value:
            return

        for subplot in self._image_widget.gridplot:
            subplot.camera.show_object(
                subplot["contours"].graphics[index].world_object,
                scale=self.zoom_components_scale.value
            )

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
        self._linear_selector_heatmap.selection = ix

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
        n_contours = len(self._image_widget.gridplot[0, 0]["contours"])

        # use the random colors
        if metric == "random":
            for subplot in self._image_widget.gridplot:
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

        for subplot in self._image_widget.gridplot:
            # first initialize using a quantitative cmap
            # this ensures that setting cmap_values will work
            subplot["contours"].cmap = "gray"

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

    def _save_eval(self, obj):
        index = self._get_selected_row()

        eval_params = self._eval_controller.get_data()
        # this overwrites hdf5 file
        self._dataframe.iloc[index].cnmf.run_eval(eval_params)
        print("Overwrote CNMF object with new eval")

    def show(self, sidecar: bool = False):
        """
        Show the widget

        Parameters
        ----------
        sidecar

        Returns
        -------

        """

        temporals = VBox([self._plot_temporal.show(), self._plot_heatmap.show()])

        iw_contour_controls = VBox([self._image_widget.widget, self._tab_contours_eval])

        plots = HBox([temporals, iw_contour_controls])

        return VBox([self._top_widget, plots])
