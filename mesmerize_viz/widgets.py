import numpy as np
from ipywidgets import widgets, VBox, HBox, Layout
from fastplotlib import GridPlot, Image, Subplot
from typing import *
import pandas as pd
from uuid import UUID
from collections import OrderedDict
import pims
import time


# formats dict to yaml-ish-style
is_pos = lambda x: 1 if x > 0 else 0
format_key = lambda d, t: "\n" * is_pos(t) + \
                          "\n".join(
                              [": ".join(["\t" * t + k, format_key(v, t + 1)]) for k, v in d.items()]
                          ) if isinstance(d, dict) else str(d)


input_readers = [
    "pims",
]


blue_circle = chr(int("0x1f535",base=16))
green_circle = chr(int("0x1f7e2",base=16))
red_circle = chr(int("0x1f534",base=16))


class _MCorrContainer:
    def __init__(
            self,
            input: Union[Subplot, np.ndarray, Image],
            mcorr: Union[Subplot, np.ndarray, Image],
            dsavg: Union[Subplot, np.ndarray, Image],
            mean: Union[Subplot, np.ndarray, Image],
            corr: Union[Subplot, np.ndarray, Image],
            shifts: Union[Subplot, np.ndarray, Image]
    ):
        self.input = input
        self.mcorr = mcorr
        self.dsavg = dsavg
        self.mean = mean
        self.corr = corr
        self.shifts = shifts


class _BaseViewer:
    def __init__(
            self,
            dataframe: pd.DataFrame,
            grid_plot_shape: Tuple[int, int],
            grid_plot_kwargs: Optional[dict] = None,
            multi_select: bool = False,
    ):
        self.dataframe: pd.DataFrame = dataframe
        self.grid_shape: Tuple[int, int] = None

        # in case the user did something weird with indexing
        self.dataframe: pd.DataFrame = dataframe.reset_index(drop=True)

        self._init_batch_list_widget(multi_select)
        self.batch_list_widget.layout = Layout(height="200px")

        self.batch_list_widget.observe(self.item_selection_changed)

        self.uuid_text_widget = widgets.Text(disabled=True, tooltip="UUID of item")
        self.params_text_widget = widgets.Textarea(disabled=True, tooltip="Parameters of item")

        self.outputs_text_widget = widgets.Textarea(disabled=True, tooltip="Output info of item")

        self.grid_plot: GridPlot = GridPlot(shape=(2, 3), **grid_plot_kwargs)

        self.frame_slider = widgets.IntSlider(value=0, min=0, description="frame index:")
        self.grid_plot.renderer.add_event_handler(self._set_frame_slider_width, "resize")

        self.frame_slider.observe(self.update_frame, "value")

        self.button_reset_view = widgets.Button(description="Reset View")
        self.button_reset_view.on_click(self.reset_grid_plot_scenes)

    def _init_batch_list_widget(self, multi_select: bool):
        options = list()
        for ix, r in self.dataframe.iterrows():
            if r["outputs"] is None:
                indicator = blue_circle
            elif r["outputs"]["success"] is True:
                indicator = green_circle
            elif r["outputs"]["success"] is False:
                indicator = red_circle
            name = r["name"]
            options.append(f"{ix}: {indicator} {name}")

        if multi_select:
            self.batch_list_widget: widgets.SelectMultiple = widgets.SelectMultiple(
                options=options,
                index=0
            )
        else:
            self.batch_list_widget: widgets.Select = widgets.Select(
                options=options,
                index=0
            )

    def _set_frame_slider_width(self, *args):
        w, h = self.grid_plot.renderer.logical_size
        self.frame_slider.layout = Layout(width=f"{w}px")

    def get_selected_index(self) -> int:
        return self.batch_list_widget.index

    def get_selected_item(self) -> pd.Series:
        if self.get_selected_index() is None:
            return False

        ix = self.get_selected_index()
        r = self.dataframe.iloc[ix]

        if r["outputs"]["success"] is False:
            self.outputs_text_widget.value = r["outputs"]["traceback"]
            return False

        return r

    def item_selection_changed(self):
        pass

    def update_frame(self, *args):
        pass

    def get_layout(self):
        uuid_params_output = VBox([self.uuid_text_widget, self.params_text_widget, self.outputs_text_widget])

        info_widgets = HBox([self.batch_list_widget, uuid_params_output])

        return VBox([
            info_widgets,
            self.button_reset_view,
            self.frame_slider,
            self.grid_plot.show(),
        ])

    def show(self):
        return self.get_layout()

    def reset_grid_plot_scenes(self, *args):
        self.grid_plot.subplots[0, 0].center_scene()


class MCorrViewer(_BaseViewer):
    def __init__(
            self,
            dataframe: pd.DataFrame,
    ):
        super(MCorrViewer, self).__init__(
            dataframe,
            grid_plot_shape=(2, 3),
            grid_plot_kwargs={"controllers": "sync"},
            multi_select=False
        )

        self.subplots = _MCorrContainer(
            input=self.grid_plot.subplots[0, 0],
            mcorr=self.grid_plot.subplots[0, 1],
            dsavg=self.grid_plot.subplots[0, 2],
            mean=self.grid_plot.subplots[1, 0],
            corr=self.grid_plot.subplots[1, 1],
            shifts=self.grid_plot.subplots[1, 2]
        )

        self.frame_slider = widgets.IntSlider(value=0, min=0, description="frame index:")
        self.grid_plot.renderer.add_event_handler(self._set_frame_slider_width, "resize")

        self.frame_slider.observe(self.update_frame, "value")

        self.button_reset_view = widgets.Button(description="Reset View")
        self.button_reset_view.on_click(self.reset_grid_plot_scenes)

        # this should become dynamic later
        self._imaging_data: _MCorrContainer = None
        self._graphics: _MCorrContainer = None
        self.ds_window = 10

        # Nothing works without this call
        # I don't know why ¯\_(ツ)_/¯
        self.item_selection_changed()

    def item_selection_changed(self, *args):
        r = self.get_selected_item()
        if r is False:
            return

        for subplot in self.grid_plot:
            subplot.scene.clear()

        self._imaging_data = _MCorrContainer(
                input=r.caiman.get_input_movie("append-tiff"),
                mcorr=r.mcorr.get_output(),
                dsavg=None,
                mean=r.caiman.get_projection("mean"),
                corr=r.caiman.get_correlation_image(),
                shifts=None
            )

        input_graphic = Image(
            self._imaging_data.input[0],
            cmap="gnuplot2"
        )

        self.frame_slider.max = self._imaging_data.mcorr.shape[0] - 1

        mcorr_graphic = Image(
            self._imaging_data.mcorr[0],
            cmap="gnuplot2"
        )

        dsavg = self._get_dsavg(frame_index=0)
        self._imaging_data.dsavg = dsavg

        dsavg_graphic = Image(
            dsavg,
            cmap="gnuplot2"
        )

        mean_graphic = Image(
            self._imaging_data.mean,
            cmap="gray"
        )

        corr_graphic = Image(
            self._imaging_data.corr,
            cmap="gray"
        )

        self._graphics = _MCorrContainer(
            input=input_graphic,
            mcorr=mcorr_graphic,
            dsavg=dsavg_graphic,
            mean=mean_graphic,
            corr=corr_graphic,
            shifts=None
        )

        for attr in ["input", "mcorr", "dsavg", "mean", "corr"]:
            subplot: Subplot = getattr(self.subplots, attr)
            subplot.add_graphic(getattr(self._graphics, attr))

        u = str(r["uuid"])

        self.uuid_text_widget.value = u

        self.params_text_widget.value = format_key(r["params"], 0)
        self.outputs_text_widget.value = format_key(r["outputs"], 0)

        # this does work for some reason if not called from the nb itself ¯\_(ツ)_/¯
        self.reset_grid_plot_scenes()

    def _get_dsavg(self, frame_index: int) -> np.ndarray:
        if self.ds_window % 2 == 1:  # make sure it's even
            self.ds_window += 1

        start = max(0, (frame_index - int(self.ds_window / 2)))
        end = min(self._imaging_data.mcorr.shape[0], (frame_index + int(self.ds_window / 2)))

        return np.nanmean(
            self._imaging_data.mcorr[start:end], axis=0
        )

    def update_frame(self, *args):
        if self.get_selected_index() is None:
            return

        ix = self.frame_slider.value

        for attr in ["input", "mcorr"]:
            graphic: Image = getattr(self._graphics, attr)
            graphic.update_data(getattr(self._imaging_data, attr)[ix])

        self._imaging_data.dsavg = self._get_dsavg(frame_index=ix)
        self._graphics.dsavg.update_data(
            self._imaging_data.dsavg
        )

    def _generate_grid_plot(self):
        pass
