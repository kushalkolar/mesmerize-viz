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

        self.grid_plot: GridPlot = GridPlot(shape=grid_plot_shape, **grid_plot_kwargs)

        self.frame_slider = widgets.IntSlider(value=0, min=0, description="frame index:")
        self.grid_plot.renderer.add_event_handler(self._set_frame_slider_width, "resize")

        self.frame_slider.observe(self.update_frame, "value")

        self.play_button = widgets.Play(
            value=self.frame_slider.value,
            min=self.frame_slider.min,
            max=self.frame_slider.max,
            step=1,
            interval=100,
        )

        widgets.jslink(
            (self.play_button, 'value'),
            (self.frame_slider, 'value')
        )

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

    def _set_frame_slider_minmax(self, minmax: Tuple[int, int]):
        self.frame_slider.min = minmax[0]
        self.frame_slider.max = minmax[1]
        self.play_button.min = minmax[0]
        self.play_button.max = minmax[1]

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

    def update_graphic(self, position: Tuple[int, int], change: str):
        pass

    def get_layout(self):
        uuid_params_output = VBox([self.uuid_text_widget, self.params_text_widget, self.outputs_text_widget])

        info_widgets = HBox([self.batch_list_widget, uuid_params_output])

        return VBox([
            info_widgets,
            HBox([self.button_reset_view, self.play_button]),
            self.frame_slider,
            self.grid_plot.show(),
        ])

    def show(self):
        return self.get_layout()

    def reset_grid_plot_scenes(self, *args):
        self.grid_plot.subplots[0, 0].center_scene()