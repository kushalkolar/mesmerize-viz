import numpy as np
from ipywidgets import widgets, VBox, HBox, Layout
from fastplotlib import GridPlot, Image
from typing import *
import pandas as pd
from uuid import UUID
from collections import OrderedDict
import pims


# formats dict to yaml-ish-style
is_pos = lambda x: 1 if x > 0 else 0
format_key = lambda d, t: "\n" * is_pos(t) + \
                          "\n".join(
                              [": ".join(["\t" * t + k, format_key(v, t + 1)]) for k, v in d.items()]
                          ) if isinstance(d, dict) else str(d)


input_readers = [
    "pims",
]


class BatchViewer:
    def __init__(
            self,
            dataframe: pd.DataFrame,
            multi_select: bool = False,
    ):
        # in case the user did something weird with indexing
        self.dataframe: pd.DataFrame = dataframe.reset_index(drop=True)
        self.grid_shape: Tuple[int, int] = None

        self.batch_list_widget_label = widgets.Label(value="batch items:")
        if multi_select:
            self.batch_list_widget: widgets.SelectMultiple = widgets.SelectMultiple(
                options=self.dataframe["name"].to_list(),
                index=0
            )
        else:
            self.batch_list_widget: widgets.Select = widgets.Select(
                options=self.dataframe["name"].to_list(),
                index=0
            )

        self.batch_list_widget.observe(self.item_selection_changed)

        self.uuid_text_widget = widgets.Text(disabled=True, tooltip="UUID of item")
        self.params_text_widget = widgets.Textarea(disabled=True, tooltip="Parameters of item")

        self.outputs_text_widget = widgets.Textarea(description="output info", disabled=True, tooltip="Output info of item")

        self.accordion = widgets.Accordion(children=[self.outputs_text_widget], titles=('output info'))

        self.grid_plot: GridPlot = GridPlot(shape=(1, 3), controllers=np.array([[0, 0, 0]]))
        self.frame_slider = widgets.IntSlider(value=0, min=0, description="frame index:")
        self.grid_plot.renderer.add_event_handler(self._set_frame_slider_width, "resize")

        self.frame_slider.observe(self.update_frame, "value")

        # this should become dynamic later
        self.current_movies: OrderedDict[str, np.ndarray] = OrderedDict([
            ("input", None),
            ("mcorr", None),
        ])

        self.current_graphics: OrderedDict[str, Image] = OrderedDict([
            ("input", None),
            ("mcorr", None),
            ("dsavg", None),
        ])

        self.ds_window = 10

        # Nothing works without this call
        # I don't know why ¯\_(ツ)_/¯
        self.item_selection_changed()

    def _set_frame_slider_width(self, *args):
        w, h = self.grid_plot.renderer.logical_size
        self.frame_slider.layout = Layout(width=f"{w}px")

    def get_selected_index(self) -> int:
        return self.batch_list_widget.index

    def item_selection_changed(self, *args):
        if self.get_selected_index() is None:
            return

        for subplot in self.grid_plot:
            subplot.scene.clear()

        ix = self.get_selected_index()
        r = self.dataframe.iloc[ix]

        self.current_movies["input"] = r.caiman.get_input_movie()
        self.current_movies["mcorr"] = r.mcorr.get_output()

        self.current_graphics["input"] = Image(
            self.current_movies["input"][0],
            cmap="gnuplot2"
        )

        self.frame_slider.max = self.current_movies["input"].shape[0] - 1

        self.current_graphics["mcorr"] = Image(
            self.current_movies["mcorr"][0],
            cmap="gnuplot2"
        )

        dsavg = self._get_dsavg(frame_index=0)

        self.current_graphics["dsavg"] = Image(
            dsavg,
            cmap="gnuplot2"
        )

        for subplot, graphic in zip(self.grid_plot, self.current_graphics.values()):
            subplot.add_graphic(graphic)

        # make graphics, remove any existing from the scene

        u = str(r["uuid"])

        self.uuid_text_widget.value = u

        self.params_text_widget.value = format_key(r["params"], 0)
        self.outputs_text_widget.value = format_key(r["outputs"], 0)

    def _get_dsavg(self, frame_index: int) -> np.ndarray:
        if self.ds_window % 2 == 1:  # make sure it's even
            self.ds_window += 1

        start = max(0, (frame_index - int(self.ds_window / 2)))
        end = min(self.current_movies["mcorr"].shape[0], (frame_index + int(self.ds_window / 2)))

        return np.nanmean(
            self.current_movies["mcorr"][start:end], axis=0
        )

    def update_frame(self, *args):
        if self.get_selected_index() is None:
            return

        ix = self.frame_slider.value

        self.current_graphics["input"].update_data(self.current_movies["input"][ix])
        self.current_graphics["mcorr"].update_data(self.current_movies["mcorr"][ix])
        self.current_graphics["dsavg"].update_data(self._get_dsavg(frame_index=ix))

    def get_layout(self):
        batch_widgets = VBox([self.batch_list_widget_label, self.batch_list_widget])

        uuid_params_output = VBox([self.uuid_text_widget, self.params_text_widget, self.accordion])

        info_widgets = HBox([batch_widgets, uuid_params_output])

        return VBox([
            info_widgets,
            self.frame_slider,
            self.grid_plot.show()
        ])

    def show(self):
        return self.get_layout()

    def _generate_grid_plot(self):
        pass
