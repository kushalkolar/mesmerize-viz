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
    def __int__(
            self,
            dataframe: pd.DataFrame,
            multi_select: bool = False,
    ):
        # in case the user did something weird with indexing
        self.dataframe: pd.DataFrame = dataframe.reset_index(drop=True)
        self.grid_shape: Tuple[int, int] = None

        self.batch_list_widget_label = widgets.Label(description="batch items:")
        if multi_select:
            self.batch_list_widget: widgets.SelectMultiple = widgets.SelectMultiple()
        else:
            self.batch_list_widget: widgets.Select = widgets.Select()

        self.batch_list_widget.observe(self.item_selection_changed)

        self.uuid_text_widget = widgets.Text(disabled=True, tooltip="UUID of item")
        self.params_text_widget = widgets.Textarea(disabled=True, tooltip="Parameters of item")

        self.outputs_text_widget = widgets.Text(disabled=True, tooltip="Output info of item")

        self.accordion = widgets.Accordion(children=[self.outputs_text_widget], titles=('output info'))

        self.grid_plot: GridPlot = GridPlot(shape=(1, 3), controllers=np.array([[0, 0, 0]]))
        self.frame_slider = widgets.IntSlider(value=0, min=0, description="frame index:")

        self.frame_slider.observe(self.update_grid_plot)

        # this should become dynamic later
        self.current_movies = OrderedDict([
            ("input", None),
            ("mcorr", None),
            ("dsavg", None),
        ])

        self.current_graphics = OrderedDict([
            ("input", None),
            ("mcorr", None),
            ("dsavg", None),
        ])

    def get_selected_index(self) -> int:
        return self.batch_list_widget.index

    def item_selection_changed(self):
        for subplot in self.grid_plot:
            subplot.scene.clear()

        ix = self.get_selected_index()
        r = self.dataframe.iloc[ix]

        self.current_movies["input"] = r.caiman.get_input_movie()
        mcorr_memmap = r.mcorr.get_output()
        self.current_movies["mcorr"] = mcorr_memmap

        self.current_graphics["input"] = Image(
            self.current_movies["input"][0],
            cmap="gnuplot2"
        )

        self.current_graphics["mcorr"] = Image(
            self.current_movies["mcorr"][0],
            cmap="gnuplot2"
        )

        dsavg = np.nanmean(mcorr_memmap[0:0+5], axis=0)

        self.current_graphics["dsavg"] = Image(
            dsavg,
            cmap="gnuplot2"
        )

        # make graphics, remove any existing from the scene

        u = str(r["uuid"])

        self.uuid_text_widget.value = u

        self.params_text_widget.value = format_key(r["params"], 0)
        self.outputs_text_widget.value = format_key(r["outputs"], 0)

    def update_grid_plot(self):
        frame_index = self.frame_slider.value

        self.current_graphics

    def get_layout(self):
        batch_list_uuid_text_params_text = VBox(
            self.batch_list_widget_label,
            self.batch_list_widget,
            self.uuid_text_widget,
            self.params_text_widget,
            self.accordion,
        )

        return VBox([
            HBox([batch_list_uuid_text_params_text, self.grid_plot.show()]),
            self.frame_slider
        ])

    def show(self):
        return self.get_layout()

    def _generate_grid_plot(self):
        pass
