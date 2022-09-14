import numpy as np
from ipywidgets import widgets, VBox, HBox, Layout
from fastplotlib import GridPlot, Image, Subplot, Line
from typing import *
import pandas as pd
from uuid import UUID
from collections import OrderedDict
import pims
import time

from mesmerize_viz.baseviewer import _BaseViewer

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


class _CNMFContainer:
    def __init__(
            self,
            input: Union[Subplot, np.ndarray, Image],
            contours: Union[Subplot, np.ndarray, Image],
            reconstructed: Union[Subplot, np.ndarray, Image],
            residuals: Union[Subplot, np.ndarray, Image],
            #temporal: Union[Subplot, np.ndarray, Image],
            background: Union[Subplot, np.ndarray, Image]
    ):
        self.input = input
        self.contours = contours
        self.reconstructed = reconstructed
        self.residuals = residuals
        #self.temporal = temporal
        self.background = background


class CNMFViewer(_BaseViewer):
    def __init__(
            self,
            dataframe: pd.DataFrame,
    ):
        super(CNMFViewer, self).__init__(
            dataframe,
            grid_plot_shape=(2, 3),
            grid_plot_kwargs={"controllers": "sync"},
            multi_select=False
        )

        self.subplots = _CNMFContainer(
            input=self.grid_plot.subplots[0, 0],
            contours=self.grid_plot.subplots[0, 1],
            reconstructed=self.grid_plot.subplots[0, 2],
            residuals=self.grid_plot.subplots[1, 0],
            #temporal=self.grid_plot.subplots[1, 1],
            background=self.grid_plot.subplots[1, 2]
        )

        self._imaging_data: _CNMFContainer = None
        self._graphics: _CNMFContainer = None
        self.ds_window = 10

        self.item_selection_changed()

    def item_selection_changed(self, *args):
        r = self.get_selected_item()
        if r is False:
            return

        for subplot in self.grid_plot:
            subplot.scene.clear()

        self._imaging_data = _CNMFContainer(
            input=r.caiman.get_input_movie(),
            contours=r.cnmf.get_contours()[0],
            residuals=r.cnmf.get_residuals(),
            reconstructed=r.cnmf.get_rcm(),
            #temporal=r.cnmf.get_temporal(),
            background=r.cnmf.get_rcb()
        )

        input_graphic = Image(
            self._imaging_data.input[0],
            cmap="gnuplot2"
        )

        self._set_frame_slider_minmax(
            (0, self._imaging_data.reconstructed.shape[0]-1)
        )

        contours_graphic: List[Line] = list()

        for coor in self._imaging_data.contours:
            zs = np.ones(coor.shape[0])
            coors_3d = np.dstack([coor[:,0], coor[:,1], zs])[0].astype(np.float32)

            colors = np.vstack([[1.,0.,0.,1.]]*coors_3d.shape[0]).astype(np.float32)

            contours_graphic.append(Line(data=coors_3d, colors=colors))

        residuals_graphic = Image(
            self._imaging_data.residuals[0],
            cmap="gnuplot2"
        )

        reconstructed_graphic = Image(
            self._imaging_data.reconstructed[0],
            cmap="gnuplot2"
        )

        background_graphic = Image(
            self._imaging_data.background[0],
            cmap="gray"
        )

        self._graphics = _CNMFContainer(
            input=input_graphic,
            contours=contours_graphic,
            residuals=residuals_graphic,
            reconstructed=reconstructed_graphic,
            # temporal,
            background=background_graphic
        )

        for attr in ["input", "residuals", "reconstructed", "background"]:
            subplot: Subplot = getattr(self.subplots, attr)
            subplot.add_graphic(getattr(self._graphics, attr))

        for line in contours_graphic:
            self.subplots.input.add_graphic(line)

        u = str(r["uuid"])

        self.uuid_text_widget.value = u

        self.params_text_widget.value = format_key(r["params"], 0)
        self.outputs_text_widget.value = format_key(r["outputs"], 0)

        # this does work for some reason if not called from the nb itself ¯\_(ツ)_/¯
        self.reset_grid_plot_scenes()

    def update_frame(self, *args):
        if self.get_selected_index() is None:
            return

        ix = self.frame_slider.value

        for attr in ["input", "residuals", "reconstructed", "background"]:
            graphic: Image = getattr(self._graphics, attr)
            graphic.update_data(getattr(self._imaging_data, attr)[ix])

    def _generate_grid_plot(self):
        pass

