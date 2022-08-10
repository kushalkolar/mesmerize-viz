import numpy as np
from ipywidgets import widgets, VBox, HBox, Layout
from fastplotlib import GridPlot, Image, Subplot
from typing import *
import pandas as pd
from uuid import UUID
from collections import OrderedDict
import pims
import time
from .baseviewer import _BaseViewer

# formats dict to yaml-ish-style
is_pos = lambda x: 1 if x > 0 else 0
format_key = lambda d, t: "\n" * is_pos(t) + \
                          "\n".join(
                              [": ".join(["\t" * t + k, format_key(v, t + 1)]) for k, v in d.items()]
                          ) if isinstance(d, dict) else str(d)

class _CNMFContainer:
    def __init__(
            self,
            input: Union[Subplot, np.ndarray, Image],
            contours: Union[Subplot, np.ndarray, Image],
            residuals: Union[Subplot, np.ndarray, Image],
            reconstructed: Union[Subplot, np.ndarray, Image],
            temporal: Union[Subplot, np.ndarray, Image],
            background: Union[Subplot, np.ndarray, Image]
    ):
        self.temporal = temporal
        self.input = input
        self.contours = contours
        self.residuals = residuals
        self.reconstructed = reconstructed
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
            residuals=self.grid_plot.subplots[0, 2],
            reconstructed=self.grid_plot.subplots[1, 0],
            temporal=self.grid_plot.subplots[1, 1],
            background=self.grid_plot.subplots[1, 2]
        )

        # this should become dynamic later
        self._imaging_data: _CNMFContainer = None
        self._graphics: _CNMFContainer = None
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

        self._imaging_data = _CNMFContainer(
                input=r.caiman.get_input_movie(),
                contours=r.cnmf.get_contours(),
                residuals=r.cnmf.get_residuals(),
                reconstructed=r.cnmf.get_rcm(),
                temporal=r.cnmf.get_temporal(),
                background=r.cnmf.get_rcb()
            )

        input_graphic = Image(
            self._imaging_data.input[0],
            cmap="gnuplot2"
        )

        self._set_frame_slider_minmax(
            (0, self._imaging_data.reconstructed.shape[0] - 1)
        )

        contours_graphic = Image(
            self._imaging_data.contours[0],
            cmap="gnuplot2"
        )

        residuals_graphic = Image(
            self._imaging_data.residuals[0],
            cmap="gnuplot2"
        )

        reconstructed_graphic = Image(
            self._imaging_data.reconstructed[0],
            cmap="gnuplot2"
        )

        temporal_graphic = Image(
            self._imaging_data.residuals[0],
            cmap="gnuplot2"
        )

        background_graphic = Image(
            self._imaging_data.background[0],
            cmap="gray")

        self._graphics = _CNMFContainer(
            input=input_graphic,
            contours=contours_graphic,
            residuals=residuals_graphic,
            reconstructed=reconstructed_graphic,
            temporal_graphic=temporal_graphic,
            background=background_graphic
        )

        for attr in ["input", "contours", "residuals", "reconstructed", "temporal", "background"]:
            subplot: Subplot = getattr(self.subplots, attr)
            subplot.add_graphic(getattr(self._graphics, attr))

        u = str(r["uuid"])

        self.uuid_text_widget.value = u

        self.params_text_widget.value = format_key(r["params"], 0)
        self.outputs_text_widget.value = format_key(r["outputs"], 0)

        # this does work for some reason if not called from the nb itself ¯\_(ツ)_/¯
        self.reset_grid_plot_scenes()


