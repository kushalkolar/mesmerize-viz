import numpy as np
from fastplotlib import GridPlot, Image, Subplot, Line
from typing import *
import pandas as pd
from .baseviewer import _BaseViewer

# formats dict to yaml-ish-style
is_pos = lambda x: 1 if x > 0 else 0
format_key = lambda d, t: "\n" * is_pos(t) + \
                          "\n".join(
                              [": ".join(["\t" * t + k, format_key(v, t + 1)]) for k, v in d.items()]
                          ) if isinstance(d, dict) else str(d)

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

class _CNMFContainer:
    def __init__(
            self,
            input: Union[Subplot, np.ndarray, Image],
            contours: Union[Subplot, np.ndarray, Image],
            residuals: Union[Subplot, np.ndarray, Image],
            reconstructed: Union[Subplot, np.ndarray, Image],
            #temporal: Union[Subplot, np.ndarray, Image],
            background: Union[Subplot, np.ndarray, Image]
    ):
        #self.temporal = temporal
        self.input = input
        self.contours = contours
        self.residuals = residuals
        self.reconstructed = reconstructed
        self.background = background

class Viewer(_BaseViewer):
    def __init__(
            self,
            dataframe: pd.DataFrame,
    ):
        super(Viewer, self).__init__(
            dataframe,
            grid_plot_shape=(2, 3),
            grid_plot_kwargs={"controllers": "sync"},
            multi_select=False
        )

        if self.get_selected_item()["algo"] == "mcorr":
            self.subplots = _MCorrContainer(
                input=self.grid_plot.subplots[0, 0],
                mcorr=self.grid_plot.subplots[0, 1],
                dsavg=self.grid_plot.subplots[0, 2],
                mean=self.grid_plot.subplots[1, 0],
                corr=self.grid_plot.subplots[1, 1],
                shifts=self.grid_plot.subplots[1, 2]
            )

            self._imaging_data: _MCorrContainer = None
            self._graphics: _MCorrContainer = None
            self.ds_window = 10

        elif self.get_selected_item()["algo"] == "cnmf":
            self.subplots = _CNMFContainer(
                input=self.grid_plot.subplots[0, 0],
                contours=self.grid_plot.subplots[0, 1],
                residuals=self.grid_plot.subplots[0, 2],
                reconstructed=self.grid_plot.subplots[1, 0],
                # temporal=self.grid_plot.subplots[1, 1],
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

        algo = r["algo"]

        for subplot in self.grid_plot:
            subplot.scene.clear()

        if algo == "mcorr":
            self._imaging_data = _MCorrContainer(
                    input=r.caiman.get_input_movie(),
                    mcorr=r.mcorr.get_output(),
                    dsavg=None,
                    mean=r.caiman.get_projection("mean"),
                    corr=r.caiman.get_corr_image(),
                    shifts=None
                )
            input_graphic = Image(
                self._imaging_data.input[0],
                cmap="gnuplot2"
            )

            self._set_frame_slider_minmax(
                (0, self._imaging_data.mcorr.shape[0] - 1)
            )

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
        elif algo == "cnmf":
            self._imaging_data = _CNMFContainer(
                input=r.caiman.get_input_movie(),
                contours=r.cnmf.get_contours()[0],
                residuals=r.cnmf.get_residuals(),
                reconstructed=r.cnmf.get_rcm(),
                # temporal=r.cnmf.get_temporal(),
                background=r.cnmf.get_rcb()
            )
            input_graphic = Image(
                self._imaging_data.input[0],
                cmap="gnuplot2"
            )

            self._set_frame_slider_minmax(
                (0, self._imaging_data.reconstructed.shape[0] - 1)
            )

            contours_graphics: List[Line] = list()

            for coor in self._imaging_data.contours:
                zs = np.ones(coor.shape[0])  # this will place it above the image
                coors_3d = np.dstack([coor[:, 0], coor[:, 1], zs])[0].astype(np.float32)

                # red color, just [R, G, B, A] -> red, green, blue, alpha (transparency)
                colors = np.vstack([[1., 0., 0., 1.]] * coors_3d.shape[0]).astype(np.float32)
                contours_graphics.append(Line(data=coors_3d, colors=colors))

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
                cmap="gray")

            self._graphics = _CNMFContainer(
                input=input_graphic,
                contours=contours_graphics,
                residuals=residuals_graphic,
                reconstructed=reconstructed_graphic,
                # temporal_graphic=temporal_graphic,
                background=background_graphic
            )

            for attr in ["input", "residuals", "reconstructed", "background"]:
                subplot: Subplot = getattr(self.subplots, attr)
                subplot.add_graphic(getattr(self._graphics, attr))

            for line in contours_graphics:
                self.subplots.input.add_graphic(line)

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

        if self.get_selected_item()["algo"] == "mcorr":
            for attr in ["input", "mcorr"]:
                graphic: Image = getattr(self._graphics, attr)
                graphic.update_data(getattr(self._imaging_data, attr)[ix])

            self._imaging_data.dsavg = self._get_dsavg(frame_index=ix)
            self._graphics.dsavg.update_data(
                self._imaging_data.dsavg
            )
        elif self.get_selected_item()["algo"] == "cnmf":
            for attr in ["input", "residuals", "reconstructed", "background"]:
                graphic: Image = getattr(self._graphics, attr)
                graphic.update_data(getattr(self._imaging_data, attr)[ix])

    def _generate_grid_plot(self):
        pass

