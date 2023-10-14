from typing import *

import pandas as pd

from ._viz_container import CNMFVizContainer


@pd.api.extensions.register_dataframe_accessor("cnmf")
class CNMFDataFrameVizExtension:
    def __init__(self, df):
        self._dataframe = df

    def viz(
            self,
            data_options: List[str] = None,
            start_index: int = 0,
            reset_timepoint_on_change: bool = False,
            data_graphic_kwargs: dict = None,
            gridplot_kwargs: dict = None,
            cmap: str = "gnuplot2",
            component_colors: str = "random",
            calcium_framerate: float = None,
            other_data_loaders: Dict[str, callable] = None,
            data_kwargs: dict = None,
            data_grid_kwargs: dict = None,
    ):
        """
        Visualize motion correction output.

        Parameters
        ----------
        data_options: list of str or list of list of str
            default [["temporal"], ["heatmap-norm"], ["input", "rcm", "rcb", "residuals"]]

            **Note:** You may add suffixes to temporal and heatmap options for "dfof", "zscore", "norm",
            examples: "temporal-dfof", "heatmap-norm", "heatmap-zscore", "heatmap-dfof", etc.

            list of data to plot, valid options are:

            +------------------+-----------------------------------------+
            | data option      | description                             |
            +------------------+-----------------------------------------+
            | "input"          | input movie                             |
            | "rcm"            | reconstructed movie, A * C              |
            | "rcb"            | reconstructed background, b * f         |
            | "residuals"      | residuals, input - (A * C) - (b * f)    |
            | "corr"           | correlation image, if computed          |
            | "pnr"            | peak-noise-ratio image, if computed     |
            | "temporal"       | temporal components overlaid            |
            | "temporal-stack" | temporal components stack               |
            | "heatmap"        | temporal components heatmap             |
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


        start_index: int, default 0
            start index item used to set the initial data in the ImageWidget

        reset_timepoint_on_change: bool, default False
            reset the timepoint in the ImageWidget when changing items/rows

        data_grid_kwargs: dict, optional
            kwargs passed to DataGrid()

        Returns
        -------
        McorrVizContainer
            widget that contains the DataGrid, params text box and ImageWidget
        """
        container = CNMFVizContainer(
            dataframe=self._dataframe,
            data=data_options,
            start_index=start_index,
            reset_timepoint_on_change=reset_timepoint_on_change,
            data_graphic_kwargs=data_graphic_kwargs,
            gridplot_kwargs=gridplot_kwargs,
            cmap=cmap,
            component_colors=component_colors,
            calcium_framerate=calcium_framerate,
            other_data_loaders=other_data_loaders,
            data_kwargs=data_kwargs,
            data_grid_kwargs=data_grid_kwargs,
        )

        return container
