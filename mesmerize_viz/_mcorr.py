from typing import *
from functools import partial
from warnings import warn

import numpy as np
import pandas as pd
from mesmerize_core import MCorrExtensions
from fastplotlib import ImageWidget
from sidecar import Sidecar
from IPython.display import display

from ipydatagrid import DataGrid
from ipywidgets import Textarea, VBox, HBox, Layout, IntSlider, Checkbox

from ._utils import DummyMovie, format_params


projs = [
    "mean",
    "max",
    "std",
]

VALID_DATA_OPTIONS = (
    "input",
    "mcorr",
    "corr",
    *projs,
)


def get_mcorr_data_mapping(series: pd.Series) -> dict:
    """
    Returns dict that maps data option str to a callable that can return the corresponding data array.

    For example, ``{"input": series.get_input_movie}`` maps "input" -> series.get_input_movie

    Parameters
    ----------
    series: pd.Series
        row/item to get mcorr mapping

    Returns
    -------
    dict
        {data label: callable}
    """

    projections = {k: partial(series.caiman.get_projection, k) for k in projs}
    m = {
        "input": series.caiman.get_input_movie,
        "mcorr": series.mcorr.get_output,
        "corr": series.caiman.get_corr_image,
        **projections
    }
    return m


class McorrVizContainer:
    """Widget that contains the DataGrid, params text box and ImageWidget"""
    @property
    def image_widget(self) -> ImageWidget:
        return self._image_widget

    @property
    def current_row(self) -> int:
        return self._current_row

    def __init__(
        self,
            dataframe: pd.DataFrame,
            data_options: List[str] = None,
            start_index: int = None,
            reset_timepoint_on_change: bool = False,
            input_movie_kwargs: dict = None,
            image_widget_kwargs: dict = None,
            data_grid_kwargs: dict = None,
    ):
        """
        Visualize motion correction output.

        Parameters
        ----------
        data_options: list of str, default ["input", "mcorr", "mean", "corr"]
            list of data to plot, valid options are:

            +-------------+-------------------------------------+
            | data option | description                         |
            +=============+=====================================+
            | input       | input movie                         |
            | mcorr       | motion corrected movie              |
            | mean        | mean projection                     |
            | max         | max projection                      |
            | std         | standard deviation projection       |
            | corr        | correlation image, if computed      |
            | pnr         | peak-noise-ratio image, if computed |
            +-------------+-------------------------------------+

        start_index: int
            start index item used to set the initial data in the ImageWidget

        reset_timepoint_on_change: bool, default False
            reset the timepoint in the ImageWidget when changing items/rows

        input_movie_kwargs: dict, optional
            kwargs passed to get_input_movie()

        image_widget_kwargs: dict, optional
            kwargs passed to ImageWidget

        data_grid_kwargs: dict, optional
            kwargs passed to DataGrid()
        """
        if data_options is None:
            # default viz
            data_options = ["input", "mcorr", "mean", "corr"]

        for d in data_options:
            if d not in VALID_DATA_OPTIONS:
                raise KeyError(f"Invalid data option: \"{d}\", valid options are:"
                               f"\n{VALID_DATA_OPTIONS}")

        if data_grid_kwargs is None:
            data_grid_kwargs = dict()

        self._dataframe = dataframe

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

        self._datagrid = DataGrid(
            df_show,  # show only a subset
            selection_mode="cell",
            layout={"height": "250px", "width": "750px"},
            base_row_size=24,
            index_name="index",
            column_widths=default_widths,
            **data_grid_kwargs
        )

        self._params_text_area = Textarea()
        self._params_text_area.layout = Layout(
            height="250px",
            max_height="250px",
            width="360px",
            max_width="500px",
            disabled=True,
        )

        # data options is private since this can't be changed once an image widget has been made
        self._data_options = data_options

        if input_movie_kwargs is None:
            input_movie_kwargs = dict()

        if image_widget_kwargs is None:
            image_widget_kwargs = dict()

        # default kwargs unless user has specified more
        default_iw_kwargs = {
            "window_funcs": {"t": (np.mean, 11)},
            "cmap": "gnuplot2",
            "grid_plot_kwargs": {"size": (700, 600)},
        }

        image_widget_kwargs = {
            **default_iw_kwargs,
            **image_widget_kwargs  # anything in default gets replaced with user-specified entries if present
        }

        self.input_movie_kwargs = input_movie_kwargs
        self.image_widget_kwargs = image_widget_kwargs

        self.reset_timepoint_on_change = reset_timepoint_on_change
        self._image_widget: ImageWidget = None

        # try to guess the start index
        if start_index is None:
            start_index = dataframe[dataframe.algo == "mcorr"].iloc[0].name

        self._current_row: int = start_index

        self._set_params_text_area(index=start_index)

        # set initial selected row
        self._datagrid.select(
            row1=start_index,
            column1=0,
            row2=start_index,
            column2=len(df_show.columns),
            clear_mode="all"
        )

        # callback when row changed
        self._datagrid.observe(self._row_changed, names="selections")

        # set the initial widget state with the start index
        data_arrays = self._get_row_data(index=start_index)

        self._image_widget = ImageWidget(
            data=data_arrays,
            names=self._data_options,
            **self.image_widget_kwargs
        )

        # mean window slider
        self._slider_mean_window = IntSlider(
            min=1,
            step=2,
            max=99,
            value=self._image_widget.window_funcs["t"].window_size,  # set from the image widget
            description="mean wind",
            description_tooltip="set a mean rolling window"
        )
        self._slider_mean_window.observe(self._set_mean_window_size, "value")

        # TODO: mean diff checkbox
        # self._checkbox_mean_diff

        self._sidecar = None
        self._widget = None

    def _set_mean_window_size(self, change):
        self._image_widget.window_funcs = {"t": (np.mean, change["new"])}

        # set same index, forces ImageWidget to run process_indices() so the image shown updates using the new window
        self._image_widget.current_index = self._image_widget.current_index

    def _set_mean_diff(self, change):
        # TODO: will do later
        pass

    def _get_row_data(self, index: int) -> List[np.ndarray]:
        data_arrays: List[np.ndarray] = list()

        data_mapping = get_mcorr_data_mapping(self._dataframe.iloc[index])

        mcorr = data_mapping["mcorr"]()

        shape = mcorr.shape
        ndim = mcorr.ndim
        size = mcorr.size

        # go through all data options user has chosen
        for option in self._data_options:
            func = data_mapping[option]

            if option == "input":
                # kwargs, such as using a specific input movie loader
                array = func(**self.input_movie_kwargs)

            else:
                # just fetch the array
                array = func()

            # for 2D images
            if array.ndim == 2:
                array = DummyMovie(array, shape=shape, ndim=ndim, size=size)

            data_arrays.append(array)

        return data_arrays

    def _get_selected_row(self) -> Union[int, None]:
        r1 = self._datagrid.selections[0]["r1"]
        r2 = self._datagrid.selections[0]["r2"]

        if r1 != r2:
            warn("Only single row selection is currently allowed")
            return

        # get corresponding dataframe index from currently visible dataframe
        # since filtering etc. is possible
        index = self._datagrid.get_visible_data().index[r1]

        return index

    def _row_changed(self, *args):
        index = self._get_selected_row()
        if index is None:
            return

        if self._current_row == index:
            return

        try:
            # fetch the data for this row
            data_arrays = self._get_row_data(index)

        except Exception as e:
            self._params_text_area.value = f"{type(e).__name__}\n" \
                                          f"{str(e)}\n\n" \
                                          f"See jupyter log for details"
            raise e

        else:
            # no exceptions, set ImageWidget
            self._image_widget.set_data(
                new_data=data_arrays,
                reset_vmin_vmax=False,
                reset_indices=self.reset_timepoint_on_change
            )
            self._set_params_text_area(index)
            self._current_row = index

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
        self._params_text_area.value = diffs + format_params(self._dataframe.iloc[index].params, 0)

    def show(self, sidecar: bool = False):
        """
        Show the widget
        """

        self._image_widget.reset_vmin_vmax()

        datagrid_params = HBox([self._datagrid, self._params_text_area])

        if self._image_widget.gridplot.canvas.__class__.__name__ == "JupyterWgpuCanvas":
            widget = VBox([
                    datagrid_params,
                    self._image_widget.show(sidecar=False),
                    self._slider_mean_window
                ])

            # TODO: remove monkeypatch once the autoscale bug is fixed in fastplotlib
            self._image_widget.gridplot[0, 0].auto_scale()

            if not sidecar:
                return widget

            if self._sidecar is None:
                self._sidecar = Sidecar()

            with self._sidecar:
                return display(widget)

        elif self._image_widget.gridplot.canvas.__class__.__name__ == "QWgpuCanvas":
            # shown the image widget in Qt window
            self._image_widget.show()
            # TODO: remove monkeypatch once the autoscale bug is fixed in fastplotlib
            self._image_widget.gridplot[0, 0].auto_scale(maintain_aspect=True)
            # return datagrid to show in jupyter
            return VBox([datagrid_params, self._slider_mean_window])

    def close(self):
        """
        Close the widget, performs cleanup
        """

        self._image_widget.close()
        self._datagrid.close()
        self._params_text_area.close()
        self._slider_mean_window.close()

        if self._sidecar is not None:
            self._sidecar.close()


@pd.api.extensions.register_dataframe_accessor("mcorr")
class MCorrDataFrameVizExtension:
    def __init__(self, df):
        self._dataframe = df

    def viz(
            self,
            data_options: List[str] = None,
            start_index: int = 0,
            reset_timepoint_on_change: bool = False,
            input_movie_kwargs=None,
            image_widget_kwargs=None,
            data_grid_kwargs: dict = None,
    ):
        """
        Visualize motion correction output.

        Parameters
        ----------
        data_options: list of str, default ["input", "mcorr", "mean", "corr"]
            list of data options to plot, valid options are:

            +-------------+-------------------------------------+
            | data option | description                         |
            +=============+=====================================+
            | input       | input movie                         |
            | mcorr       | motion corrected movie              |
            | mean        | mean projection                     |
            | max         | max projection                      |
            | std         | standard deviation projection       |
            | corr        | correlation image, if computed      |
            | pnr         | peak-noise-ratio image, if computed |
            +-------------+-------------------------------------+

        start_index: int, default 0
            start index item used to set the initial data in the ImageWidget

        reset_timepoint_on_change: bool, default False
            reset the timepoint in the ImageWidget when changing items/rows

        input_movie_kwargs: dict, optional
            kwargs passed to get_input_movie()

        image_widget_kwargs: dict, optional
            kwargs passed to ImageWidget

        data_grid_kwargs: dict, optional
            kwargs passed to DataGrid()

        Returns
        -------
        McorrVizContainer
            widget that contains the DataGrid, params text box and ImageWidget
        """

        container = McorrVizContainer(
            dataframe=self._dataframe,
            data_options=data_options,
            start_index=start_index,
            reset_timepoint_on_change=reset_timepoint_on_change,
            input_movie_kwargs=input_movie_kwargs,
            image_widget_kwargs=image_widget_kwargs,
            data_grid_kwargs=data_grid_kwargs
        )

        return container
