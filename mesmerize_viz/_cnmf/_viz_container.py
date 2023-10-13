import itertools
from _warnings import warn
from typing import *

import pandas as pd
from ipydatagrid import DataGrid
from ipywidgets import Textarea, Layout, VBox, HBox
from IPython.display import display
from sidecar import Sidecar

from ._wrapper import VALID_DATA_OPTIONS, get_cnmf_data_mapping, GridPlotWrapper
from .._utils import format_params


class CNMFVizContainer:
    """Widget that contains the DataGrid, params text box fastplotlib GridPlot, etc"""

    def __init__(
        self,
            dataframe: pd.DataFrame,
            data: List[str] = None,
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
        Visualize CNMF output and other data columns such as behavior video (optional)

        Parameters
        ----------
        dataframe: pd.DataFrame

        data: list of str
            data options, such as "input", "temporal", "contours", etc.

        start_index

        reset_timepoint_on_change

        calcium_framerate

        other_data_loaders: Dict[str, callable]
            if loading non-calcium related data arrays, provide dict of callables for opening them.
            Example, if you provide ``data = ["contours", "temporal", "behavior"]``, and the "behavior"
            column contains videos, you could provide `other_data_loads = {"behavior": LazyVideo}

        data_kwargs: dict
            kwargs passed to corresponding extension function to load data.
            example: ``{"temporal": {"component_ixs": "good"}}``

        gridplot_kwargs: List[dict]
            kwargs passed to GridPlot

        data_grid_kwargs
        """

        if data is None:
            data = [["temporal"], ["input", "rcm", "rcb", "residuals"]]

        if other_data_loaders is None:
            other_data_loaders = dict()

        # simple list of str, single gridplot
        if all(isinstance(option, str) for option in data):
            data = [data]

        if not all(isinstance(option, list) for option in data):
            raise TypeError(
                "Must pass list of str or nested list of str"
            )

        # make sure data options are valid
        for d in list(itertools.chain(*data)):
            if (d not in VALID_DATA_OPTIONS) and (d not in dataframe.columns):
                raise ValueError(
                    f"`data` options are: {VALID_DATA_OPTIONS} or a DataFrame column name: {dataframe.columns}\n"
                    f"You have passed: {d}"
                )

            if d in dataframe.columns:
                if d not in other_data_loaders.keys():
                    raise ValueError(
                        f"You have provided the non-CNMF related data option: {d}.\n"
                        f"If you provide a non-cnmf related data option you must also provide a "
                        f"data loader callable for it to `other_data_loaders`"
                    )

        self._other_data_loaders = other_data_loaders

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
            max_width="500px"
        )

        # data options is private since this can't be changed once an image widget has been made
        self._data = data

        if data_kwargs is None:
            data_kwargs = dict()

        self.data_kwargs = data_kwargs

        self.current_row: int = start_index

        self._make_gridplot(
            start_index=start_index,
            reset_timepoint_on_change=reset_timepoint_on_change,
            data_graphic_kwargs=data_graphic_kwargs,
            gridplot_kwargs=gridplot_kwargs,
            cmap=cmap,
            component_colors=component_colors,
        )

        self._set_params_text_area(index=start_index)

        # set initial selected row
        self.datagrid.select(
            row1=start_index,
            column1=0,
            row2=start_index,
            column2=len(df_show.columns),
            clear_mode="all"
        )

        # callback when row changed
        self.datagrid.observe(self._row_changed, names="selections")

        self.sidecar = None

    def _make_gridplot(
            self,
            start_index: int,
            reset_timepoint_on_change: bool,
            data_graphic_kwargs: dict,
            gridplot_kwargs: dict,
            cmap: str,
            component_colors: str,
    ):

        data_mapping = get_cnmf_data_mapping(
            self._dataframe.iloc[start_index],
            self.data_kwargs
        )

        self._gridplot_wrapper = GridPlotWrapper(
            data=self._data,
            data_mapping=data_mapping,
            reset_timepoint_on_change=reset_timepoint_on_change,
            data_graphic_kwargs=data_graphic_kwargs,
            gridplot_kwargs=gridplot_kwargs,
            cmap=cmap,
            component_colors=component_colors

        )

        self.gridplots = self._gridplot_wrapper.gridplots

    def show(self, sidecar: bool = True):
        """Show the widget"""

        # create gridplots and start render loop
        gridplots_widget = [gp.show(sidecar=False) for gp in self.gridplots]

        if "Jupyter" in self.gridplots[0].canvas.__class__.__name__:
            vbox_elements = gridplots_widget
        else:
            vbox_elements = list()

        if self.sidecar is None:
            self.sidecar = Sidecar()

        widget = VBox(
            [
                HBox([self.datagrid, self.params_text_area]), HBox([self._gridplot_wrapper.component_slider, self._gridplot_wrapper.component_int_box]),
                VBox(vbox_elements)
            ]
        )

        with self.sidecar:
            return display(widget)

    def close(self):
        """Close the widget"""
        for gp in self.gridplots:
            gp.close()

    def _get_selection_row(self) -> Union[int, None]:
        r1 = self.datagrid.selections[0]["r1"]
        r2 = self.datagrid.selections[0]["r2"]

        if r1 != r2:
            warn("Only single row selection is currently allowed")
            return

        # get corresponding dataframe index from currently visible dataframe
        # since filtering etc. is possible
        index = self.datagrid.get_visible_data().index[r1]

        return index

    def _row_changed(self, *args):
        index = self._get_selection_row()
        if index is None:
            return

        if self.current_row == index:
            return

        try:
            data_mapping = get_cnmf_data_mapping(
                self._dataframe.iloc[index],
                self.data_kwargs
            )
            self._gridplot_wrapper.change_data(data_mapping)
        except Exception as e:
            self.params_text_area.value = f"{type(e).__name__}\n" \
                                          f"{str(e)}\n\n" \
                                          f"See jupyter log for details"
            raise e

        self._set_params_text_area(index)

        self.current_row = index

    def _set_params_text_area(self, index):
        row = self._dataframe.iloc[index]
        # try and get the param diffs
        try:
            param_diffs = self._dataframe.caiman.get_params_diffs(
                algo=row["algo"],
                item_name=row["item_name"]
            ).iloc[index]

            diffs_dict = {"diffs": param_diffs}
            diffs = f"{format_params(diffs_dict, 0)}\n\n"
        except:
            diffs = ""

        # diffs and full params
        self.params_text_area.value = diffs + format_params(self._dataframe.iloc[index].params, 0)
