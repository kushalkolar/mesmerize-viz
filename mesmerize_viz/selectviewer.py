from typing import Tuple

import numpy as np
from ipywidgets import GridspecLayout, Layout, Dropdown, VBox, HBox
from functools import partial
from itertools import product


class SelectViewer:
    def __init__(
            self,
            _BaseViewer,
            grid_shape: Tuple[int, int] = None,
            multi_select: bool = False,
            algo: str = "mcorr",
    ):
        self.grid_shape = grid_shape
        self.table = GridspecLayout(self.grid_shape[0], self.grid_shape[1])
        self.algo = algo
        self.base = _BaseViewer
        self.grid_cells = np.empty(shape=self.grid_shape, dtype=object)

        self.set_grid()

        for i, j in product(range(self.grid_shape[0]), range(grid_shape[1])):
            widget = self.grid_cells[i, j]
            widget.observe(partial(self.item_selection_change, (i, j)), 'value')

    def set_grid(self):
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                self.table[i, j] = Dropdown(
                    options=['raw movie', 'mcorr movie', 'downsampled avg movie', 'correlation image', 'shifts'])
                self.grid_cells[i, j] = (self.table[i, j])

    def item_selection_change(self, *args):
        grid_position = args[0]
        change = args[1]["new"]
        self.base.update_graphic(grid_position, change)

    def get_layout(self):
        return self.table

