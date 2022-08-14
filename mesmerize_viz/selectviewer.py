import numpy as np
from ipywidgets import widgets, VBox, HBox, Layout, Dropdown, GridspecLayout
from fastplotlib import GridPlot, Image, Subplot
from typing import *
import pandas as pd
from uuid import UUID
from collections import OrderedDict
import pims
import time


class SelectViewer:
    def __init__(
            self,
            grid_plot_shape: Tuple[int, int] = None,
            multi_select: bool = False,
            algo: str = "mcorr"
    ):
        self.grid_shape = grid_plot_shape
        self.table = GridspecLayout(self.grid_shape[0], self.grid_shape[1])
        self.algo = algo

    def set_algo(self, algo: str):
        self.algo = algo
        if self.algo == "mcorr":
            for i in range(self.grid_shape[0]):
                for j in range(self.grid_shape[1]):
                    self.table[i, j] = Dropdown(
                        options=['raw movie', 'mcorr movie', 'downsampled avg movie', 'correlation image', 'shifts'])
        elif self.algo == "cnmf":
            for i in range(self.grid_shape[0]):
                for j in range(self.grid_shape[1]):
                    self.table[i, j] = Dropdown(
                        options=['input movie', 'contours', 'reconstructed', 'residuals', 'temporal', 'background'])

    def get_layout(self):
        self.set_algo("cnmf")
        return self.table
