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

    def update_alg(self, algo: str):
        if algo == "mcorr":
            for i in range(self.grid_shape[0]):
                for j in range(self.grid_shape[1]):
                    self.table[i,j] = Dropdown(
            options=['raw movie', 'mcorr movie', 'downsampled avg movie', 'correlation image', 'shifts'])
        elif algo == "cnmf":
            for i in range(self.grid_shape[0]):
                for j in range(self.grid_shape[1]):
                    self.table[i, j] = Dropdown(
            options=['input movie', 'contours', 'reconstructed movie', 'residuals', 'heatmap', 'traces'])


    def get_layout(self):
        self.update_alg("cnmf")
        return self.table
