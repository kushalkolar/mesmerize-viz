import numpy as np
from ipywidgets import widgets, VBox, HBox, Layout
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
            grid_plot_shape: Tuple[int, int],
            multi_select: bool = False
    ):
        self.grid_shape: Tuple[int, int] = None

    def get_layout(self):
        return widgets.IntSlider()