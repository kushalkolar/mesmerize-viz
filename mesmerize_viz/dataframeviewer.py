import numpy as np
from ipywidgets import widgets, VBox, HBox, Layout
from fastplotlib import GridPlot, Image, Subplot
from typing import *
import pandas as pd
from uuid import UUID
from collections import OrderedDict
import pims
import time

class DataFrameViewer:
    def __init__(
            self,
            _BaseViewer,
            SelectViewer,
    ):
        self.base = _BaseViewer
        self.select = SelectViewer
        self.tab = widgets.Tab()

        self.tab.set_title(0, 'Viewer')
        self.tab.set_title(1, 'Selector')

        self.tab.children = (self.base.get_layout(), self.select.get_layout())

    def show(self):
        return self.tab

