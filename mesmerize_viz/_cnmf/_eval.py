from typing import *
from collections import OrderedDict

from ipywidgets import FloatSlider, FloatText, HBox, VBox, link, Layout, Label


class EvalWidgets:
    def __init__(self):
        # low thresholds
        self._low_thresholds = OrderedDict(
            rval_lowest=(-1.0, -1.0, 1.0),  # (val, min, max)
            SNR_lowest=(0.5, 0., 100),
            cnn_lowest=(0.1, 0., 1.0),
        )

        # high thresholds
        self.high_thresholds = OrderedDict(
            rval_thr=(0.8, 0., 1.0),
            min_SNR=(2.5, 0., 100),
            min_cnn_thr=(0.9, 0., 1.0),
        )

        self._low_threshold_widget = list()
        for k in self._low_thresholds:
            kwargs = dict(value=self._low_thresholds[k][0], min=self._low_thresholds[k][1], max=self._low_thresholds[k][2], step=0.01, description=k)
            slider = FloatSlider(**kwargs)
            entry = FloatText(**kwargs, layout=Layout(width="150px"))

            link((slider, "value"), (entry, "value"))

            setattr(self, f"_{k}", entry)

            self._low_threshold_widget.append(HBox([slider, entry]))

        self._high_threshold_widgets = list()
        for k in self.high_thresholds:
            kwargs = dict(value=self.high_thresholds[k][0], min=self.high_thresholds[k][1], max=self.high_thresholds[k][2], step=0.01, description=k)
            slider = FloatSlider(**kwargs)
            entry = FloatText(**kwargs, layout=Layout(width="150px"))

            link((slider, "value"), (entry, "value"))

            setattr(self, f"_{k}", entry)

            self._high_threshold_widgets.append(HBox([slider, entry]))

        self.widget = VBox(
            [
                Label("Low Thresholds"),
                self._low_threshold_widget,
                Label("High Thresholds"),
                self._high_threshold_widgets
            ]
        )

    def get_params(self):
        """get the values from the GUI"""

        eval_params = dict()
        for param in self._low_thresholds:
            eval_params[param] = getattr(self, f"_{param}.value")

        for param in self._high_threshold_widgets:
            eval_params[param] = getattr(self, f"_{param}.value")

        return eval_params

    def set_param(self, param: str, value: float):
        w = getattr(self, f"_{param}")

        w.value = value
