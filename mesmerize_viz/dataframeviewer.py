from ipywidgets import Tab


class DataframeViewer:
    def __init__(
            self,
            _BaseViewer,
            SelectViewer
    ):
        self.base = _BaseViewer
        self.select = SelectViewer
        self.tab = Tab()

        self.tab.set_title(0, "Viewer")
        self.tab.set_title(1, "Selector")

        self.tab.children = (self.base.get_layout(), self.select.get_layout())

    def show(self):
        if self.base.algo == "mcorr":
            return self.base.get_layout()
        else:
            return self.tab
