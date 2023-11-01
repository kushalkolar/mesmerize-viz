# mesmerize-viz

This is currently in beta. Motion Correction and CNMF visualizations should just work.

https://www.youtube.com/watch?v=GWvaEeqA1hw

## Installation

Assuming you have `mesmerize-core` installed:

```bash
git clone https://github.com/kushalkolar/mesmerize-viz.git
cd mesmerize-viz
pip install -e .
```

If you want to use `%gui qt` you will need pyqt6:

```
pip install PyQt6
```

## Usage

See the example notebooks

## Voila app

WIP

Install voila:

```bash
pip install voila
```

Use as a voila app (as shown in the demo video).

```bash
cd mesmerize-viz
voila examples/app.ipynb --enable_nbextensions=True
```

Note that the voila app is a WIP prototype
