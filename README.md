# mesmerize-viz

Currently has basic functionality for mcorr and cnmf visualization

https://github.com/kushalkolar/mesmerize-viz/assets/9403332/490dac6e-d57c-4996-9d77-a391d67ed049

## Installation

Note that `mesmerize-viz` currently requires the latest `fastplotlib` and `pygfx` from github.

Assuming you have `mesmerize-core` installed:

```bash
git clone https://github.com/pygfx/pygfx.git
cd pygfx
pip install -e .

git clone https://github.com/kushalkolar/fastplotlib.git
cd fastplotlib
pip install -e .

git clone https://github.com/kushalkolar/mesmerize-viz.git
cd mesmerize-viz
pip install -e .
```

## Usage

See the example notebooks, or use as a voila app.

Install voila:

```bash
pip install voila
```

Use as a voila app (as shown in the demo video)

```bash
cd mesmerize-viz
voila examples/app.ipynb --enable_nbextensions=True
```
