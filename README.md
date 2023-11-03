# mesmerize-viz

This is currently in beta. Motion Correction and CNMF visualizations should just work. CNMFE will work without `"rcb"` and `"residuals"` `image_data_options`

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

### CNMF

**Export parameter variants**

https://github.com/kushalkolar/mesmerize-viz/assets/9403332/41175c80-7bdf-4210-96d4-4913ae46568e

**Explore components**

https://github.com/kushalkolar/mesmerize-viz/assets/9403332/c6d8cb7d-f99c-4771-8562-b890c9a18ae2

**Visualize component evaluation metrics**

https://github.com/kushalkolar/mesmerize-viz/assets/9403332/b2780212-c941-4306-b7de-45bfa49ab9cd

**Interactive component evaluation using metrics and manully accept or reject components**

https://github.com/kushalkolar/mesmerize-viz/assets/9403332/0e7b0b41-9360-456c-9c91-6bd74fedb11d


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
