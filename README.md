# mesmerize-viz

This is currently in beta and documentation is a WIP. Motion Correction and CNMF visualizations should just work. CNMFE will work without `"rcb"` and `"residuals"` `image_data_options`.

:exclamation: **Harware requirements** The large CNMF visualizations with contours etc. usually require either a dedicated GPU or integrated GPU with access to at least 1GB of VRAM. 

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

### Explore parameter variants

Click on different rows to view the results of different runs of motion correction, CNMF or CNMFE.

https://github.com/kushalkolar/mesmerize-viz/assets/9403332/41175c80-7bdf-4210-96d4-4913ae46568e

### Explore components

Explore components using the heatmap selector, or the component index slider. Auto-zoom into components if desired using the checkbox, set the zoom scale using the slider.

https://github.com/kushalkolar/mesmerize-viz/assets/9403332/c6d8cb7d-f99c-4771-8562-b890c9a18ae2

### Visualize component evaluation metrics

View the evaluation metrics by setting the contour colors based on the metrics. Select to show "all", or only "accepted" or only "rejected" components based on the current evaluation criteria. You can also choose to make the accepted or rejected components semi-transparent instead of entirely opague or invisible using the alpha slider.

Colormaps used:

accepted/rejected: Set1, accepted - blue, rejected - red

snr, r_values, and cnn_preds: spring: low value: pink, high value: yellow

https://github.com/kushalkolar/mesmerize-viz/assets/9403332/b2780212-c941-4306-b7de-45bfa49ab9cd

### Interactive component evaluation using metrics and manully accept or reject components

Interactively change the metric thresholds for the sliders. See the caiman docs for info on the evaluation params: https://caiman.readthedocs.io/en/latest/Getting_Started.html#component-evaluation 

After setting the metric thresholds, you can manually accept or reject components by clicking on them and pressing "a" (accept) or "r" (reject) keys on your keyboard. 

When you are happy with component evaluation, click "Save eval to disk". This overwrites the existing hdf5 file with the state of the hdf5 file as shown in the visualization, i.e. `estimates.idx_components` and `estimates.edx_components_bad` gets set with respect to the visualization. 

https://github.com/kushalkolar/mesmerize-viz/assets/9403332/0e7b0b41-9360-456c-9c91-6bd74fedb11d

## Voila app

WIP
### Explore components**

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
