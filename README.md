# mesmerize-viz
Notebook widgets built using [`fastplotlib`](https://github.com/kushalkolar/fastplotlib) and `ipywidgets` for exploring the data stored in a batch DataFrame from [`mesmerize-core`](https://github.com/nel-lab/mesmerize-core). These visualizations are built using [`fastplotlib`](https://github.com/kushalkolar/fastplotlib) that can utilize [Vulkan](https://en.wikipedia.org/wiki/Vulkan), so they are faster than `mesmerize-napari`. 

This requires [`mesmerize-core`](https://github.com/nel-lab/mesmerize-core) and [`fastplotlib`](https://github.com/kushalkolar/fastplotlib) to be installed.

This is a WIP, currently includes experimental widget for inspecting motion correction results (see video).\
This widget displays:

|Raw | MCor | Downsampled Mean |
|----|------|--------------------|
|Mean Image | Correlation image | Blank |

https://user-images.githubusercontent.com/9403332/175873146-1aeaadc1-aa06-48dd-a50e-d9f998284f47.mp4
