import fastplotlib as fpl
from pathlib import Path
import dask.array as da
from dask import delayed
from mesmerize_viz import *
import mesmerize_core as mc


def get_image_widget(df, df_idx=0):
    input_movie = df.iloc[df_idx].caiman.get_input_movie()

    figure = fpl.ImageWidget(
        data=input_movie,
        names=['Input'],
        histogram_widget=False,
        figure_kwargs={"size": (700, 560), "shape": (1, 3)},
    )
    for subplot in figure.figure:
        # sometimes the toolbar adds clutter
        subplot.toolbar = False
    return figure
    # return mviz_corr


if __name__ == '__main__':
    batch = Path(r'C:\Users\RBO\caiman_data\animal_01\session_01\batch.pickle')
    # batch = Path().home() / 'caiman_data' / 'animal_01' / 'session_01' / 'results' / 'cnmf_batch.pickle'
    df = mc.load_batch(batch)
    viz = df.cnmf.viz(start_index=-3)
    viz.show()
    fpl.run()
