from typing import *

import numpy as np
from fastplotlib.graphics._base import Graphic
from fastplotlib.graphics._collection_base import GraphicCollection


# to format params dict into yaml-like string
is_pos = lambda x: 1 if x > 0 else 0
# this doesn't work without the lambda, yes it is ugly
format_params = lambda d, t: "\n" * is_pos(t) + \
    "\n".join(
        [": ".join(["   " * t + k, format_params(v, t + 1)]) for k, v in d.items()]
    ) if isinstance(d, dict) else str(d)


class DummyMovie:
    """Really really hacky"""
    def __init__(self, image: np.ndarray, shape, ndim, size):
        self.image = image
        self.shape = shape
        self.ndim = ndim
        self.size = size

    def __getitem__(self, index: Union[int, slice]):
        if isinstance(index, tuple):
            for s in index:
                if isinstance(s, int):
                    # assumption
                    index = s
                    break

                if (s.start is None) and (s.stop is None) and (s.step is None):
                    continue
                else:
                    # assume that this is the dimension that user has asked for, and we return the image using
                    # slice size from this dimension
                    index = s

        if isinstance(index, (slice, range)):
            start, stop, step = index.start, index.stop, index.step

            if start is None:
                start = 0

            if stop is None:
                # assumption, again this is very hacky
                stop = max(self.shape)

            if step is None:
                step = 1

            r = range(start, stop, step)

            n_frames = len(r)

            return np.array([self.image] * n_frames)

        if isinstance(index, int):
            return self.image

        else:
            raise TypeError(f"DummyMovie only accept int or slice indexing, you have passed: {index}")



def get_nearest_graphics_indices(
    pos: tuple[float, float] | tuple[float, float, float],
    graphics: Sequence[Graphic] | GraphicCollection,
) -> np.ndarray[int]:
    """
    Returns indices of the nearest ``graphics`` to the passed position ``pos`` in world space
    in order of closest to furtherst. Uses the distance between ``pos`` and the center of the
    bounding sphere for each graphic.

    Parameters
    ----------
    pos: (x, y) | (x, y, z)
        position in world space, z-axis is ignored when calculating L2 norms if ``pos`` is 2D

    graphics: Sequence, i.e. array, list, tuple, etc. of Graphic | GraphicCollection
        the graphics from which to return a sorted array of graphics in order of closest
        to furthest graphic

    Returns
    -------
    ndarray[int]
        indices of the nearest nearest graphics to ``pos`` in order

    """
    if isinstance(graphics, GraphicCollection):
        graphics = graphics.graphics

    if not all(isinstance(g, Graphic) for g in graphics):
        raise TypeError("all elements of `graphics` must be Graphic objects")

    pos = np.asarray(pos)

    if pos.shape != (2,) or not pos.shape != (3,):
        raise TypeError

    # get centers
    centers = np.empty(shape=(len(graphics), len(pos)))
    for i in range(centers.shape[0]):
        centers[i] = graphics[i].world_object.get_world_bounding_sphere()[: len(pos)]

    # l2
    distances = np.linalg.norm(centers[:, : len(pos)] - pos, ord=2, axis=1)

    sort_indices = np.argsort(distances)
    return sort_indices


def get_nearest_graphics(
    pos: tuple[float, float] | tuple[float, float, float],
    graphics: Sequence[Graphic] | GraphicCollection,
) -> np.ndarray[Graphic]:
    """
    Returns the nearest ``graphics`` to the passed position ``pos`` in world space.
    Uses the distance between ``pos`` and the center of the bounding sphere for each graphic.

    Parameters
    ----------
    pos: (x, y) | (x, y, z)
        position in world space, z-axis is ignored when calculating L2 norms if ``pos`` is 2D

    graphics: Sequence, i.e. array, list, tuple, etc. of Graphic | GraphicCollection
        the graphics from which to return a sorted array of graphics in order of closest
        to furthest graphic

    Returns
    -------
    ndarray[Graphic]
        nearest graphics to ``pos`` in order

    """
    sort_indices = get_nearest_graphics_indices(pos, graphics)
    return np.asarray(graphics)[sort_indices]