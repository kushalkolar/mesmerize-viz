from typing import *
import numpy as np
import pygfx

from fastplotlib import ImageGraphic, LinearSelector, ScatterGraphic, ImageWidget, LineCollection, Figure
from ipywidgets import IntSlider, FloatSlider, BoundedIntText

from fastplotlib.graphics._collection_base import CollectionFeature
from fastplotlib.graphics._features import FeatureEvent
from fastplotlib.layouts._subplot import Subplot
from ._utils import get_nearest_graphics_indices

MARGIN: float = 1

# TODO: need to make a method for automatic MARGIN setting based on the data

class NeuronStoreComponent:

    @property
    def subscriber(self) -> LineCollection | LinearSelector | BoundedIntText | Figure:
        return self._subscriber

    def __init__(self, subscriber, data=None):
        """A Collection component of the collection store."""
        self._subscriber = subscriber
        self._data = data

        # If given a subpot with a line collection, that line collection is the data
        if isinstance(self.subscriber, Subplot):
            # hacky
            for graphic in self.subscriber.graphics:
                if isinstance(graphic, LineCollection):
                    self._data = self.subscriber[graphic.name]


    @property
    def data(self) -> np.ndarray | CollectionFeature:
        return self._data


class NeuronStore:
    @property
    def current_index(self):
        """Index of the currently selected graphic."""
        return self._current_index

    @current_index.setter
    def current_index(self, value: int):
        """Store the previous index and set the current neuron index."""
        self.previous_index = self._current_index
        self._current_index = int(value)

    @property
    def previous_color(self):
        return self._previous_color

    @previous_color.setter
    def previous_color(self, value: int):
        self._previous_index = value

    @property
    def previous_index(self):
        return self._previous_index

    @previous_index.setter
    def previous_index(self, value: int):
        self._previous_index = value

    @property
    def store(self) -> List[NeuronStoreComponent]:
        """Returns the items in the store."""
        return self._store

    @property
    def zoom_scale(self) -> float:
        """Returns the zoom scale."""
        return self._zoom_scale

    @zoom_scale.setter
    def zoom_scale(self, value: float):
        """Sets the zoom scale."""
        self._zoom_scale = value

    @property
    def auto_zoom(self) -> int:
        """  """
        return self._auto_zoom

    @auto_zoom.setter
    def auto_zoom(self, value: int):
        """  """
        self._auto_zoom = value


    def __init__(self):
        """
        NeuronStore for synchronizes and updating components of a plot (i.e. Ipywidgets.IntSlider,
        fastplotlib.LinearSelector, or fastplotlob.ImageGraphic).

        NOTE: If passing a `fastplotlib.ImageGraphic`, it is understood that there should be an associated
        `ndarray` given.
        """
        # initialize store
        self._store = list()
        # by default, current_index is zero
        self._current_index = None
        self._previous_index = None
        # store the previous color to reset when a new neuron is selected
        self._previous_color = None
        self._auto_zoom = False
        self._zoom_scale = 1

    def subscribe(self,
                  subscriber: Subplot | LineCollection | LinearSelector | BoundedIntText | IntSlider,
                  data=None) -> None:
        """
        Method for adding a subscriber to the store to be synchronized.

        Parameters
        ----------
        subscriber: fastplotlib.ImageGraphic, fastplotlib.LinearSelector, ipywidgets.IntSlider, or ipywidgets.FloatSlider
            ipywidget or fastplotlib object to be synchronized
        data: ndarray
        """
        # create a TimeStoreComponent
        component = NeuronStoreComponent(subscriber=subscriber, data=data)

        # add component to the store
        self._store.append(component)
        if isinstance(subscriber, Subplot):
            for g in subscriber.graphics:
                g.add_event_handler(self._update_store, "click")
        if isinstance(component.subscriber, LineCollection | LinearSelector):
            component.subscriber.add_event_handler(self._update_store, "selection")
        elif isinstance(component.subscriber, BoundedIntText | IntSlider):
            component.subscriber.observe(self._update_store, "value")

    def unsubscribe(self, subscriber):
        """Remove a subscriber from the store."""
        for component in self.store:
            if component.subscriber == subscriber:
                #  remove the component from the store
                self.store.remove(component)
                # remove event handler
                if isinstance(component.subscriber, (ImageGraphic, Subplot)):
                    # does this go here?
                    # for graphic in component.subscriber.graphics:
                    #     graphic.registered_callbacks.clear()
                    component.subscriber.remove_event_handler(self._update_store, "click")
                if isinstance(component.subscriber, LineCollection | LinearSelector):
                    component.subscriber.remove_event_handler(self._update_store, "selection")
                if isinstance(component.subscriber, BoundedIntText | IntSlider):
                    component.subcriber.unobserve(self._update_store)

    def _update_store(self, ev):
        """Called when event occurs and store needs to be updated."""
        # First, parse event to set index
        if isinstance(ev, pygfx.PointerEvent):
            # an image graphic or contour was selected on the screen
            # first, set current_index from the pointer event on the graphic
            index_updated = False
            for component in self.store:
                # hacky, cant subscribe every line in a line graphic?
                if hasattr(component.subscriber, "graphics"):
                    if ev.graphic == component.subscriber.graphics[0]:
                        xy = component.subscriber.map_screen_to_world(ev)[:-1]
                        nearest_idx = get_nearest_graphics_indices(xy, component.data)[0]
                        self.current_index = nearest_idx
                        index_updated = True
            if not index_updated:
                print("Clicking on line graphics is not yet supported.")
                return
        elif isinstance(ev, FeatureEvent):
            # came from heatmap component selector
            if hasattr(ev, "pick_Info"):
                if ev.info["pygfx_event"] is None:
                    # this means that the selector was not triggered by the user but that it moved due to another event
                    # so then we don't set_component_index because then infinite recursion
                    return
            index = int(ev.info["value"])
            self.current_index = index
        else:
            print(ev)
            raise TypeError(f"Unknown event: {ev}")

        # update each subscriber's data with the new index
        for component in self.store:
            if isinstance(component.subscriber, Subplot):
                if all(hasattr (component.data, attr) for attr in ["thickness", "colors"]):
                    component.data[self.current_index].thickness = 8
                    # self._previous_color = component.data[self.current_index].colors
                    # component.data[self.current_index].colors = "w"

                    if self.previous_index is not None:
                        component.data[self.previous_index].thickness = 2
                        # component.data[self.previous_index].colors = self._previous_color
                elif component.data is not None:
                    component.subscriber.graphics[0].data[:, 1] = component.data[self.current_index]

            elif isinstance(component.subscriber, LinearSelector):
                component.subscriber.value = self.current_index

            elif isinstance(component.subscriber, BoundedIntText | IntSlider):
                component.subscriber.unobserve_all()
                component.subscriber.value = self.current_index

    def _set_auto_zoom(self, value: bool):
        self.auto_zoom = value
        if self.auto_zoom:
            for component in self.store:
                if isinstance(component.subscriber, Subplot):
                    if len(component.subscriber.graphics) > 1:  # contains an image graphic and line-collection
                        component.subscriber.camera.show_object(
                            component.subscriber['contours'][self.current_index].world_object,
                            scale=self.zoom_scale
                        )
        else:
            for component in self.store:
                if isinstance(component.subscriber, Subplot):
                    component.subscriber.zoom_flag = value

    def _set_zoom_scale(self, value: float):
        print('called')
        self.zoom_scale = value
        self._set_auto_zoom(True)


class TimeStoreComponent:
    @property
    def subscriber(self) -> ImageGraphic | IntSlider | FloatSlider | LinearSelector:
        return self._subscriber

    @property
    def data(self) -> np.ndarray | None:
        return self._data

    @property
    def multiplier(self) -> int | float | None:
        return self._multiplier

    @property
    def data_filter(self) -> callable:
        return self._data_filter

    def __init__(self, subscriber, data=None, data_filter=None, multiplier=None):
        """A TimeStore component of the time store."""
        if multiplier is None:
            multiplier = 1

        self._multiplier = multiplier

        self._subscriber = subscriber

        # must have data if ImageGraphic
        if isinstance(self.subscriber, (ImageGraphic, ScatterGraphic)):
            # LazyArrayRCM has no `__array__`, using `shape` for now
            if not hasattr(data, 'shape'):
                raise ValueError("If passing in `ImageGraphic` must provide associated `ndarray` object to update "
                                 "data with.")
            self._data = data
        self._data_filter = data_filter


class TimeStore:
    @property
    def time(self):
        """Current t value that items in the store are set at."""
        return self._time

    @time.setter
    def time(self, value: int | float):
        """Set the current time."""
        self._time = int(value)

    @property
    def store(self) -> List[TimeStoreComponent]:
        """Returns the items in the store."""
        return self._store

    def __init__(self):
        """
        TimeStore for synchronizes and updating components of a plot (i.e. Ipywidgets.IntSlider,
        fastplotlib.LinearSelector, or fastplotlob.ImageGraphic).

        NOTE: If passing a `fastplotlib.ImageGraphic`, it is understood that there should be an associated
        `ndarray` given.
        """
        # initialize store
        self._store = list()
        # by default, time is zero
        self._time = 0

    def subscribe(self,
                  subscriber: ImageWidget | ImageGraphic | LinearSelector | ScatterGraphic | IntSlider | FloatSlider,
                  data: np.ndarray = None,
                  data_filter: callable = None,
                  multiplier: int | float = None) -> None:
        """
        Method for adding a subscriber to the store to be synchronized.

        Parameters
        ----------
        subscriber: fastplotlib.ImageGraphic, fastplotlib.LinearSelector, ipywidgets.IntSlider, or ipywidgets.FloatSlider
            ipywidget or fastplotlib object to be synchronized
        data: np.ndarray, optional
            If subscriber is a fastplotlib.ImageGraphic, must have an associating numpy.ndarray to update data with.
        data_filter: callable, optional
            Function to apply to data before updating. Must return data in the same shape as input.
        multiplier: int | float, optional
            Scale the current time to reflect differing timescale.
        """
        # create a TimeStoreComponent
        component = TimeStoreComponent(subscriber=subscriber,
                                       data=data,
                                       data_filter=data_filter,
                                       multiplier=multiplier)

        # add component to the store
        self._store.append(component)

        if isinstance(component.subscriber, ImageWidget):
            component.subscriber.add_event_handler(self._update_store, "current_index")
        if isinstance(component.subscriber, (IntSlider, FloatSlider)):
            component.subscriber.observe(self._update_store, "value")
        if isinstance(component.subscriber, LinearSelector):
            component.subscriber.add_event_handler(self._update_store, "selection")

    def unsubscribe(self, subscriber: ImageGraphic | LinearSelector | IntSlider | FloatSlider):
        """Remove a subscriber from the store."""
        for component in self.store:
            if component.subscriber == subscriber:
                #  remove the component from the store
                self.store.remove(component)
                # remove event handler
                if isinstance(component, (IntSlider)):
                    component.subscriber.unobserve(self._update_store)

    def _update_store(self, ev):
        """Called when event occurs and store needs to be updated."""
        # parse event to see if it originated from ipywidget or selector
        if isinstance(ev, FeatureEvent):
            # check for multiplier to adjust time
            for component in self.store:
                if isinstance(component.subscriber, LinearSelector):
                    if ev.graphic == component.subscriber:
                        self.time = ev.info["value"] / component.multiplier
        elif isinstance(ev, dict):
            self.time = ev["t"]
        else:
            self.time = ev["new"]

        for component in self.store:
            if isinstance(component.subscriber, ImageWidget):
                # user moved qslider, don't update imagewidget
                if isinstance(ev, dict) and 't' in ev:
                    pass
                else:
                    component.subscriber.current_index = {"t": self.time}
            elif isinstance(component.subscriber, ScatterGraphic):
                component.subscriber.data = component.data[self.time]
            # update ImageGraphic data no matter what
            elif isinstance(component.subscriber, ImageGraphic):
                if component.data_filter is None:
                    new_data = component.data[self.time]
                else:
                    new_data = component.data_filter(component.data[self.time])
                if new_data.shape != component.subscriber.data.value.shape:
                    raise ValueError(f"data filter function: {component.data_filter} must return data in the same shape"
                                     f"as the current data")
                component.subscriber.data = new_data
            elif isinstance(component.subscriber, LinearSelector):
                # only update if different
                if abs(component.subscriber.selection - (self.time * component.multiplier)) > MARGIN:
                    component.subscriber.selection = self.time * component.multiplier
                else:
                    # only update if different
                    if abs(component.subscriber.selection - self.time) > MARGIN:
                        component.subscriber.selection = self.time
