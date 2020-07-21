"""A simple self contained example of how Napari can be useful for visualizing data. Using modelled
charge transition data here where the 2D plot is just a 1D line repeated, and the profile line is of the
1D plot"""

import napari
from dataclasses import dataclass, asdict
import numpy as np

# For typing
from typing import Union, List, Tuple
from napari.layers.image.image import Image
from napari._vispy.vispy_figure import Fig
from vispy.scene.visuals import LinePlot


def get_front_data(data_layer):
    if data_layer.ndim > 2:
        front_data = data_layer.data[data_layer.coordinates[:-2]]
    else:
        front_data = data_layer.data[:]
    return front_data


class Window(object):
    def __init__(self):
        self.viewer = napari.Viewer()
        self.data: Image = None  # data image_layer
        self.x_array: np.ndarray = None  # Somewhere to store x_array info

        # For profiles
        self.profile_fig: Fig = None  # Napari subclass of Vispy Figure
        self._profiles: LinePlot = None  # Vispy lines

    def add_data(self, data, x=None):
        """
        Add data to napari window
        Args:
            data (np.ndarray): N dimensional data where last two dimensions form y,x of 2D data
        Returns:
            None
        """
        if x is not None:
            self.x_array = x

        scaling = list((100 / s for s in data.shape[-2:]))  # Scales 2D image to be 100x100 by default
        # There are possibly issues with trying to use fractional values because of the way the indexing works

        # Scale all other dimensions by 1 by default
        if data.ndim > 2:
            scaling = [1] * (data.ndim - 2) + scaling

        self.data = self.viewer.add_image(data, scale=tuple(scaling))

    def add_profile(self):
        """
        Add profile line to figure (and will initialize figure if first time being called)

        Returns:
            None
        """
        # If first time adding profile, initialize values
        if self.profile_fig is None:
            self.profile_fig, _ = self.viewer.window.add_docked_figure(area='right', initial_width=500)
            self._profiles = []

        profile_id = len(self._profiles)  # i.e. 0 for first profile, 1 for second etc

        # Get the 2D data which is currently being displayed (self.data contains Nd coord info)
        front_data = get_front_data(self.data)

        # Add a profile line to new axes in Figure (just plotting row 0 of data for now)
        new_profile = self.profile_fig[profile_id, 0].plot(front_data[0], marker_size=0)

        # Store for updating data in this profile in callbacks
        self._profiles.append(new_profile)

        # Attach callback so that dragging dimension bars updates plots (Note: This breaks the 3D view in napari)
        self.viewer.dims.events.connect(self.update_lines)

    def update_lines(self, *args):
        """
        Update any lines in the profile figure

        Args:
            *args (): Here so that callback can pass in arguments even if we don't use them

        Returns:
            None
        """
        # Get the 2D data which is currently displayed
        front_data = get_front_data(self.data)
        for profile in self._profiles:
            # Update profile with first row of data being shown
            profile.set_data(front_data[0], marker_size=0)

        # Rescale the figure axes (plot_widgets for vispy)
        for ax in self.profile_fig.plot_widgets:
            ax.autoscale()


def i_sense(x, mid, theta, amp, lin, const):
    """ Charge transition shape """
    arg = (x - mid) / (2 * theta)
    return -amp/2 * np.tanh(arg) + lin * (x - mid) + const


@dataclass
class TransitonInfo:
    """Dataclass to store info about transition fit"""
    x: np.ndarray = np.linspace(-10, 10, 200)
    mid: Union[float, np.ndarray] = 0.0
    theta: Union[float, np.ndarray] = 0.5
    amp: Union[float, np.ndarray] = 0.8
    lin: Union[float, np.ndarray] = 0.01
    const: Union[float, np.ndarray] = 8


def get_data_array(info: TransitonInfo):
    """
    Turn TransitionInfo into an (N+2)dimension data array where N is the number of variables that are
    arrays, and +2 for 2D plots with those variables.
    Note: Just repeating a 1D plot twice to make it a 2D array so napari can view it easily

    Args:
        info (TransitonInfo):  Info to turn into a Nd data array

    Returns:
        np.ndarray: N+2 dimensional array of transition data where N is number of variables that are arrays
    """
    # Turn into dictionary so can iterate through values
    info = asdict(info)
    x = info['x']  # x is separate from other variables

    # Get which variables are arrays instead of just values
    array_keys = []
    for k, v in info.items():
        if isinstance(v, np.ndarray) and k != 'x':
            array_keys.append(k)

    # Get meshgrids for all variables that were arrays, (here x has to go at the end to get the right shape of data)
    meshes = np.meshgrid(*[v for k, v in info.items() if k in array_keys], x, indexing='ij')

    # Make meshes into a dict using the keys we got above
    meshes = {k: v for k, v in zip(array_keys+['x'], meshes)}

    # Make a list of all of the variables either drawing from meshes, or otherwise just the single values
    vars = {}
    for k in info.keys():
        vars[k] = meshes[k] if k in meshes else info[k]

    # Evaluate the charge transition at all meshgrid positions in one go (resulting in N+1 dimension array)
    data_array = i_sense(vars['x'], vars['mid'], vars['theta'], vars['amp'], vars['lin'], vars['const'])

    # Add a y dimension to the data so that it is an N+2 dimension array (duplicate all data and then move that axis
    # to the y position (N, y, x)
    data2d_array = np.moveaxis(np.repeat([data_array], 2, axis=0), 0, -2)
    return data2d_array


if __name__ == '__main__':
    # If running in Ipython this function will exist at runtime
    get_ipython().enable_gui('qt')
    # Otherwise run everything below using "with napari.gui_qt():"

    run = '5D'
    t_params = TransitonInfo()
    if run == '2D':
        """Just Plot the 2D data with a line profile and no sliders"""
        pass
    elif run == '3D':
        """Plot 2D data with a single slider which changes the theta parameter"""
        t_params.theta = np.linspace(0.3, 2, num=20)
    elif run == '5D':
        """Plot 5D data with a slider for each of theta, amp, lin"""
        t_params.theta = np.linspace(0.3, 2, num=20)
        t_params.amp = np.linspace(0.5, 1, num=5)
        t_params.lin = np.linspace(0.0, 0.05, num=10)
    else:
        raise ValueError('Set run to one of (2D, 3D, 5D)')
    data = get_data_array(t_params)
    w = Window()
    w.add_data(data, x=t_params.x)
    w.add_profile()


