import abc
import collections
import numpy as np
from rlkit.envs.images import Renderer
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt


class MatplotLibRenderer(Renderer, metaclass=abc.ABCMeta):
    def __init__(
            self,
            modify_ax_fn=None,
            modify_fig_fn=None,
            autoscale_y=True,
            dpi=32,
            **kwargs
    ):
        super().__init__(
            create_image_format='HWC',
            normalize_image='True',
            **kwargs)
        self._dpi = dpi  # TODO: use automatically
        self.modify_fig_fn = modify_fig_fn
        self.modify_ax_fn = modify_ax_fn
        _, height, width = self.image_chw

        figsize = (height / dpi, width / dpi)
        self._figsize = figsize
        self.fig = plt.figure(
            figsize=figsize,
            dpi=dpi,
        )
        self.ax = self.fig.add_subplot(1, 1, 1)
        if modify_fig_fn:
            modify_fig_fn(self.fig)
        if modify_ax_fn:
            modify_ax_fn(self.ax)
        self._autoscale_y = autoscale_y
        self.canvas = FigureCanvas(self.fig)

    def _render_plot(self):
        # https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
        self.canvas.draw()
        flat_img = np.frombuffer(self.canvas.tostring_rgb(), dtype='uint8')
        width, height = self.fig.get_size_inches() * self.fig.get_dpi()
        width = int(width)
        height = int(height)
        img = flat_img.reshape(height, width, 3)
        return img

    def __getstate__(self):
        """You can't cloudpickle matplotlib most objects."""
        state = {k: v for k, v in self.__dict__.items()}
        state.pop('fig')
        state.pop('ax')
        state.pop('canvas')
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            self.__dict__[k] = v
        self.fig = plt.figure(
            figsize=self._figsize,
            dpi=self._dpi,
        )

        self.ax = self.fig.add_subplot(1, 1, 1)
        if self.modify_fig_fn:
            self.modify_fig_fn(self.fig)
        if self.modify_ax_fn:
            self.modify_ax_fn(self.ax)
        self.canvas = FigureCanvas(self.fig)


class ScrollingPlotRenderer(MatplotLibRenderer):
    """
    Plot the history of some number over a scrolling window.

    So something like:

    ---------------------
    |                   |
    |   |        ___/   |
    |   |     __/       |
    | y |    /          |
    |   | __/           |
    |   |_____________  |
    |        time       |
    ---------------------
    """
    def __init__(
            self,
            window_size=100,
            **kwargs
    ):
        """Render an image."""
        super().__init__(**kwargs)
        self.window_size = window_size
        self.lines = self.ax.plot(
            [], [],
        )[0]
        self.t = 0
        self.xs = collections.deque(maxlen=window_size)
        self.ys = collections.deque(maxlen=window_size)

    def reset(self):
        self.t = 0
        self.xs = collections.deque(maxlen=self.window_size)
        self.ys = collections.deque(maxlen=self.window_size)
        self.lines.set_xdata(np.array([]))
        self.lines.set_ydata(np.array([]))

    def _create_image(self, number):
        self._update_plot(number)
        return self._render_plot()

    def _update_plot(self, number):
        if number is not None:
            self.xs.append(self.t)
            self.t += 1
            self.ys.append(number)
        self.lines.set_xdata(np.array(self.xs))
        self.lines.set_ydata(np.array(self.ys))
        self.ax.relim()
        self.ax.autoscale_view(scalex=True, scaley=self._autoscale_y)

    def __getstate__(self):
        state = super().__getstate__()
        state['lines'] = None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.lines = self.ax.plot(
            [], [],
        )[0]


class TextRenderer(MatplotLibRenderer):
    """
    Plot the history of some number over a scrolling window.

    So something like:

    ---------------------
    |                   |
    |   |        ___/   |
    |   |     __/       |
    | y |    /          |
    |   | __/           |
    |   |_____________  |
    |        time       |
    ---------------------
    """
    def __init__(
            self,
            text,
            *args,
            font_size=45,
            **kwargs
    ):
        """Render an image."""
        super().__init__(*args, **kwargs)
        self._font_size = font_size
        self.ax.axis('off')
        self.prefix = ''
        self._text = None
        self._img = None

        self.set_text(text)

    def _create_image(self, number):
        return self._img

    def set_text(self, text):
        if self._text is not None:
            self._text.remove()
        self._add_text(text)
        self._img = self._render_plot()

    def _add_text(self, text):
        self._text = self.ax.text(
            0.0,
            1.0,
            self.prefix + text,
            horizontalalignment='left',
            verticalalignment='top',
            # transform=self.ax.transAxes,
            fontsize=self._font_size,
        )

    def __getstate__(self):
        state = super().__getstate__()
        state['_text'] = None
        return state
