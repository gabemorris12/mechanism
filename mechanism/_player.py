# Class for creating animations with keybindings
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


class Player(FuncAnimation):
    def __init__(self, fig, func, frames=None, init_func=None, fargs=None, save_count=None, cache_frame_data=True,
                 key_bindings=True, **kwargs):
        self.fig = fig
        self.func = func
        self.frames = frames
        self.fargs = fargs
        self.save_count = frames if save_count is None else save_count
        self.i = 0
        self.going = True

        if key_bindings:
            # noinspection PyTypeChecker
            plt.connect('key_press_event', self._pause_play)

        # noinspection PyTypeChecker
        FuncAnimation.__init__(self, fig, func, frames=self._frames(), init_func=init_func, fargs=fargs,
                               save_count=self.save_count, cache_frame_data=cache_frame_data, **kwargs)

    def _frames(self):
        while True:
            yield self.i
            self.i += 1

            if self.i == self.frames:
                self.i = 0

    def _pause_play(self, event):
        if self.going and event.key == " ":
            self.pause()
            self.going = False
        elif not self.going and event.key == " ":
            self.resume()
            self.going = True
        elif event.key == 'right':
            self.pause()
            self.going = False
            self._forward_step()
        elif event.key == 'left':
            self.pause()
            self.going = False
            self._backward_step()

    def _forward_step(self):
        self.i += 1
        if self.i == self.frames:
            self.i = 0
        self.func(self.i)
        self.fig.canvas.draw_idle()

    def _backward_step(self):
        self.i -= 1
        if self.i == -1:
            self.i = self.frames - 1
        self.func(self.i)
        self.fig.canvas.draw_idle()
