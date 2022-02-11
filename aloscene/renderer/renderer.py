import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib
from abc import ABC, abstractmethod
import numpy as np


def adapt_text_size_to_frame(size, frame_size):
    base_size_h = size * frame_size[1] / 1000
    base_size_w = size * frame_size[0] / 1000
    return base_size_h, base_size_w


def put_adapative_cv2_text(
    frame: np.array, frame_size: tuple, text: str, pos_x: float, pos_y: float, color=None, square_background=True
):
    """Put Text on the given frame with adaptive size.

    Parameters
    ----------
    frame: np.array
        Frame to put the text on
    frame_size: np.array
        Frame size (height, width)
    text: str
        Text do display
    pos_x: int
    pos_y: int
    """
    size_h, size_w = adapt_text_size_to_frame(1.0, frame_size)
    c_size_h = int(size_w * 20)
    w_size_w = int(size_w * 20)

    pos_x = int(pos_x)
    pos_y = int(pos_y)
    if square_background:
        cv2.rectangle(
            frame,
            (pos_x - 3, pos_y - c_size_h - c_size_h),
            (pos_x + (w_size_w * (len(text) + 1)), pos_y + c_size_h),
            (1, 1, 1),
            -1,
        )

    cv2.putText(
        frame,
        text,
        (int(pos_x), int(pos_y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        (size_h + size_w) / 2,
        (0, 0, 0) if color is None else color,
        1,
        cv2.LINE_AA,
    )


class View(object):

    CV = "cv"
    MATPLOTLIB = "matplotlib"

    def __init__(self, image, title=None, **kwargs):
        """ The view class is used to store the information about one view and set
        the parameters that could be automaticly changed during the scene rendering.

        Parameters
        ----------
        image: (np.ndarray)
            The image on which to display the new information. The image values must be\
            between 0 and 1. If the image is None, we assume, the view object returns its \
            own image of the scene.
        title: str
            Title of the view """
        if image.shape[-1] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        self.image = image
        self.title = title

    def render(self, method="matplotlib", location: str = None, figsize=[6.4, 4.8]):
        """Render the current view using "matplotlib" or
        opencv ("cv")

        Parameters
        ----------
        method: str
            One of ("cv", "matplotlib") or (View.CV, View.MATPLOTLIB). "matplotlib" by default.
        location: str
            Do not render using matplotlib or opencv, but save the view into the given location"""
        if location is not None:
            plt.figure(figsize=figsize, tight_layout=True)
            plt.imshow(self.image)
            plt.savefig(location)
            plt.close()

        if method == self.MATPLOTLIB:
            plt.figure(figsize=figsize, tight_layout=True)
            plt.imshow(self.image)
            plt.show()
            plt.close()

        elif method == self.CV:
            cv2.imshow("Frame" if self.title is None else self.title, self.image[:, :, ::-1])
            if cv2.waitKey(1):
                return
        else:
            raise Exception(f"render method {method} is not handle")

    def add(self, view):
        """Extend the view with an other view"""
        return View(Renderer.get_grid_view([self, view]))


class Renderer(object):
    def __init__(self):
        self.renderer_to_fn = {"cv": self._cv_render, "matplotlib": self._matplotlib_render}
        self.out = None
        self.out_shape = None

    def _cv_render(self, view: np.ndarray):
        """Render using opencv"""
        view = view[:, :, ::-1]
        cv2.imshow("Cortex view", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    def _matplotlib_render(self, view: np.ndarray):
        """Render using opencv"""
        plt.imshow(view)
        plt.show()

    @staticmethod
    def get_grid_view(views: list, cell_grid_size=None, grid_size=None, **kwargs):
        """Get a grid of view from multiple view"""

        smallest_view = None
        target_display_shape = cell_grid_size
        if target_display_shape is None:
            for v in range(len(views)):
                if smallest_view is None or views[v].image.shape[0] * views[v].image.shape[1] < smallest_view:
                    smallest_view = views[v].image.shape[0] * views[v].image.shape[1]
                    target_display_shape = views[v].image.shape
        for v in range(len(views)):
            views[v].image = cv2.resize(views[v].image, (target_display_shape[1], target_display_shape[0]))

        nb = len(views)
        while not math.sqrt(nb).is_integer():
            nb += 1
        grid_size = math.sqrt(nb) if grid_size is None else grid_size

        lines = []
        v = 0
        line = np.zeros((target_display_shape[0], target_display_shape[1] * int(grid_size), 3))
        while v < len(views):
            start = int(v % grid_size) * target_display_shape[1]
            line[:, start : start + target_display_shape[1], :] = views[v].image

            if views[v].title is not None:
                cv2.rectangle(line, (start, 0), (start + 10 + (len(views[v].title) * 6), 10), (0, 0, 0), -1)
                cv2.putText(
                    line,
                    views[v].title,
                    (start + 10, 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0.8, 0.8, 0.8),
                    1,
                    cv2.LINE_AA,
                )
                pass

            v += 1
            if v % grid_size == 0 or v == len(views):
                lines.append(line)
                line = np.zeros((target_display_shape[0], target_display_shape[1] * int(grid_size), 3))

        return np.concatenate(lines, axis=0)

    def render(
        self,
        views: list,
        renderer: str = "cv",
        cell_grid_size=None,
        record_file: str = None,
        fps=30,
        grid_size=None,
        skip_views=False,
    ):
        """Render a list of view using the given renderer.

        Parameters
        ----------
        views : list
            List of np.darray to display
        renderer : str
            String to set the renderer to use. Can be either ("cv" or "matplotlib")
        cell_grid_size : tuple
            Tuple or None. If not None, the tuple values (height, width) will be used
            to set the size of the each grid cell of the display. If only one view is used,
            the view will be resize to the cell grid size.
        record_file : str
            None by default. Used to save the rendering into one video.
        skip_views : bool, optional
            Skip views, in order to speed up the render process, by default False
        """
        if renderer not in ["cv", "matplotlib"]:
            raise ValueError("The renderer must be one of the following:{}".format(self.renderer_to_fn.keys()))
        if skip_views and record_file is None:
            raise ValueError("When skip_views is desired, a record_file must be provided.")

        view = self.get_grid_view(views, cell_grid_size=cell_grid_size, grid_size=grid_size)
        if not skip_views:
            self.renderer_to_fn[renderer](view)
        if record_file is not None and self.out is None:
            self.out_shape = (view.shape[1], view.shape[0])
            self.out = cv2.VideoWriter(record_file, cv2.VideoWriter_fourcc(*"DIVX"), fps, self.out_shape)
        if record_file is not None:
            view = (view * 255).astype(np.uint8)
            view = view[:, :, ::-1]
            self.out.write(view)

    def save(self):
        """If render() was called with a record_file, then this method will
        save the final video on the system.
        """
        if self.out is None:
            raise Exception("You can't save a video without passing a record_file to the render() method")
        self.out.release()
