import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib
from abc import ABC, abstractmethod
import numpy as np
import cv2


def adapt_text_size_to_frame(size, frame_size):
    """Adapt text size to frame.

    Parameters
    ----------
    size : float
        Size factor
    frame_size : tuple
        Frame size (height, width)
    """
    base_size_h = size * frame_size[1] / 1000
    base_size_w = size * frame_size[0] / 1000
    return base_size_h, base_size_w


def put_adapative_cv2_text(
    frame: np.array,
    frame_size: tuple,
    text: str,
    pos_x: float,
    pos_y: float,
    text_color=None,
    background_color=None,
    square_background=True,
    views_counter=None
):
    """Put Text on the given frame with adaptive size.

    Parameters
    ----------
    frame: np.ndarray
        Frame to put the text on
    frame_size: tuple
        Frame size (height, width)
    text: str
        Text to display
    pos_x: float
        Starting x position
    pos_y: float
        Starting y position
    text_color: tuple, optional
        Text RGB color, by default None.
    background_color: tuple, optional
        Background RGB color, by default None.
    square_background: bool, optional
        Add square background if True, by default True.
    views_counter: int, optional
        Number of views in the final image, by default None.
    """
    size_h, size_w = adapt_text_size_to_frame(1.0, frame_size)

    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, (size_h + size_w) / 2, -1)

    # Decrease text size if too long
    if views_counter:
        if w > (frame_size[1] / (views_counter / 2)):
            size_h /= 2
            size_w /= 2

            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, (size_h + size_w) / 2, -1)

    pos_x = int(pos_x)
    pos_y = int(pos_y)

    if square_background:
        cv2.rectangle(
            frame,
            (pos_x, int(pos_y - 2 * h)),
            (pos_x + w + 3, int(pos_y + 3 * h) ),
            (1, 1, 1) if background_color is None else background_color,
            -1,
        )

    cv2.putText(
        frame,
        text,
        (int(pos_x), int(pos_y + 2 * h)),
        cv2.FONT_HERSHEY_SIMPLEX,
        (size_h + size_w) / 2,
        (0, 0, 0) if text_color is None else text_color,
        1,
        cv2.LINE_AA,
    )


class View(object):

    CV = "cv"
    MATPLOTLIB = "matplotlib"

    def __init__(self, image, title=None, **kwargs):
        """The view class is used to store the information about one view and set
        the parameters that could be automaticly changed during the scene rendering.

        Parameters
        ----------
        image : np.ndarray
            The image on which to display the new information. The image values must be\
            between 0 and 1. If the image is None, we assume, the view object returns its \
            own image of the scene.
        title : str, optional
            Title of the view, by default None.
        """
        if image.shape[-1] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        self.image = image
        self.title = title

    def render(self, method="matplotlib", location: str = None, figsize=[6.4, 4.8]):
        """Render the current view using "matplotlib" or opencv ("cv").

        Parameters
        ----------
        method : str, optional
            One of ("cv", "matplotlib") or (View.CV, View.MATPLOTLIB). "matplotlib" by default.
        location : str, optional
            Do not render using matplotlib or opencv, but save the view into the given location.
            By default None.
        figsize : list, optional
            Figure size if using matplotlib, by default [6.4, 4.8].
            """
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
        """Extend the view with another view.

        Parameters
        ----------
        view : renderer.View
            View to add.
        """
        return View(Renderer.get_grid_view([self, view]))

    def save(self, location):
        """Save the current view into the given location.

        Parameters
        ----------
        location : str
            Path to the image to save.
        """
        cv2.imwrite(location, self.image[:, :, ::-1] * 255)


class Renderer(object):
    def __init__(self):
        self.renderer_to_fn = {"cv": self._cv_render, "matplotlib": self._matplotlib_render}
        self.out = None
        self.out_shape = None

    def _cv_render(self, view: np.ndarray):
        """Render using opencv.

        Parameters
        ----------
        view : np.ndarray
            View to render.
        """
        view = view[:, :, ::-1]
        cv2.imshow("Cortex view", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    def _matplotlib_render(self, view: np.ndarray):
        """Render using opencv.

        Parameters
        ----------
        view : np.ndarray
            View to render.
        """
        plt.imshow(view)
        plt.show()

    @classmethod
    def get_grid_view(cls, views: list, cell_grid_size=None, grid_size=None, add_title=True, **kwargs):
        """Get a grid of view from multiple view.

        Parameters
        ----------
        views : list
            List of views.
        cell_grid_size : tuple, optional
            If not None, the tuple values (height, width) will be used
            to set the size of the each grid cell of the display. By default None.
        grid_size : int, optional
            Number of views in the final grid, by default None.
        add_title : bool, optional
            Add title on the view, by default True.
        """
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

            if add_title:
                cls.add_title(line, (start, 0), views[v].title, len(views))

            v += 1
            if v % grid_size == 0 or v == len(views):
                lines.append(line)
                line = np.zeros((target_display_shape[0], target_display_shape[1] * int(grid_size), 3))

        return np.concatenate(lines, axis=0)

    @staticmethod
    def add_title(array, start, title, views_counter):
        """Add a box with view title.

        Parameters
        ----------
        array : np.ndarray
            Grid view array (will be modified inplace).
        start : tuple
            Top-left corner of title box.
        title : str
            Title of the view.
        views_counter : int
            Number of views.
        """
        if title is None:
            return
        else:
            put_adapative_cv2_text(array,
                                   array.shape,
                                   title,
                                   start[0],
                                   start[1],
                                   text_color=(1, 1, 1),
                                   background_color=(0, 0, 0),
                                   views_counter=views_counter)

    @classmethod
    def get_user_defined_grid_view(cls, views, add_title):
        """Create grid_view from list of list, without resizing views.

        view : list
            List of list of views with view[i][j] = view for line i, colomun j
        add_title : bool
            Add title on the view if True.
        """
        # create blank image
        Hf, Wf = 0, 0
        for line in views:
            hmax = max(v.image.shape[0] for v in line)
            w = sum(v.image.shape[1] for v in line)
            Hf += hmax
            Wf = max(Wf, w)
        final_view = np.zeros((Hf, Wf, 3))

        # write each view on the image
        y = 0
        for line in views:
            x = 0
            hmax = 0
            for v in line:
                h, w = v.image.shape[:2]
                final_view[y : y + h, x : x + w, :] = v.image

                if add_title:
                    cls.add_title(final_view, (x, y), v.title)
                x += w
                hmax = max(hmax, h)
            y += hmax
        return final_view

    def render(
        self,
        views: list,
        renderer: str = "cv",
        cell_grid_size=None,
        record_file: str = None,
        fps=30,
        grid_size=None,
        skip_views=False,
        add_title=True
    ):
        """Render a list of view using the given renderer.

        Parameters
        ----------
        views : list
            List of np.darray to display
        renderer : str, optional
            String to set the renderer to use. Can be either ("cv" or "matplotlib"). By default "cv".
        cell_grid_size : tuple, optional
            Tuple or None. If not None, the tuple values (height, width) will be used
            to set the size of the each grid cell of the display. If only one view is used,
            the view will be resize to the cell grid size.
        record_file : str, optional
            None by default. Used to save the rendering into one video.
        fps: int, optional
            FPS, by default 30.
        grid_size : int, optional
            Number of views in the grid, by default None.
        skip_views : bool, optional
            Skip views, in order to speed up the render process, by default False.
        add_title : bool, optional
            Add title on the view if True, by default True.
        """
        if renderer not in ["cv", "matplotlib"]:
            raise ValueError("The renderer must be one of the following:{}".format(self.renderer_to_fn.keys()))
        if skip_views and record_file is None:
            raise ValueError("When skip_views is desired, a record_file must be provided.")
        if isinstance(views[0], list):
            view = self.get_user_defined_grid_view(views, add_title)
        else:
            view = self.get_grid_view(views, cell_grid_size=cell_grid_size, grid_size=grid_size, add_title=add_title)
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
