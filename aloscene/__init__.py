ALOSCENE_ROOT = "/".join(__file__.split("/")[:-1])
from . import tensors
from .labels import Labels
from .camera_calib import CameraExtrinsic, CameraIntrinsic
from .mask import Mask
from .flow import Flow
from .depth import Depth
from .points_2d import Points2D
from .points_3d import Points3D
from .disparity import Disparity
from .pose import Pose
from .bounding_boxes_2d import BoundingBoxes2D
from .bounding_boxes_3d import BoundingBoxes3D
from .oriented_boxes_2d import OrientedBoxes2D
from .frame import Frame
from .tensors.spatial_augmented_tensor import SpatialAugmentedTensor

from .renderer import Renderer

def batch_list(tensors):
    return SpatialAugmentedTensor.batch_list(tensors)

_renderer = None
def render(
        views: list,
        renderer: str = "cv",
        size=None,
        record_file: str = None,
        fps=30,
        grid_size=None,
        skip_views=False,
    ):
    """Render a list of view.

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
    global _renderer
    _renderer = Renderer() if _renderer is None else _renderer

    _renderer.render(
        views=views,
        renderer=renderer,
        cell_grid_size=size,
        record_file=record_file,
        fps=fps,
        grid_size=grid_size,
        skip_views=skip_views
    )


def save_renderer():
    """If render() was called with a `record_file`, then this method will
    save the final video on the system. Warning: It is currently not possible
    to save multiple stream directly with aloception. Todo so, one can manually create multiple `Renderer`.
    """
    _renderer.save()
