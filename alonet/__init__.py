ALONET_ROOT = "/".join(__file__.split("/")[:-1])
from . import metrics
from . import common
from . import detr
from . import transformers
from . import raft

from . import deformable_detr
from . import callbacks

from . import detr_panoptic
from . import deformable_detr_panoptic

import pkg_resources
__version__ = pkg_resources.get_distribution("aloception").version
