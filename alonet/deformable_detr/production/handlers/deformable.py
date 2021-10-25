"""
ModelHandler defines a custom model handler.
"""

from ts.torch_handler.base_handler import BaseHandler
from alonet.deformable_detr.ops.functions import load_ops


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def initialize(self, context):
        super().initialize(context)
        self.initialized = True
        load_ops()
