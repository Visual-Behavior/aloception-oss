"""
ModelHandler defines a custom model handler.
"""

from ts.torch_handler.base_handler import BaseHandler


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def initialize(self, context):
        super().initialize(context)
        self.initialized = True
