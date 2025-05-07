import os

from alonet import ALONET_ROOT

from .base_exporter import BaseTRTExporter, BaseONNXExporter
from .base_exporter_config import BaseONNXExporterConfig, BaseTRTExporterConfig

MS_DEFORM_IM2COL_PLUGIN_LIB = os.path.join(
    ALONET_ROOT, "torch2trt/plugins/ms_deform_im2col/build/libms_deform_im2col_trt.so"
)
