import torch
import numpy as np
import torch.nn as nn

from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor


class QuantizedModel:
    """Models quantization inteface : Transforms each model Layer to QuantLayer for quantization aware training
    ConvTranspose2d ~ QuantConvTransposed2d, Conv2d ~ QuantConv2D, Linear ~ QuantLinear
    
    Parameters
    ----------
        calib_data: Iterable batched dataset
            calibration data.
        per_channel_quantization: bool.
            either to quantize each channel or not.
        method: str
            method to use to compute maximum ("mse", "entropy", "max").
        max_samples: int
            maximum samples to use.
    
    Exemples
    --------
        >>> from ... import Model
        >>> class QModel(QuantizedModel, Model):
        .       def __init__(self, ..., **kwargs):
        .           ## set QuantizedLayers description first
        .           self.set_default_desc()
        .           super(QModel).__init__(**kwargs)
        >>> quant_model = QModel(...)
        >>> quant_model.calibrate()
        >>> print(quant_model.cuda())
        
    """
    def __init__(
            self,
            calib_data,
            method="max",
            max_samples=20,
            calib_method='histogram',
            per_channel_quantization=True,
            **kwargs,
            ):
        self.method = method
        self.calib_data = calib_data
        self.max_samples = max_samples
        self.calib_method = calib_method
        self.per_channel_quantization = per_channel_quantization
    
    def set_default_desc(self, calib_method='histogram', per_channel_quantization=True):
        quant_desc_weight = QuantDescriptor(calib_method=calib_method)
        if per_channel_quantization:
            quant_desc_input = QuantDescriptor(calib_method=calib_method)
        else:
            quant_desc_input = QuantDescriptor(calib_method=calib_method, axis=None)

        quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantConvTranspose2d.set_default_quant_desc_weight(quant_desc_weight)

        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)

        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)

        ## convert Layers to QuantLayers
        quant_modules.initialize()

        ## set fake quantization to True before torch2onnx
        quant_nn.TensorQuantizer.use_fb_fake_quant = True
    
    def collect_stats(self):
        """Feed data to the network and collect statistic"""
        # Enable calibrators
        for name, module in self.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        for i, image in enumerate(self.calib_data): ## FIXME
            if isinstance(image, np.ndarray):
                image = np.expand_dims(image, axis=0)
            self(image)
            if i >= self.max_samples:
                break

        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()
            
    def compute_amax(self, **kwargs):
        # Load calib result
        for name, module in self.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(**kwargs)

    def calibrate(self):
        """Calibrates the model"""
        with torch.no_grad():
            self.collect_stats()
            self.compute_amax(method=self.method)
