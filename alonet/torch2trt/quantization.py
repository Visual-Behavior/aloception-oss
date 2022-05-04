import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules, calib
from pytorch_quantization.tensor_quant import QuantDescriptor


class QuantizedModel:
    """Models quantization inteface
    Transforms each model Layer to QuantLayer for quantization aware training.
    Performs calibration for post training quantization.
    
    Examples
    --------
        >>> from ... import Model
        >>> class QModel(QuantizedModel, Model):
        .       def __init__(self, ..., **kwargs):
        .           ## set QuantizedLayers description first
        .           self.set_default_desc()
        .           super(QModel).__init__(**kwargs)
        >>> quant_model = QModel(...)
        >>> ## for QAT just fine tune the model as layers are converted
        >>> print(quant_model.cuda())
        >>> ## for PTQ
        >>> class CalibDataset:
        >>>     def __init__(self):
        >>>         self.ds1 = Dataset1(...)    ## iterable dataset
        >>>         self.ds2 = Dataset2(...)    ## iterable dataset
        >>>         self.length = min(len(self.ds1), len(self.ds2))
        >>>         self.model_kwargs = dict(...)
        >>>
        >>>     def __getitem__(self, idx):
        >>>         return (self.ds1[idx].as_tensor(), self.ds2[idx].as_tensor()), self.model_kwargs
        >>>
        >>>     def __len__(self):
        >>>         return self.length
        >>>
        >>> calib_data = CalibDataset()
        >>> quant_model.calibrate(calib_data=calib_data, max_samples=10)
        
    """
    def set_default_desc(self,  calib_method="histogram", per_channel_quantization=True, **kwargs):
        """Converts Layers to QuantLayers
        
        Parameters
        ----------
            calib_method: str
                method to use to compute maximum ("histogram" | "max").
            per_channel_quantization: bool.
                either to calculate quzantization parameters for each channel or for the entire layer.
        
        raises
        ------
            ValueError
                When calib_method is not one of "histogram" or "max".
            
        """
        if calib_method not in ["histogram", "max"]:
            ValueError("Unknown calibration method, should be max | histogram")
            
        if per_channel_quantization:
            quant_desc_input = QuantDescriptor(calib_method=calib_method, **kwargs)
        else:
            quant_desc_input = QuantDescriptor(calib_method=calib_method, axis=None, **kwargs)
        quant_desc_weight = QuantDescriptor(calib_method=calib_method, **kwargs)

        ## input
        quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

        ## weights
        quant_nn.QuantConvTranspose2d.set_default_quant_desc_weight(quant_desc_weight)
        quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
        quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)

        ## convert Layers to QuantLayers
        quant_modules.initialize()

    def collect_stats(self, calib_data, max_samples=None):
        """Feed data to the network and collect statistic
        
        Parameters
        ----------
            calib_data: Iterable batched dataset
                calibration data.
            max_samples: int
                maximum samples to use.
        """
        max_samples = len(calib_data) if max_samples is None else min(max_samples, len(calib_data))
        # Enable calibrators
        for name, module in self.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        print('collecting stats...')
        for i in tqdm(range(max_samples)):
            args, kwargs = calib_data[i]
            self(*args, **kwargs)
            if i + 1>= max_samples:
                break

        # Disable calibrators
        for name, module in self.named_modules():
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
                        ## setting strict to False for optimized model configuration
                        module.load_calib_amax(strict=False)
                    else:
                        ## setting strict to False for optimized model configuration
                        module.load_calib_amax(strict=False, **kwargs)
                print(F"Calibrator: {module._calibrator}")
                print(F"{name:40}: {module}")

    def calibrate(
            self,
            calib_data,
            method="percentile",
            percentile=99.99,
            max_samples=10,
            ):
        """Calibrates the model
        
        Parameters
        ----------
            calib_data: Iterable dataset.
                Calibration dataset.
            method: int
                Set when description calibrator is HistogramCalibrator. one of (mse | percentile).
            percentile: float
                Calibration percentile if calibrator is HistogramCalibrator. Ignored otherwise.
            max_samples: int
                Max samples to use for calibration.
        
        Raises
        ------
            ValueError
                When method is not "mse" or "percentile".

        """
        if method not in ["mse", "percentile"]:
            raise ValueError("Method should be \"mse\" | \"percentile\"")
        with torch.no_grad():
            self.collect_stats(calib_data=calib_data, max_samples=max_samples)
            self.compute_amax(method=method, percentile=percentile)
