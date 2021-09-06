# Requirements

```
TensorRT 8
cmake >= 3.8
```

Tips for setting LD_LIBRARY_PATH in conda env: [stackoverflow](https://stackoverflow.com/questions/46826497/conda-set-ld-library-path-for-env-only)

Python packages:
```
onnx >= 1.9
pycuda < 2021.1
```

# Usage
`alonet/torch2trt/base_exporter.py` defines a base class for exporting PyTorch model to TensorRT. To export engine, simply run `export_engine()` method.

Every model in `alonet` should inherit this class in a file `trt_exporter.py` and overload the following methods and attribute(s):
```
build_torch_model()
adapt_graph()
prepare_sample_inputs()
custom_opset
```

# Under the hood
The pseude code of `exporter.export_engine()`:
1. Build PyTorch model with `build_torch_model()` method
2. Export ONNX file by tracing with dummy inputs from `prepare_sample_inputs()` with `custom_opset` if needed
3. Modify ONNX file with `adapt_graph()` for TensorRT compability
4. Export TensorRT engine


# Create custom TensorRT plugin
- [Official guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending)
- [Toy example](https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html#uff_custom_plugin)


# Using custom plugins
If a model requires some TensorRT custom plugins, those plugins need to be registered everytime before exporting/executing engine.

General steps to implement a custom plugin in *aloception*:
1. Organize all sources in `alonet/torch2trt/plugins/my_plugin`
2. In `alonet/torch2trt/plugins/make.sh`, write command to build plugin
3. In `alonet/torch2trt/__init__.py`, define a path to built module for ease of use
4. In each alonet model `trt_exporter.py`, one should implement a function responsible for loading all necessary plugins using `alonet.torch2trt.load_trt_custom_plugins(...)`. This function is responsible for building, if needed, and loading a plugin.

All necessary plugins using by an engine must be loaded before exporting/executing. An example can be found in `alonet/deformable_detr/trt_exporter.py` and `sandbox/test_detr_tensorrt_inference.py`


# What to do if I have errors when exporting engine ?

Each model will need its own trick in order to adapt to TensorRT. The workflow is iterative:
1. Export ONNX to TensorRT
2. If error, find the nodes causing error in ONNX graph, understand the error. Mostly, error will be incompability between ONNX and TensorRT. 
3. To fix error, modify PyTorch code is the easiest way, if not possible, modify ONNX graph using `onnx_graphsurgeon`.
4. Repeat step 1

Toy examples for using `onnx_graphsurgeon` can be found in [the official repo](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/examples)

For PyTorch custom ops, follow this [instruction](https://github.com/onnx/tutorials/blob/master/PyTorchCustomOperator/README.md) to add custom ONNX ops. Then when exporting ONNX to TensorRT, we replace custom ONNX ops by the name of corresponding TensorRT plugin using also `onnx_surgery`. A toy example can be found [here](https://developer.nvidia.com/blog/estimating-depth-beyond-2d-using-custom-layers-on-tensorrt-and-onnx-models/)

# Example
- PRECISION either fp32, fp16 or mix
- Default HW is 1280 1920.
## Detr
To export engine:
```
python alonet/detr/trt_exporter.py [--HW H W] [--precision PRECISION] [--verbose] 
```

## Deformable Detr
To export engine:
```
python alonet/deformable_detr/trt_exporter.py [--HW H W] [--refinement] [--precision PRECISION] [--verbose]
```
Set --refinement for Deformable Detr with box refinement.
## Example inference script
To test engine:
```
python sandbox/test_detr_tensorrt_inference.py --engine /path/to/engine/ --image_path /path/to/test/image
```

# Inference time on V100
Time measured in ms for a 1280x1920 image

|Model|FP32|FP16|
|---|---|---|
|detr-r50|50|25
|deformable-detr-r50|100|55