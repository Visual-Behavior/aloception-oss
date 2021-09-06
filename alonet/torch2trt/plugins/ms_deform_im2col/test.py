import torch

torch.manual_seed(3)
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import os
import ctypes
import re
import time

from alonet.torch2trt.TRTExecutor import TRTExecutor
from alonet.deformable_detr.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch, MSDeformAttnFunction

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

current_path = os.path.dirname(os.path.abspath(__file__))

MS_DEFORM_IM2COL_PLUGIN_LIB = "alonet/torch2trt/plugins/ms_deform_im2col/build/libms_deform_im2col_trt.so"
ctypes.CDLL(MS_DEFORM_IM2COL_PLUGIN_LIB)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list


def GiB(val):
    return val * 1 << 30


def camel_to_snake(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def get_trt_plugin(plugin_name):
    plugin = None
    for plugin_creator in PLUGIN_CREATORS:
        if plugin_creator.name == plugin_name:
            plugin = plugin_creator.create_plugin(camel_to_snake(plugin_name), None)
    if plugin is None:
        raise Exception(f"plugin {plugin_name} not found")
    return plugin


def build_test_engine(input_shape, dtype=trt.float32):
    num_level = input_shape["flatten_sampling_loc"][3]

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network:
        builder.max_batch_size = 1
        config = builder.create_builder_config()
        config.max_workspace_size = GiB(5)
        if dtype == trt.float16:
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        input_flatten_value = network.add_input(
            name="input_flatten_value", dtype=dtype, shape=input_shape["flatten_value"]
        )
        input_spatial_shapes = network.add_input(name="input_spatial_shapes", dtype=trt.int32, shape=(1, num_level, 2))
        input_start_index = network.add_input(name="input_start_index", dtype=trt.int32, shape=(1, num_level))
        input_flatten_sampling_loc = network.add_input(
            name="input_flatten_sampling_loc", dtype=dtype, shape=input_shape["flatten_sampling_loc"]
        )
        input_flatten_attn_weight = network.add_input(
            name="input_flatten_attn_weight", dtype=dtype, shape=input_shape["flatten_attn_weight"]
        )

        ms_deform_im2col_node = network.add_plugin_v2(
            inputs=[
                input_flatten_value,
                input_spatial_shapes,
                input_start_index,
                input_flatten_sampling_loc,
                input_flatten_attn_weight,
            ],
            plugin=get_trt_plugin("MsDeformIm2ColTRT"),
        )
        ms_deform_im2col_node.name = "ms_deform_im2col_node"
        ms_deform_im2col_node.get_output(0).name = "im2col_output"

        network.mark_output(ms_deform_im2col_node.get_output(0))

        return builder.build_engine(network, config)


def get_target_test_tensors(dtype=np.float32):
    test_dir = os.path.join(current_path, "test_tensors")
    test_tensors = {}
    test_shapes = {}
    for filename in os.listdir(test_dir):
        tensor_name = filename.split(".")[0]
        test_tensors[tensor_name] = np.load(os.path.join(test_dir, filename))

        if test_tensors[tensor_name].dtype == np.int64:
            test_tensors[tensor_name] = test_tensors[tensor_name].astype(np.int32)
        elif test_tensors[tensor_name].dtype == np.float32:
            test_tensors[tensor_name] = test_tensors[tensor_name].astype(dtype)

        test_shapes[tensor_name] = test_tensors[tensor_name].shape
    return test_tensors, test_shapes


def create_random_test_data(dtype=np.float32):
    N, M, D = 1, 8, 32
    Lq, L, P = 12000, 4, 4
    shapes = torch.as_tensor([(64, 64), (32, 32), (16, 16), (8, 8)], dtype=torch.long).cuda()
    level_start_index = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
    S = sum([(H * W).item() for H, W in shapes])
    value = torch.rand(N, S, M, D).cuda()
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    output_pytorch = ms_deform_attn_core_pytorch(value, shapes, sampling_locations, attention_weights).detach()
    # output_pytorch = MSDeformAttnFunction.apply(value, shapes, level_start_index, sampling_locations, attention_weights, 64).detach()
    test_tensor = {
        "flatten_value": value.cpu().numpy(),
        "spatial_shapes": shapes.cpu().numpy().astype(np.int32),
        "level_start_index": level_start_index.cpu().numpy().astype(np.int32),
        "flatten_sampling_loc": sampling_locations.cpu().numpy(),
        "flatten_attn_weight": attention_weights.cpu().numpy(),
        "output": output_pytorch.cpu().numpy(),
    }
    return test_tensor


def check_forward_equal(output_trt, output_torch):
    for i in range(len(output_trt.shape)):
        assert output_trt.shape[i] == output_torch.shape[i]
    abs_err = output_trt - output_torch
    rel_err = abs_err / output_torch
    max_abs_err = abs_err.max()
    mean_abs_err = abs_err.mean()
    max_rel_err = rel_err.max()
    mean_rel_err = rel_err.mean()
    print(f"Check forward equal: mean_abs_err: {mean_abs_err:.2e}\tmax_abs_err: {max_abs_err:.2e}")
    print(f"Check forward equal: mean_rel_err: {mean_rel_err:.2e}\tmax_rel_err: {max_rel_err:.2e}")


def main():
    # Get test data
    test_tensors = create_random_test_data()
    test_shapes = {key: test_tensors[key].shape for key in test_tensors}
    for key in test_tensors:
        print(key, test_tensors[key].shape, test_tensors[key].dtype, type(test_tensors[key]))

    # Build simple engine: input -> MsDeformIm2Col -> output
    engine = build_test_engine(test_shapes, dtype=trt.float32)
    trt_model = TRTExecutor(engine=engine)
    trt_model.print_bindings_info()

    # Run TensorRT engine
    trt_model.inputs[0].host = test_tensors["flatten_value"]
    trt_model.inputs[1].host = test_tensors["spatial_shapes"]
    trt_model.inputs[2].host = test_tensors["level_start_index"]
    trt_model.inputs[3].host = test_tensors["flatten_sampling_loc"]
    trt_model.inputs[4].host = test_tensors["flatten_attn_weight"]
    trt_model.execute()

    # Check output TensorRT vs Torch
    output_trt = trt_model.outputs[0].host
    output_torch = test_tensors["output"]
    print(output_trt.flatten())
    print(output_torch.flatten())
    check_forward_equal(output_trt, output_torch)


if __name__ == "__main__":
    main()
