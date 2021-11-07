"""Helper class for exporting PyTorch model to TensorRT engine
"""

import argparse
import os
import torch
import tensorrt as trt
import ctypes
import numpy as np
import onnx_graphsurgeon as gs

from torch.onnx import register_custom_op_symbolic

from aloscene import Frame
from alonet.deformable_detr import DeformableDetrR50, DeformableDetrR50Refinement
from alonet.torch2trt import BaseTRTExporter, MS_DEFORM_IM2COL_PLUGIN_LIB, load_trt_custom_plugins
from alonet.torch2trt.utils import get_nodes_by_op

CUSTOM_OP_VERSION = 9


def symbolic_ms_deform_attn_forward(
    g, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step
):
    return g.op(
        "alonet_custom::ms_deform_attn_forward",
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    )


register_custom_op_symbolic(
    "alonet_custom::ms_deform_attn_forward", symbolic_ms_deform_attn_forward, CUSTOM_OP_VERSION
)


def load_trt_plugins_for_deformable_detr():
    load_trt_custom_plugins(MS_DEFORM_IM2COL_PLUGIN_LIB)


class DeformableDetrTRTExporter(BaseTRTExporter):
    def __init__(self, model_name="deformable-detr-r50", weights="deformable-detr-r50", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights
        self.do_constant_folding = False
        self.custom_opset = {"alonet_custom": 1}

    def adapt_graph(self, graph: gs.Graph):
        # batch_size = graph.inputs[0].shape[0] # test

        # ======= Add nodes for MsDeformIm2ColTRT ===========
        # Replace ms_deform_attn_forward nodes by MsDeformIm2ColTRT
        # which is the custom plugin in TensorRT
        im2col_nodes = get_nodes_by_op("ms_deform_attn_forward", graph)

        def handle_ops_MsDeformIm2ColTRT(graph: gs.Graph, node: gs.Node):
            inputs = node.inputs
            inputs.pop()  # The last input is im2col_step = 64 (constant), our TRT plugin doesn't fully support this attribute
            outputs = node.outputs
            graph.layer(op="MsDeformIm2ColTRT", name=node.name + "_trt", inputs=inputs, outputs=outputs)

        for n in im2col_nodes:
            handle_ops_MsDeformIm2ColTRT(graph, n)
            # Detach old node from graph
            n.inputs.clear()
            n.outputs.clear()
            graph.nodes.remove(n)

        # ====== Handle Clip ops =======
        # min, max input in Clip must be Constant while parsing ONNX to TensorRT
        clip_nodes = get_nodes_by_op("Clip", graph)

        def handle_op_Clip(node: gs.Node):
            max_constant = np.array(np.finfo(np.float32).max, dtype=np.float32)
            print(node.inputs[1].i().inputs[0].attrs.keys())
            print(node)
            if "value" in node.inputs[1].i().inputs[0].attrs:
                min_constant = node.inputs[1].i().inputs[0].attrs["value"].values.astype(np.float32)
                if len(node.inputs[2].inputs) > 0:
                    max_constant = node.inputs[2].i().inputs[0].attrs["value"].values.astype(np.float32)
            elif "to" in node.inputs[1].i().inputs[0].attrs:
                min_constant = np.array(np.finfo(np.float32).min, dtype=np.float32)
            else:
                raise Exception("Error")
            node.inputs.pop(1)
            node.inputs.insert(1, gs.Constant(name=node.name + "_min", values=min_constant))
            node.inputs.pop(2)
            node.inputs.insert(2, gs.Constant(name=node.name + "_max", values=max_constant))

        for n in clip_nodes:
            handle_op_Clip(n)

        # ===== Handle Slice ops ======
        # axes input must be Constant
        slice_nodes = get_nodes_by_op("Slice", graph)

        def handle_op_Slice(node: gs.Node):
            axes_input = node.inputs[3].inputs[0]
            if axes_input.op == "Unsqueeze":
                axes_constant = node.inputs[3].i().inputs[0].attrs["value"].values
                node.inputs.pop(3)
                node.inputs.insert(3, gs.Constant(name=node.name + "_axes", values=axes_constant))

        for n in slice_nodes:
            handle_op_Slice(n)

        graph.toposort()
        graph.cleanup()
        return graph

    def prepare_sample_inputs(self):
        assert len(self.input_shapes) == 1, "DETR takes only 1 input"
        shape = self.input_shapes[0]
        x = torch.rand(shape, dtype=torch.float32)
        x = Frame(x, names=["C", "H", "W"]).norm_resnet()
        x = Frame.batch_list([x] * self.batch_size).to(self.device)
        tensor_input = (x.as_tensor(), x.mask.as_tensor())
        tensor_input = torch.cat(tensor_input, dim=1)  # [b, 4, H, W]
        return (tensor_input,), {"is_export_onnx": None}


if __name__ == "__main__":
    # test script
    from alonet.common.weights import vb_fodler

    load_trt_plugins_for_deformable_detr()
    device = torch.device("cuda")

    parser = argparse.ArgumentParser()
    parser.add_argument("--refinement", action="store_true", help="If set, use box refinement")
    parser.add_argument(
        "--HW", type=int, nargs=2, default=[1280, 1920], help="Height and width of input image, default 1280 1920"
    )
    BaseTRTExporter.add_argparse_args(parser)
    parser.add_argument("--image_chw")
    args = parser.parse_args()

    if args.refinement:
        model_name = "deformable-detr-r50-refinement"
        model = DeformableDetrR50Refinement(weights=model_name, aux_loss=False).eval()
    else:
        model_name = "deformable-detr-r50"
        model = DeformableDetrR50(weights=model_name, aux_loss=False).eval()

    if args.onnx_path is None:
        args.onnx_path = os.path.join(vb_fodler(), "weights", model_name, model_name + ".onnx")

    input_shape = [3] + list(args.HW)

    exporter = DeformableDetrTRTExporter(
        model=model, weights=model_name, input_shapes=(input_shape,), input_names=["img"], device=device, **vars(args)
    )

    exporter.export_engine()
