from collections import OrderedDict
import argparse
import torch
import os

from alonet import ALONET_ROOT
from alonet.torch2trt import BaseTRTExporter, SAMPLE_BILINEAR_PLUGIN_LIB, load_trt_custom_plugins
from alonet.raft import RAFT
from aloscene import Frame
from alonet.torch2trt.utils import get_nodes_by_op


def load_trt_plugins_raft():
    load_trt_custom_plugins(SAMPLE_BILINEAR_PLUGIN_LIB)


class RaftTRTExporter(BaseTRTExporter):
    def __init__(self, iters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iters = iters

    @staticmethod
    def replace_grid_sampler(graph, node, idx):
        img, grid = node.inputs[:2]

        # permute img and grid to channel_last convention
        img_tr = graph.layer(
            name=f"perm_img_{idx}",
            op="Transpose",
            inputs=[img],
            outputs=[f"img_tr_{idx}"],
            attrs=OrderedDict({"perm": [0, 2, 3, 1]}),
        )[0]

        # compute bilinear sampling with our custom plugin
        sampled_img = graph.layer(
            name=f"bilinear_sample_{idx}",
            op="BilinearSampler_TRT",
            inputs=[img_tr, grid],
            outputs=[f"sampled_img_{idx}"],
        )[0]

        # transpose back to channel first
        grid_sampler_output = node.outputs[0]
        graph.layer(
            name=f"perm_sampled_img{idx}",
            op="Transpose",
            inputs=[sampled_img],
            outputs=[grid_sampler_output],
            attrs=OrderedDict({"perm": [0, 3, 1, 2]}),
        )

        # remove the old grid_sampler node
        node.inputs.clear()
        node.outputs.clear()
        graph.nodes.remove(node)

    def adapt_graph(self, graph):

        grid_sample_nodes = get_nodes_by_op("grid_sampler", graph)
        for idx, node in enumerate(grid_sample_nodes):
            self.replace_grid_sampler(graph, node, idx)

        graph.toposort()
        graph.cleanup()

        return graph

    def prepare_sample_inputs(self):
        shape1, shape2 = self.input_shapes
        assert shape1 == shape2
        x1 = torch.rand(shape1, dtype=torch.float32)
        x2 = torch.rand(shape2, dtype=torch.float32)
        frame1 = Frame(x1, names=["B", "C", "H", "W"], normalization="01").norm_minmax_sym().batch()
        frame2 = Frame(x2, names=["B", "C", "H", "W"], normalization="01").norm_minmax_sym().batch()
        frame1 = frame1.as_tensor()
        frame2 = frame2.as_tensor()
        model_inputs = (frame1, frame2)
        model_kwargs = {"iters": self.iters, "flow_init": None, "only_last": True, "is_export_onnx": None}
        return model_inputs, model_kwargs


if __name__ == "__main__":
    load_trt_plugins_raft()
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--HW", type=int, nargs=2, default=[368, 496], help="Height and width of input image, default 368, 496"
    )
    parser.add_argument("--iters", type=int, default=12)
    # parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp32")
    BaseTRTExporter.add_argparse_args(parser)
    kwargs = vars(parser.parse_args())
    kwargs["onnx_path"] = os.path.join(ALONET_ROOT, f"raft-things_iters{kwargs['iters']}.onnx")
    kwargs["verbose"] = True
    shape = [1, 3] + kwargs.pop("HW")
    i_shapes = (shape, shape)
    i_names = ["frame1", "frame2"]
    model = RAFT(weights="raft-things")
    model.eval()

    exporter = RaftTRTExporter(model=model, input_shapes=i_shapes, input_names=i_names, **kwargs)
    # exporter.export_engine()
    exporter._torch2onnx()
    exporter._onnx2engine()
