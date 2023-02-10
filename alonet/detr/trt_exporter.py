"""Helper class for exporting PyTorch model to TensorRT engine
"""

import argparse
import os
import torch
import onnx_graphsurgeon as gs

from aloscene import Frame
from alonet.detr import DetrR50
from alonet.torch2trt import BaseTRTExporter


class DetrTRTExporter(BaseTRTExporter):
    def __init__(self, model_name="detr-r50", weights="detr-r50", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_opset = None

    def adapt_graph(self, graph: gs.Graph):
        from onnxsim import simplify
        import onnx
        from alonet.torch2trt.onnx_hack import rename_nodes_

        # no need to modify graph
        model = onnx.load(self.onnx_path)
        check = False
        model_simp, check = simplify(model)

        if check:
            print("\n[INFO] Simplified ONNX model validated. Graph optimized...")
            graph = gs.import_onnx(model_simp)
            graph.toposort()
            graph.cleanup()
        else:
            print("\n[INFO] ONNX model was not validated.")

        if self.use_scope_names:  # Rename nodes to correct profiling
            graph = rename_nodes_(graph, True)
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
    from alonet.common.pl_helpers import vb_folder

    # test script
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--HW", type=int, nargs=2, default=[1280, 1920], help="Height and width of input image, by default %(default)s"
    )
    parser.add_argument("--cpu", action="store_true", help="Compile model in CPU, by default %(default)s")
    BaseTRTExporter.add_argparse_args(parser)
    # parser.add_argument("--image_chw")
    args = parser.parse_args()
    if args.onnx_path is None:
        args.onnx_path = os.path.join(vb_folder(), "weights", "detr-r50", "detr-r50.onnx")
    device = torch.device("cpu") if args.cpu else torch.device("cuda")

    input_shape = [3] + list(args.HW)
    model = DetrR50(
        weights="detr-r50",
        tracing=True,
        aux_loss=False,
        # return_dec_outputs=True,
        # return_enc_outputs=True,
        # return_bb_outputs=True,
    )
    model = model.eval().to(device)
    exporter = DetrTRTExporter(
        model=model, input_shapes=(input_shape,), input_names=["img"], device=device, **vars(args)
    )

    exporter.export_engine()
