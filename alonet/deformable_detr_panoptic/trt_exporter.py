"""Helper class for exporting PyTorch model to TensorRT engine
"""

import argparse
import os
import torch
import onnx
import onnx_graphsurgeon as gs

from alonet.torch2trt.onnx_hack import rename_nodes_
from alonet.torch2trt import BaseTRTExporter
from aloscene import Frame


class PanopticTRTExporter(BaseTRTExporter):
    def __init__(self, *args, export_with_detr: bool = False, **kwargs):
        # Get default inputs
        input_names = kwargs.get("input_names", None)
        if input_names is None:
            if export_with_detr:
                kwargs["input_names"] = ("img",)
            else:
                kwargs["input_names"] = (
                    "dec_outputs",
                    "enc_outputs",
                    "bb_lvl0_src_outputs",
                    "bb_lvl1_src_outputs",
                    "bb_lvl2_src_outputs",
                    "bb_lvl3_src_outputs",
                    "bb_lvl3_mask_outputs",
                )
                kwargs["dynamic_axes"] = kwargs.get("dynamic_axes", None) or {"dec_outputs": {2: "num_queries"}}

        super().__init__(*args, **kwargs)
        self.custom_opset = None
        self.export_with_detr = export_with_detr

    def adapt_graph(self, graph: gs.Graph):
        # return graph  # Not optimize
        from onnxsim import simplify

        # no need to modify graph
        model = onnx.load(self.onnx_path)
        check = False
        if self.dynamic_axes is not None:
            model_simp, check = simplify(
                model,
                dynamic_input_shape=True,  # Choose optimal values for simplify
                input_shapes={key: val[1] for key, val in self.engine_builder.opt_profiles.items()},
            )
        else:
            model_simp, check = simplify(model)

        if check:
            print("\n[INFO] Simplified ONNX model validated. Graph optimized...")
            graph = gs.import_onnx(model_simp)
            graph.toposort()
            graph.cleanup()
        else:
            print("\n[INFO] ONNX model was not validated.")

        if self.use_scope_names:  # Rename nodes to correct profiling
            graph = rename_nodes_(graph, verbose=True)
        return graph

    def prepare_sample_inputs(self):
        assert len(self.input_shapes) == 1, "Panoptic takes only 1 input"
        shape = self.input_shapes[0]
        x = torch.rand(shape, dtype=torch.float32)
        x = Frame(x, names=["C", "H", "W"]).norm_resnet()
        x = Frame.batch_list([x] * self.batch_size).to(self.device)
        x = torch.cat((x.as_tensor(), x.mask.as_tensor()), dim=1)  # [b, 4, H, W]

        if self.export_with_detr:  # Image as input = export detr + panoptic
            tensor_input = (x,)
        else:  # dict as input = export only panoptic
            with torch.no_grad():
                tensor_input = self.model.detr_forward(x)  # Get Detr outputs expected

            tensor_input = {iname: tensor_input[iname].contiguous() for iname in self.input_names}
        return tensor_input, {"is_export_onnx": None}


if __name__ == "__main__":
    raise NotImplementedError("Not implemented yet")
    from alonet.common.weights import vb_fodler
    from alonet.detr_panoptic import PanopticHead
    from alonet.detr import DetrR50
    from alonet.detr.trt_exporter import DetrTRTExporter

    # test script
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--HW", type=int, nargs=2, default=[1280, 1920], help="Height and width of input image, by default %(default)s"
    )
    parser.add_argument("--cpu", action="store_true", help="Compile model in CPU, by default %(default)s")
    parser.add_argument(
        "--split_engines",
        action="store_true",
        help="Save DETR/Panoptic engines in different files, by default %(default)s",
    )
    BaseTRTExporter.add_argparse_args(parser)

    args = parser.parse_args()
    device = torch.device("cpu") if args.cpu else torch.device("cuda")
    input_shape = [3] + list(args.HW)
    pan_onnx_path = os.path.join(vb_fodler(), "weights", "detr-r50-panoptic")
    if args.split_engines:
        pan_onnx_path = os.path.join(pan_onnx_path, "panoptic-head.onnx")
    else:
        pan_onnx_path = os.path.join(pan_onnx_path, "detr-r50-panoptic.onnx")
    pan_onnx_path = args.onnx_path or pan_onnx_path

    model = PanopticHead(
        DETR_module=DetrR50(num_classes=250, background_class=None),
        weights="detr-r50-panoptic",
        tracing=True,
        aux_loss=False,
        return_pred_outputs=not args.split_engines,
    )
    model = model.eval().to(device)

    if args.split_engines:
        # 1. Export Detr engine
        print("\n[INFO] Exporting DETR engine...")
        args.onnx_path = os.path.join(os.path.split(pan_onnx_path)[0], "detr-r50.onnx")
        exporter = DetrTRTExporter(
            model=model.detr, input_shapes=(input_shape,), input_names=["img"], device=device, **vars(args)
        )
        exporter.export_engine()

        print("\n\n[INFO] Exporting PanopticHead engine...")

    # 2. Export PanopticHead engine
    args.onnx_path = pan_onnx_path
    profile = {"dec_outputs": [(6, 1, 1, 256), (6, 1, 10, 256), (6, 1, 100, 256)]} if args.split_engines else None
    exporter = PanopticTRTExporter(
        model=model,
        input_shapes=(input_shape,),
        export_with_detr=not args.split_engines,
        device=device,
        opt_profiles=profile,  # Example of profile for dynamic num of queries
        **vars(args),
    )
    exporter.export_engine()
