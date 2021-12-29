"""Helper class for exporting PyTorch model to TensorRT engine
"""

import argparse
import os
import io
import torch
import onnx
import onnx_graphsurgeon as gs
from contextlib import ExitStack, redirect_stdout

from alonet.torch2trt.onnx_hack import get_scope_names, rename_nodes_, rename_tensors_, scope_name_workaround
from alonet.torch2trt import BaseTRTExporter
from aloscene import Frame


class PanopticTRTExporter(BaseTRTExporter):
    def __init__(self, *args, num_query: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_opset = None
        if self.input_names is None:  # Default inputs
            self.input_names = (
                "dec_outputs",
                "enc_outputs",
                "bb_lvl0_src_outputs",
                "bb_lvl1_src_outputs",
                "bb_lvl2_src_outputs",
                "bb_lvl3_src_outputs",
                "bb_lvl3_mask_outputs",
            )
        self.num_query = num_query  # Fix num_query to static value

    def adapt_graph(self, graph: gs.Graph):
        # return graph  # Not optimize
        from onnxsim import simplify

        # no need to modify graph
        model = onnx.load(self.onnx_path)
        check = False
        model_simp, check = simplify(model)
        # model_simp, check = simplify(model, input_shapes={"dec_outputs": [6, 1, 10, 256]})

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
        assert len(self.input_shapes) == 1, "Panoptic takes only 1 input"
        shape = self.input_shapes[0]
        x = torch.rand(shape, dtype=torch.float32)
        x = Frame(x, names=["C", "H", "W"]).norm_resnet()
        x = Frame.batch_list([x] * self.batch_size).to(self.device)
        tensor_input = self.model.detr(x)  # Get Detr outputs expected

        if self.num_query is not None:  # Fix static num queries
            tensor_input["dec_outputs"] = tensor_input["dec_outputs"][:, :, : self.num_query]

        # return (tuple(tensor_input[iname].contiguous() for iname in self.input_names),), {}
        return {iname: tensor_input[iname].contiguous() for iname in self.input_names}, {}

    def _torch2onnx(self):
        # Prepare dummy input for tracing
        inputs, kwargs = self.prepare_sample_inputs()

        # Get sample inputs/outputs for later sanity check
        with torch.no_grad():
            m_outputs = self.model(inputs, **kwargs)
        np_inputs = tuple(inputs[iname].cpu().numpy() for iname in self.input_names)
        np_m_outputs = {}
        output_names = (
            m_outputs._fields if hasattr(m_outputs, "_fields") else ["out_" + str(i) for i in range(len(m_outputs))]
        )
        for key, val in zip(output_names, m_outputs):
            if isinstance(val, torch.Tensor):
                np_m_outputs[key] = val.cpu().numpy()
        # print("Model output keys:", m_outputs.keys())

        # Export to ONNX
        with ExitStack() as stack:
            # context managers necessary to redirect stdout and modify export trace to print scope
            if self.use_scope_names:
                buffer = stack.enter_context(io.StringIO())
                stack.enter_context(redirect_stdout(buffer))
                stack.enter_context(scope_name_workaround())
            torch.onnx.export(
                self.model,  # model being run
                inputs,  # model input (or a tuple for multiple inputs)
                self.onnx_path,  # where to save the model
                export_params=True,  # store the trained parameter weights inside the model file
                opset_version=13,  # the ONNX version to export the model to
                do_constant_folding=self.do_constant_folding,  # whether to execute constant folding for optimization
                verbose=self.verbose or self.use_scope_names,  # verbose mandatory in scope names procedure
                input_names=self.input_names,  # the model's input names
                output_names=output_names,
                custom_opsets=self.custom_opset,
                enable_onnx_checker=True,
                operator_export_type=self.operator_export_type,
                # dynamic_axes={
                #     "dec_outputs": [2],  # Query dim must be dynamic
                #     "pred_masks": [1],  # With query dynamic, pred_masks is dynamic to.
                # },
            )

            if self.use_scope_names:
                onnx_export_log = buffer.getvalue()

        # rewrite onnx graph with new scope names
        if self.use_scope_names:
            # print(onnx_export_log)
            number2scope = get_scope_names(onnx_export_log, strict=False)
            graph = gs.import_onnx(onnx.load(self.onnx_path))
            graph = rename_tensors_(graph, number2scope, verbose=True)
            onnx.save(gs.export_onnx(graph), self.onnx_path)

        print("Saved ONNX at:", self.onnx_path)
        # empty GPU memory for later TensorRT optimization
        torch.cuda.empty_cache()
        return np_inputs, np_m_outputs


if __name__ == "__main__":
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
    BaseTRTExporter.add_argparse_args(parser)

    args = parser.parse_args()
    device = torch.device("cpu") if args.cpu else torch.device("cuda")
    input_shape = [3] + list(args.HW)
    pan_onnx_path = args.onnx_path or os.path.join(vb_fodler(), "weights", "detr-r50-panoptic", "panoptic-head.onnx")

    model = PanopticHead(
        DETR_module=DetrR50(num_classes=250, background_class=None),
        weights="detr-r50-panoptic",
        tracing=True,
        aux_loss=False,
    )
    model = model.eval().to(device)

    # 1. Export Detr engine
    print("[INFO] Exporting DETR engine...")
    args.onnx_path = os.path.join(os.path.split(pan_onnx_path)[0], "detr-r50.onnx")
    model.detr.tracing = True
    exporter = DetrTRTExporter(
        model=model.detr, input_shapes=(input_shape,), input_names=["img"], device=device, **vars(args)
    )
    exporter.export_engine()
    model.detr.tracing = False  # Required for next procedure

    # 2. Export PanopticHead engine
    print("[INFO] Exporting PanopticHead engine...")
    args.onnx_path = pan_onnx_path
    exporter = PanopticTRTExporter(model=model, input_shapes=(input_shape,), device=device, **vars(args))
    exporter.export_engine()
