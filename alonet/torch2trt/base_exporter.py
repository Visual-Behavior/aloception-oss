import io
import os
import time
from typing import Dict, List, Tuple, Union


import torch
import numpy as np


try:
    import onnx
    import onnx_graphsurgeon as gs
    import tensorrt as trt
    import pycuda.driver as cuda
    prod_package_error = None
except Exception as e:
    prod_package_error = e
    pass


from contextlib import redirect_stdout, ExitStack
from alonet.torch2trt.onnx_hack import scope_name_workaround, get_scope_names, rename_tensors_
from alonet.torch2trt import TRTEngineBuilder, TRTExecutor, utils
from alonet.torch2trt.utils import get_nodes_by_op, rename_nodes_



class BaseTRTExporter:
    """
    Base class for exporting PyTorch model to TensorRT engine.
    Child class must implement the following methods/attributes:
    - adapt_graph()
    - prepare_sample_inputs()
    - custom_opset

    Workflow:
    ---------
    PyTorch model ----> ONNX -----(necessary graph modification)-----> TensorRT engine
    """

    def __init__(
        self,
        model: torch.nn.Module,
        onnx_path: str,
        input_shapes: tuple = ([3, 1280, 1920]),
        input_names: list = None,
        batch_size: int = 1,
        precision: str = "fp32",
        do_constant_folding: bool = True,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False,
        use_scope_names: bool = False,
        operator_export_type=None,
        dynamic_axes: Union[Dict[str, Dict[int, str]], Dict[str, List[int]]] = None,
        opt_profiles: Dict[str, Tuple[List[int]]] = None,
        skip_adapt_graph=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        model : torch.nn.Module
            a model loaded with trained weights
        onnx_path : str
            Onnx file path which will be exported.
            Example: /abc/xyz/my_model.onnx
        input_shapes : tuple of tuple/list, default ([3, 1280, 1920], )
            input shape must be specified when export model
        input_names : list of str
        batch_size : int, default 1
        precision : str
            TRT engine precision, either fp32, fp16, mix
            mix precision between fp32 and fp16 allow TensorRT more liberty
            to find the best combination optimization in term of execution time.
        do_constant_folding : bool, default True
            Optimized ONNX graph if True. Sometimes this optimization will make
            the ONNX graph modification more complicated.
        verbose : bool, default False
            Print out everything. Good for debugging.
        dynamic_axes : Union[Dict[str, Dict[int, str]], Dict[str, List[int]]], by default None
            Axes of tensors that will be dynamics (not shape specified), by default None. See
            `https://pytorch.org/docs/stable/onnx.html#functions <torch.onnx.export>`_.
        opt_profiles : Dict[str, Tuple[List[int]]], by default None
            Optimization profiles (one by each dynamic axis).
        operator_export_type: torch.onnx.OperatorExportTypes

        Raises
        ------
        Exception
            * Model must be instantiated with attr:`tracing` = True
            * If :attr:`dynamic_axes` is desired, :attr:`opt_profiles` must be provided with sames keys as
              :attr:`dynamic_axes`.
        """
        if prod_package_error is not None:
            raise prod_package_error

        if hasattr(model, 'tracing'):
            assert model.tracing, "Model must be instantiated with tracing=True"

        self.model = model
        self.input_names = input_names
        self.onnx_path = onnx_path
        self.do_constant_folding = do_constant_folding
        self.input_shapes = input_shapes
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = device
        self.precision = precision
        self.custom_opset = None  # to be redefine in child class if needed
        self.use_scope_names = use_scope_names
        self.operator_export_type = operator_export_type
        self.skip_adapt_graph = skip_adapt_graph
        if dynamic_axes is not None:
            assert opt_profiles is not None, "If dynamic_axes are to be used, opt_profiles must be provided"
            assert isinstance(dynamic_axes, dict)
            assert opt_profiles.keys() == dynamic_axes.keys(), "dynamic_axes and opt_profiles must have same keys"
        self.dynamic_axes = dynamic_axes
        # ===== Initiate Trt Engine builder
        onnx_dir = os.path.split(onnx_path)[0]
        onnx_file_name = os.path.split(onnx_path)[1]
        model_name = onnx_file_name.split(".")[0]

        if not self.skip_adapt_graph:
            self.adapted_onnx_path = os.path.join(onnx_dir, "trt_" + onnx_file_name)
        else:
            self.adapted_onnx_path = os.path.join(onnx_dir, onnx_file_name)

        self.engine_path = os.path.join(onnx_dir, model_name + f"_{precision.lower()}.engine")

        if self.verbose:
            trt_logger = trt.Logger(trt.Logger.VERBOSE)
        else:
            trt_logger = trt.Logger(trt.Logger.WARNING)

        self.engine_builder = TRTEngineBuilder(self.adapted_onnx_path, logger=trt_logger, opt_profiles=opt_profiles)

        if precision.lower() == "fp32":
            pass
        elif precision.lower() == "fp16":
            self.engine_builder.FP16_allowed = True
            self.engine_builder.strict_type = True
        elif precision.lower() == "mix":
            self.engine_builder.FP16_allowed = True
            self.engine_builder.strict_type = False
        else:
            raise Exception(f"precision {precision} not supported")

    def build_torch_model(self):
        """Build PyTorch model and load weight with the given name

        Returns
        -------
        model: torch.nn.Module
        """
        pass
        raise Exception("Child class should implement this method")


    def adapt_graph(self, graph):
        """Modify ONNX graph to ensure compability between ONNX and TensorRT

        Returns
        -------
        graph: onnx_graphsurgeon.Graph
        """
        return graph

    def _adapt_graph(self, graph):
        """Modify ONNX graph to ensure compability between ONNX and TensorRT

        Returns
        -------
        graph: onnx_graphsurgeon.Graph
        """
        clip_nodes = get_nodes_by_op("Clip", graph)
        def handle_op_Clip(node: gs.Node):
            max_constant = np.array(np.finfo(np.float32).max, dtype=np.float32)
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

        from onnxsim import simplify
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


        # Call the child class for specific graph adapation
        graph = self.adapt_graph(graph)
        return graph

    def prepare_sample_inputs(self) -> Tuple[Tuple[torch.Tensor], Dict[str, Union[torch.Tensor, None]]]:
        """
        Prepare sample inputs for future sanity check
        as well as to define input shapes for ONNX and TensorRT.
        Because alonet use aloscence AugmentedTensor API
        which is not supported when exporting ONNX,
        so this method should return 2 tuples, 1 for

        Returns
        -------
        inputs: tuple/list/dictionary of torch.Tensor
            model input tensors
        kwargs: Union[Dict[str: torch.Tensor], None]
            additional argument for model.forward if needed
        """
        pass
        raise Exception("Child class should implement this method")

    def _torch2onnx(self):
        """Export PyTorch model to ONNX file.
        Return sample inputs/outputs for sanity check

        Note: is_export_onnx=None mean True,
        because torch.onnx.export support only torch.Tensor or None for forward method.

        Returns
        -------
        sample_inputs: tuple[np.ndarray]
        sample_outputs: dict[str: np.ndarray]

        """
        if prod_package_error is not None:
            raise prod_package_error

        # Prepare dummy input for tracing
        inputs, kwargs = self.prepare_sample_inputs()

        # Get sample inputs/outputs for later sanity check
        if isinstance(inputs, dict):
            with torch.no_grad():
                m_outputs = self.model(inputs, **kwargs)

            # Prepare inputs for torch.export.onnx and sanity check
            np_inputs = tuple(np.array(inputs[iname].cpu()) for iname in inputs)
            inputs = (inputs,)
        else:
            with torch.no_grad():
                m_outputs = self.model(*inputs, **kwargs)

            # Prepare inputs for torch.export.onnx and sanity check
            np_inputs = tuple(np.array(i.cpu()) for i in inputs)
        inputs = (*inputs, kwargs)

        onames = m_outputs._fields if hasattr(m_outputs, "_fields") else [f"out_{i}" for i in range(len(m_outputs))]
        np_m_outputs = {key: val.cpu().numpy() for key, val in zip(onames, m_outputs) if isinstance(val, torch.Tensor)}
        # print("Model input shapes:", [val.shape for val in np_inputs])
        # print("Model output keys:", np_m_outputs.keys(), "shapes:", [val.shape for val in np_m_outputs.values()])

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
                output_names=onames,
                custom_opsets=self.custom_opset,
                enable_onnx_checker=True,
                operator_export_type=self.operator_export_type,
                dynamic_axes=self.dynamic_axes,
            )

            if self.use_scope_names:
                onnx_export_log = buffer.getvalue()

        # rewrite onnx graph with new scope names
        if self.use_scope_names:
            # print(onnx_export_log)
            number2scope = get_scope_names(onnx_export_log, strict=False)
            graph = gs.import_onnx(onnx.load(self.onnx_path))
            graph = rename_tensors_(graph, number2scope, verbose=True)
            graph = rename_nodes_(graph, True)
            onnx.save(gs.export_onnx(graph), self.onnx_path)

        print("Saved ONNX at:", self.onnx_path)
        # empty GPU memory for later TensorRT optimization
        torch.cuda.empty_cache()
        return np_inputs, np_m_outputs

    def _onnx2engine(self, **kwargs):
        """
        Export TensorRT engine from an ONNX file

        Returns
        -------
        engine: tensorrt.ICudaEngine
        """
        if prod_package_error is not None:
            raise prod_package_error

        if not self.skip_adapt_graph:
            graph = gs.import_onnx(onnx.load(self.onnx_path))
            graph.toposort()

            # === Modify ONNX graph for TensorRT compability
            graph = self._adapt_graph(graph, **kwargs)
            utils.print_graph_io(graph)
            # === Export adapted onnx for TRT engine
            onnx.save(gs.export_onnx(graph), self.adapted_onnx_path)

        # === Build engine
        self.engine_builder.export_engine(self.engine_path)
        return self.engine_builder.engine

    def sanity_check(self, engine, sample_inputs, sample_outputs):
        if self.precision.lower() == "fp32":
            threshold = 1e-4
        else:
            threshold = 1e-1
        check = True
        # Get engine info
        model = TRTExecutor(engine, stream=cuda.Stream())
        model.print_bindings_info()
        # Prepare engine inputs
        for i in range(len(sample_inputs)):
            model.inputs[i].host = np.array(sample_inputs[i]).astype(model.inputs[i].dtype)
        # GPU warm up
        [model.execute() for i in range(3)]
        # Time engine inference
        N = 20
        tic = time.time()
        [model.execute() for i in range(N)]
        toc = time.time()
        # Check engine outputs with sample outputs
        m_outputs = model.execute()
        print("==== Absolute / relavtive error:")
        for out in m_outputs:
            print('out', m_outputs[out])
            diff = m_outputs[out].astype(float) - sample_outputs[out].astype(float)
            abs_err = np.abs(diff)
            rel_err = np.abs(diff / (sample_outputs[out] + 1e-6))  # Avoid div by zero
            print(out)
            print(f"\tmean: {abs_err.mean():.2e}\t{rel_err.mean():.2e}")
            print(f"\tmax: {abs_err.max():.2e}\t{rel_err.max():.2e}")
            print(f"\tstd: {abs_err.std():.2e}\t{rel_err.std():.2e}")
            check = check & (rel_err.mean() < threshold)

        print(f"Engine execution time: {(toc - tic)/N*1000:.2f} ms")
        # if check:
        #     print("Sanity check passed")
        # else:
        #     print("Sanity check failed")
        return check

    def export_engine(self):
        sample_inputs, sample_outputs = self._torch2onnx()
        engine = self._onnx2engine()
        self.sanity_check(engine, sample_inputs, sample_outputs)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("tensorrt_exporter")
        parser.add_argument(
            "--onnx_path",
            type=str,
            default=None,
            help="/path/onnx/will/be/exported, by default set as ~/.aloception/weights/MODEL/MODEL.onnx",
        )
        parser.add_argument("--skip_adapt_graph", action="store_true", help="Skip the adapt graph")
        parser.add_argument("--batch_size", type=int, default=1, help="Engine batch size, default = 1")
        parser.add_argument("--precision", type=str, default="fp32", help="fp32/fp16/mix, default FP32")
        parser.add_argument("--verbose", action="store_true", help="Helpful when debugging")
        parser.add_argument(
            "--use_scope_names",
            action="store_true",
            help="Save scope names in onnx, to get profiles in inference by default %(default)s",
        )
        return parent_parser
