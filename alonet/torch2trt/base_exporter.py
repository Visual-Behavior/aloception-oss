from typing import Dict, List, Tuple, Union
import time
import os
import io
import numpy as np
import torch
import warnings

try:
    import onnx_graphsurgeon as gs
    import pycuda.driver as cuda
    import tensorrt as trt
    import onnx

    prod_package_error = None
except Exception as e:
    prod_package_error = e
    pass


from alonet.torch2trt.onnx_hack import scope_name_workaround, get_scope_names, rename_tensors_
from alonet.torch2trt import TRTEngineBuilder, TRTExecutor, utils
from alonet.torch2trt.utils import get_nodes_by_op, rename_nodes_
from contextlib import redirect_stdout, ExitStack


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

    PROFILING_TIME = 20

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
        profiling_verbosity: int = 0,
        calibrator=None,
        opset_version: int = 13,
        ignore_adapt_graph: bool = False,
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
        calibrator : torch2trt.calibrator.BaseCalibrator
            Quantization calibrator.
        profiling_verbosity : int
            Profiling verbosity in NVTX annotations and the engine inspector (Default 0)
                0 : LAYER_NAMES_ONLY (Print only the layer names. This is the default setting).
                1 : NONE (Do not print any layer information).
                2 : DETAILED : (Print detailed layer information including layer names and layer parameters).
            Set to 2 for more layers details (preicision, type, kernel ...) when calling the EngineInspector
        opset_version : int
                ONNX version (Default 13).

        Raises
        ------
        Exception
            * Model must be instantiated with attr:`tracing` = True
            * If :attr:`dynamic_axes` is desired, :attr:`opt_profiles` must be provided with sames keys as
              :attr:`dynamic_axes`.
        """
        if prod_package_error is not None:
            raise prod_package_error
        self._opset_version = opset_version
        self._model = model
        self._device = device
        self._verbose = verbose
        self._custom_opset = None  # to be redefine in child class if needed
        self._onnx_path = onnx_path
        self._precision = precision
        self._batch_size = batch_size
        self._input_names = input_names
        self._input_shapes = input_shapes
        self._use_scope_names = use_scope_names
        self._do_constant_folding = do_constant_folding
        self._operator_export_type = operator_export_type
        self._ignore_adapt_graph = ignore_adapt_graph

        if dynamic_axes is not None:
            assert opt_profiles is not None, "If dynamic_axes are to be used, opt_profiles must be provided"
            assert isinstance(dynamic_axes, dict)
            assert opt_profiles.keys() == dynamic_axes.keys(), "dynamic_axes and opt_profiles must have same keys"
        self._dynamic_axes = dynamic_axes

        # Initiate Trt Engine builder
        onnx_dir = os.path.split(onnx_path)[0]
        onnx_file_name = os.path.split(onnx_path)[1]
        model_name = onnx_file_name.split(".")[0]

        self._engine_path = os.path.join(onnx_dir, model_name + f"_{precision.lower()}.engine")

        if self._verbose:
            trt_logger = trt.Logger(trt.Logger.VERBOSE)
        else:
            trt_logger = trt.Logger(trt.Logger.WARNING)

        self._engine_builder = TRTEngineBuilder(
            self._onnx_path, logger=trt_logger, opt_profiles=opt_profiles, calibrator=calibrator
        )

        if profiling_verbosity == 0:
            self._engine_builder.profiling_verbosity = "LAYER_NAMES_ONLY"
        elif profiling_verbosity == 1:
            self._engine_builder.profiling_verbosity = "NONE"
        elif profiling_verbosity == 2:
            self._engine_builder.profiling_verbosity = "DETAILED"
        else:
            raise AttributeError("unknown profiling_verbosity")
        if precision.lower() == "fp32":
            pass
        elif precision.lower() == "int8":
            self._engine_builder.INT8_allowed = True
            self._engine_builder.strict_type = True
        elif precision.lower() == "fp16":
            self._engine_builder.FP16_allowed = True
            self._engine_builder.strict_type = True
        elif precision.lower() == "mix":
            self._engine_builder.FP16_allowed = True
            self._engine_builder.strict_type = False
        else:
            raise Exception(f"precision {precision} not supported")

    @property
    def onnx_path(self) -> str:
        # Flexibility for some engines
        return self._onnx_path

    @property
    def model(self) -> torch.nn.Module:
        """
        PyTorch model

        Returns
        -------
        model: torch.nn.Module
        """
        return self._model

    @model.setter
    def model(self, value: torch.nn.Module) -> None:
        """
        Set PyTorch model

        Parameters
        ----------
        value: torch.nn.Module
        """
        self._model = value

    @property
    def input_shapes(self) -> tuple:
        """
        Input shapes of the model during tracing

        Returns
        -------
        input_shapes: tuple
        """
        return self._input_shapes

    @input_shapes.setter
    def input_shapes(self, value: tuple) -> None:
        """
        Set input shapes of the model during tracing

        Parameters
        ----------
        value: tuple
        """
        self._input_shapes = value

    @property
    def input_names(self) -> list:
        """
        Input names of the model during tracing

        Returns
        -------
        input_names: list
        """
        return self._input_names

    @input_names.setter
    def input_names(self, value: list) -> None:
        """
        Set input names of the model during tracing

        Parameters
        ----------
        value: list
        """
        self._input_names = value

    @property
    def batch_size(self) -> int:
        """
        Batch size of the model during tracing
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        """
        Set batch size of the model during tracing
        """
        self._batch_size = value

    @property
    def build_torch_model(self) -> torch.nn.Module:
        """Build PyTorch model and load weight with the given name

        Returns
        -------
        model: torch.nn.Module
        """
        pass
        raise Exception("Child class should implement this method")

    def specific_adapt_graph(self, graph: gs.Graph) -> gs.Graph:
        """Modify ONNX graph to ensure compability between ONNX and TensorRT

        Returns
        -------
        graph: onnx_graphsurgeon.Graph
        """
        warnings.warn("specific_adapt_graph should be implemented in the child class if needed")
        return graph

    def _adapt_graph(self, graph: gs.Graph, **kwargs) -> gs.Graph:
        """Modify ONNX graph to ensure compability between ONNX and TensorRT

        Returns
        -------
        graph: onnx_graphsurgeon.Graph
        """
        try:
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
        except:
            print("[INFO] BaseExporter: Cannot handle clip. Clip handling will be ignored")
            pass

        model = onnx.load(self._onnx_path)
        check = False
        if self._dynamic_axes is None:
            from onnxsim import simplify

            model_simp, check = simplify(model)

        if check:
            print("\n[INFO] Simplified ONNX model validated. Graph optimized...")
            graph = gs.import_onnx(model_simp)
            graph.toposort()
            graph.cleanup()
        else:
            print("\n[INFO] ONNX model was not validated.")

        # Call the child class for specific graph adapation
        graph = self.specific_adapt_graph(graph)
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

    def _torch2onnx(self) -> Tuple[Tuple[np.ndarray], Dict[str, np.ndarray]]:
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
                m_outputs = self._model(inputs, **kwargs)

            # Prepare inputs for torch.export.onnx and sanity check
            np_inputs = tuple(np.array(inputs[iname].cpu()) for iname in inputs)
            inputs = (inputs,)
        else:
            with torch.no_grad():
                m_outputs = self._model(*inputs, **kwargs)

            # Prepare inputs for torch.export.onnx and sanity check
            np_inputs = tuple(np.array(i.cpu()) for i in inputs)
        inputs = (*inputs, kwargs)

        onames = m_outputs._fields if hasattr(m_outputs, "_fields") else [f"out_{i}" for i in range(len(m_outputs))]
        np_m_outputs = {key: val.cpu().numpy() for key, val in zip(onames, m_outputs) if isinstance(val, torch.Tensor)}

        # Convert to list for dynamic axese assertions
        self._input_names = list(self._input_names)
        onames = list(onames)

        # Export to ONNX
        with ExitStack() as stack:
            # context managers necessary to redirect stdout and modify export trace to print scope
            if self._use_scope_names:
                buffer = stack.enter_context(io.StringIO())
                stack.enter_context(redirect_stdout(buffer))
                stack.enter_context(scope_name_workaround())
            torch.onnx.export(
                self._model,  # model being run
                inputs,  # model input (or a tuple for multiple inputs)
                self._onnx_path,  # where to save the model
                export_params=True,  # store the trained parameter weights inside the model file
                output_names=onames,
                input_names=self._input_names,  # the model's input names
                dynamic_axes=self._dynamic_axes,
                custom_opsets=self._custom_opset,
                opset_version=self._opset_version,  # the ONNX version to export the model to
                do_constant_folding=self._do_constant_folding,  # whether to execute constant folding for optimization
                verbose=self._verbose or self._use_scope_names,  # verbose mandatory in scope names procedure
                operator_export_type=self._operator_export_type,
            )

            if self._use_scope_names:
                onnx_export_log = buffer.getvalue()

        graph = gs.import_onnx(onnx.load(self._onnx_path))
        if not self._ignore_adapt_graph:
            graph.toposort()

            # Modify ONNX graph for TensorRT compability
            graph = self._adapt_graph(graph, **kwargs)
            utils.print_graph_io(graph)
            # Export adapted onnx for TRT engine
            onnx.save(gs.export_onnx(graph), self._onnx_path)

        # rewrite onnx graph with new scope names
        if self._use_scope_names:
            number2scope = get_scope_names(onnx_export_log, strict=False)
            graph = gs.import_onnx(onnx.load(self._onnx_path))
            graph = rename_tensors_(graph, number2scope, verbose=True)
            graph = rename_nodes_(graph, True)
            onnx.save(gs.export_onnx(graph), self._onnx_path)

        print("Saved ONNX at:", self._onnx_path)

        # empty GPU memory for later TensorRT optimization
        torch.cuda.empty_cache()

        return np_inputs, np_m_outputs

    def _onnx2engine(self, **kwargs) -> trt.ICudaEngine:
        """
        Export TensorRT engine from an ONNX file

        Returns
        -------
        engine: tensorrt.ICudaEngine
        """
        if prod_package_error is not None:
            raise prod_package_error

        # Build engine
        self._engine_builder.export_engine(self._engine_path)
        return self._engine_builder.engine

    def _sanity_check(
        self, engine: trt.ICudaEngine, sample_inputs: Tuple[np.ndarray], sample_outputs: Dict[str, np.ndarray]
    ) -> bool:
        """
        Perform a sanity check on the TensorRT engine

        Returns
        -------
        bool
        """
        if self._precision.lower() == "fp32":
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
        tic = time.time()
        [model.execute() for i in range(self.PROFILING_TIME)]
        toc = time.time()

        # Check engine outputs with sample outputs
        m_outputs = model.execute()
        print("Absolute / relavtive error:")
        for out in m_outputs:
            print("out", m_outputs[out])
            diff = m_outputs[out].astype(float) - sample_outputs[out].astype(float)
            abs_err = np.abs(diff)
            rel_err = np.abs(diff / (sample_outputs[out] + 1e-6))  # Avoid div by zero
            print(out)
            print(f"\tmean: {abs_err.mean():.2e}\t{rel_err.mean():.2e}")
            print(f"\tmax: {abs_err.max():.2e}\t{rel_err.max():.2e}")
            print(f"\tstd: {abs_err.std():.2e}\t{rel_err.std():.2e}")
            check = check & (rel_err.mean() < threshold)

        print(f"Engine execution time: {(toc - tic)/self.PROFILING_TIME*1000:.2f} ms")
        return check

    def export_engine(self) -> trt.ICudaEngine:
        """
        Export TensorRT engine from PyTorch model

        Returns
        -------
        engine: tensorrt.ICudaEngine
        """
        sample_inputs, sample_outputs = self._torch2onnx()
        engine = self._onnx2engine()
        self._sanity_check(engine, sample_inputs, sample_outputs)
