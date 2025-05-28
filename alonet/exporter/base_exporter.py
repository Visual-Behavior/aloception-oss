import sys
import subprocess
import os
from typing import Dict, List, Tuple, Union
from abc import ABC, abstractmethod
import io
import numpy as np
import torch
import warnings

import onnx_graphsurgeon as gs
from onnxsim import simplify
import onnx


from alonet.exporter.onnx_hack import scope_name_workaround, get_scope_names, rename_tensors_

from alonet.exporter.utils import get_nodes_by_op, rename_nodes_, print_graph_io
from contextlib import redirect_stdout, ExitStack


class BaseExporter(ABC):
    def __init__(self, save_path: str):
        """
        BaseExporter

        Parameters
        ----------

        save_path: str
            Path to save output model
        """
        self._save_path = save_path

    @property
    def save_path(self) -> str:
        return self._save_path

    @save_path.setter
    def save_path(self, new_path: str) -> None:
        self._save_path = new_path

    @abstractmethod
    def export(self):
        raise NotImplementedError("This method must be implemented in child class.")


class BaseONNXExporter(BaseExporter):
    def __init__(
        self,
        model: torch.nn.Module,
        save_path: str,
        input_shapes: tuple = ([3, 1280, 1920]),
        input_names: list = None,
        batch_size: int = 1,
        do_constant_folding: bool = True,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False,
        use_scope_names: bool = False,
        operator_export_type=None,
        opt_profiles: Dict[str, Tuple[List[int]]] = None,
        opset_version: int = 13,
        ignore_adapt_graph: bool = False,
    ):
        """
        Parameters
        ----------
        model : torch.nn.Module
            a model loaded with trained weights
        save_path : str
            Onnx file path which will be exported.
            Example: /abc/xyz/my_model.onnx
        input_shapes : tuple of tuple/list, default ([3, 1280, 1920], )
            input shape must be specified when export model
        input_names : list of str
            Name of inputs to onnx
        batch_size : int
            Batch size of inputs. Default: 1
        do_constant_folding : bool
            Optimized ONNX graph if True. Sometimes this optimization will make
            the ONNX graph modification more complicated. Default True
        verbose : bool
            Print out everything. Good for debugging. Default False.
        opt_profiles : Dict[str, Tuple[List[int]]]
            Optimization profiles (one by each dynamic axis). Default None
        operator_export_type: torch.onnx.OperatorExportTypes
            Type of operator export of torch.onnx. Default None.
        opset_version : int
                ONNX version (Default 13).

        Raises
        ------
        Exception
            * Model must be instantiated with attr:`tracing` = True
            * If :attr:`dynamic_axes` is desired, :attr:`opt_profiles` must be provided with sames keys as
              :attr:`dynamic_axes`.
        """
        super().__init__(save_path)
        self._opset_version = opset_version
        self._model = model
        self._device = device
        self._verbose = verbose
        self._custom_opset = None  # to be redefine in child class if needed
        self._batch_size = batch_size
        self._input_names = input_names
        self._input_shapes = input_shapes
        self._use_scope_names = use_scope_names
        self._do_constant_folding = do_constant_folding
        self._operator_export_type = operator_export_type
        self._ignore_adapt_graph = ignore_adapt_graph

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
        model = onnx.load(self.save_path)
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

    def export(self) -> Tuple[Tuple[np.ndarray], Dict[str, np.ndarray]]:
        """Export PyTorch model to ONNX file.
        Return sample inputs/outputs for sanity check

        Note: is_export_onnx=None mean True,
        because torch.onnx.export support only torch.Tensor or None for forward method.

        Returns
        -------
        sample_inputs: tuple[np.ndarray]
        sample_outputs: dict[str: np.ndarray]

        """
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
                self.save_path,  # where to save the model
                export_params=True,  # store the trained parameter weights inside the model file
                output_names=onames,
                input_names=self._input_names,  # the model's input names
                custom_opsets=self._custom_opset,
                opset_version=self._opset_version,  # the ONNX version to export the model to
                do_constant_folding=self._do_constant_folding,  # whether to execute constant folding for optimization
                verbose=self._verbose or self._use_scope_names,  # verbose mandatory in scope names procedure
                operator_export_type=self._operator_export_type,
            )

            if self._use_scope_names:
                onnx_export_log = buffer.getvalue()

        graph = gs.import_onnx(onnx.load(self.save_path))
        if not self._ignore_adapt_graph:
            graph.toposort()

            # Modify ONNX graph for TensorRT compability
            graph = self._adapt_graph(graph, **kwargs)
            print_graph_io(graph)
            # Export adapted onnx for TRT engine
            onnx.save(gs.export_onnx(graph), self.save_path)

        # rewrite onnx graph with new scope names
        if self._use_scope_names:
            number2scope = get_scope_names(onnx_export_log, strict=False)
            graph = gs.import_onnx(onnx.load(self.save_path))
            graph = rename_tensors_(graph, number2scope, verbose=True)
            graph = rename_nodes_(graph, True)
            onnx.save(gs.export_onnx(graph), self.save_path)

        print("Saved ONNX at:", self.save_path)

        torch.cuda.empty_cache()

        return np_inputs, np_m_outputs


class BaseTRTExporter(BaseExporter):
    def __init__(
        self,
        onnx_path: str,
        precision: str = "fp32",
        verbose: bool = False,
        opt_profiles: Dict[str, Tuple[List[int]]] = None,
        profiling_verbosity: int = 0,
        calibrator=None,
    ):
        """
        Parameters
        ----------
        onnx_path: str
            Path to the onnx file to compile to TensorRT.
        precision : str
            TRT engine precision, either fp32, fp16, mix
            mix precision between fp32 and fp16 allow TensorRT more liberty
            to find the best combination optimization in term of execution time.
        verbose : bool
            Print out everything. Default False
        opt_profiles : Dict[str, Tuple[List[int]]].
            Optimization profiles (one by each dynamic axis). Default None
        profiling_verbosity : int
            Profiling verbosity in NVTX annotations and the engine inspector (Default 0)
                0 : LAYER_NAMES_ONLY (Print only the layer names. This is the default setting).
                1 : NONE (Do not print any layer information).
                2 : DETAILED : (Print detailed layer information including layer names and layer parameters).
            Set to 2 for more layers details (preicision, type, kernel ...) when calling the EngineInspector
        calibrator : torch2trt.calibrator.BaseCalibrator
            Quantization calibrator.

        Raises
        ------
        Exception
            * Model must be instantiated with attr:`tracing` = True
            * If :attr:`dynamic_axes` is desired, :attr:`opt_profiles` must be provided with sames keys as
              :attr:`dynamic_axes`.
        """

        try:
            # import pycuda and tensorrt
            import pycuda.driver as cuda
            import tensorrt as trt
            from alonet.exporter import trt_engine_builder
        except Exception as e:
            raise ImportError(
                "pycuda and tensorrt are not installed."
                "Please install using the requirement file at alonet/torch2trt/requirements.txt"
            )

        # Initiate Trt Engine builder
        onnx_dir = os.path.split(onnx_path)[0]
        onnx_file_name = os.path.split(onnx_path)[1]
        model_name = onnx_file_name.split(".")[0]
        engine_path = os.path.join(onnx_dir, model_name + f"_{precision.lower()}.engine")
        super().__init__(engine_path)

        self._verbose = verbose
        self._onnx_path = onnx_path
        self._precision = precision

        if self._verbose:
            trt_logger = trt.Logger(trt.Logger.VERBOSE)
        else:
            trt_logger = trt.Logger(trt.Logger.WARNING)

        self._engine_builder = trt_engine_builder(
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

    def export(self, **kwargs):
        """
        Export TensorRT engine from an ONNX file

        Returns
        -------
        engine: tensorrt.ICudaEngine
        """

        # Build engine
        self._engine_builder.export_engine(self.save_path)
        return self._engine_builder.engine
