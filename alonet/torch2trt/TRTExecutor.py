from collections import defaultdict
from typing import Union

try:
    import pycuda
    import pycuda.autoinit as cuda_init

    pycuda_error = None
except Exception as pycuda_error:
    pass

try:
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    tensorrt_error = None

    class CustomProfiler(trt.IProfiler):
        def __init__(self):
            trt.IProfiler.__init__(self)
            self.reset()

        def report_layer_time(self, layer_name, ms):
            self.timing[layer_name].append(ms)

        def reset(self):
            self.timing = defaultdict(list)


except Exception as tensorrt_error:
    pass

from alonet.torch2trt.utils import allocate_buffers, allocate_dynamic_mem, execute_async, execute_sync, get_bindings


# MS_DEFORM_IM2COL_PLUGIN_LIB = "alonet/torch2trt/plugins/ms_deform_im2col/build/libms_deform_im2col_trt.so"
# ctypes.CDLL(MS_DEFORM_IM2COL_PLUGIN_LIB)
# trt.init_libnvinfer_plugins(TRT_LOGGER, '')
# PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list


class TRTExecutor:
    """
    A helper class to execute a TensorRT engine.

    Attributes:
    -----------
    stream: pycuda.driver.Stream
    engine: tensorrt.ICudaEngine
    context: tensorrt.IExecutionContext
    inputs/outputs: list[HostDeviceMem]
        see trt_helper.py
    bindings: list[int]
        pointers in GPU for each input/output of the engine
    dict_inputs/dict_outputs: dict[str, HostDeviceMem]
        key = input node name
        value = HostDeviceMem of corresponding binding

    """

    def __init__(
        self,
        engine,
        stream,
        sync_mode: bool = False,
        verbose_logger: bool = False,
        profiling: bool = False,
    ):
        """
        Parameters
        ----------
        engine: if str, path to engine file, if tensorrt.ICudaEngine, serialized engine
        stream: pycuda.driver.Stream
            if None, one will be created by allocate_buffers function
        sync_mode: bool, default = False.
            True/False enable the synchronized/asynchonized execution of TensorRT engine
        logger: tensorrt.ILogger, logger to print info in terminal
        """
        if tensorrt_error is not None:
            raise tensorrt_error
        self.sync_mode = sync_mode
        self.stream = stream
        if verbose_logger:
            self.logger = trt.Logger(trt.Logger.VERBOSE)
        else:
            self.logger = TRT_LOGGER
        if isinstance(engine, str):
            with open(engine, "rb") as f, trt.Runtime(self.logger) as runtime:
                print("Reading engine  ...")
                self.engine = runtime.deserialize_cuda_engine(f.read())
                assert self.engine is not None, "Read engine failed"
                print("Engine loaded")
        else:
            self.engine = engine
        self.context = self.engine.create_execution_context()
        if profiling:
            self.context.profiler = CustomProfiler()
        # Allocate_buffer take into account if engine has dynamic axes
        self.inputs, self.outputs, self.stream, self.has_dynamic_axes = allocate_buffers(
            self.context, self.stream, self.sync_mode
        )
        self.dict_inputs = {mem_obj.name: mem_obj for mem_obj in self.inputs}
        self.dict_outputs = {mem_obj.name: mem_obj for mem_obj in self.outputs}

    @property
    def bindings(self):
        # Be carefull, call bindings after set all shapes
        if self.has_dynamic_axes or not hasattr(self, "_bindings"):
            self._bindings = get_bindings(self.context, self.dict_inputs, self.dict_outputs)
        return self._bindings

    def print_bindings_info(self):
        print("ID / Name / isInput / shape / dtype")
        for i in range(self.engine.num_bindings):
            print(
                f"Binding: {i}, name: {self.engine.get_binding_name(i)}, input: {self.engine.binding_is_input(i)}, \
                    shape: {self.engine.get_binding_shape(i)}, dtype: {self.engine.get_binding_dtype(i)}"
            )

    def execute(self):
        if self.has_dynamic_axes:
            # Set input shape in context to update output shapes
            allocate_dynamic_mem(self.context, self.dict_inputs, self.dict_outputs)

        if self.sync_mode:
            execute_sync(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs)
        else:
            execute_async(
                self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream
            )
        return {out.name: out.host for out in self.outputs}

    def set_binding_shape(self, binding: int, shape: tuple):
        self.context.set_binding_shape(binding, shape)

    def allocate_mem(self):
        self.inputs, self.outputs, self.stream, self.has_dynamic_axes = allocate_buffers(
            self.context, self.stream, self.sync_mode
        )
        self.dict_inputs = {mem_obj.name: mem_obj for mem_obj in self.inputs}
        self.dict_outputs = {mem_obj.name: mem_obj for mem_obj in self.outputs}

    def __call__(self, *inputs, **kwargs):
        for i, tensor in enumerate(inputs):
            self.inputs[i].host = tensor
        return self.execute()
