import os

try:
    import pycuda.driver as cuda
    import onnx_graphsurgeon as gs
    import tensorrt as trt

    prod_package_error = None
except Exception as prod_package_error:
    pass


import ctypes
from alonet import ALONET_ROOT


def load_trt_custom_plugins(lib_path: str):
    """
    Register custom plugins for TensorRT from built source.
    If the built file does not exitste, this function will try
    to run alonet/torch2trt/plugins/make.sh, a script responsible
    for building all plugins in alonet/torch2trt/plugins

    Parameters
    -----------
    lib_path: str
        Path relative to aloception root, point to built plugin
    """
    if prod_package_error is not None:
        raise prod_package_error

    if not os.path.isfile(lib_path):
        import subprocess

        make_file = os.path.join(ALONET_ROOT, "torch2trt/plugins/make.sh")
        subprocess.call(["sh", make_file, ALONET_ROOT])
    ctypes.CDLL(lib_path)
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.INFO), "")


def print_graph_io(graph):
    """
    Parameters
    ----------
    graph: gs.Graph
    """
    if prod_package_error is not None:
        raise prod_package_error
    # Print inputs:
    print("\n=====ONNX graph inputs =====")
    for i in graph.inputs:
        print(i)
    # Print outputs:
    print("=====ONNX graph outputs =====")
    for i in graph.outputs:
        print(i)
    print("\n")


def io_name_handler(graph):
    """
    Parameters
    ----------
    graph: gs.Graph
    """
    if prod_package_error is not None:
        raise prod_package_error
    # for now, no need to handle IO names
    return graph


def get_node_by_name(name, onnx_graph):
    """
    Parameters
    ----------
    graph: gs.Graph
    """
    for n in onnx_graph.nodes:
        if name in n.name:
            return n
    return None


def get_nodes_by_op(op_name, onnx_graph):
    nodes = []
    for n in onnx_graph.nodes:
        if n.op == op_name:
            nodes.append(n)
    return nodes


def get_nodes_by_prefix(prefix, onnx_graph):
    """
    Parameters
    ----------
    graph: gs.Graph
    """
    nodes = []
    for n in onnx_graph.nodes:
        if n.name.startswith(prefix):
            nodes.append(n)
    return nodes


def GiB(val):
    """Calculate Gibibit in bits, used to set workspace for TensorRT engine builder."""
    return val * 1 << 30


def is_dynamic(shape: tuple):
    return any(dim is None or dim < 0 for dim in shape)


class HostDeviceMem(object):
    """
    Simple helper class to store useful data of an engine's binding

    Attributes
    ----------
    host_mem : np.ndarray
        data stored in CPU
    device_mem : pycuda.driver.DeviceAllocation
        represent data pointer in GPU
    shape : tuple
    dtype : np dtype
    location : trt.TensorLocation
        Device location information
    name: str
        name of the binding
    """

    def __init__(self, host_mem, device_mem, shape, dtype, name=""):
        self.host = host_mem
        self.device = device_mem
        self.shape = shape
        self.dtype = dtype
        self.name = name

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class DynamicHostDeviceMem(HostDeviceMem):
    """
    Class to store useful data of an engine's binding. Allocate memory on each host setting

    Attributes
    ----------
    host_mem : np.ndarray
        data stored in CPU
    device_mem : pycuda.driver.DeviceAllocation
        represent data pointer in GPU
    shape : tuple
    dtype : np dtype
    location : trt.TensorLocation
        Device location information
    name: str
        name of the binding
    """

    @property
    def host(self):
        return self._host

    @host.setter
    def host(self, new_host):
        if new_host is None:
            self.release()
        else:
            self.allocate_mem(tuple(new_host.shape))
        self._host = new_host

    def allocate_mem(self, new_shape: tuple):

        if prod_package_error is not None:
            raise prod_package_error

        if self._host is None or not hasattr(self.shape, "__iter__") or tuple(new_shape) != tuple(self.shape):
            # Allocate buffer with new shape, after release memory
            self.release()
            self.shape = new_shape
            self._host = cuda.pagelocked_empty(trt.volume(self.shape), self.dtype)
            self.device = cuda.mem_alloc(self._host.nbytes)  # New pointer

    def release(self):
        if hasattr(self, "device") and self.device is not None:
            self.device.free()  # Freeze memory allocated
        self.shape = None


def allocate_buffers(context, stream=None, sync_mode=True):
    """
    Read bindings' information in ExecutionContext, create pagelocked np.ndarray in CPU,
    allocate corresponding memory in GPU.

    Parameters
    ----------
    context : trt.IExecutionContext
    stream : pycuda.driver.Stream, optional
    sync_mode : bool, optional

    Returns
    -------
    inputs: list[HostDeviceMem]
    outputs: list[HostDeviceMem]
    bindings: list[int]
        list of pointers in GPU for each bindings
    stream: pycuda.driver.Stream
        used for memory transfers between CPU-GPU

    Notes
    -----
    If a binding is dynamic, create a :class:`DynamicHostDeviceMem` with :attr:`host` = :attr:`shape` = None
    (new host will set this properties)

    """
    if prod_package_error is not None:
        raise prod_package_error

    inputs = []
    outputs = []
    has_dynamic_axes = False
    if stream is None and not sync_mode:
        stream = cuda.Stream()
    for binding in context.engine:
        binding_idx = context.engine.get_binding_index(binding)
        shape = context.engine.get_binding_shape(binding_idx)
        dtype = trt.nptype(context.engine.get_binding_dtype(binding))
        if is_dynamic(shape):
            # Not allocate buffers because is a dynamic binding
            host_mem = device_mem = None
            has_dynamic_axes = True
            mem_obj = DynamicHostDeviceMem(host_mem, device_mem, shape, dtype, binding)
        else:
            size = trt.volume(shape) * context.engine.max_batch_size
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            mem_obj = HostDeviceMem(host_mem, device_mem, shape, dtype, binding)
        # Append to the appropriate list.
        if context.engine.binding_is_input(binding):
            inputs.append(mem_obj)
        else:
            outputs.append(mem_obj)
    return inputs, outputs, stream, has_dynamic_axes


def allocate_dynamic_mem(context, dict_inputs, dict_outputs):
    """Set all input shapes in context to convert a dynamic context in static (output fix shapes).
    Then, allocate output shapes

    Parameters
    ----------
    context : trt.IExecutionContext
    dict_inputs : Dict[str, HostDeviceMem]
    dict_outputs : Dict[str, HostDeviceMem]
    """
    assert all(
        mem_obj.device is not None and mem_obj.host is not None and not is_dynamic(mem_obj.shape)
        for mem_obj in dict_inputs.values()
    ), "When there are dynamic axes, all inputs shape must be set (model[i].host = array for all inputs)"

    # Set input shape in context to update output shapes
    for iname, mem_obj in dict_inputs.items():
        idx = context.engine.get_binding_index(iname)
        assert context.set_binding_shape(idx, mem_obj.shape), f"Impossible to set shape {mem_obj.shape} into {iname}"

    # Create empty tensors to allocate output buffers
    for oname, mem_obj in dict_outputs.items():
        idx = context.engine.get_binding_index(oname)
        shape = context.get_binding_shape(idx)  # Shape updated with inputs.set_binding_shape
        if isinstance(mem_obj, DynamicHostDeviceMem):  # Buffer allocate
            mem_obj.allocate_mem(shape)
    return True


def get_bindings(context, dict_inputs, dict_outputs):
    """Get input/output pointers and add them into a list

    Parameters
    ----------
    context : trt.IExecutionContext
    dict_inputs : Dict[str, HostDeviceMem]
    dict_outputs : Dict[str, HostDeviceMem]

    Returns
    -------
    list
        List of input/output pointers
    """
    bindings = [None] * (len(dict_inputs) + len(dict_outputs))

    # Get bindings for inputs
    for name, mem_obj in dict_inputs.items():
        idx = context.engine.get_binding_index(name)
        bindings[idx] = int(mem_obj.device) if mem_obj.device is not None else None

    # Get bindings for outputs
    for name, mem_obj in dict_outputs.items():
        idx = context.engine.get_binding_index(name)
        bindings[idx] = int(mem_obj.device) if mem_obj.device is not None else None
    return bindings


def execute_async(context, bindings, inputs, outputs, stream):
    """
    Execute an TensorRT engine.

    Parameters
    ----------
    context: tensorrt.IExecutionContext
    bindings: list[int]
        list of pointers in GPU for each bindings
    inputs: list[HostDeviceMem]
    outputs: list[HostDeviceMem]
    stream: pycuda.driver.Stream
        used for memory transfers between CPU-GPU

    Returns
    -------
    list : np.ndarray
        For each outputs of the engine
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    check = context.execute_async(bindings=bindings, stream_handle=stream.handle)
    assert check, "Kernel execution failed"
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    for out in outputs:
        out.host = out.host.reshape(out.shape)
    return [out.host for out in outputs]


def execute_sync(context, bindings, inputs, outputs):
    """
    Execute an TensorRT engine.

    Parameters
    -----------
    context: tensorrt.IExecutionContext
    bindings: list[int]
        list of pointers in GPU for each bindings
    inputs: list[HostDeviceMem]
    outputs: list[HostDeviceMem]
    stream: pycuda.driver.Stream
        used for memory transfers between CPU-GPU

    Parameters
    ----------
    list[np.ndarray] for each outputs of the engine
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod(inp.device, inp.host) for inp in inputs]
    # Run inference.
    check = context.execute_v2(bindings=bindings)
    assert check, "Kernel execution failed"
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh(out.host, out.device) for out in outputs]
    # Return only the host outputs.
    for out in outputs:
        out.host = out.host.reshape(out.shape)
    return [out.host for out in outputs]



def rename_nodes_(graph, verbose=False):

    dont_rename = [v.name for v in graph.inputs + graph.outputs]

    for node in graph.nodes:
        if node.name not in dont_rename:
            # Replace name by output name to include in profiling
            node.name = node.outputs[0].name
            # If the node does not have name, try to replace by inputs tensors to it
            try:
                id_node = int(node.name)
                node_is_int = True
            except:
                node_is_int = False

            if node_is_int:
                for inode in node.inputs:
                    try:  # Only for named inputs
                        int(inode.name)
                        inode_is_int = True
                    except:
                        inode_is_int = False

                    # Input named, change tensor name
                    if not inode_is_int:
                        new_name = inode.name + "_" + str(id_node)
                        if verbose:
                            print(f"  changed {node.name} to {new_name}")
                        node.name = new_name

    return graph
