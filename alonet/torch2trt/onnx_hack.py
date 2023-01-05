import torch


class scope_name_workaround(object):
    """Workaround to preserve tensors scope names during onnx export"""

    def __init__(self):
        self.backup = None

    def __enter__(self):
        def _tracing_name(self_, tracing_state):
            if not tracing_state._traced_module_stack:
                return None
            module = tracing_state._traced_module_stack[-1]
            for name, child in module.named_children():
                if child is self_:
                    return name
            return None

        def _slow_forward(self_, *input, **kwargs):
            tracing_state = torch._C._get_tracing_state()
            if not tracing_state or isinstance(self_.forward, torch._C.ScriptMethod):
                return self_.forward(*input, **kwargs)
            if not hasattr(tracing_state, "_traced_module_stack"):
                tracing_state._traced_module_stack = []
            name = _tracing_name(self_, tracing_state)
            if name:
                tracing_state.push_scope("%s[%s]" % (self_._get_name(), name))
            else:
                tracing_state.push_scope(self_._get_name())
            tracing_state._traced_module_stack.append(self_)
            try:
                result = self_.forward(*input, **kwargs)
            finally:
                tracing_state.pop_scope()
                tracing_state._traced_module_stack.pop()
            return result

        self.backup = torch.nn.Module._slow_forward
        setattr(torch.nn.Module, "_slow_forward", _slow_forward)

    def __exit__(self, type, value, tb):
        setattr(torch.nn.Module, "_slow_forward", self.backup)


def get_scope_names(onnx_export_log, strict=True):
    """Correspondance between tensor names given during ONNX with tensor scope name

    Parameters
    ----------
    onnx_export_log : str
        string output of redirected stdout during torch.onnx.export(..., verbose=True)

    Returns
    -------
    number2scope: dict
        dict of pairs <number_name: scope_name>. For exemple : {"34" : "AlexNet/Sequential[classifier]/Linear[4]"}
    """
    lines = onnx_export_log.split("\n")
    number2scope = {}
    scope_dict = {}
    for line in lines:
        if "scope: " in line:
            name = line.split(":", 1)[0].strip().replace("%", "")
            scope = line.split("scope: ", 1)[1].split(" # ")[0].strip()
            if scope not in scope_dict:
                scope_dict[scope] = 0
            elif strict:
                raise ValueError(f"Identical scope encountered more than once: {scope}")
            else:
                scope_dict[scope] += 1
                scope = f"{scope}_{scope_dict[scope]}"
            number2scope[name] = scope
    return number2scope


def rename_tensors_(graph, number2scope, verbose=False):
    """Rename tensors in graph with their scope name instead of number name

    Parameters
    ----------
    graph:
        graph loaded with onnx_graphsurgeon
    number2score: dict
        dict of pairs <number_name: scope_name>. For exemple : {"34" : "AlexNet/Sequential[classifier]/Linear[4]"}

    Returns
    -------
        modified graph (the graph is modified inplace)
    """
    dont_rename = [v.name for v in graph.inputs + graph.outputs]
    for key, val in graph.tensors().items():
        if key in number2scope and key not in dont_rename:
            val.name = number2scope[key]
            if verbose:
                print(f"  changed {key} to {number2scope[key]}")
    return graph


def rename_nodes_(graph, verbose=False):
    """Rename node names in graph with their outputs or inputs to improve timing measure.
    Must be used after call :func:`rename_tensors_`

    Parameters
    ----------
    graph:
        graph loaded with onnx_graphsurgeon

    Returns
    -------
        modified graph (the graph is modified inplace)
    """
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

def _add_grid_sampler_to_opset13():
        ## Credits https://github.com/pytorch/pytorch/issues/27212#issuecomment-1059773074

        from torch.onnx import register_custom_op_symbolic
        import torch.onnx.symbolic_helper as sym_help

        def grid_sampler(g, input, grid, mode, padding_mode, align_corners):
            mode = sym_help._maybe_get_const(mode, "i")
            padding_mode = sym_help._maybe_get_const(padding_mode, "i")
            mode_str = ['bilinear', 'nearest', 'bicubic'][mode]
            padding_mode_str = ['zeros', 'border', 'reflection'][padding_mode]
            align_corners = int(sym_help._maybe_get_const(align_corners, "b"))

            return g.op("com.microsoft::GridSample", input, grid,
                        mode_s=mode_str,
                        padding_mode_s=padding_mode_str,
                        align_corners_i=align_corners)
            
        register_custom_op_symbolic('::grid_sampler', grid_sampler, 1)
