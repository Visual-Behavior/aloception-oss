try:
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    prod_package_error = None
except Exception as prod_package_error:
    pass


from typing import Dict, List, Tuple


def GiB(val):
    return val * 1 << 30


class TRTEngineBuilder:
    """
    Work with TensorRT 8. Should work fine with TensorRT 7.2.3 (not tested)

    Helper class to build TensorRT engine from ONNX graph file (including weights).
    The graph must have defined input shape. For more detail, please see TensorRT Developer Guide:
    https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#python_topics
    """

    def __init__(
        self,
        onnx_file_path: str,
        FP16_allowed: bool = False,
        INT8_allowed: bool = False,
        strict_type: bool = False,
        calibrator: bool = None,
        logger=None,
        opt_profiles: Dict[str, Tuple[List[int]]] = None,
    ):
        """
        Parameters
        -----------
        onnx_file_path: str
            path to ONNX graph file
        FP16_allowed: bool
            Enable FP16 precision for engine builder
        INT8_allowed: bool
            Enable FP16 precision for engine builder, user must provide also a calibrator
        strict_type: bool
            Ensure that the builder understands to force the precision
        calibrator: extended instance from tensorrt.IInt8Calibrator
            Used for INT8 quantization
        opt_profiles : Dict[str, Tuple[List[int]]], by default None
            Optimization profiles (one by each dynamic axis), with the minimum, minimum and maximum values.

        Raises
        ------
        Exception
            If :attr:`opt_profiles` is desired, each profile must be a set of
            [:value:`min_shape`/:value:`optimal_shape`/:value:`max_shape`]
        """
        if prod_package_error is not None:
            raise prod_package_error
        logger = logger if logger is not None else trt.Logger
        self.FP16_allowed = FP16_allowed
        self.INT8_allowed = INT8_allowed
        self.onnx_file_path = onnx_file_path
        self.calibrator = calibrator
        self.max_workspace_size = GiB(8)
        self.strict_type = strict_type
        self.logger = logger
        self.engine = None
        if opt_profiles is not None:
            assert isinstance(opt_profiles, dict)
            assert all([len(op) == 3 for op in opt_profiles.values()]), "Each profile must be a set of min/opt/max"
        self.opt_profiles = opt_profiles

    def set_workspace_size(self, workspace_size_GiB: int):
        self.max_workspace_size = GiB(workspace_size_GiB)

    def setup_profile(self, builder, config):
        """Setup builder engine to add custom optimization profiles

        Parameters
        ----------
        builder : trt.Builder
            Builder to create optimization profile
        config : trt.IBuilderConfig
            IBuilderConfig to add the new profile

        Notes
        -----
        See `https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes`_ for more
        information
        """
        profile = builder.create_optimization_profile()
        for tname, profiles in self.opt_profiles.items():
            profile.set_shape(tname, *profiles)
        config.add_optimization_profile(profile)

    def check_dynamic_axes(self, engine):
        """Setup each dynamic axis on engine with their profiles to check whether or not they are dynamic

        Parameters
        ----------
        engine : trt.ICudaEngine
            Engine exported

        Raises
        ------
        Exception
            The engine was not correctly exported: it is not possible to configure the dimensions of the tensioners
            with their corresponding profiles.
        """
        context = engine.create_execution_context()
        shapes_specified = context.all_binding_shapes_specified
        for tname, profiles in self.opt_profiles.items():
            binding_idx = engine.get_binding_index(tname)
            for shape in profiles:
                if not context.set_binding_shape(binding_idx, shape):
                    error = f"Impossible to generate dynamic axis for '{tname}'. "
                    error += "Proof a fixed profile (min=opt=max) with different sample_inputs shapes "
                    error += "to find the possible error."
                    raise Exception(error)
        assert not shapes_specified, "Incorrect engine. All shapes are statics"

    def get_engine(self):
        """Setup engine builder, read ONNX graph and build TensorRT engine.

        Returns
        -------
        trt.ICudaEngine
            Engine created from ONNX graph.

        Raises
        ------
        RuntimeError
            INT8 not supported by the platform.
        Exception
            TRT export engine error. It was not possible to export the engine.
        """
        global network_creation_flag
        with trt.Builder(self.logger) as builder, builder.create_network(
            network_creation_flag
        ) as network, trt.OnnxParser(network, self.logger) as parser:
            builder.max_batch_size = 1
            config = builder.create_builder_config()
            config.max_workspace_size = self.max_workspace_size
            # FP16
            if self.FP16_allowed:
                config.set_flag(trt.BuilderFlag.FP16)
            # INT8
            if self.INT8_allowed:
                if not builder.platform_has_fast_int8:
                    raise RuntimeError('INT8 not supported on this platform')
                config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = self.calibrator
            if self.strict_type:
                config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            # Add optimization profile (used for dynamic shapes)
            if self.opt_profiles is not None:
                self.setup_profile(builder, config)

            # Load and build model
            with open(self.onnx_file_path, "rb") as model:
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        parser_error = parser.get_error(error)
                        print(parser_error)
                        print("code:", parser_error.code())
                        print("desc:", parser_error.desc())
                        print("file:", parser_error.file())
                        print("func:", parser_error.func())
                        print("line:", parser_error.line())
                        print("node:", parser_error.node())
                    return None
                else:
                    print("ONNX file is loaded")

            print("Building engine...")
            engine = builder.build_engine(network, config)

            if engine is None:
                raise Exception("TRT export engine error. Check log")

            # Sanity check for dynamic axes
            if self.opt_profiles is not None:
                self.check_dynamic_axes(engine)

            print("Engine built")
            self.engine = engine
        return engine

    def export_engine(self, engine_path: str):
        """Seriazlize TensorRT engine

        Parameters
        ----------
        engine_path : str
            Path to save the engine

        Returns
        -------
        str
            Path where engine was exported
        """
        engine = self.get_engine()
        assert engine is not None, "Error while parsing engine from ONNX"
        with open(engine_path, "wb") as f:
            print("Seriazlized engine: " + engine_path)
            f.write(engine.serialize())
        print("Engine exported\n")
        return engine_path
