import inspect
import shutil
import textwrap
import ast
import os
import getpass
import hashlib
import numpy as np
from typing import Annotated, get_origin, get_args, Any, Optional

from docc.sdfg import (
    Scalar,
    PrimitiveType,
    Pointer,
    Structure,
    Array,
    Type,
    StructuredSDFG,
    StructuredSDFGBuilder,
)
from docc.compiler.docc_program import DoccProgram
from docc.compiler.compiled_sdfg import CompiledSDFG
from docc.python.ast_parser import ASTParser


def _compile_wrapper(self, output_folder=None):
    """Wrapper to allow StructuredSDFG.compile() to return a CompiledSDFG."""
    lib_path = self._compile(output_folder)
    return CompiledSDFG(lib_path, self)


# Monkey-patch StructuredSDFG to add compile method
StructuredSDFG.compile = _compile_wrapper


def _map_python_type(dtype):
    """Map Python/numpy types to SDFG types."""
    # If it is already a sdfg Type, return it
    if isinstance(dtype, Type):
        return dtype

    # Handle Annotated for Arrays
    if get_origin(dtype) is Annotated:
        args = get_args(dtype)
        base_type = args[0]
        metadata = args[1:]

        if base_type is np.ndarray:
            # Convention: Annotated[np.ndarray, shape, dtype]
            shape = metadata[0]
            elem_type = Scalar(PrimitiveType.Double)  # Default

            if len(metadata) > 1:
                possible_dtype = metadata[1]
                elem_type = _map_python_type(possible_dtype)

            return Pointer(elem_type)

    # Handle numpy.ndarray[Shape, DType]
    if get_origin(dtype) is np.ndarray:
        args = get_args(dtype)
        # args[0] is shape, args[1] is dtype
        if len(args) >= 2:
            elem_type = _map_python_type(args[1])
            return Pointer(elem_type)

    # Simple mapping for python types
    if dtype is float or dtype is np.float64:
        return Scalar(PrimitiveType.Double)
    elif dtype is np.float32:
        return Scalar(PrimitiveType.Float)
    elif dtype is bool or dtype is np.bool_:
        return Scalar(PrimitiveType.Bool)
    elif dtype is int or dtype is np.int64:
        return Scalar(PrimitiveType.Int64)
    elif dtype is np.int32:
        return Scalar(PrimitiveType.Int32)
    elif dtype is np.int16:
        return Scalar(PrimitiveType.Int16)
    elif dtype is np.int8:
        return Scalar(PrimitiveType.Int8)
    elif dtype is np.uint64:
        return Scalar(PrimitiveType.UInt64)
    elif dtype is np.uint32:
        return Scalar(PrimitiveType.UInt32)
    elif dtype is np.uint16:
        return Scalar(PrimitiveType.UInt16)
    elif dtype is np.uint8:
        return Scalar(PrimitiveType.UInt8)

    # Handle Python classes - map to Structure type
    if inspect.isclass(dtype):
        # Use the class name as the structure name
        return Pointer(Structure(dtype.__name__))

    return dtype


class PythonProgram(DoccProgram):

    def __init__(
        self,
        func,
        target: str = "none",
        category: str = "server",
        instrumentation_mode: Optional[str] = None,
        capture_args: Optional[bool] = None,
        remote_tuning: bool = False,
    ):
        super().__init__(
            name=func.__name__,
            target=target,
            category=category,
            instrumentation_mode=instrumentation_mode,
            capture_args=capture_args,
            remote_tuning=remote_tuning,
        )
        self.func = func
        self._last_structure_member_info = {}

    def __call__(self, *args: Any) -> Any:
        # JIT compile and run
        compiled = self.compile(*args)
        res = compiled(*args)

        # Handle return value conversion based on annotation
        sig = inspect.signature(self.func)
        ret_annotation = sig.return_annotation

        if ret_annotation is not inspect.Signature.empty:
            if get_origin(ret_annotation) is Annotated:
                type_args = get_args(ret_annotation)
                if len(type_args) >= 1 and type_args[0] is np.ndarray:
                    shape = None
                    if len(type_args) >= 2:
                        shape = type_args[1]

                    if shape is not None:
                        try:
                            return np.ctypeslib.as_array(res, shape=shape)
                        except Exception:
                            pass

        # Try to infer return shape from metadata
        if hasattr(compiled, "get_return_shape"):
            shape = compiled.get_return_shape(*args)
            if shape is not None:
                try:
                    return np.ctypeslib.as_array(res, shape=shape)
                except Exception:
                    pass

        return res

    def compile(
        self,
        *args: Any,
        output_folder: Optional[str] = None,
        instrumentation_mode: Optional[str] = None,
        capture_args: Optional[bool] = None,
    ) -> CompiledSDFG:
        original_output_folder = output_folder

        # Resolve options
        if instrumentation_mode is None:
            instrumentation_mode = self.instrumentation_mode
        if capture_args is None:
            capture_args = self.capture_args

        # Check environment variable DOCC_CI
        docc_ci = os.environ.get("DOCC_CI", "")
        if docc_ci:
            if docc_ci == "regions":
                if instrumentation_mode is None:
                    instrumentation_mode = "ols"
            elif docc_ci == "arg-capture":
                if capture_args is None:
                    capture_args = True
            else:
                # Full mode (or unknown value treated as full)
                if instrumentation_mode is None:
                    instrumentation_mode = "ols"
                if capture_args is None:
                    capture_args = True

        # Defaults
        if instrumentation_mode is None:
            instrumentation_mode = ""
        if capture_args is None:
            capture_args = False

        # 1. Analyze arguments and shapes
        arg_types = []
        shape_values = []  # List of unique shape values found
        shape_sources = []  # List of (arg_idx, dim_idx) for each unique shape value

        # Mapping from (arg_idx, dim_idx) -> unique_shape_idx
        arg_shape_mapping = {}

        # First pass: collect scalar integer arguments and their values
        sig = inspect.signature(self.func)
        params = list(sig.parameters.items())
        scalar_int_params = {}  # Maps value -> parameter name (first one wins)
        for i, ((name, param), arg) in enumerate(zip(params, args)):
            if isinstance(arg, (int, np.integer)) and not isinstance(
                arg, (bool, np.bool_)
            ):
                val = int(arg)
                if val not in scalar_int_params:
                    scalar_int_params[val] = name

        for i, arg in enumerate(args):
            t = self._infer_type(arg)
            arg_types.append(t)

            if isinstance(arg, np.ndarray):
                for dim_idx, dim_val in enumerate(arg.shape):
                    # Check if we've seen this value
                    if dim_val in shape_values:
                        # Reuse
                        u_idx = shape_values.index(dim_val)
                    else:
                        # New
                        u_idx = len(shape_values)
                        shape_values.append(dim_val)
                        shape_sources.append((i, dim_idx))

                    arg_shape_mapping[(i, dim_idx)] = u_idx

        # Detect scalar-shape equivalences: which shape indices have a matching scalar param
        # Maps unique_shape_idx -> scalar parameter name
        shape_to_scalar = {}
        for s_idx, s_val in enumerate(shape_values):
            if s_val in scalar_int_params:
                shape_to_scalar[s_idx] = scalar_int_params[s_val]

        # 2. Signature - include scalar-shape equivalences for correct caching
        mapping_sig = sorted(arg_shape_mapping.items())
        equiv_sig = sorted(shape_to_scalar.items())
        type_sig = ", ".join(self._type_to_str(t) for t in arg_types)
        signature = f"{type_sig}|{mapping_sig}|{equiv_sig}"

        if output_folder is None:
            filename = inspect.getsourcefile(self.func)
            hash_input = f"{filename}|{self.name}|{self.target}|{self.category}|{self.capture_args}|{self.instrumentation_mode}|{signature}".encode(
                "utf-8"
            )
            stable_id = hashlib.sha256(hash_input).hexdigest()[:16]

            docc_tmp = os.environ.get("DOCC_TMP")
            if docc_tmp:
                output_folder = f"{docc_tmp}/{self.name}-{stable_id}"
            else:
                user = os.getenv("USER")
                if not user:
                    user = getpass.getuser()
                output_folder = f"/tmp/{user}/DOCC/{self.name}-{stable_id}"

        if original_output_folder is None and signature in self.cache:
            return self.cache[signature]

        # 3. Build SDFG
        if os.path.exists(output_folder):
            # Multiple python processes running the same code?
            shutil.rmtree(output_folder)
        sdfg, out_args, out_shapes = self._build_sdfg(
            arg_types, args, arg_shape_mapping, len(shape_values), shape_to_scalar
        )
        sdfg.validate()

        # Tensor targets keep tensor nodes
        if self.target != "onnx":
            sdfg.expand()

        # Simplify pipelines
        sdfg.simplify()

        # Normalization for scheduling
        if self.target != "none":
            sdfg.normalize()

        sdfg.dump(output_folder)

        # Schedule if target is specified
        if self.target != "none":
            sdfg.schedule(self.target, self.category, self.remote_tuning)

        self.last_sdfg = sdfg

        lib_path = sdfg._compile(
            output_folder=output_folder,
            target=self.target,
            instrumentation_mode=instrumentation_mode,
            capture_args=capture_args,
        )

        # Build ONNX model from JSON if target is onnx (after _compile creates the JSON)
        if self.target == "onnx":
            from docc.python.onnx_model_builder import convert_json_to_onnx

            onnx_model_path = convert_json_to_onnx(output_folder)
            if onnx_model_path:
                print(f"Generated ONNX models: {onnx_model_path}")

        # 5. Create CompiledSDFG
        compiled = CompiledSDFG(
            lib_path,
            sdfg,
            shape_sources,
            self._last_structure_member_info,
            out_args,
            out_shapes,
        )

        # Cache if using default output folder
        if original_output_folder is None:
            self.cache[signature] = compiled

        return compiled

    def to_sdfg(self, *args: Any) -> StructuredSDFG:
        arg_types = [self._infer_type(arg) for arg in args]

        # Build shape mapping
        shape_values = []
        shape_sources = []
        arg_shape_mapping = {}

        sig = inspect.signature(self.func)
        params = list(sig.parameters.items())
        scalar_int_params = {}
        for i, ((name, param), arg) in enumerate(zip(params, args)):
            if isinstance(arg, (int, np.integer)) and not isinstance(
                arg, (bool, np.bool_)
            ):
                val = int(arg)
                if val not in scalar_int_params:
                    scalar_int_params[val] = name

        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                for dim_idx, dim_val in enumerate(arg.shape):
                    if dim_val in shape_values:
                        u_idx = shape_values.index(dim_val)
                    else:
                        u_idx = len(shape_values)
                        shape_values.append(dim_val)
                        shape_sources.append((i, dim_idx))
                    arg_shape_mapping[(i, dim_idx)] = u_idx

        shape_to_scalar = {}
        for s_idx, s_val in enumerate(shape_values):
            if s_val in scalar_int_params:
                shape_to_scalar[s_idx] = scalar_int_params[s_val]

        sdfg, _, _ = self._build_sdfg(
            arg_types, args, arg_shape_mapping, len(shape_values), shape_to_scalar
        )
        return sdfg

    def _convert_inputs(self, args: tuple) -> tuple:
        return args

    def _convert_outputs(self, result: Any, original_args: tuple) -> Any:
        return result

    def _get_signature(self, arg_types):
        return ", ".join(self._type_to_str(t) for t in arg_types)

    def _type_to_str(self, t):
        if isinstance(t, Scalar):
            return f"Scalar({t.primitive_type})"
        elif isinstance(t, Array):
            return f"Array({self._type_to_str(t.element_type)}, {t.num_elements})"
        elif isinstance(t, Pointer):
            return f"Pointer({self._type_to_str(t.pointee_type)})"
        elif isinstance(t, Structure):
            return f"Structure({t.name})"
        return str(t)

    def _infer_type(self, arg):
        if isinstance(arg, (float, np.float64)):
            return Scalar(PrimitiveType.Double)
        elif isinstance(arg, np.float32):
            return Scalar(PrimitiveType.Float)
        elif isinstance(arg, (bool, np.bool_)):
            return Scalar(PrimitiveType.Bool)
        elif isinstance(arg, (int, np.int64)):
            return Scalar(PrimitiveType.Int64)
        elif isinstance(arg, np.int32):
            return Scalar(PrimitiveType.Int32)
        elif isinstance(arg, np.int16):
            return Scalar(PrimitiveType.Int16)
        elif isinstance(arg, np.int8):
            return Scalar(PrimitiveType.Int8)
        elif isinstance(arg, np.uint64):
            return Scalar(PrimitiveType.UInt64)
        elif isinstance(arg, np.uint32):
            return Scalar(PrimitiveType.UInt32)
        elif isinstance(arg, np.uint16):
            return Scalar(PrimitiveType.UInt16)
        elif isinstance(arg, np.uint8):
            return Scalar(PrimitiveType.UInt8)
        elif isinstance(arg, np.ndarray):
            # Map dtype
            if arg.dtype == np.float64:
                elem_type = Scalar(PrimitiveType.Double)
            elif arg.dtype == np.float32:
                elem_type = Scalar(PrimitiveType.Float)
            elif arg.dtype == np.bool_:
                elem_type = Scalar(PrimitiveType.Bool)
            elif arg.dtype == np.int64:
                elem_type = Scalar(PrimitiveType.Int64)
            elif arg.dtype == np.int32:
                elem_type = Scalar(PrimitiveType.Int32)
            elif arg.dtype == np.int16:
                elem_type = Scalar(PrimitiveType.Int16)
            elif arg.dtype == np.int8:
                elem_type = Scalar(PrimitiveType.Int8)
            elif arg.dtype == np.uint64:
                elem_type = Scalar(PrimitiveType.UInt64)
            elif arg.dtype == np.uint32:
                elem_type = Scalar(PrimitiveType.UInt32)
            elif arg.dtype == np.uint16:
                elem_type = Scalar(PrimitiveType.UInt16)
            elif arg.dtype == np.uint8:
                elem_type = Scalar(PrimitiveType.UInt8)
            else:
                raise ValueError(f"Unsupported numpy dtype: {arg.dtype}")

            return Pointer(elem_type)
        elif isinstance(arg, str):
            # Explicitly reject strings - they are not supported
            raise ValueError(f"Unsupported argument type: {type(arg)}")
        else:
            # Check if it's a class instance
            if hasattr(arg, "__class__") and not isinstance(arg, type):
                # It's an instance of a class, return pointer to Structure
                return Pointer(Structure(arg.__class__.__name__))
            raise ValueError(f"Unsupported argument type: {type(arg)}")

    def _build_sdfg(
        self,
        arg_types,
        args,
        arg_shape_mapping,
        num_unique_shapes,
        shape_to_scalar=None,
    ):
        if shape_to_scalar is None:
            shape_to_scalar = {}
        sig = inspect.signature(self.func)

        # Handle return type - always void for SDFG, output args used for returns
        return_type = Scalar(PrimitiveType.Void)
        infer_return_type = True

        # Parse return annotation to determine output arguments if possible
        explicit_returns = []
        if sig.return_annotation is not inspect.Signature.empty:
            infer_return_type = False

            # Helper to normalize annotation to list of types
            def normalize_annotation(ann):
                # Handle Tuple[type, ...]
                origin = get_origin(ann)
                if origin is tuple:
                    type_args = get_args(ann)
                    # Tuple[()] or Tuple w/o args
                    if not type_args:
                        return []
                    # Tuple[int, float]
                    if len(type_args) > 0 and type_args[-1] is not Ellipsis:
                        return [_map_python_type(t) for t in type_args]
                    # Tuple[int, ...] - not supported for fixed number of returns yet?
                    # For now assume fixed tuple
                    return [_map_python_type(t) for t in type_args]
                else:
                    return [_map_python_type(ann)]

            explicit_returns = normalize_annotation(sig.return_annotation)
            for rt in explicit_returns:
                if not isinstance(rt, Type):
                    # Fallback if map failed (e.g. invalid annotation)
                    infer_return_type = True
                    explicit_returns = []
                    break

        builder = StructuredSDFGBuilder(f"{self.name}_sdfg", return_type)

        # Add pre-defined return arguments if we know them
        if not infer_return_type:
            for i, dtype in enumerate(explicit_returns):
                # Scalar -> Pointer(Scalar)
                # Array -> Already Pointer(Scalar). Keep it.
                arg_type = dtype
                if isinstance(dtype, Scalar):
                    arg_type = Pointer(dtype)

                builder.add_container(f"_docc_ret_{i}", arg_type, is_argument=True)

        # Register structure types for any class arguments
        # Also track member name to index mapping for each structure
        structures_to_register = {}
        structure_member_info = {}  # Maps struct_name -> {member_name: (index, type)}
        for i, (arg, dtype) in enumerate(zip(args, arg_types)):
            if isinstance(dtype, Pointer) and dtype.has_pointee_type():
                pointee = dtype.pointee_type
                if isinstance(pointee, Structure):
                    struct_name = pointee.name
                    if struct_name not in structures_to_register:
                        # Get class from arg to introspect members
                        if hasattr(arg, "__dict__"):
                            # Use __dict__ to get only instance attributes
                            # Sort by name to ensure consistent ordering
                            # Note: This alphabetical ordering is used to define the
                            # structure layout and must match the order expected by
                            # the backend code generation
                            member_types = []
                            member_names = []
                            for attr_name, attr_value in sorted(arg.__dict__.items()):
                                if not attr_name.startswith("_"):
                                    # Infer member type from instance attribute
                                    # Check bool before int since bool is subclass of int
                                    member_type = None
                                    if isinstance(attr_value, bool):
                                        member_type = Scalar(PrimitiveType.Bool)
                                    elif isinstance(attr_value, (int, np.int64)):
                                        member_type = Scalar(PrimitiveType.Int64)
                                    elif isinstance(attr_value, (float, np.float64)):
                                        member_type = Scalar(PrimitiveType.Double)
                                    elif isinstance(attr_value, np.int32):
                                        member_type = Scalar(PrimitiveType.Int32)
                                    elif isinstance(attr_value, np.float32):
                                        member_type = Scalar(PrimitiveType.Float)
                                    # TODO: Consider using np.integer and np.floating abstract types
                                    # for more comprehensive numpy type coverage
                                    # TODO: Add support for nested structures and arrays

                                    if member_type is not None:
                                        member_types.append(member_type)
                                        member_names.append(attr_name)

                            if member_types:
                                structures_to_register[struct_name] = member_types
                                # Build member name to (index, type) mapping
                                structure_member_info[struct_name] = {
                                    name: (idx, mtype)
                                    for idx, (name, mtype) in enumerate(
                                        zip(member_names, member_types)
                                    )
                                }

        # Store structure_member_info for later use in CompiledSDFG
        self._last_structure_member_info = structure_member_info

        # Register all discovered structures with the builder
        for struct_name, member_types in structures_to_register.items():
            builder.add_structure(struct_name, member_types)

        # Register arguments
        params = list(sig.parameters.items())
        if len(params) != len(arg_types):
            raise ValueError(
                f"Argument count mismatch: expected {len(params)}, got {len(arg_types)}"
            )

        array_info = {}

        # Add regular arguments
        for i, ((name, param), dtype, arg) in enumerate(zip(params, arg_types, args)):
            builder.add_container(name, dtype, is_argument=True)

            # If it's an array, prepare shape info
            if isinstance(arg, np.ndarray):
                shapes = []
                for dim_idx in range(arg.ndim):
                    u_idx = arg_shape_mapping[(i, dim_idx)]
                    # Use scalar parameter name if there's an equivalence, otherwise _sX
                    if u_idx in shape_to_scalar:
                        shapes.append(shape_to_scalar[u_idx])
                    else:
                        shapes.append(f"_s{u_idx}")

                array_info[name] = {"ndim": arg.ndim, "shapes": shapes}

        # Add unified shape arguments only for shapes without scalar equivalents
        for i in range(num_unique_shapes):
            if i not in shape_to_scalar:
                builder.add_container(
                    f"_s{i}", Scalar(PrimitiveType.Int64), is_argument=True
                )

        # Create symbol table for parser
        symbol_table = {}
        for i, ((name, param), dtype, arg) in enumerate(zip(params, arg_types, args)):
            symbol_table[name] = dtype

        for i in range(num_unique_shapes):
            if i not in shape_to_scalar:
                symbol_table[f"_s{i}"] = Scalar(PrimitiveType.Int64)

        # Parse AST
        source_lines, start_line = inspect.getsourcelines(self.func)
        source = textwrap.dedent("".join(source_lines))
        tree = ast.parse(source)
        ast.increment_lineno(tree, start_line - 1)
        func_def = tree.body[0]

        filename = inspect.getsourcefile(self.func)
        function_name = self.func.__name__

        parser = ASTParser(
            builder,
            array_info,
            symbol_table,
            filename,
            function_name,
            infer_return_type=infer_return_type,
            globals_dict=self.func.__globals__,
            structure_member_info=structure_member_info,
        )
        for node in func_def.body:
            parser.visit(node)

        sdfg = builder.move()
        # Mark return arguments metadata
        out_args = []
        for name in sdfg.arguments:
            if name.startswith("_docc_ret_"):
                out_args.append(name)

        return sdfg, out_args, parser.captured_return_shapes


def native(
    func=None,
    *,
    target="none",
    category="desktop",
    instrumentation_mode=None,
    capture_args=None,
):
    """Decorator to create a PythonProgram from a Python function.

    Example:
        @native
        def my_function(x: np.ndarray) -> np.ndarray:
            return x * 2

        result = my_function(np.array([1.0, 2.0, 3.0]))
    """
    if func is None:
        return lambda f: PythonProgram(
            f,
            target=target,
            category=category,
            instrumentation_mode=instrumentation_mode,
            capture_args=capture_args,
        )
    return PythonProgram(
        func,
        target=target,
        category=category,
        instrumentation_mode=instrumentation_mode,
        capture_args=capture_args,
    )
