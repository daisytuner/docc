import inspect
import json
import shutil
import textwrap
import ast
import os
import getpass
import hashlib
import ml_dtypes
import numpy as np
from typing import Annotated, get_origin, get_args, Any, Optional
import time

from docc.sdfg import (
    Scalar,
    PrimitiveType,
    Pointer,
    Structure,
    Array,
    Type,
    Tensor,
    StructuredSDFG,
    StructuredSDFGBuilder,
    DoccMetrics,
)
from docc.compiler.docc_program import DoccProgram, DoccOptions
from docc.compiler.compiled_sdfg import CompiledSDFG
from docc.python.ast_parser import ASTParser
from docc.python.type_system import element_type_from_sdfg_type, scalar_type_for_dtype


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

    # Handle a parametrized numpy dtype generic, e.g. numpy.dtype[numpy.float64]
    # (produced by npt.NDArray[RealT] -> ndarray[Any, dtype[float64]]).
    if get_origin(dtype) is np.dtype:
        inner = get_args(dtype)
        if inner:
            return _map_python_type(inner[0])

    # Simple mapping for python/numpy scalar types via the shared dtype table.
    scalar = scalar_type_for_dtype(dtype)
    if scalar is not None:
        return scalar

    # Handle Python classes - map to Structure type
    if inspect.isclass(dtype):
        # Use the class name as the structure name
        return Pointer(Structure(dtype.__name__))

    return dtype


class PythonProgram(DoccProgram):

    def __init__(self, func, options: Optional[DoccOptions] = None):
        super().__init__(
            name=func.__name__,
            options=options or DoccOptions(),
        )
        self.func = func
        self._last_structure_member_info = {}

    def __call__(self, *args: Any) -> Any:
        # JIT compile and run. CompiledSDFG validates the call mode (numpy /
        # cupy / torch) and rejects GPU arrays on non-device-resident artifacts.
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
    ) -> CompiledSDFG:
        original_output_folder = output_folder

        metrics = DoccMetrics()
        compile_start_time = time.perf_counter()
        metrics.add_metric("function", self.name, "source")
        metrics.add_frontend_source_info("python")

        # Binary reuse (DOCC_REUSE_BINARIES) reloads a previously built .so and
        # its persisted SDFG; DoccOptions forces the SDFG dump that produces it.
        docc_reuse_binaries = self.options.reuse_binaries

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

        # 2. Signature - include scalar-shape equivalences for correct caching
        mapping_sig = sorted(arg_shape_mapping.items())
        type_sig = ", ".join(self._type_to_str(t) for t in arg_types)
        signature = f"{type_sig}|{mapping_sig}"

        # With concrete (non-symbolic) shapes the SDFG bakes in the actual sizes,
        # so identically typed calls with different shapes must not alias in the
        # cache; fold the shape values into the signature in that mode.
        if not self.options.symbolic_shapes:
            signature = f"{signature}|concrete{shape_values}"

        # In-memory cache key: the structural signature plus the resolved compile
        # options, so repeated in-process compiles with different
        # instrumentation/arg-capture/remote-tuning do not alias to the first
        # built binary (the on-disk hash already accounts for these options).
        mem_cache_key = f"{signature}|{self.options.capture_args}|{self.options.instrumentation_mode}|{self.options.remote_tuning}"

        if output_folder is None:
            source_path = inspect.getsourcefile(self.func)
            hash_input = f"{source_path}|{self.name}|{self.options.target}|{self.options.category}|{self.options.capture_args}|{self.options.instrumentation_mode}|{self.options.remote_tuning}|{signature}".encode(
                "utf-8"
            )
            stable_id = hashlib.sha256(hash_input).hexdigest()[:16]
            filename = os.path.basename(inspect.getsourcefile(self.func))

            # Isolate parallel pytest-xdist workers so concurrent compilations of
            # identically named kernels don't share (and clobber) one build dir.
            worker = os.environ.get("PYTEST_XDIST_WORKER", "")
            worker_suffix = f"-{worker}" if worker else ""

            docc_tmp = os.environ.get("DOCC_TMP")
            if docc_tmp:
                output_folder = f"{docc_tmp}/{filename}-{self.name}-{self.options.target}-{stable_id}{worker_suffix}"
            else:
                user = os.getenv("USER")
                if not user:
                    user = getpass.getuser()
                output_folder = (
                    f"/tmp/{user}/DOCC/{self.name}-{stable_id}{worker_suffix}"
                )

        if original_output_folder is None and mem_cache_key in self.cache:
            return self.cache[mem_cache_key]

        # 3. Reuse a previously built binary if requested and available.
        # Structure arguments need per-member layout info that is only produced
        # while parsing the kernel, so reuse is limited to plain array/scalar
        # kernels; anything else falls through to a full rebuild.
        has_struct_args = any(
            isinstance(t, Pointer)
            and t.has_pointee_type()
            and isinstance(t.pointee_type, Structure)
            for t in arg_types
        )
        if docc_reuse_binaries and not has_struct_args:
            reused = self._try_reuse_binary(output_folder, shape_sources)
            if reused is not None:
                if original_output_folder is None:
                    self.cache[mem_cache_key] = reused

                metrics.capture_env_vars()
                metrics.append_to(output_folder)
                return reused

        # 4. Build SDFG
        if os.path.exists(output_folder):
            # Multiple python processes running the same code?
            shutil.rmtree(output_folder)

        sdfg, out_args, out_shapes, out_strides = self._build_sdfg(
            arg_types, args, arg_shape_mapping, shape_values
        )
        parse_sdfg_time = time.perf_counter() - compile_start_time
        metrics.add_metric(
            "parse_to_sdfg_time_ms", round(parse_sdfg_time * 1000), "compile_times"
        )

        lib_path = self.sdfg_pipe(
            sdfg,
            output_folder,
            metrics=metrics,
        )

        # Persist the return-value layout so a later DOCC_REUSE_BINARIES run can
        # rebuild the CompiledSDFG without re-parsing the kernel.
        if output_folder:
            self._persist_return_layout(output_folder, sdfg, out_shapes, out_strides)

        # 5. Create CompiledSDFG
        compiled = CompiledSDFG(
            lib_path,
            sdfg,
            shape_sources,
            self._last_structure_member_info,
            out_args,
            out_shapes,
            out_strides,
            device_resident=self._device_resident,
            device_backend=self._device_backend,
            target=self.options.target,
        )

        # Cache if using default output folder
        if original_output_folder is None:
            self.cache[mem_cache_key] = compiled

        compile_time_ms = round((time.perf_counter() - compile_start_time) * 1000)
        metrics.add_metric("compile_time_ms", compile_time_ms, "compile_times")
        metrics.capture_env_vars()
        metrics.append_to(output_folder)

        return compiled

    def _persist_return_layout(
        self, output_folder: str, sdfg: StructuredSDFG, out_shapes, out_strides
    ) -> None:
        """Stamp the return-value layout into the persisted SDFG metadata.

        The return shapes/strides are discovered while parsing the kernel and
        are not otherwise recoverable from the SDFG structure. Persisting them
        (into the same ``py5.post_sched.json`` the reuse path loads) lets a later
        ``DOCC_REUSE_BINARIES`` run reconstruct the CompiledSDFG without
        re-parsing/recompiling.
        """
        json_path = os.path.join(output_folder, f"{sdfg.name}.py5.post_sched.json")
        if not os.path.exists(json_path):
            return
        try:
            with open(json_path) as f:
                data = json.load(f)
            metadata = data.setdefault("metadata", {})
            metadata["output_shapes"] = json.dumps(out_shapes)
            metadata["output_strides"] = json.dumps(out_strides)
            with open(json_path, "w") as f:
                json.dump(data, f)
        except (OSError, ValueError):
            pass

    def _try_reuse_binary(
        self, output_folder: Optional[str], shape_sources
    ) -> Optional[CompiledSDFG]:
        """Reload a cached ``.so`` + normalized SDFG instead of recompiling.

        Mirrors the strictness of the pytorch/mlir binary-reuse path: when the
        cache directory does not exist yet this returns ``None`` so the caller
        performs a first build (no error); but when the directory *does* exist
        and a required artifact is missing, it raises ``ValueError`` so a broken
        or stale cache surfaces loudly instead of silently recompiling. The
        calling convention (device residency) and return-value layout are
        restored from the persisted SDFG metadata so arguments are marshalled
        exactly as they were at build time.
        """
        if not output_folder:
            return None

        # Cache directory absent -> first build. Let the caller build it; this
        # is the one case the mlir/pytorch frontends also treat as non-fatal.
        if not os.path.exists(output_folder):
            return None

        sdfg_name = f"{self.name}_sdfg"
        lib_path = os.path.join(output_folder, f"lib{sdfg_name}.so")
        json_path = os.path.join(output_folder, f"{sdfg_name}.py5.post_sched.json")
        if not os.path.exists(lib_path):
            raise ValueError(f"Tried reusing binary '{lib_path}' but does not exist")
        if not os.path.exists(json_path):
            raise ValueError(f"Tried loading SDFG '{json_path}' but does not exist")

        sdfg = StructuredSDFG.from_file(json_path)

        # Return arguments are recoverable directly from the SDFG signature.
        out_args = [name for name in sdfg.arguments if name.startswith("_docc_ret_")]

        # Return-value layout was persisted into the SDFG metadata at build time.
        out_shapes = {}
        out_strides = {}
        shapes_meta = sdfg.metadata("output_shapes")
        strides_meta = sdfg.metadata("output_strides")
        if shapes_meta:
            try:
                out_shapes = json.loads(shapes_meta)
            except ValueError:
                out_shapes = {}
        if strides_meta:
            try:
                out_strides = json.loads(strides_meta)
            except ValueError:
                out_strides = {}

        # Restore the device-residency calling convention chosen at compile time;
        # otherwise a device-resident binary would be fed host pointers.
        self._device_resident = sdfg.metadata("device_resident") == "1"
        backend = sdfg.metadata("device_backend")
        self._device_backend = backend or None

        return CompiledSDFG(
            lib_path,
            sdfg,
            shape_sources,
            {},
            out_args,
            out_shapes,
            out_strides,
            device_resident=self._device_resident,
            device_backend=self._device_backend,
            target=self.options.target,
        )

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

        sdfg, _, _, _ = self._build_sdfg(
            arg_types, args, arg_shape_mapping, shape_values
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
        if isinstance(arg, np.ndarray):
            elem_type = scalar_type_for_dtype(arg.dtype)
            if elem_type is None:
                raise ValueError(f"Unsupported numpy dtype: {arg.dtype}")
            return Pointer(elem_type)
        if isinstance(arg, str):
            # Explicitly reject strings - they are not supported
            raise ValueError(f"Unsupported argument type: {type(arg)}")

        scalar = scalar_type_for_dtype(type(arg))
        if scalar is not None:
            return scalar

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
        shape_values,
    ):
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
                            member_shapes = []
                            for attr_name, attr_value in sorted(arg.__dict__.items()):
                                if not attr_name.startswith("_"):
                                    # Infer member type from instance attribute
                                    # Check bool before int since bool is subclass of int
                                    member_type = None
                                    member_shape = None
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
                                    elif isinstance(attr_value, np.ndarray):
                                        # Array member: stored as a pointer field
                                        # (struct-of-arrays). Record the concrete
                                        # shape so attribute access can build a
                                        # tensor view over the member pointer.
                                        member_type = self._infer_type(attr_value)
                                        member_shape = [
                                            str(int(s)) for s in attr_value.shape
                                        ]
                                    # TODO: Consider using np.integer and np.floating abstract types
                                    # for more comprehensive numpy type coverage
                                    # TODO: Add support for nested structures

                                    if member_type is not None:
                                        member_types.append(member_type)
                                        member_names.append(attr_name)
                                        member_shapes.append(member_shape)

                            if member_types:
                                structures_to_register[struct_name] = member_types
                                # Build member name to (index, type, shape) mapping.
                                # shape is None for scalar members and a list of
                                # dimension-size strings for array members.
                                structure_member_info[struct_name] = {
                                    name: (idx, mtype, shape)
                                    for idx, (name, mtype, shape) in enumerate(
                                        zip(member_names, member_types, member_shapes)
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

        # Add regular arguments
        tensor_table = {}
        for i, ((name, param), dtype, arg) in enumerate(zip(params, arg_types, args)):
            builder.add_container(name, dtype, is_argument=True)

            # Store layout information for arrays
            if isinstance(arg, np.ndarray):
                element_type = element_type_from_sdfg_type(dtype)

                shapes = []
                for dim_idx in range(arg.ndim):
                    dim_val = arg.shape[dim_idx]
                    if dim_val == 1:
                        # Always use literal "1" for size-1 dimensions to enable
                        # proper broadcasting detection
                        shapes.append("1")
                    elif not self.options.symbolic_shapes:
                        # Bake in the concrete integer size instead of a symbol.
                        shapes.append(str(int(dim_val)))
                    else:
                        u_idx = arg_shape_mapping[(i, dim_idx)]
                        shapes.append(f"_s{u_idx}")

                strides = []
                if arg.flags["C_CONTIGUOUS"]:
                    # Row-major: stride[i] = product of shapes[i+1:]
                    for dim_idx in range(arg.ndim):
                        if dim_idx == arg.ndim - 1:
                            strides.append("1")
                        else:
                            suffix_shapes = shapes[dim_idx + 1 :]
                            if len(suffix_shapes) == 1:
                                strides.append(suffix_shapes[0])
                            else:
                                strides.append("(" + " * ".join(suffix_shapes) + ")")
                elif arg.flags["F_CONTIGUOUS"]:
                    # Column-major: stride[i] = product of shapes[:i]
                    for dim_idx in range(arg.ndim):
                        if dim_idx == 0:
                            strides.append("1")
                        else:
                            prefix_shapes = shapes[:dim_idx]
                            if len(prefix_shapes) == 1:
                                strides.append(prefix_shapes[0])
                            else:
                                strides.append("(" + " * ".join(prefix_shapes) + ")")
                else:
                    # Non-contiguous: use actual stride values
                    for dim_idx in range(arg.ndim):
                        stride_val = arg.strides[dim_idx] // arg.itemsize
                        strides.append(f"{stride_val}")

                offset = "0"
                tensor_table[name] = Tensor(element_type, shapes, strides, offset)

            elif isinstance(arg, np.generic):
                # NumPy scalar types (np.float64, np.int32, etc.) should be treated
                # as 0-d arrays for type promotion purposes - they trigger full
                # promotion, unlike Python literals which adapt to the array dtype
                element_type = element_type_from_sdfg_type(dtype)
                tensor_table[name] = Tensor(element_type, [], [], "0")

        # Add unified shape arguments only for shapes without scalar equivalents
        # and skip size-1 dimensions (they use literal "1" instead). Symbolic
        # shapes are disabled entirely when the concrete sizes are baked in.
        if self.options.symbolic_shapes:
            for i in range(len(shape_values)):
                if shape_values[i] != 1:
                    builder.add_container(
                        f"_s{i}", Scalar(PrimitiveType.Int64), is_argument=True
                    )
                    builder.add_assumption_lb(f"_s{i}", "1")  # Shapes must be positive
                    builder.add_assumption_const(f"_s{i}", True)  # Shapes are constant

        # Create symbol table for parser
        container_table = {}
        for i, ((name, param), dtype, arg) in enumerate(zip(params, arg_types, args)):
            container_table[name] = dtype

        if self.options.symbolic_shapes:
            for i in range(len(shape_values)):
                if shape_values[i] != 1:
                    container_table[f"_s{i}"] = Scalar(PrimitiveType.Int64)

        # Parse AST
        source_lines, start_line = inspect.getsourcelines(self.func)
        source = textwrap.dedent("".join(source_lines))
        tree = ast.parse(source)
        ast.increment_lineno(tree, start_line - 1)
        func_def = tree.body[0]

        filename = inspect.getsourcefile(self.func)
        function_name = self.func.__name__

        # Combine globals with closure variables (closure takes precedence)
        combined_globals = dict(self.func.__globals__)
        if self.func.__closure__ is not None and self.func.__code__.co_freevars:
            for name, cell in zip(
                self.func.__code__.co_freevars, self.func.__closure__
            ):
                combined_globals[name] = cell.cell_contents

        parser = ASTParser(
            builder,
            tensor_table,
            container_table,
            filename,
            function_name,
            infer_return_type=infer_return_type,
            globals_dict=combined_globals,
            structure_member_info=structure_member_info,
        )
        for node in func_def.body:
            parser.visit(node)

        # Emit hoisted allocations at function entry
        parser.memory_handler.emit_allocations()

        sdfg = builder.move()
        # Mark return arguments metadata
        out_args = []
        for name in sdfg.arguments:
            if name.startswith("_docc_ret_"):
                out_args.append(name)

        return (
            sdfg,
            out_args,
            parser.captured_return_shapes,
            parser.captured_return_strides,
        )


def native(func=None, **options: Any):
    """Decorator to create a PythonProgram from a Python function.

    Keyword options are forwarded to :class:`DoccOptions` (e.g. ``target``,
    ``category``, ``instrumentation_mode``, ``capture_args``, ``remote_tuning``,
    ``einsum``, ``symbolic_shapes``).

    Example:
        @native
        def my_function(x: np.ndarray) -> np.ndarray:
            return x * 2

        result = my_function(np.array([1.0, 2.0, 3.0]))
    """
    docc_options = DoccOptions.from_kwargs(**options)
    if func is None:
        return lambda f: PythonProgram(f, docc_options)
    return PythonProgram(func, docc_options)
