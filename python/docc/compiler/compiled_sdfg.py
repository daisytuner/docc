import ctypes
import re
import time
import warnings
from docc.sdfg import Scalar, Array, Pointer, Structure, PrimitiveType

import numpy as np
import ml_dtypes


class DoccPerformanceWarning(UserWarning):
    """Warning emitted when a slower-than-necessary execution path is taken."""


def warn_device_residency_failed(backend):
    """Emit a one-time warning that device-residency promotion did not apply.

    Args:
        backend: The GPU backend name (e.g. "cuda", "rocm") for context.
    """
    warnings.warn(
        f"Device residency could not be enabled for this function on backend {backend}. "
        f"The function is only partially offloaded; function arguments stay in host memory before "
        f"and after execution.",
        DoccPerformanceWarning,
        stacklevel=3,
    )


def warn_host_to_device_copies(backend):
    """Emit a one-time performance warning about unnecessary host-to-device copies.

    Args:
        backend: The GPU backend name (e.g. "cuda", "rocm") for context.
    """
    warnings.warn(
        f"Device residency is enabled for this function on backend {backend}. "
        f"Function arguments are passed from host memory slowing down execution. "
        f"Provide device-resident arrays to avoid unnecessary copies and improve performance.",
        DoccPerformanceWarning,
        stacklevel=3,
    )


def idiv(a, b):
    """Integer division (floor division for positive numbers)."""
    return int(a) // int(b)


def _is_device_array(arg):
    """Return True if ``arg`` already lives in device memory.

    cupy arrays and CUDA torch tensors expose ``__cuda_array_interface__``; a
    torch tensor reports its location via ``is_cuda``. Host arrays (numpy, CPU
    torch tensors) return False.
    """
    if getattr(arg, "__cuda_array_interface__", None) is not None:
        return True
    is_cuda = getattr(arg, "is_cuda", None)
    if is_cuda is not None:
        return bool(is_cuda)
    return False


def _device_array_ptr(arg):
    """Extract the raw device pointer from a GPU array (cupy or torch.cuda).

    Both cupy arrays and CUDA torch tensors expose ``__cuda_array_interface__``;
    torch tensors additionally expose ``data_ptr()``. Returns an integer address.
    """
    cai = getattr(arg, "__cuda_array_interface__", None)
    if cai is not None:
        return cai["data"][0]
    data_ptr = getattr(arg, "data_ptr", None)
    if callable(data_ptr):
        return data_ptr()
    raise TypeError(
        f"Device-resident execution requires a GPU array exposing "
        f"__cuda_array_interface__ or data_ptr(), got {type(arg).__name__}"
    )


# Evaluation context for shape expressions
_EVAL_GLOBALS = {"idiv": idiv}

# Pre-compiled regex for _convert_to_python_syntax
_FUNC_CALL_PATTERN = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\(([^()]+)\)")
_PLACEHOLDER_PATTERN = re.compile(
    r"@@@FUNC@@@([a-zA-Z_][a-zA-Z0-9_]*)@@@(.+?)@@@END@@@"
)
_KNOWN_FUNCTIONS = frozenset(
    {"int", "float", "abs", "min", "max", "sum", "len", "idiv"}
)

# Argument type constants for fast dispatch
_ARG_TYPE_OUTPUT_ARRAY = 0
_ARG_TYPE_OUTPUT_SCALAR = 1
_ARG_TYPE_SHAPE = 2
_ARG_TYPE_USER_ARRAY = 3
_ARG_TYPE_USER_STRUCT = 4
_ARG_TYPE_USER_SCALAR = 5

# Call modes. A single call uses exactly one mode (mixing array kinds is not
# allowed). The mode, combined with whether the artifact is device-resident,
# fully determines the execution path:
#
#   mode      | non-device-resident        | device-resident
#   ----------|----------------------------|-------------------------------
#   NumPyCPU  | host execution (default)   | host->device copy + warning
#   NumPyGPU  | rejected (TypeError)       | zero-copy device execution
#   TorchCPU  | host execution             | host->device copy + warning
#   TorchGPU  | rejected (TypeError)       | zero-copy device execution
_CALL_MODE_NUMPY_CPU = "NumPyCPU"
_CALL_MODE_NUMPY_GPU = "NumPyGPU"
_CALL_MODE_TORCH_CPU = "TorchCPU"
_CALL_MODE_TORCH_GPU = "TorchGPU"

# Modes whose data lives in device memory; these require a device-resident
# artifact, otherwise the call is rejected.
_GPU_CALL_MODES = frozenset({_CALL_MODE_NUMPY_GPU, _CALL_MODE_TORCH_GPU})


def _is_torch_tensor(arg):
    """Return True if ``arg`` is a torch tensor (without importing torch)."""
    return type(arg).__module__.split(".", 1)[0] == "torch"


def _classify_array_kind(arg):
    """Classify a single argument into one of the four call modes.

    Returns one of the ``_CALL_MODE_*`` constants for array arguments, or None
    for mode-agnostic values (Python/numpy scalars, structures, ...). Neither
    torch nor cupy is imported: classification relies on the defining module and
    the array's own attributes.
    """
    root = type(arg).__module__.split(".", 1)[0]
    if root == "torch":
        return (
            _CALL_MODE_TORCH_GPU
            if getattr(arg, "is_cuda", False)
            else _CALL_MODE_TORCH_CPU
        )
    if root == "cupy":
        return _CALL_MODE_NUMPY_GPU
    if isinstance(arg, np.ndarray):
        return _CALL_MODE_NUMPY_CPU
    # Any other object exposing the CUDA array interface is a device array.
    if getattr(arg, "__cuda_array_interface__", None) is not None:
        return _CALL_MODE_NUMPY_GPU
    return None


# Pre-cache ctypes.c_int64 for speed
_c_int64 = ctypes.c_int64
_ctypes_cast = ctypes.cast
_ctypes_addressof = ctypes.addressof
_ctypes_byref = ctypes.byref
_ctypes_pointer = ctypes.pointer
_ctypes_c_void_p = ctypes.c_void_p

# Map primitive types to numpy dtypes for fast buffer allocation
_PRIMITIVE_TO_NP_DTYPE = {
    PrimitiveType.Float: np.float32,
    PrimitiveType.Double: np.float64,
    PrimitiveType.Int8: np.int8,
    PrimitiveType.Int16: np.int16,
    PrimitiveType.Int32: np.int32,
    PrimitiveType.Int64: np.int64,
    PrimitiveType.UInt8: np.uint8,
    PrimitiveType.UInt16: np.uint16,
    PrimitiveType.UInt32: np.uint32,
    PrimitiveType.UInt64: np.uint64,
    PrimitiveType.Bool: np.bool_,
    PrimitiveType.Half: np.float16,
    PrimitiveType.BFloat: ml_dtypes.bfloat16,
}

# Pre-computed dtype map for numpy conversion
_NUMPY_DTYPE_MAP = {
    PrimitiveType.Float: np.float32,
    PrimitiveType.Double: np.float64,
    PrimitiveType.Int8: np.int8,
    PrimitiveType.Int16: np.int16,
    PrimitiveType.Int32: np.int32,
    PrimitiveType.Int64: np.int64,
    PrimitiveType.UInt8: np.uint8,
    PrimitiveType.UInt16: np.uint16,
    PrimitiveType.UInt32: np.uint32,
    PrimitiveType.UInt64: np.uint64,
    PrimitiveType.Bool: np.bool_,
    PrimitiveType.Half: np.float16,
    PrimitiveType.BFloat: ml_dtypes.bfloat16,
}

_CTYPES_MAP = {
    PrimitiveType.Bool: ctypes.c_bool,
    PrimitiveType.Int8: ctypes.c_int8,
    PrimitiveType.Int16: ctypes.c_int16,
    PrimitiveType.Int32: ctypes.c_int32,
    PrimitiveType.Int64: ctypes.c_int64,
    PrimitiveType.UInt8: ctypes.c_uint8,
    PrimitiveType.UInt16: ctypes.c_uint16,
    PrimitiveType.UInt32: ctypes.c_uint32,
    PrimitiveType.UInt64: ctypes.c_uint64,
    PrimitiveType.Float: ctypes.c_float,
    PrimitiveType.Double: ctypes.c_double,
    # Half and BFloat are 2 bytes, use c_uint16 for raw storage
    PrimitiveType.Half: ctypes.c_uint16,
    PrimitiveType.BFloat: ctypes.c_uint16,
}


class CompiledSDFG:
    def __init__(
        self,
        lib_path,
        sdfg,
        shape_sources=None,
        structure_member_info=None,
        output_args=None,
        output_shapes=None,
        output_strides=None,
        device_resident=False,
        device_backend=None,
        target=None,
    ):
        self.lib_path = lib_path
        self.sdfg = sdfg
        self.shape_sources = shape_sources or []
        self.structure_member_info = structure_member_info or {}
        self.lib = ctypes.CDLL(lib_path)
        self.func = getattr(self.lib, sdfg.name)

        # Check for output args
        self.output_args = output_args or []
        if not self.output_args and hasattr(sdfg, "metadata"):
            out_args_str = sdfg.metadata("output_args")
            if out_args_str:
                self.output_args = out_args_str.split(",")

        self.output_shapes = output_shapes or {}
        self.output_strides = output_strides or {}

        # Device residency: set by the DeviceResidentArgPromotion pass when all
        # pointer arguments were promoted to device-resident storage. When active,
        # the compiled function expects device pointers (no host<->device copies at
        # the boundary) and produces device-resident outputs. Communicated
        # explicitly via the constructor (pass return value), not via metadata.
        self.device_resident = bool(device_resident)
        self.device_backend = device_backend or (
            "cuda" if self.device_resident else None
        )
        # Warn at most once per artifact when host inputs must be copied to device.
        self._warned_host_to_device = False

        # Compilation target (e.g. "cuda"/"rocm"/"sequential"). Used to inform,
        # once, when a GPU target ran on the host because device-residency
        # promotion did not apply to this artifact.
        self.target = target
        self._warned_residency_failed = False

        # Cache for ctypes structure definitions
        self._ctypes_structures = {}

        # Set up argument types
        self.arg_names = sdfg.arguments
        self.arg_types = []
        self.arg_sdfg_types = []  # Keep track of original sdfg types
        for arg_name in sdfg.arguments:
            arg_type = sdfg.type(arg_name)
            self.arg_sdfg_types.append(arg_type)
            ct_type = self._get_ctypes_type(arg_type)
            self.arg_types.append(ct_type)

        self.func.argtypes = self.arg_types

        # Set up return type
        self.func.restype = self._get_ctypes_type(sdfg.return_type)

        # Pre-compute argument classification for fast __call__
        self._precompute_arg_metadata()

    def _precompute_arg_metadata(self):
        """Pre-compute argument metadata for fast __call__ dispatch."""
        output_args_set = set(self.output_args)

        # Build shape source lookup: s_idx -> (u_idx, dim_idx)
        # Also pre-compute the shape keys
        self._shape_sources_list = []  # [(s_idx, u_idx, dim_idx, key_str), ...]
        for i, (u_idx, dim_idx) in enumerate(self.shape_sources):
            self._shape_sources_list.append((i, u_idx, dim_idx, f"_s{i}"))

        # Classify each argument using tuple-based info for faster access
        # Each entry is (arg_type, *type_specific_data)
        self._arg_info = []
        user_arg_counter = 0

        # For output ordering (avoid sorting at runtime)
        output_order = []

        for i, arg_name in enumerate(self.arg_names):
            if arg_name in output_args_set:
                # Output argument
                target_type = self.arg_types[i]
                base_type = target_type._type_
                sdfg_type = self.arg_sdfg_types[i]

                # Pre-compute primitive type for return processing
                primitive_type = None
                if isinstance(sdfg_type, Pointer) and sdfg_type.has_pointee_type():
                    pointee = sdfg_type.pointee_type
                    if isinstance(pointee, Scalar):
                        primitive_type = pointee.primitive_type

                if arg_name in self.output_shapes:
                    dims = self.output_shapes[arg_name]
                    # Always compile shape expressions - they may depend on runtime values
                    compiled_dims = []
                    for d in dims:
                        d_str = str(d)
                        expr = self._convert_to_python_syntax(d_str)
                        compiled_dims.append(compile(expr, "<shape>", "eval"))

                    # Pre-compile stride expressions if available
                    compiled_strides = None
                    if arg_name in self.output_strides:
                        compiled_strides = []
                        for s in self.output_strides[arg_name]:
                            expr = self._convert_to_python_syntax(str(s))
                            compiled_strides.append(compile(expr, "<stride>", "eval"))

                    # Get numpy dtype for fast allocation
                    np_dtype = (
                        _PRIMITIVE_TO_NP_DTYPE.get(primitive_type, np.float64)
                        if primitive_type
                        else np.float64
                    )

                    # Tuple: (arg_type, name, base_type, target_type, compiled_dims, compiled_strides, primitive_type, np_dtype)
                    info_idx = len(self._arg_info)
                    self._arg_info.append(
                        (
                            _ARG_TYPE_OUTPUT_ARRAY,
                            arg_name,
                            base_type,
                            target_type,
                            compiled_dims,
                            compiled_strides,
                            primitive_type,
                            np_dtype,
                        )
                    )
                    output_order.append((int(arg_name.split("_")[-1]), info_idx))
                else:
                    # Scalar return
                    info_idx = len(self._arg_info)
                    self._arg_info.append(
                        (_ARG_TYPE_OUTPUT_SCALAR, arg_name, base_type, primitive_type)
                    )
                    output_order.append((int(arg_name.split("_")[-1]), info_idx))

            elif arg_name.startswith("_s") and arg_name[2:].isdigit():
                # Shape symbol argument - tuple: (arg_type, s_idx, key_str)
                s_idx = int(arg_name[2:])
                self._arg_info.append((_ARG_TYPE_SHAPE, s_idx, f"_s{s_idx}"))
            else:
                # User argument
                sdfg_type = self.arg_sdfg_types[i]
                target_type = self.arg_types[i]
                is_struct_ptr = (
                    sdfg_type
                    and isinstance(sdfg_type, Pointer)
                    and sdfg_type.has_pointee_type()
                    and isinstance(sdfg_type.pointee_type, Structure)
                )

                if is_struct_ptr:
                    struct_name = sdfg_type.pointee_type.name
                    struct_class = self._create_ctypes_structure(struct_name)
                    members = self.structure_member_info[struct_name]
                    sorted_members = tuple(
                        sorted(members.items(), key=lambda x: x[1][0])
                    )
                    # Tuple: (arg_type, user_idx, name, struct_class, sorted_members)
                    self._arg_info.append(
                        (
                            _ARG_TYPE_USER_STRUCT,
                            user_arg_counter,
                            arg_name,
                            struct_class,
                            sorted_members,
                        )
                    )
                elif hasattr(target_type, "contents"):
                    # Array user arg - tuple: (arg_type, user_idx, name, target_type)
                    self._arg_info.append(
                        (_ARG_TYPE_USER_ARRAY, user_arg_counter, arg_name, target_type)
                    )
                else:
                    # Scalar user arg - tuple: (arg_type, user_idx, name, target_type)
                    self._arg_info.append(
                        (_ARG_TYPE_USER_SCALAR, user_arg_counter, arg_name, target_type)
                    )
                user_arg_counter += 1

        self._num_user_args = user_arg_counter

        # Pre-sort output order and build position map
        output_order.sort(key=lambda x: x[0])
        self._output_order = tuple(idx for _, idx in output_order)
        # Map from _arg_info index to result position (for O(1) lookup)
        self._output_pos_map = {idx: pos for pos, idx in enumerate(self._output_order)}

    def _convert_to_python_syntax(self, expr_str):
        result = expr_str

        while True:
            match = _FUNC_CALL_PATTERN.search(result)
            if not match:
                break

            name = match.group(1)
            index = match.group(2)

            if name.lower() in _KNOWN_FUNCTIONS:
                # Use unique delimiters that won't appear in expressions
                placeholder = f"@@@FUNC@@@{name}@@@{index}@@@END@@@"
                result = result[: match.start()] + placeholder + result[match.end() :]
            else:
                result = (
                    result[: match.start()] + f"{name}[{index}]" + result[match.end() :]
                )

        result = _PLACEHOLDER_PATTERN.sub(r"\1(\2)", result)

        return result

    def _create_ctypes_structure(self, struct_name):
        """Create a ctypes Structure class for the given structure name."""
        if struct_name in self._ctypes_structures:
            return self._ctypes_structures[struct_name]

        if struct_name not in self.structure_member_info:
            raise ValueError(f"Structure '{struct_name}' not found in member info")

        # Get member info: {member_name: (index, type)}
        members = self.structure_member_info[struct_name]
        # Sort by index to get correct order
        sorted_members = sorted(members.items(), key=lambda x: x[1][0])

        # Build _fields_ for ctypes.Structure
        fields = []
        for member_name, member_info in sorted_members:
            member_type = member_info[1]
            ct_type = self._get_ctypes_type(member_type)
            fields.append((member_name, ct_type))

        # Create the ctypes Structure class dynamically
        class CStructure(ctypes.Structure):
            _fields_ = fields

        self._ctypes_structures[struct_name] = CStructure
        return CStructure

    def _get_ctypes_type(self, sdfg_type):
        if isinstance(sdfg_type, Scalar):
            return _CTYPES_MAP.get(sdfg_type.primitive_type, ctypes.c_void_p)
        elif isinstance(sdfg_type, Array):
            # Arrays are passed as pointers
            elem_type = _CTYPES_MAP.get(sdfg_type.primitive_type, ctypes.c_void_p)
            return ctypes.POINTER(elem_type)
        elif isinstance(sdfg_type, Pointer):
            # Check if pointee is a Structure
            # Note: has_pointee_type() is guaranteed to exist on Pointer instances from C++ bindings
            if sdfg_type.has_pointee_type():
                pointee = sdfg_type.pointee_type
                if isinstance(pointee, Structure):
                    # Create ctypes structure and return pointer to it
                    struct_class = self._create_ctypes_structure(pointee.name)
                    return ctypes.POINTER(struct_class)
                elif isinstance(pointee, Scalar):
                    elem_type = _CTYPES_MAP.get(pointee.primitive_type, ctypes.c_void_p)
                    return ctypes.POINTER(elem_type)
            return ctypes.c_void_p
        return ctypes.c_void_p

    def _convert_return_value(self, func_result, shape_symbol_values):
        return_type = self.sdfg.return_type

        if isinstance(return_type, Pointer):
            if return_type.has_pointee_type():
                pointee = return_type.pointee_type
                if isinstance(pointee, Scalar):
                    # Pointer to scalar element type - need to determine array size
                    # Get return shape from metadata if available
                    return_shape_str = self.sdfg.metadata("return_shape")
                    if return_shape_str:
                        # Strip brackets (metadata may be "[10,10]" format)
                        return_shape_str = return_shape_str.strip("[]")
                        shape = []
                        for dim_str in return_shape_str.split(","):
                            try:
                                eval_str = self._convert_to_python_syntax(str(dim_str))
                                val = eval(eval_str, _EVAL_GLOBALS, shape_symbol_values)
                                shape.append(int(val))
                            except Exception:
                                # Can't evaluate shape, return raw pointer
                                return func_result

                        # Determine numpy dtype from primitive type
                        dtype = _NUMPY_DTYPE_MAP.get(pointee.primitive_type, np.float64)

                        # Calculate total size
                        total_size = 1
                        for dim in shape:
                            total_size *= dim

                        # Create numpy array from pointer
                        ct_type = _CTYPES_MAP.get(
                            pointee.primitive_type, ctypes.c_double
                        )
                        arr_type = ct_type * total_size
                        # For Half/BFloat, use np.frombuffer since np.ctypeslib.as_array
                        # doesn't support these types (PEP 3118 buffer format limitation)
                        if pointee.primitive_type in (
                            PrimitiveType.Half,
                            PrimitiveType.BFloat,
                        ):
                            byte_size = total_size * 2  # Half and BFloat are 2 bytes
                            arr = np.frombuffer(
                                (ctypes.c_char * byte_size).from_address(
                                    ctypes.cast(func_result, ctypes.c_void_p).value
                                ),
                                dtype=dtype,
                            ).copy()
                        else:
                            arr = np.ctypeslib.as_array(
                                ctypes.cast(
                                    func_result, ctypes.POINTER(arr_type)
                                ).contents
                            )
                        return arr.reshape(shape)
                    else:
                        # No shape info - try to infer from input shapes
                        # For identity-like operations, the output shape matches input
                        if len(self.shape_sources) > 0 and len(shape_symbol_values) > 0:
                            # Use first input's shape as a fallback
                            shape = []
                            for i in range(len(self.shape_sources)):
                                if f"_s{i}" in shape_symbol_values:
                                    shape.append(shape_symbol_values[f"_s{i}"])

                            if shape:
                                dtype = _NUMPY_DTYPE_MAP.get(
                                    pointee.primitive_type, np.float64
                                )

                                total_size = 1
                                for dim in shape:
                                    total_size *= dim

                                ct_type = _CTYPES_MAP.get(
                                    pointee.primitive_type, ctypes.c_double
                                )
                                arr_type = ct_type * total_size
                                # For Half/BFloat, use np.frombuffer since np.ctypeslib.as_array
                                # doesn't support these types (PEP 3118 buffer format limitation)
                                if pointee.primitive_type in (
                                    PrimitiveType.Half,
                                    PrimitiveType.BFloat,
                                ):
                                    byte_size = (
                                        total_size * 2
                                    )  # Half and BFloat are 2 bytes
                                    arr = np.frombuffer(
                                        (ctypes.c_char * byte_size).from_address(
                                            ctypes.cast(
                                                func_result, ctypes.c_void_p
                                            ).value
                                        ),
                                        dtype=dtype,
                                    ).copy()
                                else:
                                    arr = np.ctypeslib.as_array(
                                        ctypes.cast(
                                            func_result, ctypes.POINTER(arr_type)
                                        ).contents
                                    )
                                return arr.reshape(shape)

                        # Can't determine shape, return raw pointer
                        return func_result
        elif isinstance(return_type, Scalar):
            return func_result

        return func_result

    def _ensure_device_array(self, arg, keepalive, writebacks):
        """Return a device array for ``arg``, copying host inputs to the device.

        Device-resident artifacts consume raw device pointers. For backwards
        compatibility, callers may still pass host arrays (numpy arrays or CPU
        torch tensors); these are copied to the device here and a one-time
        performance warning is emitted. Device arrays are returned unchanged.

        The copied device array is appended to ``keepalive`` so it outlives the
        call, and the ``(host, device)`` pair is recorded in ``writebacks`` so
        that in-place writes (e.g. output arguments) are mirrored back to the
        original host array after execution.
        """
        if _is_device_array(arg):
            return arg

        import cupy

        host = arg
        # Convert a CPU torch tensor to numpy first; cupy.asarray handles numpy.
        if hasattr(host, "detach") and hasattr(host, "numpy"):
            host = host.detach().numpy()
        device_arg = cupy.asarray(host)
        keepalive.append(device_arg)
        writebacks.append((arg, device_arg))

        if not self._warned_host_to_device:
            self._warned_host_to_device = True
            warn_host_to_device_copies(self.device_backend or "cuda")

        return device_arg

    def _call_device(self, *args):
        """Execute with device-resident arguments (cupy / torch.cuda).

        Inputs are passed as raw device pointers and output arrays are allocated
        on the device, so no host<->device copies happen at the call boundary.
        Outputs are returned as cupy arrays (zero-copy interoperable with torch
        via DLPack / __cuda_array_interface__).
        """
        import cupy

        _eval = eval
        _GLOBALS = _EVAL_GLOBALS
        # 1. Build shape_symbol_values from shape sources
        shape_symbol_values = {}
        for s_idx, u_idx, dim_idx, key in self._shape_sources_list:
            if u_idx < len(args):
                shape_symbol_values[key] = args[u_idx].shape[dim_idx]

        # 2. Process arguments
        converted_args = []
        keepalive = []  # keep device buffers / ctypes scalars alive
        writebacks = []  # (host_arg, device_arg) for host inputs copied to device
        return_buffers = []  # (buf, size, dims, compiled_strides, primitive_type)
        # Track whether the caller supplied device arrays. When all array inputs
        # are host arrays (numpy / CPU torch), the caller works in host space and
        # expects host outputs, so device output buffers are copied back to host.
        any_device_input = False

        for info in self._arg_info:
            arg_type = info[0]

            if arg_type == _ARG_TYPE_OUTPUT_ARRAY:
                target_type = info[3]
                compiled_dims = info[4]
                compiled_strides = info[5]
                np_dtype = info[7]

                size = 1
                dims = []
                for code in compiled_dims:
                    d = int(_eval(code, _GLOBALS, shape_symbol_values))
                    dims.append(d)
                    size *= d

                buf = cupy.empty(size, dtype=np_dtype)
                keepalive.append(buf)
                return_buffers.append((buf, size, dims, compiled_strides, info[6]))
                converted_args.append(_ctypes_cast(int(buf.data.ptr), target_type))

            elif arg_type == _ARG_TYPE_OUTPUT_SCALAR:
                base_type = info[2]
                primitive_type = info[3]
                buf = base_type()
                keepalive.append(buf)
                return_buffers.append((buf, 1, None, None, primitive_type))
                converted_args.append(_ctypes_byref(buf))

            elif arg_type == _ARG_TYPE_SHAPE:
                converted_args.append(_c_int64(shape_symbol_values.get(info[2], 0)))

            elif arg_type == _ARG_TYPE_USER_ARRAY:
                arg = args[info[1]]
                shape_symbol_values[info[2]] = arg
                if _is_device_array(arg):
                    any_device_input = True
                arg = self._ensure_device_array(arg, keepalive, writebacks)
                converted_args.append(_ctypes_cast(_device_array_ptr(arg), info[3]))

            elif arg_type == _ARG_TYPE_USER_STRUCT:
                raise NotImplementedError(
                    "Structure arguments are not supported for device-resident "
                    "execution."
                )

            else:  # _ARG_TYPE_USER_SCALAR
                arg = args[info[1]]
                shape_symbol_values[info[2]] = arg
                converted_args.append(info[3](arg))

        # 3. Call the function
        func_result = self.func(*converted_args)

        # 3b. Mirror device results back into host inputs that were copied to the
        # device, so in-place writes (e.g. output arguments) are visible to the
        # caller. Read-only inputs are unchanged and copy back identically.
        for host_arg, device_arg in writebacks:
            host_view = cupy.asnumpy(device_arg)
            if isinstance(host_arg, np.ndarray):
                np.copyto(host_arg, host_view.reshape(host_arg.shape))
            elif hasattr(host_arg, "copy_"):  # torch CPU tensor
                import torch

                host_arg.copy_(
                    torch.from_numpy(host_view.reshape(tuple(host_arg.shape)))
                )

        # 4. Process returns using pre-sorted order
        if not return_buffers:
            return None

        # Host callers get host (numpy) outputs; device callers get cupy outputs.
        host_mode = not any_device_input

        num_outputs = len(return_buffers)
        results = [None] * num_outputs

        buf_idx = 0
        for i, info in enumerate(self._arg_info):
            arg_type = info[0]
            if arg_type not in (_ARG_TYPE_OUTPUT_ARRAY, _ARG_TYPE_OUTPUT_SCALAR):
                continue

            result_pos = self._output_pos_map[i]
            buf, size, dims, compiled_strides, primitive_type = return_buffers[buf_idx]
            buf_idx += 1

            if arg_type == _ARG_TYPE_OUTPUT_SCALAR:
                results[result_pos] = buf.value
            else:
                arr = buf
                if dims and len(dims) > 1:
                    arr = arr.reshape(dims)
                if host_mode:
                    arr = cupy.asnumpy(arr)
                results[result_pos] = arr

        if len(results) == 1:
            return results[0]
        return tuple(results) if results else None

    def _resolve_call_mode(self, args):
        """Classify the call's array arguments into exactly one call mode.

        All array arguments must be the same kind; mixing numpy / cupy / CPU
        torch / CUDA torch arrays in a single call is rejected. Calls without any
        array argument default to ``NumPyCPU`` (host execution).
        """
        modes = set()
        for info in self._arg_info:
            arg_type = info[0]
            if arg_type == _ARG_TYPE_USER_ARRAY or arg_type == _ARG_TYPE_USER_STRUCT:
                kind = _classify_array_kind(args[info[1]])
                if kind is not None:
                    modes.add(kind)

        if not modes:
            return _CALL_MODE_NUMPY_CPU
        if len(modes) > 1:
            raise TypeError(
                "Mixed array kinds are not allowed in a single call; every array "
                f"argument must be the same kind, but got {sorted(modes)}. "
                "Provide all inputs (and outputs) as one of: numpy arrays, cupy "
                "arrays, CPU torch tensors, or CUDA torch tensors."
            )
        return next(iter(modes))

    def __call__(self, *args):
        """Execute the compiled artifact, dispatching by call mode.

        Exactly one call mode is allowed per call (no mixing of array kinds).
        GPU modes (cupy / CUDA torch) require a device-resident artifact; host
        modes (numpy / CPU torch) run on the device with a one-time performance
        warning when the artifact is device-resident, otherwise on the host.
        """
        mode = self._resolve_call_mode(args)

        if mode in _GPU_CALL_MODES and not self.device_resident:
            raise TypeError(
                f"{mode} arguments were provided, but this artifact is not "
                "device-resident. GPU arrays can only be used with a "
                "device-resident artifact (a fully-offloadable kernel compiled "
                "for a GPU target). Provide host arrays (numpy arrays or CPU "
                "torch tensors) instead."
            )

        # Device-resident artifacts consume/produce device pointers directly.
        # Host inputs are copied to the device inside _call_device (with a
        # one-time warning); device inputs are consumed zero-copy.
        if self.device_resident:
            return self._call_device(*args)

        # Host execution path. CPU torch tensors are converted to numpy views so
        # the ctypes boundary can take their data pointer.
        # Inform once when a GPU target falls back to host execution because the
        # device-residency optimization did not apply to this artifact.
        if self.target in ("cuda", "rocm") and not self._warned_residency_failed:
            self._warned_residency_failed = True
            warn_device_residency_failed(self.target)

        if mode == _CALL_MODE_TORCH_CPU:
            args = tuple(
                a.detach().cpu().contiguous().numpy() if _is_torch_tensor(a) else a
                for a in args
            )
        return self._call_host(*args)

    def _call_host(self, *args):
        # Ultra-fast path using pre-computed tuple-based argument info
        # Local variable caching for speed
        _eval = eval
        _GLOBALS = _EVAL_GLOBALS
        _np_empty = np.empty

        # 1. Build shape_symbol_values from shape sources (pre-computed list)
        shape_symbol_values = {}
        for s_idx, u_idx, dim_idx, key in self._shape_sources_list:
            if u_idx < len(args):
                shape_symbol_values[key] = args[u_idx].shape[dim_idx]

        # 2. Process arguments using tuple-based dispatch
        converted_args = []
        structure_refs = (
            []
        )  # Keep refs alive (includes numpy arrays for output buffers)
        return_buffers = (
            []
        )  # List of (np_arr, size, dims, compiled_strides, primitive_type)
        # Structs whose scalar members must be copied back into the Python
        # object after the call: (python_obj, c_struct, sorted_members).
        struct_writebacks = []

        for info in self._arg_info:
            arg_type = info[0]

            if arg_type == _ARG_TYPE_OUTPUT_ARRAY:
                # info = (type, name, base_type, target_type, compiled_dims, compiled_strides, primitive_type, np_dtype)
                target_type = info[3]
                compiled_dims = info[4]
                compiled_strides = info[5]
                np_dtype = info[7]

                # Evaluate size from compiled code objects
                size = 1
                dims = []
                for code in compiled_dims:
                    d = int(_eval(code, _GLOBALS, shape_symbol_values))
                    dims.append(d)
                    size *= d

                # Use numpy for fast allocation (much faster than ctypes)
                buf_arr = _np_empty(size, dtype=np_dtype)
                structure_refs.append(buf_arr)  # Keep alive
                return_buffers.append((buf_arr, size, dims, compiled_strides, info[6]))
                # Get pointer directly from numpy array interface
                ptr = buf_arr.ctypes.data
                converted_args.append(_ctypes_cast(ptr, target_type))

            elif arg_type == _ARG_TYPE_OUTPUT_SCALAR:
                # info = (type, name, base_type, primitive_type)
                base_type = info[2]
                primitive_type = info[3]
                buf = base_type()
                structure_refs.append(buf)
                return_buffers.append((buf, 1, None, None, primitive_type))
                converted_args.append(_ctypes_byref(buf))

            elif arg_type == _ARG_TYPE_SHAPE:
                # info = (type, s_idx, key_str)
                converted_args.append(_c_int64(shape_symbol_values.get(info[2], 0)))

            elif arg_type == _ARG_TYPE_USER_ARRAY:
                # info = (type, user_idx, name, target_type)
                user_idx = info[1]
                arg = args[user_idx]
                shape_symbol_values[info[2]] = arg  # For indirect access
                # Direct pointer access - faster than data_as()
                converted_args.append(_ctypes_cast(arg.ctypes.data, info[3]))

            elif arg_type == _ARG_TYPE_USER_STRUCT:
                # info = (type, user_idx, name, struct_class, sorted_members)
                arg = args[info[1]]
                shape_symbol_values[info[2]] = arg
                struct_class = info[3]
                struct_values = {}
                for m in info[4]:
                    member_name = m[0]
                    if not hasattr(arg, member_name):
                        continue
                    member_value = getattr(arg, member_name)
                    member_type = m[1][1]
                    if isinstance(member_value, np.ndarray):
                        # Array member: pass the data pointer (struct-of-arrays).
                        struct_values[member_name] = member_value.ctypes.data_as(
                            self._get_ctypes_type(member_type)
                        )
                    else:
                        struct_values[member_name] = member_value
                c_struct = struct_class(**struct_values)
                structure_refs.append(c_struct)
                converted_args.append(_ctypes_pointer(c_struct))
                # Record for scalar-member write-back after the call.
                struct_writebacks.append((arg, c_struct, info[4]))

            else:  # _ARG_TYPE_USER_SCALAR
                # info = (type, user_idx, name, target_type)
                arg = args[info[1]]
                shape_symbol_values[info[2]] = arg
                converted_args.append(info[3](arg))

        # 3. Call the function
        func_result = self.func(*converted_args)

        # 3b. Copy scalar struct members modified in-place back into the Python
        # object. Array members share the numpy buffer via their data pointer,
        # but scalar members are passed by value and must be mirrored back so
        # writes like `domain.dtcourant = ...` are visible to the caller.
        for py_obj, c_struct, sorted_members in struct_writebacks:
            for m in sorted_members:
                member_name = m[0]
                if isinstance(m[1][1], Pointer):
                    continue  # array member: written in place via the pointer
                if hasattr(py_obj, member_name):
                    setattr(py_obj, member_name, getattr(c_struct, member_name))

        # 4. Process returns using pre-sorted order
        if not return_buffers:
            if func_result is not None:
                return self._convert_return_value(func_result, shape_symbol_values)
            return None

        # return_buffers: [(np_arr_or_ctypes_scalar, size, dims, compiled_strides, primitive_type), ...]
        num_outputs = len(return_buffers)
        results = [None] * num_outputs

        buf_idx = 0
        for i, info in enumerate(self._arg_info):
            arg_type = info[0]
            if arg_type not in (_ARG_TYPE_OUTPUT_ARRAY, _ARG_TYPE_OUTPUT_SCALAR):
                continue

            result_pos = self._output_pos_map[i]
            buf, size, dims, compiled_strides, primitive_type = return_buffers[buf_idx]
            buf_idx += 1

            if arg_type == _ARG_TYPE_OUTPUT_SCALAR:
                # Scalar - buf is a ctypes scalar
                results[result_pos] = buf.value
            else:
                # Array - buf is already a numpy array
                arr = buf
                if dims and len(dims) > 1:
                    # Need to reshape
                    if compiled_strides:
                        try:
                            itemsize = arr.itemsize
                            byte_strides = tuple(
                                int(_eval(s, {}, shape_symbol_values)) * itemsize
                                for s in compiled_strides
                            )
                            arr = np.lib.stride_tricks.as_strided(
                                arr, shape=dims, strides=byte_strides
                            )
                        except:
                            arr = arr.reshape(dims)
                    else:
                        arr = arr.reshape(dims)
                elif dims and len(dims) == 1:
                    pass  # Already 1D with correct size
                results[result_pos] = arr

        if len(results) == 1:
            return results[0]
        return tuple(results) if results else None

    def get_return_shape(self, *args):
        shape_str = self.sdfg.metadata("return_shape")
        if not shape_str:
            return None

        shape_exprs = shape_str.split(",")

        # Reconstruct shape values
        shape_values = {}
        for i, (arg_idx, dim_idx) in enumerate(self.shape_sources):
            arg = args[arg_idx]
            if np is not None and isinstance(arg, np.ndarray):
                val = arg.shape[dim_idx]
                shape_values[f"_s{i}"] = val

        # Add scalar arguments to shape_values
        # We assume the first len(args) arguments in sdfg.arguments correspond to the user arguments
        if hasattr(self.sdfg, "arguments"):
            for arg_name, arg_val in zip(self.sdfg.arguments, args):
                if isinstance(arg_val, (int, np.integer)):
                    shape_values[arg_name] = int(arg_val)

        evaluated_shape = []
        for expr in shape_exprs:
            # Simple evaluation using eval with shape_values
            # Warning: eval is unsafe, but here expressions come from our compiler
            try:
                val = eval(expr, _EVAL_GLOBALS, shape_values)
                evaluated_shape.append(int(val))
            except Exception:
                return None

        return tuple(evaluated_shape)
