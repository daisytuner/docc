import ctypes
from docc.sdfg import Scalar, Array, Pointer, Structure, PrimitiveType

import numpy as np


def idiv(a, b):
    """Integer division (floor division for positive numbers)."""
    return int(a) // int(b)


# Evaluation context for shape expressions
_EVAL_GLOBALS = {"idiv": idiv}

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

    def _convert_to_python_syntax(self, expr_str):
        import re

        result = expr_str

        while True:
            pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\(([^()]+)\)"
            match = re.search(pattern, result)
            if not match:
                break

            name = match.group(1)
            index = match.group(2)

            # Skip known function names
            known_functions = {
                "int",
                "float",
                "abs",
                "min",
                "max",
                "sum",
                "len",
                "idiv",
            }
            if name.lower() in known_functions:
                # Use unique delimiters that won't appear in expressions
                placeholder = f"@@@FUNC@@@{name}@@@{index}@@@END@@@"
                result = result[: match.start()] + placeholder + result[match.end() :]
            else:
                result = (
                    result[: match.start()] + f"{name}[{index}]" + result[match.end() :]
                )

        result = re.sub(
            r"@@@FUNC@@@([a-zA-Z_][a-zA-Z0-9_]*)@@@(.+?)@@@END@@@", r"\1(\2)", result
        )

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
        for member_name, (index, member_type) in sorted_members:
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
                        dtype_map = {
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
                        }
                        dtype = dtype_map.get(pointee.primitive_type, np.float64)

                        # Calculate total size
                        total_size = 1
                        for dim in shape:
                            total_size *= dim

                        # Create numpy array from pointer
                        ct_type = _CTYPES_MAP.get(
                            pointee.primitive_type, ctypes.c_double
                        )
                        arr_type = ct_type * total_size
                        arr = np.ctypeslib.as_array(
                            ctypes.cast(func_result, ctypes.POINTER(arr_type)).contents
                        )
                        return arr.reshape(shape).astype(dtype)
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
                                dtype_map = {
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
                                }
                                dtype = dtype_map.get(
                                    pointee.primitive_type, np.float64
                                )

                                total_size = 1
                                for dim in shape:
                                    total_size *= dim

                                ct_type = _CTYPES_MAP.get(
                                    pointee.primitive_type, ctypes.c_double
                                )
                                arr_type = ct_type * total_size
                                arr = np.ctypeslib.as_array(
                                    ctypes.cast(
                                        func_result, ctypes.POINTER(arr_type)
                                    ).contents
                                )
                                return arr.reshape(shape).astype(dtype)

                        # Can't determine shape, return raw pointer
                        return func_result
        elif isinstance(return_type, Scalar):
            return func_result

        return func_result

    def __call__(self, *args):
        # Identify user arguments vs implicit arguments (shapes, return values)

        # 1. Compute shape symbol values from user args input
        shape_symbol_values = {}
        for u_idx, dim_idx in self.shape_sources:
            if u_idx < len(args):
                val = args[u_idx].shape[dim_idx]
                s_idx = self.shape_sources.index((u_idx, dim_idx))
                shape_symbol_values[f"_s{s_idx}"] = val

        # Add input arrays to the shape context for expressions with indirect access
        # This allows evaluating expressions like A_row[0] at runtime
        user_arg_idx = 0
        for name in self.arg_names:
            if name in self.output_args:
                continue
            if name.startswith("_s") and name[2:].isdigit():
                continue

            # Must be a user parameter - add it to shape context if it's an array
            if user_arg_idx < len(args):
                val = args[user_arg_idx]
                if isinstance(val, (int, float, np.integer, np.floating)):
                    shape_symbol_values[name] = val
                elif np is not None and isinstance(val, np.ndarray):
                    # Add numpy arrays to context for indirect access shape evaluation
                    shape_symbol_values[name] = val
                user_arg_idx += 1

        param_arg_idx = 0
        for name in self.arg_names:
            if name in self.output_args:
                continue
            if name.startswith("_s") and name[2:].isdigit():
                continue

            # Must be a user parameter
            if param_arg_idx < len(args):
                val = args[param_arg_idx]
                if isinstance(val, (int, float, np.integer, np.floating)):
                    shape_symbol_values[name] = val
                param_arg_idx += 1

        converted_args = []
        structure_refs = []
        return_buffers = {}

        next_user_arg_idx = 0

        for i, arg_name in enumerate(self.arg_names):
            target_type = self.arg_types[i]

            if arg_name in self.output_args:
                base_type = target_type._type_

                # If array (pointer type) and we have shape info, we need to allocate array.
                # If not in output_shapes, assume scalar return (pointer to single value).
                if arg_name in self.output_shapes:
                    size = 1
                    dims = self.output_shapes[arg_name]
                    # Evaluate
                    for dim_str in dims:
                        try:
                            # Convert SDFG parentheses notation to Python bracket notation
                            # e.g., "A_row(0)" -> "A_row[0]"
                            eval_str = self._convert_to_python_syntax(str(dim_str))
                            val = eval(eval_str, _EVAL_GLOBALS, shape_symbol_values)
                            size *= int(val)
                        except Exception as e:
                            raise RuntimeError(
                                f"Could not evaluate shape {dim_str} for {arg_name}: {e}"
                            )

                    buf_type = base_type * size
                    buf = buf_type()
                    return_buffers[arg_name] = (buf, size, dims)
                    converted_args.append(
                        ctypes.cast(ctypes.addressof(buf), target_type)
                    )
                    continue

                # Scalar Return (Pointer(Scalar))
                buf = base_type()
                return_buffers[arg_name] = (buf, 1, None)
                converted_args.append(ctypes.byref(buf))
                continue

            if arg_name.startswith("_s") and arg_name[2:].isdigit():
                s_idx = int(arg_name[2:])
                if f"_s{s_idx}" in shape_symbol_values:
                    val = shape_symbol_values[f"_s{s_idx}"]
                    converted_args.append(ctypes.c_int64(val))
                else:
                    converted_args.append(ctypes.c_int64(0))
                continue

            # User Argument
            if next_user_arg_idx >= len(args):
                raise ValueError("Not enough arguments provided")

            arg = args[next_user_arg_idx]
            next_user_arg_idx += 1

            # ... Conversion logic (numpy to ctypes) ...
            sdfg_type = self.arg_sdfg_types[i]

            if np is not None and isinstance(arg, np.ndarray):
                if hasattr(target_type, "contents"):
                    converted_args.append(arg.ctypes.data_as(target_type))
                else:
                    converted_args.append(arg)
            elif (
                sdfg_type
                and isinstance(sdfg_type, Pointer)
                and sdfg_type.has_pointee_type()
                and isinstance(sdfg_type.pointee_type, Structure)
            ):
                # Struct logic
                struct_name = sdfg_type.pointee_type.name
                struct_class = self._ctypes_structures.get(struct_name)
                members = self.structure_member_info[struct_name]
                sorted_members = sorted(members.items(), key=lambda x: x[1][0])
                struct_values = {}
                for member_name, (index, member_type) in sorted_members:
                    if hasattr(arg, member_name):
                        struct_values[member_name] = getattr(arg, member_name)
                c_struct = struct_class(**struct_values)
                structure_refs.append(c_struct)
                converted_args.append(ctypes.pointer(c_struct))
            else:
                converted_args.append(
                    target_type(arg)
                )  # Explicit cast to ensure int stays int

        func_result = self.func(*converted_args)

        # Process returns
        results = []
        sorted_ret_names = sorted(
            return_buffers.keys(), key=lambda x: int(x.split("_")[-1])
        )

        for name in sorted_ret_names:
            buf, size, dims = return_buffers[name]
            if size == 1 and dims is None:
                # Scalar
                # buf is c_double / c_int instance
                results.append(buf.value)
            else:
                # Array
                # buf is (c_double * size) instance.
                # Convert to numpy
                if np is not None:
                    # Create numpy array from buffer
                    arr = np.ctypeslib.as_array(buf)  # 1D
                    if dims:
                        # Reshape
                        try:
                            shape = []
                            for dim_str in dims:
                                eval_str = self._convert_to_python_syntax(str(dim_str))
                                val = eval(eval_str, _EVAL_GLOBALS, shape_symbol_values)
                                shape.append(int(val))

                            # Use strides directly if available
                            if name in self.output_strides:
                                stride_strs = self.output_strides[name]
                                try:
                                    # Evaluate stride expressions and convert to byte strides
                                    itemsize = arr.itemsize
                                    byte_strides = tuple(
                                        int(
                                            eval(
                                                self._convert_to_python_syntax(str(s)),
                                                {},
                                                shape_symbol_values,
                                            )
                                        )
                                        * itemsize
                                        for s in stride_strs
                                    )
                                    arr = np.lib.stride_tricks.as_strided(
                                        arr, shape=shape, strides=byte_strides
                                    )
                                except:
                                    arr = arr.reshape(shape)
                            else:
                                arr = arr.reshape(shape)
                        except:
                            pass
                    results.append(arr)
                else:
                    # fallback list
                    results.append(list(buf))

        if len(results) == 1:
            return results[0]
        elif len(results) > 1:
            return tuple(results)

        # No output args - check if function has a non-void return type
        if func_result is not None:
            return self._convert_return_value(func_result, shape_symbol_values)

        return None

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
