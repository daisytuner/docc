from docc.sdfg import (
    Scalar,
    PrimitiveType,
    Pointer,
    TaskletCode,
    CMathFunction,
)


class PythonHandler:
    """Handler for Python built-in functions (min, max, type casts, etc.)."""

    def __init__(self, expression_visitor):
        self._ev = expression_visitor
        self.function_handlers = {
            "min": self._handle_min_max,
            "max": self._handle_min_max,
            "int": self._handle_python_cast,
            "float": self._handle_python_cast,
            "bool": self._handle_python_cast,
        }

    def has_handler(self, func_name):
        """Check if this handler can handle the given function name."""
        return func_name in self.function_handlers

    def handle_python_call(self, node, func_name):
        """Handle a call to a Python built-in function."""
        if func_name in self.function_handlers:
            return self.function_handlers[func_name](node, func_name)
        raise NotImplementedError(f"Python function {func_name} not supported")

    # Expose parent properties for convenience
    @property
    def builder(self):
        return self._ev.builder

    @property
    def container_table(self):
        return self._ev.container_table

    def _add_read(self, block, expr_str, debug_info=None):
        return self._ev._add_read(block, expr_str, debug_info)

    def _is_int(self, operand):
        return self._ev._is_int(operand)

    def visit(self, node):
        return self._ev.visit(node)

    def _handle_min_max(self, node, func_name):
        """Handle Python's built-in min() and max() functions."""
        args = [self.visit(arg) for arg in node.args]
        if len(args) != 2:
            raise NotImplementedError(f"{func_name} only supported with 2 arguments")

        # Check types
        is_float = False
        arg_types = []

        for arg in args:
            name = arg
            if "(" in arg and arg.endswith(")"):
                name = arg.split("(")[0]

            if name in self.container_table:
                t = self.container_table[name]
                if isinstance(t, Pointer):
                    t = t.base_type

                if t.primitive_type == PrimitiveType.Double:
                    is_float = True
                    arg_types.append(PrimitiveType.Double)
                else:
                    arg_types.append(PrimitiveType.Int64)
            elif self._is_int(arg):
                arg_types.append(PrimitiveType.Int64)
            else:
                # Assume float constant
                is_float = True
                arg_types.append(PrimitiveType.Double)

        dtype = Scalar(PrimitiveType.Double if is_float else PrimitiveType.Int64)

        tmp_name = self.builder.find_new_name("_tmp_")
        self.builder.add_container(tmp_name, dtype, False)
        self.container_table[tmp_name] = dtype

        if is_float:
            # Cast args if necessary
            casted_args = []
            for i, arg in enumerate(args):
                if arg_types[i] != PrimitiveType.Double:
                    # Create temp double
                    tmp_cast = self.builder.find_new_name("_cast_")
                    self.builder.add_container(
                        tmp_cast, Scalar(PrimitiveType.Double), False
                    )
                    self.container_table[tmp_cast] = Scalar(PrimitiveType.Double)

                    # Assign int to double (implicit cast)
                    self.builder.add_assignment(tmp_cast, arg)
                    casted_args.append(tmp_cast)
                else:
                    casted_args.append(arg)

            block = self.builder.add_block()
            t_out = self.builder.add_access(block, tmp_name)

            intrinsic_name = (
                CMathFunction.fmax if func_name == "max" else CMathFunction.fmin
            )
            t_task = self.builder.add_cmath(block, intrinsic_name)

            for i, arg in enumerate(casted_args):
                t_arg, arg_sub = self._add_read(block, arg)
                self.builder.add_memlet(
                    block, t_arg, "void", t_task, f"_in{i+1}", arg_sub
                )
        else:
            block = self.builder.add_block()
            t_out = self.builder.add_access(block, tmp_name)

            # Use int_smax/int_smin tasklet
            opcode = None
            if func_name == "max":
                opcode = TaskletCode.int_smax
            else:
                opcode = TaskletCode.int_smin
            t_task = self.builder.add_tasklet(block, opcode, ["_in1", "_in2"], ["_out"])

            for i, arg in enumerate(args):
                t_arg, arg_sub = self._add_read(block, arg)
                self.builder.add_memlet(
                    block, t_arg, "void", t_task, f"_in{i+1}", arg_sub
                )

        self.builder.add_memlet(block, t_task, "_out", t_out, "void", "")
        return tmp_name

    def _handle_python_cast(self, node, func_name):
        """Handle Python type casts: int(), float(), bool()"""
        if len(node.args) != 1:
            raise NotImplementedError(f"{func_name}() cast requires exactly 1 argument")

        arg = self.visit(node.args[0])

        # Determine target type based on cast function
        if func_name == "int":
            target_dtype = Scalar(PrimitiveType.Int64)
        elif func_name == "float":
            target_dtype = Scalar(PrimitiveType.Double)
        elif func_name == "bool":
            target_dtype = Scalar(PrimitiveType.Bool)
        else:
            raise NotImplementedError(f"Cast to {func_name} not supported")

        # Determine source type
        source_dtype = None
        name = arg
        if "(" in arg and arg.endswith(")"):
            name = arg.split("(")[0]

        if name in self.container_table:
            source_dtype = self.container_table[name]
            if isinstance(source_dtype, Pointer):
                source_dtype = source_dtype.base_type
        elif self._is_int(arg):
            source_dtype = Scalar(PrimitiveType.Int64)
        elif arg == "true" or arg == "false":
            source_dtype = Scalar(PrimitiveType.Bool)
        else:
            # Assume float constant
            source_dtype = Scalar(PrimitiveType.Double)

        # Create temporary variable for result
        tmp_name = self.builder.find_new_name("_tmp_")
        self.builder.add_container(tmp_name, target_dtype, False)
        self.container_table[tmp_name] = target_dtype

        # Use tasklet assign opcode for casting (as specified in problem statement)
        block = self.builder.add_block()
        t_src, src_sub = self._add_read(block, arg)
        t_dst = self.builder.add_access(block, tmp_name)
        t_task = self.builder.add_tasklet(block, TaskletCode.assign, ["_in"], ["_out"])
        self.builder.add_memlet(block, t_src, "void", t_task, "_in", src_sub)
        self.builder.add_memlet(block, t_task, "_out", t_dst, "void", "")

        return tmp_name
