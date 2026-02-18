import ast
from docc.sdfg import Scalar, PrimitiveType, Pointer, Tensor
from docc.python.ast_utils import get_debug_info


class SciPyHandler:
    """Handler for SciPy functions (scipy.special, scipy.signal, etc.)."""

    def __init__(self, expression_visitor):
        self._ev = expression_visitor
        # Nested structure: submodule -> {func_name -> handler}
        self.function_handlers = {
            "special": {
                "softmax": self._handle_softmax,
            },
            "signal": {
                "correlate2d": self._handle_correlate2d_expr,
            },
        }

    def has_handler(self, submodule, func_name):
        """Check if this handler can handle the given submodule.func_name."""
        return (
            submodule in self.function_handlers
            and func_name in self.function_handlers[submodule]
        )

    def handle_scipy_call(self, node, submodule, func_name):
        """Handle a call to a SciPy function."""
        if self.has_handler(submodule, func_name):
            return self.function_handlers[submodule][func_name](node, func_name)
        raise NotImplementedError(
            f"SciPy function scipy.{submodule}.{func_name} not supported"
        )

    # Expose parent properties for convenience
    @property
    def tensor_table(self):
        return self._ev.tensor_table

    @property
    def builder(self):
        return self._ev.builder

    @property
    def container_table(self):
        return self._ev.container_table

    def _get_unique_id(self):
        return self._ev._get_unique_id()

    def visit(self, node):
        return self._ev.visit(node)

    def _create_array_temp(self, shape, dtype):
        """Create a temporary array with the given shape and dtype."""
        return self._ev.numpy_visitor._create_array_temp(shape, dtype)

    # ========== scipy.special Functions ==========

    def _handle_softmax(self, node, func_name):
        """Handle scipy.special.softmax."""
        args = node.args
        keywords = {kw.arg: kw.value for kw in node.keywords}

        array_node = args[0]
        array_name = self.visit(array_node)

        if array_name not in self.tensor_table:
            raise ValueError(f"Softmax input must be an array, got {array_name}")

        input_tensor = self.tensor_table[array_name]
        input_shape = input_tensor.shape
        ndim = len(input_shape)

        axis = None
        if len(args) > 1:
            axis = args[1]
        elif "axis" in keywords:
            axis = keywords["axis"]

        axes = []
        if axis is None:
            axes = list(range(ndim))
        elif isinstance(axis, ast.Constant):
            val = axis.value
            if val < 0:
                val += ndim
            axes = [val]
        elif isinstance(axis, ast.Tuple):
            for elt in axis.elts:
                if isinstance(elt, ast.Constant):
                    val = elt.value
                    if val < 0:
                        val += ndim
                    axes.append(val)
        elif (
            isinstance(axis, ast.UnaryOp)
            and isinstance(axis.op, ast.USub)
            and isinstance(axis.operand, ast.Constant)
        ):
            val = -axis.operand.value
            if val < 0:
                val += ndim
            axes = [val]
        else:
            try:
                val = int(self.visit(axis))
                if val < 0:
                    val += ndim
                axes = [val]
            except:
                raise NotImplementedError("Dynamic axis not supported")

        dtype = Scalar(PrimitiveType.Double)

        tmp_name = self._create_array_temp(input_shape, dtype)

        output_tensor = self.tensor_table[tmp_name]
        self.builder.add_reduce_op(
            func_name, array_name, input_tensor, tmp_name, output_tensor, axes, False
        )

        return tmp_name

    # ========== scipy.signal Functions ==========

    def is_correlate2d(self, node):
        """Check if a node represents a scipy.signal.correlate2d call."""
        if not isinstance(node, ast.Call):
            return False

        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "correlate2d":
                return True
        elif isinstance(node.func, ast.Name):
            if node.func.id == "correlate2d":
                return True

        return False

    def handle_correlate2d(self, target, value_node):
        """Handle scipy.signal.correlate2d (2D correlation/convolution).

        Args:
            target: The assignment target (ast.Name or string)
            value_node: The correlate2d call node

        Returns:
            True if handled successfully, False otherwise
        """
        if not self.is_correlate2d(value_node):
            return False

        args = value_node.args
        if len(args) < 2:
            return False

        in1_node = args[0]
        in2_node = args[1]

        in1_name = self.visit(in1_node)
        in2_name = self.visit(in2_node)

        if in1_name not in self.tensor_table:
            return False
        if in2_name not in self.tensor_table:
            return False

        in1_info = self.tensor_table[in1_name]
        in2_info = self.tensor_table[in2_name]

        # Check dimensions
        if len(in1_info.shape) != 2 or len(in2_info.shape) != 2:
            raise NotImplementedError(
                "Only 2D convolution is currently supported via scipy.signal mapping"
            )

        in1_shape = in1_info.shape
        in2_shape = in2_info.shape

        # Scipy Correlate2d / Convolve2d
        # Default mode is 'full', boundary 'fill', fillvalue 0

        mode = "full"
        # Parse kwargs
        for keyword in value_node.keywords:
            if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                mode = keyword.value.value

        # Also check positional args for mode
        if len(args) > 2 and isinstance(args[2], ast.Constant):
            mode = args[2].value

        if mode != "valid" and mode != "full" and mode != "same":
            raise NotImplementedError(f"Unsupported convolution mode: {mode}")

        # Map to ConvNode
        # Treat as N=1, C_in=1, C_out=1

        shape_strs = ["1", "1"] + [str(s) for s in in1_shape]
        kernel_shape_strs = [str(s) for s in in2_shape]

        # Default strides 1
        strides = ["1", "1"]
        dilations = ["1", "1"]
        group = "1"
        output_channels = "1"

        pads = ["0", "0", "0", "0"]

        if mode == "valid":
            pads = ["0", "0", "0", "0"]
        elif mode == "full":
            # Padding is kernel_size - 1 on both sides
            h_k = kernel_shape_strs[0]
            w_k = kernel_shape_strs[1]
            pad_h = f"({h_k} - 1)"
            pad_w = f"({w_k} - 1)"
            pads = [pad_h, pad_w, pad_h, pad_w]
        elif mode == "same":
            # Padding is kernel_size // 2
            h_k = kernel_shape_strs[0]
            w_k = kernel_shape_strs[1]
            pad_h = f"idiv({h_k}, 2)"
            pad_w = f"idiv({w_k}, 2)"
            pads = [pad_h, pad_w, pad_h, pad_w]

        target_name = ""
        if isinstance(target, ast.Name):
            target_name = target.id
        elif isinstance(target, str):
            target_name = target

        if not target_name:
            return False

        if self.builder.exists(target_name):
            # Ensure shape is inferred
            pass
        else:
            # Infer shape
            out_shape = []
            H1 = str(in1_shape[0])
            W1 = str(in1_shape[1])
            H2 = str(in2_shape[0])
            W2 = str(in2_shape[1])

            if mode == "valid":
                out_shape = [f"({H1} - {H2} + 1)", f"({W1} - {W2} + 1)"]
            elif mode == "same":
                out_shape = [H1, W1]
            elif mode == "full":
                out_shape = [f"({H1} + {H2} - 1)", f"({W1} + {W2} - 1)"]

            # Use Double type (float)
            dtype = Scalar(PrimitiveType.Double)
            ptr_type = Pointer(dtype)

            self.builder.add_container(target_name, ptr_type, False)

            # Update parser state
            self.container_table[target_name] = ptr_type
            self.tensor_table[target_name] = Tensor(dtype, out_shape)

            # Allocate memory for the result
            block_alloc = self.builder.add_block()

            # Calculate size: shape[0] * shape[1] * sizeof(double)
            # Assuming double (8 bytes)
            size_expr = f"(({out_shape[0]}) * ({out_shape[1]}))"
            total_size_expr = f"({size_expr} * 8)"

            t_malloc = self.builder.add_malloc(block_alloc, total_size_expr)
            t_ptr = self.builder.add_access(block_alloc, target_name)
            self.builder.add_memlet(
                block_alloc, t_malloc, "_ret", t_ptr, "void", "", ptr_type
            )

        debug_info = get_debug_info(
            value_node, getattr(self.builder, "filename", ""), ""
        )

        self.builder.add_conv(
            in1_name,
            in2_name,
            target_name,
            shape_strs,
            kernel_shape_strs,
            strides,
            pads,
            dilations,
            output_channels,
            group,
            debug_info,
        )
        return True

    def _handle_correlate2d_expr(self, node, func_name):
        """Handle scipy.signal.correlate2d as an expression (creates temp array).

        This wrapper is used when correlate2d appears in an expression context
        rather than a direct assignment.
        """
        # Create a temporary name for the result
        tmp_name = self.builder.find_new_name("_corr2d_")
        # Delegate to the main handler
        success = self.handle_correlate2d(tmp_name, node)
        if not success:
            raise NotImplementedError("Failed to handle correlate2d expression")
        return tmp_name
