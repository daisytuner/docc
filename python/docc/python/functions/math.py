from docc.sdfg import (
    Scalar,
    PrimitiveType,
    CMathFunction,
)


class MathHandler:
    """Handler for Python math module functions and built-in math operations."""

    def __init__(self, expression_visitor):
        self._ev = expression_visitor

        # Mapping of math function names to CMathFunction enum values
        self.math_funcs = {
            # Trigonometric functions
            "sin": CMathFunction.sin,
            "cos": CMathFunction.cos,
            "tan": CMathFunction.tan,
            "asin": CMathFunction.asin,
            "acos": CMathFunction.acos,
            "atan": CMathFunction.atan,
            "atan2": CMathFunction.atan2,
            # Hyperbolic functions
            "sinh": CMathFunction.sinh,
            "cosh": CMathFunction.cosh,
            "tanh": CMathFunction.tanh,
            "asinh": CMathFunction.asinh,
            "acosh": CMathFunction.acosh,
            "atanh": CMathFunction.atanh,
            # Exponential and logarithmic functions
            "exp": CMathFunction.exp,
            "exp2": CMathFunction.exp2,
            "expm1": CMathFunction.expm1,
            "log": CMathFunction.log,
            "log2": CMathFunction.log2,
            "log10": CMathFunction.log10,
            "log1p": CMathFunction.log1p,
            # Power functions
            "pow": CMathFunction.pow,
            "sqrt": CMathFunction.sqrt,
            "cbrt": CMathFunction.cbrt,
            "hypot": CMathFunction.hypot,
            # Rounding and remainder functions
            "abs": CMathFunction.fabs,
            "fabs": CMathFunction.fabs,
            "ceil": CMathFunction.ceil,
            "floor": CMathFunction.floor,
            "trunc": CMathFunction.trunc,
            "fmod": CMathFunction.fmod,
            "remainder": CMathFunction.remainder,
            # Floating-point manipulation functions
            "copysign": CMathFunction.copysign,
            # Other functions
            "fma": CMathFunction.fma,
        }

    # Expose parent properties for convenience
    @property
    def builder(self):
        return self._ev.builder

    @property
    def container_table(self):
        return self._ev.container_table

    def _add_read(self, block, expr_str, debug_info=None):
        return self._ev._add_read(block, expr_str, debug_info)

    def visit(self, node):
        return self._ev.visit(node)

    def has_handler(self, func_name):
        """Check if this handler can handle the given function name."""
        return func_name in self.math_funcs

    def handle_math_call(self, node, func_name):
        """Handle a call to a math function (e.g., sin, cos, sqrt, etc.)."""
        if func_name not in self.math_funcs:
            raise NotImplementedError(f"Math function {func_name} not supported")

        args = [self.visit(arg) for arg in node.args]

        tmp_name = self.builder.find_new_name("_tmp_")
        dtype = Scalar(PrimitiveType.Double)
        self.builder.add_container(tmp_name, dtype, False)
        self.container_table[tmp_name] = dtype

        block = self.builder.add_block()
        t_out = self.builder.add_access(block, tmp_name)

        t_task = self.builder.add_cmath(block, self.math_funcs[func_name])

        for i, arg in enumerate(args):
            t_arg, arg_sub = self._add_read(block, arg)
            self.builder.add_memlet(block, t_arg, "void", t_task, f"_in{i+1}", arg_sub)

        self.builder.add_memlet(block, t_task, "_out", t_out, "void", "")
        return tmp_name
