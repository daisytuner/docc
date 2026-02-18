import ast
from docc.sdfg import DebugInfo


def is_negative_index(node):
    """Check if an AST node represents a negative constant index.

    Returns (True, abs_value) if the node is a negative constant,
    (False, None) otherwise.
    """
    # Handle -1, -2, etc. (UnaryOp with USub)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant) and isinstance(
            node.operand.value, int
        ):
            return True, node.operand.value
    # Handle negative constants directly (rare but possible)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and node.value < 0
    ):
        return True, -node.value
    return False, None


def normalize_negative_index(idx_node, dim_size_str):
    """Create an AST node that normalizes a negative index.

    If idx_node is negative, returns an AST Name node with expression
    "(dim_size - abs_value)". Otherwise returns the original node.
    """
    is_neg, abs_val = is_negative_index(idx_node)
    if is_neg:
        # Create: (dim_size - abs_value)
        return ast.Name(id=f"({dim_size_str} - {abs_val})", ctx=ast.Load())
    return idx_node


def contains_ufunc_outer(node):
    """Check if an AST node contains a ufunc outer call (e.g., np.add.outer).

    Returns (True, ufunc_name, outer_node) if found, (False, None, None) otherwise.
    """

    class UfuncOuterFinder(ast.NodeVisitor):
        def __init__(self):
            self.found = False
            self.ufunc_name = None
            self.outer_node = None

        def visit_Call(self, node):
            # Check for np.add.outer, np.subtract.outer, etc.
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "outer"
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id in ["numpy", "np"]
            ):
                self.found = True
                self.ufunc_name = node.func.value.attr
                self.outer_node = node
                return  # Don't visit children once found
            self.generic_visit(node)

    finder = UfuncOuterFinder()
    finder.visit(node)
    return finder.found, finder.ufunc_name, finder.outer_node


def get_debug_info(node, filename, function_name=""):
    if hasattr(node, "lineno"):
        return DebugInfo(
            filename,
            function_name,
            node.lineno,
            node.col_offset + 1,
            (
                node.end_lineno
                if hasattr(node, "end_lineno") and node.end_lineno is not None
                else node.lineno
            ),
            (
                node.end_col_offset + 1
                if hasattr(node, "end_col_offset") and node.end_col_offset is not None
                else node.col_offset + 1
            ),
        )
    return DebugInfo()


class ArrayToElementRewriter(ast.NodeTransformer):
    def __init__(self, loop_vars, tensor_table):
        self.loop_vars = loop_vars
        self.tensor_table = tensor_table

    def visit_Name(self, node):
        if node.id in self.tensor_table:
            # Replace with subscript
            indices = [ast.Name(id=lv, ctx=ast.Load()) for lv in self.loop_vars]
            return ast.Subscript(
                value=ast.Name(id=node.id, ctx=ast.Load()),
                slice=(
                    ast.Tuple(elts=indices, ctx=ast.Load())
                    if len(indices) > 1
                    else indices[0]
                ),
                ctx=ast.Load(),
            )
        return node


class SliceRewriter(ast.NodeTransformer):
    # Functions that operate on whole arrays, not element-wise
    # Their arguments should NOT be transformed to indexed access
    ARRAY_WISE_FUNCTIONS = {
        "flip",
        "fliplr",
        "flipud",  # Array reversal
        "dot",
        "matmul",
        "outer",
        "inner",  # Linear algebra
        "sum",
        "max",
        "min",
        "mean",
        "std",
        "var",
        "prod",  # Reductions
        "transpose",
        "reshape",
        "ravel",
        "flatten",  # Shape operations
        "sort",
        "argsort",
        "argmax",
        "argmin",  # Sorting/searching
        "cumsum",
        "cumprod",  # Cumulative operations
        "concatenate",
        "stack",
        "vstack",
        "hstack",  # Stacking
    }

    def __init__(self, loop_vars, tensor_table, expr_visitor):
        self.loop_vars = loop_vars
        self.tensor_table = tensor_table
        self.expr_visitor = expr_visitor

    def visit_Name(self, node):
        if node.id in self.tensor_table and self.loop_vars:
            ndim = len(self.tensor_table[node.id].shape)
            if ndim <= len(self.loop_vars) and ndim > 0:
                # For broadcasting: use the LAST ndim loop vars
                # e.g., for 1D array with 2 loop vars [i, j], use [j]
                indices = [
                    ast.Name(id=lv, ctx=ast.Load()) for lv in self.loop_vars[-ndim:]
                ]
                return ast.Subscript(
                    value=ast.Name(id=node.id, ctx=ast.Load()),
                    slice=(
                        ast.Tuple(elts=indices, ctx=ast.Load())
                        if len(indices) > 1
                        else indices[0]
                    ),
                    ctx=ast.Load(),
                )
        return node

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.MatMult):
            if self.loop_vars:
                indices = [ast.Name(id=lv, ctx=ast.Load()) for lv in self.loop_vars]
                return ast.Subscript(
                    value=node,
                    slice=(
                        ast.Tuple(elts=indices, ctx=ast.Load())
                        if len(indices) > 1
                        else indices[0]
                    ),
                    ctx=ast.Load(),
                )
        return self.generic_visit(node)

    def visit_Call(self, node):
        """Handle function calls - don't transform arguments of array-wise functions."""
        func_name = None
        if isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id

        if func_name in self.ARRAY_WISE_FUNCTIONS:
            # Don't transform arguments - return node as-is
            # The function handler will process the slice correctly
            return node

        # For element-wise functions, transform arguments normally
        return self.generic_visit(node)

    def visit_Subscript(self, node):
        # First check if this subscript has any slice indices that need loop vars
        indices = []
        if isinstance(node.slice, ast.Tuple):
            indices = node.slice.elts
        else:
            indices = [node.slice]

        has_slice = any(isinstance(idx, ast.Slice) for idx in indices)

        # If no slices (all point indices), don't transform this subscript at all
        # The array is already fully indexed (e.g., _fict_[0])
        if not has_slice:
            return node

        # We need to visit the value to get its string representation for tensor_table lookup
        # But DO NOT transform it - the slices in THIS subscript will be replaced by loop vars
        value_str = self.expr_visitor.visit(node.value)
        if value_str not in self.tensor_table:
            return node

        ndim = len(self.tensor_table[value_str].shape)
        if len(indices) < ndim:
            indices = list(indices)
            for _ in range(ndim - len(indices)):
                indices.append(ast.Slice(lower=None, upper=None, step=None))

        new_indices = []
        slice_counter = 0

        for i, idx in enumerate(indices):
            if isinstance(idx, ast.Slice):
                if slice_counter >= len(self.loop_vars):
                    raise ValueError("Rank mismatch in slice assignment")

                loop_var = self.loop_vars[slice_counter]
                slice_counter += 1

                # Determine step value and check if negative
                step_str = "1"
                step_is_negative = False
                if idx.step:
                    step_str = self.expr_visitor.visit(idx.step)
                    # Check for negative step
                    if step_str.startswith("-") or step_str.startswith("(-"):
                        step_is_negative = True
                    elif isinstance(idx.step, ast.UnaryOp) and isinstance(
                        idx.step.op, ast.USub
                    ):
                        step_is_negative = True

                shapes = self.tensor_table[value_str].shape
                dim_size = shapes[i] if i < len(shapes) else f"_{value_str}_shape_{i}"

                # Compute start based on step direction
                if step_is_negative:
                    # Negative step: default start is (shape - 1)
                    if idx.lower:
                        start_str = self.expr_visitor.visit(idx.lower)
                        if start_str.startswith("-"):
                            start_str = f"({dim_size} {start_str})"
                    else:
                        start_str = f"({dim_size} - 1)"
                else:
                    # Positive step: default start is 0
                    start_str = "0"
                    if idx.lower:
                        start_str = self.expr_visitor.visit(idx.lower)
                        if start_str.startswith("-"):
                            start_str = f"({dim_size} {start_str})"

                if step_str == "1":
                    if start_str == "0":
                        term = loop_var
                    else:
                        term = f"({start_str} + {loop_var})"
                else:
                    term = f"({start_str} + {loop_var} * {step_str})"
                new_indices.append(ast.Name(id=term, ctx=ast.Load()))
            else:
                # Handle non-slice indices - need to normalize negative indices
                shapes = self.tensor_table[value_str].shape
                dim_size = shapes[i] if i < len(shapes) else f"_{value_str}_shape_{i}"
                normalized_idx = normalize_negative_index(idx, dim_size)
                new_indices.append(self.visit(normalized_idx))

        if len(new_indices) == 1:
            node.slice = new_indices[0]
        else:
            node.slice = ast.Tuple(elts=new_indices, ctx=ast.Load())

        return node


def get_unique_id(counter_ref):
    counter_ref[0] += 1
    return counter_ref[0]
