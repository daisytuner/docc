import ast
import copy
import inspect
import textwrap
from docc.sdfg import (
    Scalar,
    PrimitiveType,
    Pointer,
    TaskletCode,
    DebugInfo,
    Structure,
    CMathFunction,
    Tensor,
)
from docc.python.ast_utils import (
    SliceRewriter,
    get_debug_info,
    contains_ufunc_outer,
    normalize_negative_index,
)
from docc.python.types import (
    sdfg_type_from_type,
    element_type_from_sdfg_type,
)
from docc.python.functions.numpy import NumPyHandler
from docc.python.functions.math import MathHandler
from docc.python.functions.python import PythonHandler
from docc.python.functions.scipy import SciPyHandler


class ASTParser(ast.NodeVisitor):
    def __init__(
        self,
        builder,
        tensor_table,
        container_table,
        filename="",
        function_name="",
        infer_return_type=False,
        globals_dict=None,
        unique_counter_ref=None,
        structure_member_info=None,
    ):
        self.builder = builder

        # Lookup tables for variables
        self.tensor_table = tensor_table
        self.container_table = container_table

        # Debug info
        self.filename = filename
        self.function_name = function_name

        # Context
        self.infer_return_type = infer_return_type
        self.globals_dict = globals_dict if globals_dict is not None else {}
        self._unique_counter_ref = (
            unique_counter_ref if unique_counter_ref is not None else [0]
        )
        self._access_cache = {}
        self.structure_member_info = (
            structure_member_info if structure_member_info is not None else {}
        )
        self.captured_return_shapes = {}  # Map param name to shape string list
        self.captured_return_strides = {}  # Map param name to stride string list
        self.shapes_runtime_info = (
            {}
        )  # Map array name to runtime shapes (separate from Tensor)

        # Initialize handlers - they receive 'self' to access expression visitor methods
        self.numpy_visitor = NumPyHandler(self)
        self.math_handler = MathHandler(self)
        self.python_handler = PythonHandler(self)
        self.scipy_handler = SciPyHandler(self)

    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        return str(node.value)

    def visit_Name(self, node):
        name = node.id
        if name not in self.container_table and self.globals_dict is not None:
            if name in self.globals_dict:
                val = self.globals_dict[name]
                if isinstance(val, (int, float)):
                    return str(val)
        return name

    def visit_Add(self, node):
        return "+"

    def visit_Sub(self, node):
        return "-"

    def visit_Mult(self, node):
        return "*"

    def visit_Div(self, node):
        return "/"

    def visit_FloorDiv(self, node):
        return "//"

    def visit_Mod(self, node):
        return "%"

    def visit_Pow(self, node):
        return "**"

    def visit_Eq(self, node):
        return "=="

    def visit_NotEq(self, node):
        return "!="

    def visit_Lt(self, node):
        return "<"

    def visit_LtE(self, node):
        return "<="

    def visit_Gt(self, node):
        return ">"

    def visit_GtE(self, node):
        return ">="

    def visit_And(self, node):
        return "&"

    def visit_Or(self, node):
        return "|"

    def visit_BitAnd(self, node):
        return "&"

    def visit_BitOr(self, node):
        return "|"

    def visit_BitXor(self, node):
        return "^"

    def visit_LShift(self, node):
        return "<<"

    def visit_RShift(self, node):
        return ">>"

    def visit_Not(self, node):
        return "!"

    def visit_USub(self, node):
        return "-"

    def visit_UAdd(self, node):
        return "+"

    def visit_Invert(self, node):
        return "~"

    def visit_BoolOp(self, node):
        op = self.visit(node.op)
        values = [f"({self.visit(v)} != 0)" for v in node.values]
        expr_str = f"{f' {op} '.join(values)}"

        tmp_name = self.builder.find_new_name()
        dtype = Scalar(PrimitiveType.Bool)
        self.builder.add_container(tmp_name, dtype, False)

        self.builder.begin_if(expr_str)
        self._add_assign_constant(tmp_name, "true", dtype)
        self.builder.begin_else()
        self._add_assign_constant(tmp_name, "false", dtype)
        self.builder.end_if()

        self.container_table[tmp_name] = dtype
        return tmp_name

    def visit_UnaryOp(self, node):
        if (
            isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
        ):
            return f"-{node.operand.value}"

        op = self.visit(node.op)
        operand = self.visit(node.operand)

        if operand in self.tensor_table and op == "-":
            return self.numpy_visitor.handle_array_negate(operand)

        assert operand in self.container_table, f"Undefined variable: {operand}"
        tmp_name = self.builder.find_new_name()
        dtype = self.container_table[operand]
        self.builder.add_container(tmp_name, dtype, False)
        self.container_table[tmp_name] = dtype

        block = self.builder.add_block()
        t_src, src_sub = self._add_read(block, operand)
        t_dst = self.builder.add_access(block, tmp_name)

        if isinstance(node.op, ast.Not):
            t_const = self.builder.add_constant(
                block, "true", Scalar(PrimitiveType.Bool)
            )
            t_task = self.builder.add_tasklet(
                block, TaskletCode.int_xor, ["_in1", "_in2"], ["_out"]
            )
            self.builder.add_memlet(block, t_src, "void", t_task, "_in1", src_sub)
            self.builder.add_memlet(block, t_const, "void", t_task, "_in2", "")
            self.builder.add_memlet(block, t_task, "_out", t_dst, "void", "")
        elif op == "-":
            if dtype.primitive_type == PrimitiveType.Int64:
                t_const = self.builder.add_constant(block, "0", dtype)
                t_task = self.builder.add_tasklet(
                    block, TaskletCode.int_sub, ["_in1", "_in2"], ["_out"]
                )
                self.builder.add_memlet(block, t_const, "void", t_task, "_in1", "")
                self.builder.add_memlet(block, t_src, "void", t_task, "_in2", src_sub)
                self.builder.add_memlet(block, t_task, "_out", t_dst, "void", "")
            else:
                t_task = self.builder.add_tasklet(
                    block, TaskletCode.fp_neg, ["_in"], ["_out"]
                )
                self.builder.add_memlet(block, t_src, "void", t_task, "_in", src_sub)
                self.builder.add_memlet(block, t_task, "_out", t_dst, "void", "")
        elif op == "~":
            t_const = self.builder.add_constant(
                block, "-1", Scalar(PrimitiveType.Int64)
            )
            t_task = self.builder.add_tasklet(
                block, TaskletCode.int_xor, ["_in1", "_in2"], ["_out"]
            )
            self.builder.add_memlet(block, t_src, "void", t_task, "_in1", src_sub)
            self.builder.add_memlet(block, t_const, "void", t_task, "_in2", "")
            self.builder.add_memlet(block, t_task, "_out", t_dst, "void", "")
        else:
            t_task = self.builder.add_tasklet(
                block, TaskletCode.assign, ["_in"], ["_out"]
            )
            self.builder.add_memlet(block, t_src, "void", t_task, "_in", src_sub)
            self.builder.add_memlet(block, t_task, "_out", t_dst, "void", "")

        return tmp_name

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.MatMult):
            return self.numpy_visitor.handle_numpy_matmul_op(node.left, node.right)

        left = self.visit(node.left)
        op = self.visit(node.op)
        right = self.visit(node.right)

        left_is_array = left in self.tensor_table
        right_is_array = right in self.tensor_table

        if left_is_array or right_is_array:
            op_map = {"+": "add", "-": "sub", "*": "mul", "/": "div", "**": "pow"}
            if op in op_map:
                return self.numpy_visitor.handle_array_binary_op(
                    op_map[op], left, right
                )
            else:
                raise NotImplementedError(f"Array operation {op} not supported")

        tmp_name = self.builder.find_new_name()

        left_is_int = self._is_int(left)
        right_is_int = self._is_int(right)
        dtype = Scalar(PrimitiveType.Double)
        if left_is_int and right_is_int and op not in ["/", "**"]:
            dtype = Scalar(PrimitiveType.Int64)

        if not self.builder.exists(tmp_name):
            self.builder.add_container(tmp_name, dtype, False)
            self.container_table[tmp_name] = dtype

        real_left = left
        real_right = right
        if dtype.primitive_type == PrimitiveType.Double:
            if left_is_int:
                left_cast = self.builder.find_new_name()
                self.builder.add_container(
                    left_cast, Scalar(PrimitiveType.Double), False
                )
                self.container_table[left_cast] = Scalar(PrimitiveType.Double)

                c_block = self.builder.add_block()
                t_src, src_sub = self._add_read(c_block, left)
                t_dst = self.builder.add_access(c_block, left_cast)
                t_task = self.builder.add_tasklet(
                    c_block, TaskletCode.assign, ["_in"], ["_out"]
                )
                self.builder.add_memlet(c_block, t_src, "void", t_task, "_in", src_sub)
                self.builder.add_memlet(c_block, t_task, "_out", t_dst, "void", "")

                real_left = left_cast

            if right_is_int:
                right_cast = self.builder.find_new_name()
                self.builder.add_container(
                    right_cast, Scalar(PrimitiveType.Double), False
                )
                self.container_table[right_cast] = Scalar(PrimitiveType.Double)

                c_block = self.builder.add_block()
                t_src, src_sub = self._add_read(c_block, right)
                t_dst = self.builder.add_access(c_block, right_cast)
                t_task = self.builder.add_tasklet(
                    c_block, TaskletCode.assign, ["_in"], ["_out"]
                )
                self.builder.add_memlet(c_block, t_src, "void", t_task, "_in", src_sub)
                self.builder.add_memlet(c_block, t_task, "_out", t_dst, "void", "")

                real_right = right_cast

        if op == "**":
            block = self.builder.add_block()
            t_left, left_sub = self._add_read(block, real_left)
            t_right, right_sub = self._add_read(block, real_right)
            t_out = self.builder.add_access(block, tmp_name)

            t_task = self.builder.add_cmath(block, CMathFunction.pow)
            self.builder.add_memlet(block, t_left, "void", t_task, "_in1", left_sub)
            self.builder.add_memlet(block, t_right, "void", t_task, "_in2", right_sub)
            self.builder.add_memlet(block, t_task, "_out", t_out, "void", "")

            return tmp_name
        elif op == "%":
            block = self.builder.add_block()
            t_left, left_sub = self._add_read(block, real_left)
            t_right, right_sub = self._add_read(block, real_right)
            t_out = self.builder.add_access(block, tmp_name)

            if dtype.primitive_type == PrimitiveType.Int64:
                t_rem1 = self.builder.add_tasklet(
                    block, TaskletCode.int_srem, ["_in1", "_in2"], ["_out"]
                )
                self.builder.add_memlet(block, t_left, "void", t_rem1, "_in1", left_sub)
                self.builder.add_memlet(
                    block, t_right, "void", t_rem1, "_in2", right_sub
                )

                rem1_name = self.builder.find_new_name()
                self.builder.add_container(rem1_name, dtype, False)
                t_rem1_out = self.builder.add_access(block, rem1_name)
                self.builder.add_memlet(block, t_rem1, "_out", t_rem1_out, "void", "")

                t_add = self.builder.add_tasklet(
                    block, TaskletCode.int_add, ["_in1", "_in2"], ["_out"]
                )
                self.builder.add_memlet(block, t_rem1_out, "void", t_add, "_in1", "")
                self.builder.add_memlet(
                    block, t_right, "void", t_add, "_in2", right_sub
                )

                add_name = self.builder.find_new_name()
                self.builder.add_container(add_name, dtype, False)
                t_add_out = self.builder.add_access(block, add_name)
                self.builder.add_memlet(block, t_add, "_out", t_add_out, "void", "")

                t_rem2 = self.builder.add_tasklet(
                    block, TaskletCode.int_srem, ["_in1", "_in2"], ["_out"]
                )
                self.builder.add_memlet(block, t_add_out, "void", t_rem2, "_in1", "")
                self.builder.add_memlet(
                    block, t_right, "void", t_rem2, "_in2", right_sub
                )
                self.builder.add_memlet(block, t_rem2, "_out", t_out, "void", "")

                return tmp_name
            else:
                t_rem1 = self.builder.add_tasklet(
                    block, TaskletCode.fp_rem, ["_in1", "_in2"], ["_out"]
                )
                self.builder.add_memlet(block, t_left, "void", t_rem1, "_in1", left_sub)
                self.builder.add_memlet(
                    block, t_right, "void", t_rem1, "_in2", right_sub
                )

                rem1_name = self.builder.find_new_name()
                self.builder.add_container(rem1_name, dtype, False)
                t_rem1_out = self.builder.add_access(block, rem1_name)
                self.builder.add_memlet(block, t_rem1, "_out", t_rem1_out, "void", "")

                t_add = self.builder.add_tasklet(
                    block, TaskletCode.fp_add, ["_in1", "_in2"], ["_out"]
                )
                self.builder.add_memlet(block, t_rem1_out, "void", t_add, "_in1", "")
                self.builder.add_memlet(
                    block, t_right, "void", t_add, "_in2", right_sub
                )

                add_name = self.builder.find_new_name()
                self.builder.add_container(add_name, dtype, False)
                t_add_out = self.builder.add_access(block, add_name)
                self.builder.add_memlet(block, t_add, "_out", t_add_out, "void", "")

                t_rem2 = self.builder.add_tasklet(
                    block, TaskletCode.fp_rem, ["_in1", "_in2"], ["_out"]
                )
                self.builder.add_memlet(block, t_add_out, "void", t_rem2, "_in1", "")
                self.builder.add_memlet(
                    block, t_right, "void", t_rem2, "_in2", right_sub
                )
                self.builder.add_memlet(block, t_rem2, "_out", t_out, "void", "")

                return tmp_name

        tasklet_code = None
        if dtype.primitive_type == PrimitiveType.Int64:
            if op == "+":
                tasklet_code = TaskletCode.int_add
            elif op == "-":
                tasklet_code = TaskletCode.int_sub
            elif op == "*":
                tasklet_code = TaskletCode.int_mul
            elif op == "/":
                tasklet_code = TaskletCode.int_sdiv
            elif op == "//":
                tasklet_code = TaskletCode.int_sdiv
            elif op == "&":
                tasklet_code = TaskletCode.int_and
            elif op == "|":
                tasklet_code = TaskletCode.int_or
            elif op == "^":
                tasklet_code = TaskletCode.int_xor
            elif op == "<<":
                tasklet_code = TaskletCode.int_shl
            elif op == ">>":
                tasklet_code = TaskletCode.int_lshr
        else:
            if op == "+":
                tasklet_code = TaskletCode.fp_add
            elif op == "-":
                tasklet_code = TaskletCode.fp_sub
            elif op == "*":
                tasklet_code = TaskletCode.fp_mul
            elif op == "/":
                tasklet_code = TaskletCode.fp_div
            elif op == "//":
                tasklet_code = TaskletCode.fp_div
            else:
                raise NotImplementedError(f"Operation {op} not supported for floats")

        block = self.builder.add_block()
        t_left, left_sub = self._add_read(block, real_left)
        t_right, right_sub = self._add_read(block, real_right)
        t_out = self.builder.add_access(block, tmp_name)

        t_task = self.builder.add_tasklet(
            block, tasklet_code, ["_in1", "_in2"], ["_out"]
        )

        self.builder.add_memlet(block, t_left, "void", t_task, "_in1", left_sub)
        self.builder.add_memlet(block, t_right, "void", t_task, "_in2", right_sub)
        self.builder.add_memlet(block, t_task, "_out", t_out, "void", "")

        return tmp_name

    def visit_Attribute(self, node):
        if node.attr == "shape":
            if isinstance(node.value, ast.Name) and node.value.id in self.tensor_table:
                return f"_shape_proxy_{node.value.id}"

        if node.attr == "T":
            value_name = None
            if isinstance(node.value, ast.Name):
                value_name = node.value.id
            elif isinstance(node.value, ast.Subscript):
                value_name = self.visit(node.value)

            if value_name and value_name in self.tensor_table:
                return self.numpy_visitor.handle_transpose_expr(node)

        if isinstance(node.value, ast.Name) and node.value.id == "math":
            val = ""
            if node.attr == "pi":
                val = "M_PI"
            elif node.attr == "e":
                val = "M_E"

            if val:
                tmp_name = self.builder.find_new_name()
                dtype = Scalar(PrimitiveType.Double)
                self.builder.add_container(tmp_name, dtype, False)
                self.container_table[tmp_name] = dtype
                self._add_assign_constant(tmp_name, val, dtype)
                return tmp_name

        if isinstance(node.value, ast.Name):
            obj_name = node.value.id
            attr_name = node.attr

            if obj_name in self.container_table:
                obj_type = self.container_table[obj_name]
                if isinstance(obj_type, Pointer) and obj_type.has_pointee_type():
                    pointee_type = obj_type.pointee_type
                    if isinstance(pointee_type, Structure):
                        struct_name = pointee_type.name

                        if (
                            struct_name in self.structure_member_info
                            and attr_name in self.structure_member_info[struct_name]
                        ):
                            member_index, member_type = self.structure_member_info[
                                struct_name
                            ][attr_name]
                        else:
                            raise RuntimeError(
                                f"Member '{attr_name}' not found in structure '{struct_name}'. "
                                f"Available members: {list(self.structure_member_info.get(struct_name, {}).keys())}"
                            )

                        tmp_name = self.builder.find_new_name()

                        self.builder.add_container(tmp_name, member_type, False)
                        self.container_table[tmp_name] = member_type

                        block = self.builder.add_block()
                        obj_access = self.builder.add_access(block, obj_name)
                        tmp_access = self.builder.add_access(block, tmp_name)

                        tasklet = self.builder.add_tasklet(
                            block, TaskletCode.assign, ["_in"], ["_out"]
                        )

                        subset = "0," + str(member_index)
                        self.builder.add_memlet(
                            block, obj_access, "", tasklet, "_in", subset
                        )
                        self.builder.add_memlet(block, tasklet, "_out", tmp_access, "")

                        return tmp_name

        raise NotImplementedError(f"Attribute access {node.attr} not supported")

    def visit_Compare(self, node):
        left = self.visit(node.left)
        if len(node.ops) > 1:
            raise NotImplementedError("Chained comparisons not supported yet")

        op = self.visit(node.ops[0])
        right = self.visit(node.comparators[0])

        left_is_array = left in self.tensor_table
        right_is_array = right in self.tensor_table

        if left_is_array or right_is_array:
            return self.numpy_visitor.handle_array_compare(
                left, op, right, left_is_array, right_is_array
            )

        expr_str = f"{left} {op} {right}"

        tmp_name = self.builder.find_new_name()
        dtype = Scalar(PrimitiveType.Bool)
        self.builder.add_container(tmp_name, dtype, False)

        self.builder.begin_if(expr_str)
        self.builder.add_transition(tmp_name, "true")
        self.builder.begin_else()
        self.builder.add_transition(tmp_name, "false")
        self.builder.end_if()

        self.container_table[tmp_name] = dtype
        return tmp_name

    def visit_Subscript(self, node):
        value_str = self.visit(node.value)

        if value_str.startswith("_shape_proxy_"):
            array_name = value_str[len("_shape_proxy_") :]
            if isinstance(node.slice, ast.Constant):
                idx = node.slice.value
            elif isinstance(node.slice, ast.Index):
                idx = node.slice.value.value
            else:
                try:
                    idx = int(self.visit(node.slice))
                except:
                    raise NotImplementedError(
                        "Dynamic shape indexing not fully supported yet"
                    )

            if array_name in self.tensor_table:
                return self.tensor_table[array_name].shape[idx]

            return f"_{array_name}_shape_{idx}"

        if value_str in self.tensor_table:
            tensor = self.tensor_table[value_str]
            ndim = len(tensor.shape)
            shapes = tensor.shape

            if isinstance(node.slice, ast.Tuple):
                indices_nodes = node.slice.elts
            else:
                indices_nodes = [node.slice]

            all_full_slices = True
            for idx in indices_nodes:
                if isinstance(idx, ast.Slice):
                    if idx.lower is not None or idx.upper is not None:
                        all_full_slices = False
                        break
                    # Also check for non-trivial step (step != None and step != 1)
                    if idx.step is not None:
                        # Check if step is a constant 1; if not, it's not a full slice
                        if isinstance(idx.step, ast.Constant) and idx.step.value == 1:
                            pass  # step=1 is equivalent to no step
                        else:
                            all_full_slices = False
                            break
                else:
                    all_full_slices = False
                    break

            if all_full_slices:
                return value_str

            has_slices = any(isinstance(idx, ast.Slice) for idx in indices_nodes)
            if has_slices:
                return self._handle_expression_slicing(
                    node, value_str, indices_nodes, shapes, ndim
                )

            if len(indices_nodes) == 1 and self._is_array_index(indices_nodes[0]):
                if self.builder:
                    return self._handle_gather(value_str, indices_nodes[0])

            if isinstance(node.slice, ast.Tuple):
                indices = [self.visit(elt) for elt in node.slice.elts]
            else:
                indices = [self.visit(node.slice)]

            if len(indices) != ndim:
                raise ValueError(
                    f"Array {value_str} has {ndim} dimensions, but accessed with {len(indices)} indices"
                )

            normalized_indices = []
            for i, idx_str in enumerate(indices):
                shape_val = shapes[i]
                if isinstance(idx_str, str) and (
                    idx_str.startswith("-") or idx_str.startswith("(-")
                ):
                    normalized_indices.append(f"({shape_val} + {idx_str})")
                else:
                    normalized_indices.append(idx_str)

            subscript_str = ",".join(normalized_indices)
            access_str = f"{value_str}({subscript_str})"

            if isinstance(node.ctx, ast.Load):
                tmp_name = self.builder.find_new_name()
                self.builder.add_container(tmp_name, tensor.element_type, False)
                self.container_table[tmp_name] = tensor.element_type

                block = self.builder.add_block()
                t_src = self.builder.add_access(block, value_str)
                t_dst = self.builder.add_access(block, tmp_name)
                t_task = self.builder.add_tasklet(
                    block, TaskletCode.assign, ["_in"], ["_out"]
                )
                self.builder.add_memlet(
                    block, t_src, "void", t_task, "_in", subscript_str, tensor
                )
                self.builder.add_memlet(
                    block, t_task, "_out", t_dst, "void", "", tensor.element_type
                )

                return tmp_name

            return access_str

        slice_val = self.visit(node.slice)
        access_str = f"{value_str}({slice_val})"
        return access_str

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name) and node.target.id in self.tensor_table:
            # Convert to slice assignment: target[:] = target op value
            ndim = len(self.tensor_table[node.target.id].shape)

            slices = []
            for _ in range(ndim):
                slices.append(ast.Slice(lower=None, upper=None, step=None))

            if ndim == 1:
                slice_arg = slices[0]
            else:
                slice_arg = ast.Tuple(elts=slices, ctx=ast.Load())

            slice_node = ast.Subscript(
                value=node.target, slice=slice_arg, ctx=ast.Store()
            )

            new_node = ast.Assign(
                targets=[slice_node],
                value=ast.BinOp(left=node.target, op=node.op, right=node.value),
            )
            self.visit_Assign(new_node)
        else:
            new_node = ast.Assign(
                targets=[node.target],
                value=ast.BinOp(left=node.target, op=node.op, right=node.value),
            )
            self.visit_Assign(new_node)

    def visit_Assign(self, node):
        # Handle multiple targets: a = b = c or a, b = expr
        if len(node.targets) > 1:
            tmp_name = self.builder.find_new_name()
            # Assign value to temporary
            val_assign = ast.Assign(
                targets=[ast.Name(id=tmp_name, ctx=ast.Store())], value=node.value
            )
            ast.copy_location(val_assign, node)
            self.visit_Assign(val_assign)

            # Assign temporary to targets
            for target in node.targets:
                assign = ast.Assign(
                    targets=[target], value=ast.Name(id=tmp_name, ctx=ast.Load())
                )
                ast.copy_location(assign, node)
                self.visit_Assign(assign)
            return
        target = node.targets[0]

        # Handle tuple unpacking: I, J, K = expr1, expr2, expr3
        if isinstance(target, ast.Tuple):
            if isinstance(node.value, ast.Tuple):
                if len(target.elts) != len(node.value.elts):
                    raise ValueError("Tuple unpacking size mismatch")
                for tgt, val in zip(target.elts, node.value.elts):
                    assign = ast.Assign(targets=[tgt], value=val)
                    ast.copy_location(assign, node)
                    self.visit_Assign(assign)
                return
            else:
                raise NotImplementedError(
                    "Tuple unpacking from non-tuple values not supported"
                )

        # Special cases, where rhs is not just a simple expression but requires special handling
        if self.numpy_visitor.is_gemm(node.value):
            if self.numpy_visitor.handle_gemm(target, node.value):
                return
            if self.numpy_visitor.handle_dot(target, node.value):
                return
        if self.numpy_visitor.is_outer(node.value):
            if self.numpy_visitor.handle_outer(target, node.value):
                return
        if self.scipy_handler.is_correlate2d(node.value):
            if self.scipy_handler.handle_correlate2d(target, node.value):
                return
        if self.numpy_visitor.is_transpose(node.value):
            if self.numpy_visitor.handle_transpose(target, node.value):
                return

        # Handle subscript assignments: a[i] = val or a[i, j] = val
        if isinstance(target, ast.Subscript):
            debug_info = get_debug_info(node, self.filename, self.function_name)

            target_name = self.visit(target.value)
            indices = []
            if isinstance(target.slice, ast.Tuple):
                indices = target.slice.elts
            else:
                indices = [target.slice]

            # Handle slice assignment separately
            has_slice = False
            for idx in indices:
                if isinstance(idx, ast.Slice):
                    has_slice = True
                    break

            if has_slice:
                self._handle_slice_assignment(
                    target, node.value, target_name, indices, debug_info
                )
                return

            # Handle rhs and store in scalar tmp
            rhs_tmp = self.visit(node.value)

            block = self.builder.add_block(debug_info)
            t_task = self.builder.add_tasklet(
                block, TaskletCode.assign, ["_in"], ["_out"], debug_info
            )

            t_src, src_sub = self._add_read(block, rhs_tmp, debug_info)
            self.builder.add_memlet(
                block, t_src, "void", t_task, "_in", src_sub, None, debug_info
            )

            lhs_expr = self.visit(target)
            if "(" in lhs_expr and lhs_expr.endswith(")"):
                subset = lhs_expr[lhs_expr.find("(") + 1 : -1]
                tensor_dst = self.tensor_table[target_name]

                t_dst = self.builder.add_access(block, target_name, debug_info)
                self.builder.add_memlet(
                    block, t_task, "_out", t_dst, "void", subset, tensor_dst, debug_info
                )
            else:
                t_dst = self.builder.add_access(block, target_name, debug_info)
                self.builder.add_memlet(
                    block, t_task, "_out", t_dst, "void", "", None, debug_info
                )
            return

        # Fallback: lhs is a simple scalar assignments
        if not isinstance(target, ast.Name):
            raise NotImplementedError("Only assignment to variables supported")

        target_name = target.id
        rhs_tmp = self.visit(node.value)
        debug_info = get_debug_info(node, self.filename, self.function_name)

        if not self.builder.exists(target_name):
            if isinstance(node.value, ast.Constant):
                val = node.value.value
                if isinstance(val, int):
                    dtype = Scalar(PrimitiveType.Int64)
                elif isinstance(val, float):
                    dtype = Scalar(PrimitiveType.Double)
                elif isinstance(val, bool):
                    dtype = Scalar(PrimitiveType.Bool)
                else:
                    raise NotImplementedError(f"Cannot infer type for {val}")

                self.builder.add_container(target_name, dtype, False)
                self.container_table[target_name] = dtype
            else:
                self.builder.add_container(
                    target_name, self.container_table[rhs_tmp], False
                )
                self.container_table[target_name] = self.container_table[rhs_tmp]

        if rhs_tmp in self.tensor_table:
            self.tensor_table[target_name] = self.tensor_table[rhs_tmp]

        # Also copy shapes_runtime_info if available
        if rhs_tmp in self.shapes_runtime_info:
            self.shapes_runtime_info[target_name] = self.shapes_runtime_info[rhs_tmp]

        # Distinguish assignments: scalar -> tasklet, pointer -> reference_memlet
        src_type = self.container_table.get(rhs_tmp)
        dst_type = self.container_table[target_name]
        if src_type and isinstance(src_type, Pointer) and isinstance(dst_type, Pointer):
            block = self.builder.add_block(debug_info)
            t_src = self.builder.add_access(block, rhs_tmp, debug_info)
            t_dst = self.builder.add_access(block, target_name, debug_info)
            self.builder.add_reference_memlet(
                block, t_src, t_dst, "0", src_type, debug_info
            )
        elif (src_type and isinstance(src_type, Scalar)) or isinstance(
            dst_type, Scalar
        ):
            block = self.builder.add_block(debug_info)
            t_dst = self.builder.add_access(block, target_name, debug_info)
            t_task = self.builder.add_tasklet(
                block, TaskletCode.assign, ["_in"], ["_out"], debug_info
            )

            if src_type:
                t_src = self.builder.add_access(block, rhs_tmp, debug_info)
            else:
                t_src = self.builder.add_constant(block, rhs_tmp, dst_type, debug_info)

            self.builder.add_memlet(
                block, t_src, "void", t_task, "_in", "", None, debug_info
            )
            self.builder.add_memlet(
                block, t_task, "_out", t_dst, "void", "", None, debug_info
            )

    def visit_Expr(self, node):
        self.visit(node.value)

    def visit_If(self, node):
        cond = self.visit(node.test)
        debug_info = get_debug_info(node, self.filename, self.function_name)
        self.builder.begin_if(f"{cond} != false", debug_info)

        for stmt in node.body:
            self.visit(stmt)

        if node.orelse:
            self.builder.begin_else(debug_info)
            for stmt in node.orelse:
                self.visit(stmt)

        self.builder.end_if()

    def visit_While(self, node):
        if node.orelse:
            raise NotImplementedError("while-else is not supported")

        debug_info = get_debug_info(node, self.filename, self.function_name)
        self.builder.begin_while(debug_info)

        # Evaluate condition inside the loop so it's re-evaluated each iteration
        cond = self.visit(node.test)

        # Create if-break pattern: if condition is false, break
        self.builder.begin_if(f"{cond} == false", debug_info)
        self.builder.add_break(debug_info)
        self.builder.end_if()

        for stmt in node.body:
            self.visit(stmt)

        self.builder.end_while()

    def visit_Break(self, node):
        debug_info = get_debug_info(node, self.filename, self.function_name)
        self.builder.add_break(debug_info)

    def visit_Continue(self, node):
        debug_info = get_debug_info(node, self.filename, self.function_name)
        self.builder.add_continue(debug_info)

    def visit_For(self, node):
        if node.orelse:
            raise NotImplementedError("while-else is not supported")
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError("Only simple for loops supported")

        var = node.target.id
        debug_info = get_debug_info(node, self.filename, self.function_name)

        # Check if iterating over a range() call
        if (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            args = node.iter.args
            if len(args) == 1:
                start = "0"
                end = self.visit(args[0])
                step = "1"
            elif len(args) == 2:
                start = self.visit(args[0])
                end = self.visit(args[1])
                step = "1"
            elif len(args) == 3:
                start = self.visit(args[0])
                end = self.visit(args[1])

                # Special handling for step to avoid creating tasklets for constants
                step_node = args[2]
                if isinstance(step_node, ast.Constant):
                    step = str(step_node.value)
                elif (
                    isinstance(step_node, ast.UnaryOp)
                    and isinstance(step_node.op, ast.USub)
                    and isinstance(step_node.operand, ast.Constant)
                ):
                    step = f"-{step_node.operand.value}"
                else:
                    step = self.visit(step_node)
            else:
                raise ValueError("Invalid range arguments")

            if not self.builder.exists(var):
                self.builder.add_container(var, Scalar(PrimitiveType.Int64), False)
                self.container_table[var] = Scalar(PrimitiveType.Int64)

            self.builder.begin_for(var, start, end, step, debug_info)

            for stmt in node.body:
                self.visit(stmt)

            self.builder.end_for()
            return

        # Check if iterating over an ndarray (for x in array)
        if isinstance(node.iter, ast.Name):
            iter_name = node.iter.id
            if iter_name in self.tensor_table:
                arr_info = self.tensor_table[iter_name]
                if len(arr_info.shape) == 0:
                    raise NotImplementedError("Cannot iterate over 0-dimensional array")

                # Get the size of the first dimension
                arr_size = arr_info.shape[0]

                # Create a hidden index variable for the loop
                idx_var = self.builder.find_new_name()
                if not self.builder.exists(idx_var):
                    self.builder.add_container(
                        idx_var, Scalar(PrimitiveType.Int64), False
                    )
                    self.container_table[idx_var] = Scalar(PrimitiveType.Int64)

                # Determine the type of the loop variable (element type)
                # For a 1D array, it's a scalar; for ND array, it's a view of N-1 dimensions
                if len(arr_info.shape) == 1:
                    # Element is a scalar - get the element type from the array's type
                    arr_type = self.container_table.get(iter_name)
                    if isinstance(arr_type, Pointer):
                        elem_type = arr_type.pointee_type
                    else:
                        elem_type = Scalar(PrimitiveType.Double)  # Default fallback

                    if not self.builder.exists(var):
                        self.builder.add_container(var, elem_type, False)
                        self.container_table[var] = elem_type
                else:
                    # For multi-dimensional arrays, create a view/slice
                    # The loop variable becomes a pointer to the sub-array
                    inner_shapes = arr_info.shape[1:]
                    inner_ndim = len(arr_info.shape) - 1

                    arr_type = self.container_table.get(iter_name)
                    if isinstance(arr_type, Pointer):
                        elem_type = arr_type  # Keep as pointer type for views
                    else:
                        elem_type = Pointer(Scalar(PrimitiveType.Double))

                    if not self.builder.exists(var):
                        self.builder.add_container(var, elem_type, False)
                        self.container_table[var] = elem_type

                    # Register the view in tensor_table
                    self.tensor_table[var] = Tensor(
                        element_type_from_sdfg_type(elem_type), inner_shapes
                    )

                # Begin the for loop
                self.builder.begin_for(idx_var, "0", str(arr_size), "1", debug_info)

                # Generate the assignment: var = array[idx_var]
                # Create an AST node for the assignment and visit it
                assign_node = ast.Assign(
                    targets=[ast.Name(id=var, ctx=ast.Store())],
                    value=ast.Subscript(
                        value=ast.Name(id=iter_name, ctx=ast.Load()),
                        slice=ast.Name(id=idx_var, ctx=ast.Load()),
                        ctx=ast.Load(),
                    ),
                )
                ast.copy_location(assign_node, node)
                self.visit_Assign(assign_node)

                # Visit the loop body
                for stmt in node.body:
                    self.visit(stmt)

                self.builder.end_for()
                return

        raise NotImplementedError(
            f"Only range() loops and iteration over ndarrays supported, got: {ast.dump(node.iter)}"
        )

    def visit_Return(self, node):
        if node.value is None:
            debug_info = get_debug_info(node, self.filename, self.function_name)
            self.builder.add_return("", debug_info)
            return

        if isinstance(node.value, ast.Tuple):
            values = node.value.elts
        else:
            values = [node.value]

        parsed_values = [self.visit(v) for v in values]
        debug_info = get_debug_info(node, self.filename, self.function_name)

        if self.infer_return_type:
            for i, res in enumerate(parsed_values):
                ret_name = f"_docc_ret_{i}"
                if not self.builder.exists(ret_name):
                    dtype = Scalar(PrimitiveType.Double)
                    if res in self.container_table:
                        dtype = self.container_table[res]
                    elif isinstance(values[i], ast.Constant):
                        val = values[i].value
                        if isinstance(val, int):
                            dtype = Scalar(PrimitiveType.Int64)
                        elif isinstance(val, float):
                            dtype = Scalar(PrimitiveType.Double)
                        elif isinstance(val, bool):
                            dtype = Scalar(PrimitiveType.Bool)

                    # Wrap Scalar in Pointer. Keep Arrays/Pointers as is.
                    arg_type = dtype
                    if isinstance(dtype, Scalar):
                        arg_type = Pointer(dtype)

                    self.builder.add_container(ret_name, arg_type, is_argument=True)
                    self.container_table[ret_name] = arg_type

                    if res in self.tensor_table:
                        self.tensor_table[ret_name] = self.tensor_table[res]

            self.infer_return_type = False

        for i, res in enumerate(parsed_values):
            ret_name = f"_docc_ret_{i}"
            typ = self.container_table.get(ret_name)

            is_array_return = False
            if res in self.tensor_table:
                # Only treat as array return if it has dimensions
                # 0-d arrays (scalars) should be handled by scalar assignment
                if len(self.tensor_table[res].shape) > 0:
                    is_array_return = True
            elif res in self.container_table:
                if isinstance(self.container_table[res], Pointer):
                    is_array_return = True

            # Simple Scalar Assignment
            if not is_array_return:
                block = self.builder.add_block(debug_info)
                t_dst = self.builder.add_access(block, ret_name, debug_info)

                t_src, src_sub = self._add_read(block, res, debug_info)

                t_task = self.builder.add_tasklet(
                    block, TaskletCode.assign, ["_in"], ["_out"], debug_info
                )
                self.builder.add_memlet(
                    block, t_src, "void", t_task, "_in", src_sub, None, debug_info
                )
                self.builder.add_memlet(
                    block, t_task, "_out", t_dst, "void", "0", None, debug_info
                )

            # Array Assignment (Copy)
            else:
                # Record shape for metadata
                if res in self.tensor_table:
                    # Prefer runtime shapes if available (for indirect access patterns)
                    # Fall back to regular shapes otherwise
                    res_info = self.tensor_table[res]
                    if res in self.shapes_runtime_info:
                        shape = self.shapes_runtime_info[res]
                    else:
                        shape = res_info.shape
                    # Convert to string expressions
                    self.captured_return_shapes[ret_name] = [str(s) for s in shape]

                    # Return arrays are always contiguous - compute fresh strides
                    contiguous_strides = self.numpy_visitor._compute_strides(shape, "C")
                    self.captured_return_strides[ret_name] = [
                        str(s) for s in contiguous_strides
                    ]

                    # Always overwrite tensor_table for return arrays with contiguous strides
                    # (source tensor may have non-standard strides from views/flip)
                    self.tensor_table[ret_name] = Tensor(
                        res_info.element_type, shape, contiguous_strides
                    )

                # Copy Logic using visit_Assign
                ndim = 1
                if ret_name in self.tensor_table:
                    ndim = len(self.tensor_table[ret_name].shape)

                slice_node = ast.Slice(lower=None, upper=None, step=None)
                if ndim > 1:
                    target_slice = ast.Tuple(elts=[slice_node] * ndim, ctx=ast.Load())
                else:
                    target_slice = slice_node

                target_sub = ast.Subscript(
                    value=ast.Name(id=ret_name, ctx=ast.Load()),
                    slice=target_slice,
                    ctx=ast.Store(),
                )

                # Value node reconstruction
                if isinstance(values[i], ast.Name):
                    val_node = values[i]
                else:
                    val_node = ast.Name(id=res, ctx=ast.Load())

                assign_node = ast.Assign(targets=[target_sub], value=val_node)
                self.visit_Assign(assign_node)

        # Add control flow return to exit the function/path
        self.builder.add_return("", debug_info)

    def visit_Call(self, node):
        func_name = ""
        module_name = ""
        submodule_name = ""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == "math":
                    module_name = "math"
                    func_name = node.func.attr
                elif node.func.value.id in ["numpy", "np"]:
                    module_name = "numpy"
                    func_name = node.func.attr
                else:
                    array_name = node.func.value.id
                    method_name = node.func.attr
                    if array_name in self.tensor_table and method_name == "astype":
                        return self.numpy_visitor.handle_numpy_astype(node, array_name)
                    elif array_name in self.tensor_table and method_name == "copy":
                        return self.numpy_visitor.handle_numpy_copy(node, array_name)
            elif isinstance(node.func.value, ast.Attribute):
                if (
                    isinstance(node.func.value.value, ast.Name)
                    and node.func.value.value.id == "scipy"
                ):
                    module_name = "scipy"
                    submodule_name = node.func.value.attr
                    func_name = node.func.attr
                elif (
                    isinstance(node.func.value.value, ast.Name)
                    and node.func.value.value.id in ["numpy", "np"]
                    and node.func.attr == "outer"
                ):
                    ufunc_name = node.func.value.attr
                    return self.numpy_visitor.handle_ufunc_outer(node, ufunc_name)

        elif isinstance(node.func, ast.Name):
            func_name = node.func.id

        if module_name == "numpy":
            if self.numpy_visitor.has_handler(func_name):
                return self.numpy_visitor.handle_numpy_call(node, func_name)

        if module_name == "math":
            if self.math_handler.has_handler(func_name):
                return self.math_handler.handle_math_call(node, func_name)

        if module_name == "scipy":
            if self.scipy_handler.has_handler(submodule_name, func_name):
                return self.scipy_handler.handle_scipy_call(
                    node, submodule_name, func_name
                )

        if self.python_handler.has_handler(func_name):
            return self.python_handler.handle_python_call(node, func_name)

        if func_name in self.globals_dict:
            obj = self.globals_dict[func_name]
            if inspect.isfunction(obj):
                return self._handle_inline_call(node, obj)

        raise NotImplementedError(f"Function call {func_name} not supported")

    def _handle_inline_call(self, node, func_obj):
        try:
            source_lines, start_line = inspect.getsourcelines(func_obj)
            source = textwrap.dedent("".join(source_lines))
            tree = ast.parse(source)
            func_def = tree.body[0]
        except Exception as e:
            raise NotImplementedError(
                f"Could not parse function {func_obj.__name__}: {e}"
            )

        arg_vars = [self.visit(arg) for arg in node.args]

        if len(arg_vars) != len(func_def.args.args):
            raise NotImplementedError(
                f"Argument count mismatch for {func_obj.__name__}"
            )

        suffix = f"_{func_obj.__name__}_{self.builder.find_new_name()}"
        res_name = f"_res{suffix}"

        # Combine globals with closure variables of the inlined function
        combined_globals = dict(self.globals_dict)
        closure_constants = {}  # name -> value for numeric closure vars
        if func_obj.__closure__ is not None and func_obj.__code__.co_freevars:
            for name, cell in zip(func_obj.__code__.co_freevars, func_obj.__closure__):
                val = cell.cell_contents
                combined_globals[name] = val
                # Track numeric constants for injection
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    closure_constants[name] = val

        class VariableRenamer(ast.NodeTransformer):
            BUILTINS = {
                "range",
                "len",
                "int",
                "float",
                "bool",
                "str",
                "list",
                "dict",
                "tuple",
                "set",
                "print",
                "abs",
                "min",
                "max",
                "sum",
                "enumerate",
                "zip",
                "map",
                "filter",
                "sorted",
                "reversed",
                "True",
                "False",
                "None",
            }

            def __init__(self, suffix, globals_dict):
                self.suffix = suffix
                self.globals_dict = globals_dict

            def visit_Name(self, node):
                if node.id in self.globals_dict or node.id in self.BUILTINS:
                    return node
                return ast.Name(id=f"{node.id}{self.suffix}", ctx=node.ctx)

            def visit_Return(self, node):
                if node.value:
                    val = self.visit(node.value)
                    return ast.Assign(
                        targets=[ast.Name(id=res_name, ctx=ast.Store())],
                        value=val,
                    )
                return node

        renamer = VariableRenamer(suffix, combined_globals)
        new_body = [renamer.visit(stmt) for stmt in func_def.body]

        param_assignments = []

        # Inject closure constants as assignments
        for name, val in closure_constants.items():
            if isinstance(val, int):
                self.container_table[name] = Scalar(PrimitiveType.Int64)
                self.builder.add_container(name, Scalar(PrimitiveType.Int64), False)
                val_node = ast.Constant(value=val)
            else:
                self.container_table[name] = Scalar(PrimitiveType.Double)
                self.builder.add_container(name, Scalar(PrimitiveType.Double), False)
                val_node = ast.Constant(value=val)
            assign = ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())], value=val_node
            )
            param_assignments.append(assign)

        for arg_def, arg_val in zip(func_def.args.args, arg_vars):
            param_name = f"{arg_def.arg}{suffix}"

            if arg_val in self.container_table:
                self.container_table[param_name] = self.container_table[arg_val]
                self.builder.add_container(
                    param_name, self.container_table[arg_val], False
                )
                val_node = ast.Name(id=arg_val, ctx=ast.Load())
            elif self._is_int(arg_val):
                self.container_table[param_name] = Scalar(PrimitiveType.Int64)
                self.builder.add_container(
                    param_name, Scalar(PrimitiveType.Int64), False
                )
                val_node = ast.Constant(value=int(arg_val))
            else:
                try:
                    val = float(arg_val)
                    self.container_table[param_name] = Scalar(PrimitiveType.Double)
                    self.builder.add_container(
                        param_name, Scalar(PrimitiveType.Double), False
                    )
                    val_node = ast.Constant(value=val)
                except ValueError:
                    val_node = ast.Name(id=arg_val, ctx=ast.Load())

            assign = ast.Assign(
                targets=[ast.Name(id=param_name, ctx=ast.Store())], value=val_node
            )
            param_assignments.append(assign)

        final_body = param_assignments + new_body

        # Create a new parser instance for the inlined function
        parser = ASTParser(
            self.builder,
            self.tensor_table,
            self.container_table,
            globals_dict=combined_globals,
            unique_counter_ref=self._unique_counter_ref,
        )

        for stmt in final_body:
            parser.visit(stmt)

        return res_name

    def _add_assign_constant(self, target_name, value_str, dtype):
        block = self.builder.add_block()
        t_const = self.builder.add_constant(block, value_str, dtype)
        t_dst = self.builder.add_access(block, target_name)
        t_task = self.builder.add_tasklet(block, TaskletCode.assign, ["_in"], ["_out"])
        self.builder.add_memlet(block, t_const, "void", t_task, "_in", "")
        self.builder.add_memlet(block, t_task, "_out", t_dst, "void", "")

    def _handle_expression_slicing(self, node, value_str, indices_nodes, shapes, ndim):
        """Handle slicing in expressions (e.g., arr[1:, :, k+1]).

        Uses a zero-copy view when possible (positive step, no indirect access).
        Falls back to copy-based approach for complex cases.
        """
        if not self.builder:
            raise ValueError("Builder required for expression slicing")

        # Try view-based approach first (zero-copy)
        if self._can_use_slice_view(indices_nodes):
            return self._create_slice_view(value_str, indices_nodes, shapes, ndim)

        # Fall back to copy-based approach for complex cases
        return self._handle_expression_slicing_copy(
            node, value_str, indices_nodes, shapes, ndim
        )

    def _can_use_slice_view(self, indices_nodes):
        """Check if slicing can be expressed as a zero-copy view.

        Views can be used when:
        - All steps are non-zero constants (positive or negative)
        - No indirect array access in slice parameters

        Returns True if a view can be used, False if a copy is required.
        """
        for idx in indices_nodes:
            if isinstance(idx, ast.Slice):
                # Check for zero step (invalid)
                if idx.step is not None:
                    if isinstance(idx.step, ast.Constant):
                        if idx.step.value == 0:
                            return False  # Zero step is invalid
                    elif isinstance(idx.step, ast.UnaryOp) and isinstance(
                        idx.step.op, ast.USub
                    ):
                        # Negative step like -2 is OK
                        pass
                    elif self._contains_indirect_access(idx.step):
                        return False  # Dynamic step requires copy

                # Check for indirect access in slice bounds
                if idx.lower is not None and self._contains_indirect_access(idx.lower):
                    return False
                if idx.upper is not None and self._contains_indirect_access(idx.upper):
                    return False
            else:
                # Fixed index: check for indirect access
                if self._contains_indirect_access(idx):
                    return False
        return True

    def _create_slice_view(self, value_str, indices_nodes, shapes, ndim):
        """Create a zero-copy view for array slicing.

        This creates a new tensor that shares data with the source but has
        adjusted shape, strides, and offset to represent the sliced region.

        For positive step A[start:stop:step, ...] on dimension i:
        - new_shape[i] = ceil((stop - start) / step)
        - new_stride[i] = old_stride[i] * step
        - offset contribution = start * old_stride[i]

        For negative step A[start:stop:step, ...] (e.g., ::-1):
        - Default start = shape - 1 (last element)
        - Default stop = -1 (before first element)
        - new_shape[i] = ceil((start - stop) / abs(step))
        - new_stride[i] = old_stride[i] * step (negative)
        - offset contribution = start * old_stride[i] (points to last element)

        For a fixed index A[k, ...] on dimension i (dimension reduction):
        - offset contribution = k * old_stride[i]
        - dimension is removed from output
        """
        in_tensor = self.tensor_table[value_str]
        in_shape = in_tensor.shape
        dtype = in_tensor.element_type

        # Get input strides (compute if not available)
        in_strides = (
            in_tensor.strides
            if hasattr(in_tensor, "strides") and in_tensor.strides
            else None
        )
        if in_strides is None:
            in_strides = self.numpy_visitor._compute_strides(in_shape, "C")

        # Get base offset from input tensor
        in_offset = getattr(in_tensor, "offset", "0") or "0"

        # Build output shape, strides, and compute offset
        out_shape = []
        out_strides = []
        offset_terms = []
        if in_offset != "0":
            offset_terms.append(str(in_offset))

        for i, idx in enumerate(indices_nodes):
            shape_val = shapes[i] if i < len(shapes) else f"_{value_str}_shape_{i}"
            stride_val = in_strides[i] if i < len(in_strides) else "1"

            if isinstance(idx, ast.Slice):
                # Determine step value and sign
                step_str = "1"
                step_is_negative = False
                step_value = 1

                if idx.step is not None:
                    if isinstance(idx.step, ast.Constant):
                        step_value = idx.step.value
                        step_str = str(step_value)
                        step_is_negative = step_value < 0
                    elif isinstance(idx.step, ast.UnaryOp) and isinstance(
                        idx.step.op, ast.USub
                    ):
                        # Handle -N syntax
                        if isinstance(idx.step.operand, ast.Constant):
                            step_value = -idx.step.operand.value
                            step_str = str(step_value)
                            step_is_negative = True
                        else:
                            step_str = self.visit(idx.step)
                    else:
                        step_str = self.visit(idx.step)

                if step_is_negative:
                    # Negative step: iterate from end to start
                    # Default start = shape - 1, default stop = -1 (before 0)
                    if idx.lower is not None:
                        start_str = self.visit(idx.lower)
                        if isinstance(start_str, str) and (
                            start_str.startswith("-") or start_str.startswith("(-")
                        ):
                            start_str = f"({shape_val} + {start_str})"
                    else:
                        start_str = f"({shape_val} - 1)"

                    if idx.upper is not None:
                        stop_str = self.visit(idx.upper)
                        if isinstance(stop_str, str) and (
                            stop_str.startswith("-") or stop_str.startswith("(-")
                        ):
                            stop_str = f"({shape_val} + {stop_str})"
                    else:
                        stop_str = "-1"

                    # Shape for negative step: ceil((start - stop) / abs(step))
                    abs_step = abs(step_value)
                    if abs_step == 1:
                        dim_size = f"({start_str} - {stop_str})"
                    else:
                        dim_size = f"(({start_str} - {stop_str} + {abs_step} - 1) / {abs_step})"
                    out_shape.append(dim_size)

                    # Stride for negative step: old_stride * step (negative)
                    out_strides.append(f"({stride_val} * {step_str})")

                    # Offset: start * old_stride (points to first element to access)
                    offset_terms.append(f"({start_str} * {stride_val})")
                else:
                    # Positive step (original logic)
                    start_str = "0"
                    if idx.lower is not None:
                        start_str = self.visit(idx.lower)
                        if isinstance(start_str, str) and (
                            start_str.startswith("-") or start_str.startswith("(-")
                        ):
                            start_str = f"({shape_val} + {start_str})"

                    stop_str = str(shape_val)
                    if idx.upper is not None:
                        stop_str = self.visit(idx.upper)
                        if isinstance(stop_str, str) and (
                            stop_str.startswith("-") or stop_str.startswith("(-")
                        ):
                            stop_str = f"({shape_val} + {stop_str})"

                    # Compute new shape: ceil((stop - start) / step)
                    if step_str == "1":
                        dim_size = f"({stop_str} - {start_str})"
                    else:
                        dim_size = f"idiv({stop_str} - {start_str} + {step_str} - 1, {step_str})"
                    out_shape.append(dim_size)

                    # Compute new stride: old_stride * step
                    if step_str == "1":
                        out_strides.append(stride_val)
                    else:
                        out_strides.append(f"({stride_val} * {step_str})")

                    # Add offset contribution: start * stride
                    if start_str != "0":
                        offset_terms.append(f"({start_str} * {stride_val})")
            else:
                # Fixed index: dimension is removed, just add offset
                index_str = self.visit(idx)
                if isinstance(index_str, str) and (
                    index_str.startswith("-") or index_str.startswith("(-")
                ):
                    index_str = f"({shape_val} + {index_str})"
                offset_terms.append(f"({index_str} * {stride_val})")

        # Combine offset terms
        if not offset_terms:
            out_offset = "0"
        elif len(offset_terms) == 1:
            out_offset = offset_terms[0]
        else:
            out_offset = " + ".join(offset_terms)

        # Create new pointer container
        tmp_name = self.builder.find_new_name("_slice_view_")
        ptr_type = Pointer(dtype)
        self.builder.add_container(tmp_name, ptr_type, False)
        self.container_table[tmp_name] = ptr_type

        # Create output tensor with new shape, strides, and offset
        # Offset is stored in the Tensor (like Tensor.flip() does)
        # Reference memlet just creates the pointer alias with "0" offset
        if out_shape:
            out_tensor = Tensor(dtype, out_shape, out_strides, out_offset)
            self.tensor_table[tmp_name] = out_tensor
        else:
            # Scalar result (all indices were fixed)
            self.builder.add_container(tmp_name, dtype, False)
            self.container_table[tmp_name] = dtype

        # Create reference memlet (offset is handled by tensor's offset property)
        debug_info = DebugInfo()
        block = self.builder.add_block(debug_info)
        t_src = self.builder.add_access(block, value_str, debug_info)
        t_dst = self.builder.add_access(block, tmp_name, debug_info)
        self.builder.add_reference_memlet(block, t_src, t_dst, "0", ptr_type)

        return tmp_name

    def _handle_expression_slicing_copy(
        self, node, value_str, indices_nodes, shapes, ndim
    ):
        """Copy-based slicing for cases that cannot use views.

        This allocates a new array and copies elements using nested loops.
        Used for negative steps or indirect access patterns.
        """
        dtype = Scalar(PrimitiveType.Double)
        if value_str in self.container_table:
            t = self.container_table[value_str]
            if isinstance(t, Pointer) and t.has_pointee_type():
                dtype = t.pointee_type

        result_shapes = []
        result_shapes_runtime = []
        slice_info = []
        index_info = []

        for i, idx in enumerate(indices_nodes):
            shape_val = shapes[i] if i < len(shapes) else f"_{value_str}_shape_{i}"

            if isinstance(idx, ast.Slice):
                start_str = "0"
                start_str_runtime = "0"
                if idx.lower is not None:
                    if self._contains_indirect_access(idx.lower):
                        start_str, start_str_runtime = (
                            self._materialize_indirect_access(
                                idx.lower, return_original_expr=True
                            )
                        )
                    else:
                        start_str = self.visit(idx.lower)
                        start_str_runtime = start_str
                    if isinstance(start_str, str) and (
                        start_str.startswith("-") or start_str.startswith("(-")
                    ):
                        start_str = f"({shape_val} + {start_str})"
                        start_str_runtime = f"({shape_val} + {start_str_runtime})"

                stop_str = str(shape_val)
                stop_str_runtime = str(shape_val)
                if idx.upper is not None:
                    if self._contains_indirect_access(idx.upper):
                        stop_str, stop_str_runtime = self._materialize_indirect_access(
                            idx.upper, return_original_expr=True
                        )
                    else:
                        stop_str = self.visit(idx.upper)
                        stop_str_runtime = stop_str
                    if isinstance(stop_str, str) and (
                        stop_str.startswith("-") or stop_str.startswith("(-")
                    ):
                        stop_str = f"({shape_val} + {stop_str})"
                        stop_str_runtime = f"({shape_val} + {stop_str_runtime})"

                step_str = "1"
                if idx.step is not None:
                    step_str = self.visit(idx.step)

                # Compute dimension size accounting for step: ceil((stop - start) / step)
                # For symbolic expressions, use integer ceiling formula: idiv(n + d - 1, d)
                if step_str == "1":
                    dim_size = f"({stop_str} - {start_str})"
                    dim_size_runtime = f"({stop_str_runtime} - {start_str_runtime})"
                else:
                    dim_size = (
                        f"idiv({stop_str} - {start_str} + {step_str} - 1, {step_str})"
                    )
                    dim_size_runtime = f"idiv({stop_str_runtime} - {start_str_runtime} + {step_str} - 1, {step_str})"
                result_shapes.append(dim_size)
                result_shapes_runtime.append(dim_size_runtime)
                slice_info.append((i, start_str, stop_str, step_str))
            else:
                if self._contains_indirect_access(idx):
                    index_str = self._materialize_indirect_access(idx)
                else:
                    index_str = self.visit(idx)
                if isinstance(index_str, str) and (
                    index_str.startswith("-") or index_str.startswith("(-")
                ):
                    index_str = f"({shape_val} + {index_str})"
                index_info.append((i, index_str))

        tmp_name = self.builder.find_new_name("_slice_tmp_")
        result_ndim = len(result_shapes)

        if result_ndim == 0:
            self.builder.add_container(tmp_name, dtype, False)
            self.container_table[tmp_name] = dtype
        else:
            size_str = "1"
            for dim in result_shapes:
                size_str = f"({size_str} * {dim})"

            element_size = self.builder.get_sizeof(dtype)
            total_size = f"({size_str} * {element_size})"

            ptr_type = Pointer(dtype)
            self.builder.add_container(tmp_name, ptr_type, False)
            self.container_table[tmp_name] = ptr_type
            tensor_info = Tensor(dtype, result_shapes)
            self.shapes_runtime_info[tmp_name] = (
                result_shapes_runtime  # Store runtime shapes separately
            )
            self.tensor_table[tmp_name] = tensor_info

            debug_info = DebugInfo()
            block_alloc = self.builder.add_block(debug_info)
            t_malloc = self.builder.add_malloc(block_alloc, total_size)
            t_ptr = self.builder.add_access(block_alloc, tmp_name, debug_info)
            self.builder.add_memlet(
                block_alloc, t_malloc, "_ret", t_ptr, "void", "", ptr_type, debug_info
            )

        loop_vars = []
        debug_info = DebugInfo()

        for dim_idx, (orig_dim, start_str, stop_str, step_str) in enumerate(slice_info):
            loop_var = self.builder.find_new_name(f"_slice_loop_{dim_idx}_")
            loop_vars.append((loop_var, orig_dim, start_str, step_str))

            if not self.builder.exists(loop_var):
                self.builder.add_container(loop_var, Scalar(PrimitiveType.Int64), False)
                self.container_table[loop_var] = Scalar(PrimitiveType.Int64)

            # Account for step in loop count: ceil((stop - start) / step)
            if step_str == "1":
                count_str = f"({stop_str} - {start_str})"
            else:
                count_str = (
                    f"idiv({stop_str} - {start_str} + {step_str} - 1, {step_str})"
                )
            self.builder.begin_for(loop_var, "0", count_str, "1", debug_info)

        src_indices = [""] * ndim
        dst_indices = []

        for orig_dim, index_str in index_info:
            src_indices[orig_dim] = index_str

        for loop_var, orig_dim, start_str, step_str in loop_vars:
            if step_str == "1":
                src_indices[orig_dim] = f"({start_str} + {loop_var})"
            else:
                src_indices[orig_dim] = f"({start_str} + {loop_var} * {step_str})"
            dst_indices.append(loop_var)

        src_linear = self._compute_linear_index(src_indices, shapes, value_str, ndim)
        if result_ndim > 0:
            dst_linear = self._compute_linear_index(
                dst_indices, result_shapes, tmp_name, result_ndim
            )
        else:
            dst_linear = "0"

        block = self.builder.add_block(debug_info)
        t_src = self.builder.add_access(block, value_str, debug_info)
        t_dst = self.builder.add_access(block, tmp_name, debug_info)
        t_task = self.builder.add_tasklet(
            block, TaskletCode.assign, ["_in"], ["_out"], debug_info
        )

        self.builder.add_memlet(
            block, t_src, "void", t_task, "_in", src_linear, None, debug_info
        )
        self.builder.add_memlet(
            block, t_task, "_out", t_dst, "void", dst_linear, None, debug_info
        )

        for _ in loop_vars:
            self.builder.end_for()

        return tmp_name

    def _compute_linear_index(self, indices, shapes, array_name, ndim):
        """Compute linear index from multi-dimensional indices."""
        if ndim == 0:
            return "0"

        linear_index = ""
        for i in range(ndim):
            term = str(indices[i])
            for j in range(i + 1, ndim):
                shape_val = shapes[j] if j < len(shapes) else f"_{array_name}_shape_{j}"
                term = f"(({term}) * {shape_val})"

            if i == 0:
                linear_index = term
            else:
                linear_index = f"({linear_index} + {term})"

        return linear_index

    def _is_array_index(self, node):
        """Check if a node represents an array that could be used as an index (gather)."""
        if isinstance(node, ast.Name):
            return node.id in self.tensor_table
        return False

    def _handle_gather(self, value_str, index_node, debug_info=None):
        """Handle gather operation: x[indices] where indices is an array."""
        if debug_info is None:
            debug_info = DebugInfo()

        if isinstance(index_node, ast.Name):
            idx_array_name = index_node.id
        else:
            idx_array_name = self.visit(index_node)

        if idx_array_name not in self.tensor_table:
            raise ValueError(f"Gather index must be an array, got {idx_array_name}")

        idx_shapes = self.tensor_table[idx_array_name].shape
        idx_ndim = len(idx_shapes)

        if idx_ndim != 1:
            raise NotImplementedError("Only 1D index arrays supported for gather")

        result_shape = idx_shapes[0] if idx_shapes else f"_{idx_array_name}_shape_0"

        # For runtime evaluation, prefer shapes_runtime_info if available
        # This ensures we use expressions that can be evaluated at runtime
        if idx_array_name in self.shapes_runtime_info:
            runtime_shapes = self.shapes_runtime_info[idx_array_name]
            result_shape_runtime = runtime_shapes[0] if runtime_shapes else result_shape
        else:
            result_shape_runtime = result_shape

        dtype = Scalar(PrimitiveType.Double)
        if value_str in self.container_table:
            t = self.container_table[value_str]
            if isinstance(t, Pointer) and t.has_pointee_type():
                dtype = t.pointee_type

        idx_dtype = Scalar(PrimitiveType.Int64)
        if idx_array_name in self.container_table:
            t = self.container_table[idx_array_name]
            if isinstance(t, Pointer) and t.has_pointee_type():
                idx_dtype = t.pointee_type

        tmp_name = self.builder.find_new_name("_gather_")

        element_size = self.builder.get_sizeof(dtype)
        total_size = f"({result_shape} * {element_size})"

        ptr_type = Pointer(dtype)
        self.builder.add_container(tmp_name, ptr_type, False)
        self.container_table[tmp_name] = ptr_type
        self.tensor_table[tmp_name] = Tensor(dtype, [result_shape])
        # Store runtime evaluable shape for this gather result
        self.shapes_runtime_info[tmp_name] = [result_shape_runtime]

        block_alloc = self.builder.add_block(debug_info)
        t_malloc = self.builder.add_malloc(block_alloc, total_size)
        t_ptr = self.builder.add_access(block_alloc, tmp_name, debug_info)
        self.builder.add_memlet(
            block_alloc, t_malloc, "_ret", t_ptr, "void", "", ptr_type, debug_info
        )

        loop_var = self.builder.find_new_name("_gather_i_")
        self.builder.add_container(loop_var, Scalar(PrimitiveType.Int64), False)
        self.container_table[loop_var] = Scalar(PrimitiveType.Int64)

        idx_var = self.builder.find_new_name("_gather_idx_")
        self.builder.add_container(idx_var, idx_dtype, False)
        self.container_table[idx_var] = idx_dtype

        self.builder.begin_for(loop_var, "0", str(result_shape), "1", debug_info)

        block_load_idx = self.builder.add_block(debug_info)
        idx_arr_access = self.builder.add_access(
            block_load_idx, idx_array_name, debug_info
        )
        idx_var_access = self.builder.add_access(block_load_idx, idx_var, debug_info)
        tasklet_load = self.builder.add_tasklet(
            block_load_idx, TaskletCode.assign, ["_in"], ["_out"], debug_info
        )
        self.builder.add_memlet(
            block_load_idx,
            idx_arr_access,
            "void",
            tasklet_load,
            "_in",
            loop_var,
            None,
            debug_info,
        )
        self.builder.add_memlet(
            block_load_idx,
            tasklet_load,
            "_out",
            idx_var_access,
            "void",
            "",
            None,
            debug_info,
        )

        block_gather = self.builder.add_block(debug_info)
        src_access = self.builder.add_access(block_gather, value_str, debug_info)
        dst_access = self.builder.add_access(block_gather, tmp_name, debug_info)
        tasklet_gather = self.builder.add_tasklet(
            block_gather, TaskletCode.assign, ["_in"], ["_out"], debug_info
        )

        self.builder.add_memlet(
            block_gather,
            src_access,
            "void",
            tasklet_gather,
            "_in",
            idx_var,
            None,
            debug_info,
        )
        self.builder.add_memlet(
            block_gather,
            tasklet_gather,
            "_out",
            dst_access,
            "void",
            loop_var,
            None,
            debug_info,
        )

        self.builder.end_for()

        return tmp_name

    def _get_max_array_ndim_in_expr(self, node):
        """Get the maximum array dimensionality in an expression."""
        max_ndim = 0

        class NdimVisitor(ast.NodeVisitor):
            def __init__(self, tensor_table):
                self.tensor_table = tensor_table
                self.max_ndim = 0

            def visit_Name(self, node):
                if node.id in self.tensor_table:
                    ndim = len(self.tensor_table[node.id].shape)
                    self.max_ndim = max(self.max_ndim, ndim)
                return self.generic_visit(node)

        visitor = NdimVisitor(self.tensor_table)
        visitor.visit(node)
        return visitor.max_ndim

    def _handle_broadcast_slice_assignment(
        self, target, value, target_name, indices, target_ndim, value_ndim, debug_info
    ):
        """Handle slice assignment with broadcasting (e.g., 2D -= 1D)."""
        # Number of broadcast dimensions (outer loops)
        broadcast_dims = target_ndim - value_ndim

        shapes = self.tensor_table[target_name].shape

        # Create outer loops for broadcast dimensions
        outer_loop_vars = []
        for i in range(broadcast_dims):
            loop_var = self.builder.find_new_name(f"_bcast_iter_{i}_")
            outer_loop_vars.append(loop_var)

            if not self.builder.exists(loop_var):
                self.builder.add_container(loop_var, Scalar(PrimitiveType.Int64), False)
                self.container_table[loop_var] = Scalar(PrimitiveType.Int64)

            dim_size = shapes[i] if i < len(shapes) else f"_{target_name}_shape_{i}"
            self.builder.begin_for(loop_var, "0", dim_size, "1", debug_info)

        # Create a row view (reference) for the inner dimensions
        row_view_name = self.builder.find_new_name("_row_view_")

        # Get inner shape for the row view
        inner_shapes = shapes[broadcast_dims:] if len(shapes) > broadcast_dims else []

        # Determine element type from the target
        target_type = self.container_table.get(target_name)
        if isinstance(target_type, Pointer) and target_type.has_pointee_type():
            element_type = target_type.pointee_type
        else:
            element_type = Scalar(PrimitiveType.Double)

        # Create pointer type for row view
        row_type = Pointer(element_type)
        self.builder.add_container(row_view_name, row_type, False)
        self.container_table[row_view_name] = row_type

        # Register row view in tensor_table
        self.tensor_table[row_view_name] = Tensor(element_type, list(inner_shapes))

        # Create reference memlet: row_view = &target[i, 0, 0, ...]
        # The index is: outer_loop_vars joined, then zeros for inner dims
        ref_index_parts = outer_loop_vars[:]
        for _ in range(value_ndim):
            ref_index_parts.append("0")

        # Compute linearized index for reference
        # For target[i, j] with shape (n, m), linear index for row i is i * m
        linear_idx = outer_loop_vars[0] if outer_loop_vars else "0"
        for dim_idx in range(1, broadcast_dims):
            dim_size = (
                shapes[dim_idx]
                if dim_idx < len(shapes)
                else f"_{target_name}_shape_{dim_idx}"
            )
            linear_idx = f"({linear_idx}) * ({dim_size}) + {outer_loop_vars[dim_idx]}"

        # Multiply by inner dimension sizes to get the start of the row
        for dim_idx in range(broadcast_dims, target_ndim):
            dim_size = (
                shapes[dim_idx]
                if dim_idx < len(shapes)
                else f"_{target_name}_shape_{dim_idx}"
            )
            linear_idx = f"({linear_idx}) * ({dim_size})"

        # Create the reference memlet block
        block = self.builder.add_block(debug_info)
        t_src = self.builder.add_access(block, target_name, debug_info)
        t_dst = self.builder.add_access(block, row_view_name, debug_info)
        self.builder.add_reference_memlet(
            block, t_src, t_dst, linear_idx, row_type, debug_info
        )

        # Now handle the inner slice assignment with the row view
        # Create inner indices (all slices for the inner dimensions)
        inner_indices = [
            ast.Slice(lower=None, upper=None, step=None) for _ in range(value_ndim)
        ]

        # Create new target using row view
        new_target = ast.Subscript(
            value=ast.Name(id=row_view_name, ctx=ast.Load()),
            slice=(
                ast.Tuple(elts=inner_indices, ctx=ast.Load())
                if len(inner_indices) > 1
                else inner_indices[0]
            ),
            ctx=ast.Store(),
        )

        # Recursively handle the inner assignment (now same-dimension)
        self._handle_slice_assignment(
            new_target, value, row_view_name, inner_indices, debug_info
        )

        # Close outer loops
        for _ in outer_loop_vars:
            self.builder.end_for()

    def _handle_slice_assignment(
        self, target, value, target_name, indices, debug_info=None
    ):
        if debug_info is None:
            debug_info = DebugInfo()

        # Add missing dimensions
        tensor_info = self.tensor_table[target_name]
        ndim = len(tensor_info.shape)
        if len(indices) < ndim:
            indices = list(indices)
            for _ in range(ndim - len(indices)):
                indices.append(ast.Slice(lower=None, upper=None, step=None))

        # Handle ufunc outer case separately to preserve slice shape info
        has_outer, ufunc_name, outer_node = contains_ufunc_outer(value)
        if has_outer:
            self._handle_ufunc_outer_slice_assignment(
                target, value, target_name, indices, debug_info
            )
            return

        # Count slice dimensions to determine effective target dimensionality
        target_slice_ndim = sum(1 for idx in indices if isinstance(idx, ast.Slice))
        value_max_ndim = self._get_max_array_ndim_in_expr(value)

        if (
            target_slice_ndim > 0
            and value_max_ndim > 0
            and target_slice_ndim > value_max_ndim
        ):
            # Broadcasting case: use row-by-row approach with reference memlets
            self._handle_broadcast_slice_assignment(
                target,
                value,
                target_name,
                indices,
                target_slice_ndim,
                value_max_ndim,
                debug_info,
            )
            return

        loop_vars = []
        new_target_indices = []

        for i, idx in enumerate(indices):
            if isinstance(idx, ast.Slice):
                loop_var = self.builder.find_new_name(f"_slice_iter_{len(loop_vars)}_")
                loop_vars.append(loop_var)

                if not self.builder.exists(loop_var):
                    self.builder.add_container(
                        loop_var, Scalar(PrimitiveType.Int64), False
                    )
                    self.container_table[loop_var] = Scalar(PrimitiveType.Int64)

                start_str = "0"
                if idx.lower:
                    start_str = self.visit(idx.lower)
                    if start_str.startswith("-"):
                        dim_size = (
                            str(tensor_info.shape[i])
                            if i < len(tensor_info.shape)
                            else f"_{target_name}_shape_{i}"
                        )
                        start_str = f"({dim_size} {start_str})"

                stop_str = ""
                if idx.upper and not (
                    isinstance(idx.upper, ast.Constant) and idx.upper.value is None
                ):
                    stop_str = self.visit(idx.upper)
                    if stop_str.startswith("-") or stop_str.startswith("(-"):
                        dim_size = (
                            str(tensor_info.shape[i])
                            if i < len(tensor_info.shape)
                            else f"_{target_name}_shape_{i}"
                        )
                        stop_str = f"({dim_size} {stop_str})"
                else:
                    stop_str = (
                        str(tensor_info.shape[i])
                        if i < len(tensor_info.shape)
                        else f"_{target_name}_shape_{i}"
                    )

                step_str = "1"
                if idx.step:
                    step_str = self.visit(idx.step)

                count_str = f"({stop_str} - {start_str})"

                self.builder.begin_for(loop_var, "0", count_str, "1", debug_info)
                self.container_table[loop_var] = Scalar(PrimitiveType.Int64)
                new_target_indices.append(
                    ast.Name(
                        id=f"{start_str} + {loop_var} * {step_str}", ctx=ast.Load()
                    )
                )
            else:
                dim_size = (
                    tensor_info.shape[i]
                    if i < len(tensor_info.shape)
                    else f"_{target_name}_shape_{i}"
                )
                normalized_idx = normalize_negative_index(idx, dim_size)
                new_target_indices.append(normalized_idx)

        rewriter = SliceRewriter(loop_vars, self.tensor_table, self)
        new_value = rewriter.visit(copy.deepcopy(value))

        new_target = copy.deepcopy(target)
        if len(new_target_indices) == 1:
            new_target.slice = new_target_indices[0]
        else:
            new_target.slice = ast.Tuple(elts=new_target_indices, ctx=ast.Load())

        rhs_temp = self.visit(new_value)
        block = self.builder.add_block(debug_info)
        t_task = self.builder.add_tasklet(
            block, TaskletCode.assign, ["_in"], ["_out"], debug_info
        )

        t_src, src_sub = self._add_read(block, rhs_temp, debug_info)
        self.builder.add_memlet(
            block, t_src, "void", t_task, "_in", src_sub, None, debug_info
        )

        lhs_expr = self.visit(new_target)
        if "(" in lhs_expr and lhs_expr.endswith(")"):
            subset = lhs_expr[lhs_expr.find("(") + 1 : -1]
            tensor_dst = self.tensor_table[target_name]

            t_dst = self.builder.add_access(block, target_name, debug_info)
            self.builder.add_memlet(
                block, t_task, "_out", t_dst, "void", subset, tensor_dst, debug_info
            )
        else:
            t_dst = self.builder.add_access(block, target_name, debug_info)
            self.builder.add_memlet(
                block, t_task, "_out", t_dst, "void", "", None, debug_info
            )

        for _ in loop_vars:
            self.builder.end_for()

    def _handle_ufunc_outer_slice_assignment(
        self, target, value, target_name, indices, debug_info=None
    ):
        """Handle slice assignment where RHS contains a ufunc outer operation.

        Example: path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))

        The strategy is:
        1. Evaluate the entire RHS expression, which will create a temporary array
           containing the result of the ufunc outer (potentially wrapped in other ops)
        2. Copy the temporary result to the target slice

        This avoids the loop transformation that would destroy slice shape info.
        """
        if debug_info is None:
            from docc.sdfg import DebugInfo

            debug_info = DebugInfo()

        # Evaluate the full RHS expression
        # This will:
        # - Create temp arrays for ufunc outer results
        # - Apply any wrapping operations (np.minimum, etc.)
        # - Return the name of the final result array
        result_name = self.visit(value)

        # Now we need to copy result to target slice
        # Count slice dimensions to determine if we need loops
        target_slice_ndim = sum(1 for idx in indices if isinstance(idx, ast.Slice))

        if target_slice_ndim == 0:
            # No slices on target - just simple assignment
            target_str = self.visit(target)
            block = self.builder.add_block(debug_info)
            t_src, src_sub = self._add_read(block, result_name, debug_info)
            t_dst = self.builder.add_access(block, target_str, debug_info)
            t_task = self.builder.add_tasklet(
                block, TaskletCode.assign, ["_in"], ["_out"], debug_info
            )
            self.builder.add_memlet(
                block, t_src, "void", t_task, "_in", src_sub, None, debug_info
            )
            self.builder.add_memlet(
                block, t_task, "_out", t_dst, "void", "", None, debug_info
            )
            return

        # We have slices on the target - need to create loops for copying
        # Get target array info
        target_shapes = self.tensor_table[target_name].shape

        loop_vars = []
        new_target_indices = []

        for i, idx in enumerate(indices):
            if isinstance(idx, ast.Slice):
                loop_var = self.builder.find_new_name(f"_copy_iter_{len(loop_vars)}_")
                loop_vars.append(loop_var)

                if not self.builder.exists(loop_var):
                    self.builder.add_container(
                        loop_var, Scalar(PrimitiveType.Int64), False
                    )
                    self.container_table[loop_var] = Scalar(PrimitiveType.Int64)

                start_str = "0"
                if idx.lower:
                    start_str = self.visit(idx.lower)

                stop_str = ""
                if idx.upper and not (
                    isinstance(idx.upper, ast.Constant) and idx.upper.value is None
                ):
                    stop_str = self.visit(idx.upper)
                else:
                    stop_str = (
                        target_shapes[i]
                        if i < len(target_shapes)
                        else f"_{target_name}_shape_{i}"
                    )

                step_str = "1"
                if idx.step:
                    step_str = self.visit(idx.step)

                count_str = f"({stop_str} - {start_str})"

                self.builder.begin_for(loop_var, "0", count_str, "1", debug_info)
                self.container_table[loop_var] = Scalar(PrimitiveType.Int64)

                new_target_indices.append(
                    ast.Name(
                        id=f"{start_str} + {loop_var} * {step_str}", ctx=ast.Load()
                    )
                )
            else:
                # Handle non-slice indices - need to normalize negative indices
                dim_size = (
                    target_shapes[i]
                    if i < len(target_shapes)
                    else f"_{target_name}_shape_{i}"
                )
                normalized_idx = normalize_negative_index(idx, dim_size)
                new_target_indices.append(normalized_idx)

        # Create assignment block: target[i,j,...] = result[i,j,...]
        block = self.builder.add_block(debug_info)

        # Access nodes
        t_src = self.builder.add_access(block, result_name, debug_info)
        t_dst = self.builder.add_access(block, target_name, debug_info)
        t_task = self.builder.add_tasklet(
            block, TaskletCode.assign, ["_in"], ["_out"], debug_info
        )

        # Source index - just use loop vars for flat array from ufunc outer
        # The ufunc outer result is a flat array of size M*N
        if len(loop_vars) == 2:
            # 2D case: result is indexed as i * N + j
            # Get the second dimension size from target shapes
            n_dim = (
                target_shapes[1]
                if len(target_shapes) > 1
                else f"_{target_name}_shape_1"
            )
            src_index = f"(({loop_vars[0]}) * ({n_dim}) + ({loop_vars[1]}))"
        elif len(loop_vars) == 1:
            src_index = loop_vars[0]
        else:
            # General case - compute linear index
            src_terms = []
            stride = "1"
            for i in range(len(loop_vars) - 1, -1, -1):
                if stride == "1":
                    src_terms.insert(0, loop_vars[i])
                else:
                    src_terms.insert(0, f"({loop_vars[i]} * {stride})")
                if i > 0:
                    dim_size = (
                        target_shapes[i]
                        if i < len(target_shapes)
                        else f"_{target_name}_shape_{i}"
                    )
                    stride = (
                        f"({stride} * {dim_size})" if stride != "1" else str(dim_size)
                    )
            src_index = " + ".join(src_terms) if src_terms else "0"

        # Target index - compute linear index (row-major order)
        # For 2D array with shape (M, N): linear_index = i * N + j
        target_index_parts = []
        for idx in new_target_indices:
            if isinstance(idx, ast.Name):
                target_index_parts.append(idx.id)
            else:
                target_index_parts.append(self.visit(idx))

        # Convert to linear index
        if len(target_index_parts) == 2:
            # 2D case
            n_dim = (
                target_shapes[1]
                if len(target_shapes) > 1
                else f"_{target_name}_shape_1"
            )
            target_index = (
                f"(({target_index_parts[0]}) * ({n_dim}) + ({target_index_parts[1]}))"
            )
        elif len(target_index_parts) == 1:
            target_index = target_index_parts[0]
        else:
            # General case - compute linear index with strides
            stride = "1"
            target_index = "0"
            for i in range(len(target_index_parts) - 1, -1, -1):
                idx_part = target_index_parts[i]
                if stride == "1":
                    term = idx_part
                else:
                    term = f"(({idx_part}) * ({stride}))"

                if target_index == "0":
                    target_index = term
                else:
                    target_index = f"({term} + {target_index})"

                if i > 0:
                    dim_size = (
                        target_shapes[i]
                        if i < len(target_shapes)
                        else f"_{target_name}_shape_{i}"
                    )
                    stride = (
                        f"({stride} * {dim_size})" if stride != "1" else str(dim_size)
                    )

        # Connect memlets
        self.builder.add_memlet(
            block, t_src, "void", t_task, "_in", src_index, None, debug_info
        )
        self.builder.add_memlet(
            block, t_task, "_out", t_dst, "void", target_index, None, debug_info
        )

        # End loops
        for _ in loop_vars:
            self.builder.end_for()

    def _contains_indirect_access(self, node):
        """Check if an AST node contains any indirect array access."""
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name):
                arr_name = node.value.id
                if arr_name in self.tensor_table:
                    return True
        elif isinstance(node, ast.BinOp):
            return self._contains_indirect_access(
                node.left
            ) or self._contains_indirect_access(node.right)
        elif isinstance(node, ast.UnaryOp):
            return self._contains_indirect_access(node.operand)
        return False

    def _materialize_indirect_access(
        self, node, debug_info=None, return_original_expr=False
    ):
        """Materialize an array access into a scalar variable using tasklet+memlets."""
        if not self.builder:
            expr = self.visit(node)
            return (expr, expr) if return_original_expr else expr

        if debug_info is None:
            debug_info = DebugInfo()

        if not isinstance(node, ast.Subscript):
            expr = self.visit(node)
            return (expr, expr) if return_original_expr else expr

        if not isinstance(node.value, ast.Name):
            expr = self.visit(node)
            return (expr, expr) if return_original_expr else expr

        arr_name = node.value.id
        if arr_name not in self.tensor_table:
            expr = self.visit(node)
            return (expr, expr) if return_original_expr else expr

        dtype = Scalar(PrimitiveType.Int64)
        if arr_name in self.container_table:
            t = self.container_table[arr_name]
            if isinstance(t, Pointer) and t.has_pointee_type():
                dtype = t.pointee_type

        tmp_name = self.builder.find_new_name("_idx_")
        self.builder.add_container(tmp_name, dtype, False)
        self.container_table[tmp_name] = dtype

        ndim = len(self.tensor_table[arr_name].shape)
        shapes = self.tensor_table[arr_name].shape

        if isinstance(node.slice, ast.Tuple):
            indices = [self.visit(elt) for elt in node.slice.elts]
        else:
            indices = [self.visit(node.slice)]

        materialized_indices = []
        for idx_str in indices:
            if "(" in idx_str and idx_str.endswith(")"):
                materialized_indices.append(idx_str)
            else:
                materialized_indices.append(idx_str)

        linear_index = self._compute_linear_index(
            materialized_indices, shapes, arr_name, ndim
        )

        block = self.builder.add_block(debug_info)
        t_src = self.builder.add_access(block, arr_name, debug_info)
        t_dst = self.builder.add_access(block, tmp_name, debug_info)
        t_task = self.builder.add_tasklet(
            block, TaskletCode.assign, ["_in"], ["_out"], debug_info
        )

        self.builder.add_memlet(
            block, t_src, "void", t_task, "_in", linear_index, None, debug_info
        )
        self.builder.add_memlet(
            block, t_task, "_out", t_dst, "void", "", None, debug_info
        )

        if return_original_expr:
            original_expr = f"{arr_name}({linear_index})"
            return (tmp_name, original_expr)

        return tmp_name

    def _get_unique_id(self):
        self._unique_counter_ref[0] += 1
        return self._unique_counter_ref[0]

    def _element_type(self, name):
        if name in self.container_table:
            return element_type_from_sdfg_type(self.container_table[name])
        else:  # Constant
            if self._is_int(name):
                return Scalar(PrimitiveType.Int64)
            else:
                return Scalar(PrimitiveType.Double)

    def _is_int(self, operand):
        try:
            if operand.lstrip("-").isdigit():
                return True
        except ValueError:
            pass

        name = operand
        if "(" in operand and operand.endswith(")"):
            name = operand.split("(")[0]

        if name in self.container_table:
            t = self.container_table[name]

            def is_int_ptype(pt):
                return pt in [
                    PrimitiveType.Int64,
                    PrimitiveType.Int32,
                    PrimitiveType.Int8,
                    PrimitiveType.Int16,
                    PrimitiveType.UInt64,
                    PrimitiveType.UInt32,
                    PrimitiveType.UInt8,
                    PrimitiveType.UInt16,
                ]

            if isinstance(t, Scalar):
                return is_int_ptype(t.primitive_type)

            if type(t).__name__ == "Array" and hasattr(t, "element_type"):
                et = t.element_type
                if callable(et):
                    et = et()
                if isinstance(et, Scalar):
                    return is_int_ptype(et.primitive_type)

            if type(t).__name__ == "Pointer":
                if hasattr(t, "pointee_type"):
                    et = t.pointee_type
                    if callable(et):
                        et = et()
                    if isinstance(et, Scalar):
                        return is_int_ptype(et.primitive_type)
                if hasattr(t, "element_type"):
                    et = t.element_type
                    if callable(et):
                        et = et()
                    if isinstance(et, Scalar):
                        return is_int_ptype(et.primitive_type)

        return False

    def _add_read(self, block, expr_str, debug_info=None):
        try:
            if (block, expr_str) in self._access_cache:
                return self._access_cache[(block, expr_str)]
        except TypeError:
            pass

        if debug_info is None:
            debug_info = DebugInfo()

        if "(" in expr_str and expr_str.endswith(")"):
            name = expr_str.split("(")[0]
            subset = expr_str[expr_str.find("(") + 1 : -1]
            access = self.builder.add_access(block, name, debug_info)
            try:
                self._access_cache[(block, expr_str)] = (access, subset)
            except TypeError:
                pass
            return access, subset

        if self.builder.exists(expr_str):
            access = self.builder.add_access(block, expr_str, debug_info)
            subset = ""
            if expr_str in self.container_table:
                sym_type = self.container_table[expr_str]
                if isinstance(sym_type, Pointer):
                    if expr_str in self.tensor_table:
                        ndim = len(self.tensor_table[expr_str].shape)
                        if ndim == 0:
                            subset = "0"
                    else:
                        subset = "0"
            try:
                self._access_cache[(block, expr_str)] = (access, subset)
            except TypeError:
                pass
            return access, subset

        dtype = Scalar(PrimitiveType.Double)
        if self._is_int(expr_str):
            dtype = Scalar(PrimitiveType.Int64)
        elif expr_str == "true" or expr_str == "false":
            dtype = Scalar(PrimitiveType.Bool)

        const_node = self.builder.add_constant(block, expr_str, dtype, debug_info)
        try:
            self._access_cache[(block, expr_str)] = (const_node, "")
        except TypeError:
            pass
        return const_node, ""
