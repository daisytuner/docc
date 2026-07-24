import ast
import copy
import inspect
import textwrap
import numpy as np
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
from docc.python.memory import ManagedMemoryHandler


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
        memory_handler=None,
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
        # Map variable name -> tuple/list AST for tuple-valued variables
        # (e.g. `shp = (a.shape[0], a.shape[1])`), so they can later be resolved
        # as array-creation shape arguments.
        self.tuple_table = {}

        # Memory manager for hoisted allocations (shared with inline parsers)
        self.memory_handler = (
            memory_handler
            if memory_handler is not None
            else ManagedMemoryHandler(builder)
        )

        # Initialize handlers - they receive 'self' to access expression visitor methods
        self.numpy_visitor = NumPyHandler(self)
        self.math_handler = MathHandler(self)
        self.python_handler = PythonHandler(self)

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
                # Module-global constant array (e.g. LULESH's `gamma`). Bake it
                # into the SDFG as an initialized array container the first time
                # it is referenced.
                if isinstance(val, np.ndarray) and self.builder is not None:
                    return self._materialize_global_array(name, val)
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
            elif op in ("&", "|", "^", "<<", ">>"):
                return self.numpy_visitor.handle_array_bitwise_op(op, left, right)
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

            t_task = self.builder.add_cmath(
                block, CMathFunction.pow, dtype.primitive_type
            )
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

        # For indexed array accesses like "arr(i,j)", we need to pass the Tensor type
        # to ensure correct type inference during validation
        left_type = self._get_memlet_type_for_access(real_left, left_sub)
        right_type = self._get_memlet_type_for_access(real_right, right_sub)

        self.builder.add_memlet(
            block, t_left, "void", t_task, "_in1", left_sub, left_type
        )
        self.builder.add_memlet(
            block, t_right, "void", t_task, "_in2", right_sub, right_type
        )
        self.builder.add_memlet(block, t_task, "_out", t_out, "void", "")

        return tmp_name

    def visit_Attribute(self, node):
        if node.attr == "shape":
            if isinstance(node.value, ast.Name) and node.value.id in self.tensor_table:
                return f"_shape_proxy_{node.value.id}"
            # Array-member attribute: e.g. domain.x.shape
            if isinstance(node.value, ast.Attribute):
                resolved = self.visit(node.value)
                if isinstance(resolved, str) and resolved in self.tensor_table:
                    return f"_shape_proxy_{resolved}"

        if node.attr == "size":
            # arr.size / domain.member.size -> total element count (product of
            # the shape dimensions) as a scalar expression string, analogous to
            # how `.shape[i]` returns a raw dimension expression. Usable in
            # symbolic contexts (comparisons, ranges, arithmetic).
            target = None
            if isinstance(node.value, ast.Name) and node.value.id in self.tensor_table:
                target = node.value.id
            elif isinstance(node.value, ast.Attribute):
                resolved = self.visit(node.value)
                if isinstance(resolved, str) and resolved in self.tensor_table:
                    target = resolved
            if target is not None:
                shape = self.tensor_table[target].shape
                if not shape:
                    return "1"
                size_expr = f"({shape[0]})"
                for d in shape[1:]:
                    size_expr = f"({size_expr} * ({d}))"
                return size_expr

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

        # NumPy floating-point constants: np.inf, np.nan, np.pi, np.e
        if isinstance(node.value, ast.Name) and node.value.id in ("numpy", "np"):
            const_map = {
                "inf": "INFINITY",
                "infty": "INFINITY",
                "Inf": "INFINITY",
                "nan": "NAN",
                "NAN": "NAN",
                "pi": "M_PI",
                "e": "M_E",
            }
            val = const_map.get(node.attr, "")
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
                            member_index, member_type, member_shape = (
                                self.structure_member_info[struct_name][attr_name]
                            )
                        else:
                            raise RuntimeError(
                                f"Member '{attr_name}' not found in structure '{struct_name}'. "
                                f"Available members: {list(self.structure_member_info.get(struct_name, {}).keys())}"
                            )

                        subset = "0," + str(member_index)

                        if isinstance(member_type, Pointer):
                            # Struct-of-arrays pointer member: expose the
                            # member's *value* (the element pointer) as a tensor
                            # view. This is exactly a GEP + load, encoded with two
                            # canonical memlets:
                            #   1. a reference memlet computes the address of the
                            #      member field  ->  field_ptr = &obj->member  (T**)
                            #   2. a dereference memlet loads the stored pointer
                            #      ->  tmp = *field_ptr  (T*)
                            # Keeping these separate preserves the memlet
                            # contracts (reference = address-of, dereference =
                            # load) instead of overloading the reference memlet in
                            # codegen.
                            field_ptr_type = Pointer(member_type)  # T**
                            field_name = self.builder.find_new_name()
                            self.builder.add_container(
                                field_name, field_ptr_type, False
                            )
                            self.container_table[field_name] = field_ptr_type

                            tmp_name = self.builder.find_new_name()  # T*
                            self.builder.add_container(tmp_name, member_type, False)
                            self.container_table[tmp_name] = member_type

                            elem_type = member_type.pointee_type
                            shape = list(member_shape) if member_shape else []
                            self.tensor_table[tmp_name] = Tensor(elem_type, shape)

                            block = self.builder.add_block()
                            obj_access = self.builder.add_access(block, obj_name)
                            field_access = self.builder.add_access(block, field_name)
                            tmp_access = self.builder.add_access(block, tmp_name)
                            # field_ptr = &obj->member  (address of the field)
                            self.builder.add_reference_memlet(
                                block, obj_access, field_access, subset, None
                            )
                            # tmp = *field_ptr  (load the element pointer)
                            self.builder.add_dereference_memlet(
                                block, field_access, tmp_access, True, field_ptr_type
                            )
                            return tmp_name

                        tmp_name = self.builder.find_new_name()

                        self.builder.add_container(tmp_name, member_type, False)
                        self.container_table[tmp_name] = member_type

                        block = self.builder.add_block()
                        obj_access = self.builder.add_access(block, obj_name)
                        tmp_access = self.builder.add_access(block, tmp_name)

                        tasklet = self.builder.add_tasklet(
                            block, TaskletCode.assign, ["_in"], ["_out"]
                        )

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
        self.builder.add_assignments(tmp_name, "true")
        self.builder.begin_else()
        self.builder.add_assignments(tmp_name, "false")
        self.builder.end_if()

        self.container_table[tmp_name] = dtype
        return tmp_name

    def visit_Subscript(self, node):
        # Compile-time constant folding for subscripts on global constant
        # collections, e.g. `XI["M"]["mask"]` where XI is a global dict of bit
        # masks. Folds to the numeric literal so it can be used in expressions.
        ok, const_val = self._try_const_eval(node)
        if (
            ok
            and isinstance(const_val, (int, float))
            and not isinstance(const_val, bool)
        ):
            return str(const_val)

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

            # np.newaxis / None indexing: arr[:, None], arr[None, :], ...
            if any(self._is_newaxis(idx) for idx in indices_nodes):
                return self._handle_newaxis_subscript(value_str, indices_nodes)

            # Partial indexing: fewer indices than dimensions implies trailing
            # full slices, e.g. A[i] on a 2-D array == A[i, :] (row view).
            if len(indices_nodes) < ndim:
                indices_nodes = list(indices_nodes) + [
                    ast.Slice(lower=None, upper=None, step=None)
                    for _ in range(ndim - len(indices_nodes))
                ]

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

            # Fancy indexing with a constant integer sequence on one axis
            # (e.g. x[:, (1, 2, 3, 4, 5, 7)]). Handle before the slice path,
            # which would otherwise treat the sequence as a dimension-dropping
            # scalar index.
            if any(self._const_int_sequence(idx) is not None for idx in indices_nodes):
                return self._handle_fancy_index(
                    node, value_str, indices_nodes, shapes, ndim
                )

            has_slices = any(isinstance(idx, ast.Slice) for idx in indices_nodes)
            if has_slices:
                return self._handle_expression_slicing(
                    node, value_str, indices_nodes, shapes, ndim
                )

            if len(indices_nodes) == 1 and self._is_array_valued_index(
                indices_nodes[0]
            ):
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
        # Scatter-add: arr[idx] += vals where idx is an integer index array.
        if isinstance(node.target, ast.Subscript) and isinstance(node.op, ast.Add):
            tgt = node.target
            idx_nodes = (
                tgt.slice.elts if isinstance(tgt.slice, ast.Tuple) else [tgt.slice]
            )
            if len(idx_nodes) == 1 and self._is_array_index(idx_nodes[0]):
                target_name = self.visit(tgt.value)
                if target_name in self.tensor_table:
                    debug_info = get_debug_info(node, self.filename, self.function_name)
                    self._handle_scatter_assignment(
                        target_name,
                        idx_nodes[0],
                        node.value,
                        accumulate=True,
                        debug_info=debug_info,
                    )
                    return

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
            return

        # Array-member attribute target: e.g. `domain.x += domain.xd * dt`.
        # Resolve the member to its view container, then rewrite as a whole-array
        # slice assignment `member[:] = member op value` (writes propagate back
        # through the struct member pointer).
        if isinstance(node.target, ast.Attribute):
            resolved = self.visit(node.target)
            if isinstance(resolved, str) and resolved in self.tensor_table:
                ndim = len(self.tensor_table[resolved].shape)
                slices = [
                    ast.Slice(lower=None, upper=None, step=None) for _ in range(ndim)
                ]
                slice_arg = (
                    slices[0] if ndim == 1 else ast.Tuple(elts=slices, ctx=ast.Load())
                )
                slice_node = ast.Subscript(
                    value=ast.Name(id=resolved, ctx=ast.Load()),
                    slice=slice_arg,
                    ctx=ast.Store(),
                )
                new_node = ast.Assign(
                    targets=[slice_node],
                    value=ast.BinOp(
                        left=ast.Name(id=resolved, ctx=ast.Load()),
                        op=node.op,
                        right=node.value,
                    ),
                )
                ast.copy_location(new_node, node)
                self.visit_Assign(new_node)
                return

        new_node = ast.Assign(
            targets=[node.target],
            value=ast.BinOp(left=node.target, op=node.op, right=node.value),
        )
        self.visit_Assign(new_node)

    def visit_Assign(self, node):
        # Handle multiple targets: a = b = c or a, b = expr
        if len(node.targets) > 1:
            rhs_result = self.visit(node.value)
            if isinstance(rhs_result, str) and rhs_result in self.container_table:
                val_node = ast.Name(id=rhs_result, ctx=ast.Load())
                ast.copy_location(val_node, node)
            else:
                # Literals / expressions without a container: re-emit directly.
                val_node = node.value

            # Assign the evaluated value to each target
            for target in node.targets:
                assign = ast.Assign(targets=[target], value=val_node)
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

            # RHS is not a literal tuple (e.g. a call returning multiple values,
            # such as `x1, y1, z1 = collect_nodes(domain, ...)`). Evaluate it;
            # inline calls with a tuple return yield a list of result names.
            result = self.visit(node.value)
            if isinstance(result, (list, tuple)):
                if len(target.elts) != len(result):
                    raise ValueError("Tuple unpacking size mismatch")
                for tgt, name in zip(target.elts, result):
                    assign = ast.Assign(
                        targets=[tgt], value=ast.Name(id=name, ctx=ast.Load())
                    )
                    ast.copy_location(assign, node)
                    self.visit_Assign(assign)
                return
            raise NotImplementedError(
                "Tuple unpacking from non-tuple values not supported"
            )

        # Tuple/list-valued variable: `shp = (a.shape[0], a.shape[1])`. Record
        # it so it can later be resolved as an array-creation shape argument.
        # Only stored (not lowered); other uses of the variable are unaffected.
        if isinstance(target, ast.Name) and isinstance(
            node.value, (ast.Tuple, ast.List)
        ):
            self.tuple_table[target.id] = node.value
            return

        # Special cases, where rhs is not just a simple expression but requires special handling
        if self.numpy_visitor.is_gemm(node.value):
            if self.numpy_visitor.handle_gemm(target, node.value):
                return
            if self.numpy_visitor.handle_dot(target, node.value):
                return
        if self.numpy_visitor.is_outer(node.value):
            if self.numpy_visitor.handle_outer(target, node.value):
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

            # Partial indexing on the LHS: `b[i] = <sub-array>` on an N-D array
            # (N > number of indices) means `b[i, :] = ...`. Pad implicit
            # trailing full slices so it routes to slice-assignment. This only
            # applies to genuine scalar partial indices -- never a boolean mask
            # (`a[a < 0] = ...`) or a fancy/scatter index, which address the
            # array as a whole and must reach their dedicated handlers below.
            if target_name in self.tensor_table:
                ndim = len(self.tensor_table[target_name].shape)
                is_special_index = len(indices) == 1 and (
                    self._is_boolean_mask_index(indices[0])
                    or self._is_array_valued_index(indices[0])
                )
                if not is_special_index and 0 < len(indices) < ndim:
                    indices = list(indices) + [
                        ast.Slice(lower=None, upper=None, step=None)
                        for _ in range(ndim - len(indices))
                    ]

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

            # Handle boolean-mask assignment: target[mask] = value
            if (
                len(indices) == 1
                and target_name in self.tensor_table
                and self._is_boolean_mask_index(indices[0])
            ):
                self._handle_masked_assignment(
                    target, indices[0], node.value, target_name, node
                )
                return

            # Scatter assignment: target[idx] = value where idx is an index array.
            if (
                len(indices) == 1
                and target_name in self.tensor_table
                and self._is_array_index(indices[0])
            ):
                self._handle_scatter_assignment(
                    target_name,
                    indices[0],
                    node.value,
                    accumulate=False,
                    debug_info=debug_info,
                )
                return

            # Handle rhs and store in scalar tmp
            rhs_tmp = self.visit(node.value)

            # Evaluate the LHS (index) expression before creating the store
            # block/tasklet.
            lhs_expr = self.visit(target)

            block = self.builder.add_block(debug_info)
            t_task = self.builder.add_tasklet(
                block, TaskletCode.assign, ["_in"], ["_out"], debug_info
            )

            t_src, src_sub = self._add_read(block, rhs_tmp, debug_info)
            self.builder.add_memlet(
                block, t_src, "void", t_task, "_in", src_sub, None, debug_info
            )

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

        # Assignment to an array-member attribute: `domain.dxx = value`.
        # Resolve the member view and perform a whole-array slice assignment so
        # the write propagates back through the struct member pointer.
        if isinstance(target, ast.Attribute):
            # Scalar struct member: `domain.dtcourant = <scalar>`. Store the
            # value back into obj[0, member_index] so it round-trips through the
            # marshalled struct (checked before the array path, which would
            # otherwise emit a spurious member load).
            scalar_member = self._struct_scalar_member(target)
            if scalar_member is not None:
                obj_name, member_index, member_type = scalar_member
                self._store_scalar_member(
                    obj_name, member_index, member_type, node.value, node
                )
                return

            resolved = self.visit(target)
            if isinstance(resolved, str) and resolved in self.tensor_table:
                ndim = len(self.tensor_table[resolved].shape)
                slices = [
                    ast.Slice(lower=None, upper=None, step=None) for _ in range(ndim)
                ]
                slice_arg = (
                    slices[0] if ndim == 1 else ast.Tuple(elts=slices, ctx=ast.Load())
                )
                slice_node = ast.Subscript(
                    value=ast.Name(id=resolved, ctx=ast.Load()),
                    slice=slice_arg,
                    ctx=ast.Store(),
                )
                new_assign = ast.Assign(targets=[slice_node], value=node.value)
                ast.copy_location(new_assign, node)
                self.visit_Assign(new_assign)
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

    def visit_Raise(self, node):
        # No-op: `raise` statements are ignored. Exceptions are not modelled in
        # the SDFG IR, so error-signalling paths (e.g. lulesh's
        # `raise VolumeError(...)`) are simply dropped rather than lowered.
        # We deliberately do NOT visit the exception expression so that the
        # exception constructor/attribute (which has no SDFG lowering) is not
        # evaluated.
        pass

    def visit_Expr(self, node):
        self.visit(node.value)

    def visit_IfExp(self, node):
        # Conditional expression: `body if test else orelse`.
        # Lower it to an if/else that assigns both branches to a temporary,
        # reusing the statement-level if and assignment handling.
        tmp_name = self.builder.find_new_name()
        assign_body = ast.Assign(
            targets=[ast.Name(id=tmp_name, ctx=ast.Store())], value=node.body
        )
        assign_else = ast.Assign(
            targets=[ast.Name(id=tmp_name, ctx=ast.Store())], value=node.orelse
        )
        if_node = ast.If(test=node.test, body=[assign_body], orelse=[assign_else])
        ast.copy_location(if_node, node)
        ast.fix_missing_locations(if_node)
        self.visit(if_node)
        return tmp_name

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
            # Emit frees for all deferred allocations before returning
            if self.memory_handler.has_allocations():
                self.memory_handler.emit_frees()
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

        # Emit frees for all deferred allocations before returning
        if self.memory_handler.has_allocations():
            self.memory_handler.emit_frees()

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

        if self.python_handler.has_handler(func_name):
            return self.python_handler.handle_python_call(node, func_name)

        if func_name in self.globals_dict:
            obj = self.globals_dict[func_name]
            if inspect.isfunction(obj):
                return self._handle_inline_call(node, obj)

        raise NotImplementedError(f"Function call {func_name} not supported")

    @staticmethod
    def _stmts_always_return(body):
        """True if executing `body` always hits a `return` (so any statements
        that follow it are only reachable when it does not)."""
        if not body:
            return False
        last = body[-1]
        if isinstance(last, ast.Return):
            return True
        if isinstance(last, ast.If):
            return ASTParser._stmts_always_return(
                last.body
            ) and ASTParser._stmts_always_return(last.orelse)
        return False

    def _lift_early_returns(self, body):
        """Rewrite early `return`s into if/else so the inliner's return->assign
        conversion preserves control flow.

        The inliner turns each `return X` into `res = X`; without this, a guard
        like ``if c: return a`` followed by ``return b`` would emit
        ``if c: res = a`` then unconditionally ``res = b`` (fallback always
        wins). Here, statements following an `if` whose body always returns are
        moved into that `if`'s `else`, and dead code after a `return` is dropped.
        """
        result = []
        for i, stmt in enumerate(body):
            if isinstance(stmt, ast.If):
                new_if = copy.copy(stmt)
                new_if.body = self._lift_early_returns(stmt.body)
                new_if.orelse = self._lift_early_returns(stmt.orelse)
                rest = body[i + 1 :]
                if rest and self._stmts_always_return(new_if.body):
                    # The remaining statements are only reachable when the guard
                    # is false: fold them into the else branch.
                    new_if.orelse = list(new_if.orelse) + self._lift_early_returns(rest)
                    result.append(new_if)
                    return result
                result.append(new_if)
            else:
                result.append(stmt)
                if isinstance(stmt, ast.Return):
                    # Statements after a return are unreachable.
                    return result
        return result

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

        if len(node.args) != len(func_def.args.args):
            raise NotImplementedError(
                f"Argument count mismatch for {func_obj.__name__}"
            )

        # Compile-time tuple/list literal arguments cannot be resolved to a data
        # container (they have no runtime storage). Substitute them directly
        # into the inlined body instead - e.g. passing `nodes=(0, 1, 2, 3)` so
        # that an internal `a, b, c, d = nodes` unpacks the literal tuple.
        literal_arg_types = (ast.Tuple, ast.List)
        substitutions = {}
        arg_vars = []
        for arg_def, arg in zip(func_def.args.args, node.args):
            if isinstance(arg, literal_arg_types):
                substitutions[arg_def.arg] = arg
                arg_vars.append(None)
            elif (
                isinstance(arg, ast.Name)
                and arg.id in self.globals_dict
                and isinstance(self.globals_dict[arg.id], (dict, list, tuple, set))
            ):
                # A global compile-time-constant collection (e.g. a dict of bit
                # masks). It has no runtime container; substitute the reference
                # into the body so constant subscripts fold at compile time.
                substitutions[arg_def.arg] = arg
                arg_vars.append(None)
            else:
                arg_vars.append(self.visit(arg))

        suffix = f"_{func_obj.__name__}_{self._get_unique_id()}"
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

            def __init__(self, suffix, globals_dict, substitutions):
                self.suffix = suffix
                self.globals_dict = globals_dict
                self.substitutions = substitutions

            def visit_Name(self, node):
                # Inline compile-time literal arguments (e.g. tuple constants).
                if isinstance(node.ctx, ast.Load) and node.id in self.substitutions:
                    return copy.deepcopy(self.substitutions[node.id])
                if node.id in self.globals_dict or node.id in self.BUILTINS:
                    return node
                return ast.Name(id=f"{node.id}{self.suffix}", ctx=node.ctx)

            def visit_Return(self, node):
                if node.value:
                    val = self.visit(node.value)
                    if isinstance(val, ast.Tuple):
                        # Multi-value return: assign each element to its own
                        # result container (res_name_0, res_name_1, ...).
                        self.return_count = len(val.elts)
                        assigns = []
                        for i, elt in enumerate(val.elts):
                            assigns.append(
                                ast.Assign(
                                    targets=[
                                        ast.Name(id=f"{res_name}_{i}", ctx=ast.Store())
                                    ],
                                    value=elt,
                                )
                            )
                        return assigns
                    self.return_count = 1
                    return ast.Assign(
                        targets=[ast.Name(id=res_name, ctx=ast.Store())],
                        value=val,
                    )
                return node

        renamer = VariableRenamer(suffix, combined_globals, substitutions)
        renamer.return_count = 0
        new_body = []
        for stmt in self._lift_early_returns(func_def.body):
            transformed = renamer.visit(stmt)
            if isinstance(transformed, list):
                new_body.extend(transformed)
            elif transformed is not None:
                new_body.append(transformed)

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
            if arg_def.arg in substitutions:
                # Literal argument already substituted into the body.
                continue
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
        # Share memory_handler so hoisted allocations go to main function entry
        parser = ASTParser(
            self.builder,
            self.tensor_table,
            self.container_table,
            globals_dict=combined_globals,
            unique_counter_ref=self._unique_counter_ref,
            structure_member_info=self.structure_member_info,
            memory_handler=self.memory_handler,
        )

        for stmt in final_body:
            parser.visit(stmt)

        if getattr(renamer, "return_count", 0) > 1:
            return [f"{res_name}_{i}" for i in range(renamer.return_count)]
        return res_name

    def _add_assign_constant(self, target_name, value_str, dtype):
        block = self.builder.add_block()
        t_const = self.builder.add_constant(block, value_str, dtype)
        t_dst = self.builder.add_access(block, target_name)
        t_task = self.builder.add_tasklet(block, TaskletCode.assign, ["_in"], ["_out"])
        self.builder.add_memlet(block, t_const, "void", t_task, "_in", "")
        self.builder.add_memlet(block, t_task, "_out", t_dst, "void", "")

    def _materialize_global_array(self, name, val):
        """Bake a module-global constant numpy array into the SDFG.

        Allocates an array container (hoisted to function entry) and initializes
        every element with a constant-assign tasklet. The container is cached
        per parser so repeated references (e.g. `gamma[i]` inside a loop) reuse
        the same array. Only 1-/N-D arrays of scalar dtype are supported.
        """
        if not hasattr(self, "_global_array_cache"):
            self._global_array_cache = {}
        if name in self._global_array_cache:
            return self._global_array_cache[name]

        arr = np.ascontiguousarray(val)
        try:
            dtype = sdfg_type_from_type(arr.dtype.type)
        except ValueError:
            raise NotImplementedError(
                f"Unsupported dtype for global array '{name}': {arr.dtype}"
            )
        if not isinstance(dtype, Scalar):
            raise NotImplementedError(
                f"Global array '{name}' must have a scalar element type"
            )

        shape = [str(int(d)) for d in arr.shape]
        tmp_name = self.numpy_visitor._create_array_temp(shape, dtype, zero_init=False)
        # Register the cache entry before emitting init so any (unexpected)
        # recursive reference reuses the same container.
        self._global_array_cache[name] = tmp_name

        int_primitives = (
            PrimitiveType.Int64,
            PrimitiveType.Int32,
            PrimitiveType.Int16,
            PrimitiveType.Int8,
            PrimitiveType.UInt64,
            PrimitiveType.UInt32,
            PrimitiveType.UInt16,
            PrimitiveType.UInt8,
        )
        is_int = dtype.primitive_type in int_primitives
        is_bool = dtype.primitive_type == PrimitiveType.Bool

        flat = arr.ravel(order="C")
        block = self.builder.add_block()
        t_dst = self.builder.add_access(block, tmp_name)
        for lin, v in enumerate(flat):
            if is_bool:
                val_str = "true" if bool(v) else "false"
            elif is_int:
                val_str = str(int(v))
            else:
                val_str = repr(float(v))
            t_const = self.builder.add_constant(block, val_str, dtype)
            t_task = self.builder.add_tasklet(
                block, TaskletCode.assign, ["_in"], ["_out"]
            )
            self.builder.add_memlet(block, t_const, "void", t_task, "_in", "")
            self.builder.add_memlet(block, t_task, "_out", t_dst, "void", str(lin))

        return tmp_name

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
        # An array-typed struct member used as an index, e.g. domain.xd[domain.nodelist].
        if self._is_array_member_attribute(node):
            return True
        return False

    def _expr_array_ndim(self, node):
        """Best-effort number of dimensions of an array-valued expression.

        Side-effect-free (inspects metadata only). Returns the ndim, or None if
        the expression is scalar or not a recognizable array. Used to decide
        whether a subscript index (e.g. a column view ``n[:, i]``) is itself an
        array and should be treated as a gather index rather than a scalar.
        """
        if isinstance(node, ast.Name):
            if node.id in self.tensor_table:
                return len(self.tensor_table[node.id].shape)
            return None
        if self._is_array_member_attribute(node):
            obj_type = self.container_table.get(node.value.id)
            member = self.structure_member_info[obj_type.pointee_type.name][node.attr]
            member_shape = member[2]
            return len(member_shape) if member_shape else 1
        if isinstance(node, ast.Subscript):
            base_ndim = self._expr_array_ndim(node.value)
            if base_ndim is None:
                return None
            idx_nodes = (
                node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            )
            consumed = 0  # base axes consumed by non-newaxis indices
            added = 0  # axes contributed by slices/newaxis/array (fancy) indices
            for idx in idx_nodes:
                if self._is_newaxis(idx):
                    added += 1
                elif isinstance(idx, ast.Slice):
                    added += 1
                    consumed += 1
                else:
                    # A scalar index drops one axis; an array-valued (fancy)
                    # index such as `lm[ielem]` replaces one base axis with its
                    # own dimensions (a gather).
                    idx_ndim = self._expr_array_ndim(idx)
                    if idx_ndim is not None and idx_ndim >= 1:
                        added += idx_ndim
                    consumed += 1
            remaining = base_ndim - consumed
            if remaining < 0:
                return None
            return added + remaining
        return None

    def _is_array_valued_index(self, node):
        """True if `node` yields an array (>=1D) usable as a gather index.

        Covers array Names, array-member attributes, and array-valued
        subscript expressions such as a column view ``n[:, i]``.
        """
        if self._is_array_index(node):
            return True
        ndim = self._expr_array_ndim(node)
        return ndim is not None and ndim >= 1

    def _try_const_eval(self, node):
        """Best-effort compile-time evaluation of an expression built from
        global constants (int/float/str/dict/list/tuple) and constant subscript
        access, e.g. ``XI["M"]["mask"]``. Returns ``(True, value)`` on success,
        ``(False, None)`` otherwise.
        """
        if isinstance(node, ast.Constant):
            return True, node.value
        if isinstance(node, ast.Name):
            if node.id in self.globals_dict:
                val = self.globals_dict[node.id]
                if isinstance(val, (int, float, str, dict, list, tuple)):
                    return True, val
            return False, None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            ok, v = self._try_const_eval(node.operand)
            return (True, -v) if ok else (False, None)
        if isinstance(node, ast.Subscript):
            ok_base, base = self._try_const_eval(node.value)
            if not ok_base:
                return False, None
            ok_idx, key = self._try_const_eval(node.slice)
            if not ok_idx:
                return False, None
            try:
                return True, base[key]
            except (KeyError, IndexError, TypeError):
                return False, None
        return False, None

    def _resolve_tuple(self, node):
        """Return the element AST list if `node` denotes a tuple/list.

        A tuple is either a literal (``ast.Tuple``/``ast.List``) or a ``Name``
        bound to one via a prior ``shp = (...)`` assignment (tracked in
        ``tuple_table``). Returns ``None`` if `node` is not a tuple. This is the
        single entry point every tuple consumer (shapes, index sequences, ...)
        should use so variable-bound tuples work everywhere literals do.
        """
        if isinstance(node, (ast.Tuple, ast.List)):
            return node.elts
        if isinstance(node, ast.Name):
            bound = self.tuple_table.get(node.id)
            if bound is not None:
                return bound.elts
        return None

    def _const_int_sequence(self, node):
        """Return a list of ints if `node` is a constant integer tuple/list.

        Used to detect fancy indexing with a compile-time-constant index list,
        e.g. `x[:, (1, 2, 3, 4, 5, 7)]`. Returns None otherwise.
        """
        elts = self._resolve_tuple(node)
        if elts is None:
            return None
        values = []
        for elt in elts:
            if (
                isinstance(elt, ast.Constant)
                and isinstance(elt.value, int)
                and not isinstance(elt.value, bool)
            ):
                values.append(elt.value)
            elif (
                isinstance(elt, ast.UnaryOp)
                and isinstance(elt.op, ast.USub)
                and isinstance(elt.operand, ast.Constant)
                and isinstance(elt.operand.value, int)
            ):
                values.append(-elt.operand.value)
            else:
                return None
        return values if values else None

    def _handle_fancy_index(self, node, value_str, indices_nodes, shapes, ndim):
        """Handle fancy indexing with a constant integer sequence on one axis.

        Supports the form `A[:, ..., (i0, i1, ...), ..., :]` where exactly one
        axis is indexed by a compile-time-constant list/tuple of integers and
        every other axis is a full slice. Produces a materialized (copied)
        contiguous array whose selected axis has length ``len(sequence)``; for
        each selected position ``k`` the corresponding slab ``A[..., seq[k], ...]``
        is copied into the output. Other fancy-indexing shapes raise clearly.
        """
        # Pad implicit trailing full slices so indices cover all dimensions.
        padded = list(indices_nodes)
        while len(padded) < ndim:
            padded.append(ast.Slice(lower=None, upper=None, step=None))
        if len(padded) != ndim:
            raise NotImplementedError(
                "Fancy indexing with more indices than array dimensions is not "
                "supported"
            )

        fancy_axis = None
        cols = None
        for i, idx in enumerate(padded):
            seq = self._const_int_sequence(idx)
            if seq is not None:
                if fancy_axis is not None:
                    raise NotImplementedError(
                        "Fancy indexing on more than one axis is not supported"
                    )
                fancy_axis = i
                cols = seq
            elif isinstance(idx, ast.Slice):
                is_full = (
                    idx.lower is None
                    and idx.upper is None
                    and (
                        idx.step is None
                        or (isinstance(idx.step, ast.Constant) and idx.step.value == 1)
                    )
                )
                if not is_full:
                    raise NotImplementedError(
                        "Fancy indexing combined with a partial slice is not "
                        "supported"
                    )
            else:
                raise NotImplementedError(
                    "Fancy indexing combined with a scalar index is not supported"
                )

        # Normalize negative column indices against the (constant) axis length.
        axis_len_str = str(shapes[fancy_axis])
        try:
            axis_len_int = int(axis_len_str)
        except ValueError:
            axis_len_int = None
        norm_cols = []
        for c in cols:
            if c < 0:
                if axis_len_int is None:
                    raise NotImplementedError(
                        "Negative fancy-index values require a statically known "
                        "axis length"
                    )
                c += axis_len_int
            norm_cols.append(c)

        # Element type of the source.
        dtype = Scalar(PrimitiveType.Double)
        if value_str in self.container_table:
            t = self.container_table[value_str]
            if isinstance(t, Pointer) and t.has_pointee_type():
                dtype = t.pointee_type

        # Allocate a contiguous output array (copy).
        out_shape = list(shapes)
        out_shape[fancy_axis] = str(len(norm_cols))
        tmp_name = self.builder.find_new_name("_fancy_tmp_")
        ptr_type = Pointer(dtype)
        self.builder.add_container(tmp_name, ptr_type, False)
        self.container_table[tmp_name] = ptr_type
        self.tensor_table[tmp_name] = Tensor(dtype, out_shape)

        size_str = "1"
        for dim in out_shape:
            size_str = f"({size_str} * {dim})"
        element_size = self.builder.get_sizeof(dtype)
        total_size = f"({size_str} * {element_size})"

        debug_info = DebugInfo()
        block_alloc = self.builder.add_block(debug_info)
        t_malloc = self.builder.add_malloc(block_alloc, total_size)
        t_ptr = self.builder.add_access(block_alloc, tmp_name, debug_info)
        self.builder.add_memlet(
            block_alloc, t_malloc, "_ret", t_ptr, "void", "", ptr_type, debug_info
        )

        # Copy each selected slab: out[..., k:k+1, ...] = A[..., col:col+1, ...].
        def _make_subscript(base_name, k):
            elts = [ast.Slice(lower=None, upper=None, step=None) for _ in range(ndim)]
            elts[fancy_axis] = ast.Slice(
                lower=ast.Constant(value=k),
                upper=ast.Constant(value=k + 1),
                step=None,
            )
            sl = elts[0] if ndim == 1 else ast.Tuple(elts=elts, ctx=ast.Load())
            sub = ast.Subscript(
                value=ast.Name(id=base_name, ctx=ast.Load()), slice=sl, ctx=ast.Load()
            )
            ast.copy_location(sub, node)
            ast.fix_missing_locations(sub)
            return sub

        for k, col in enumerate(norm_cols):
            dst = _make_subscript(tmp_name, k)
            dst.ctx = ast.Store()
            src = _make_subscript(value_str, col)
            assign = ast.Assign(targets=[dst], value=src)
            ast.copy_location(assign, node)
            ast.fix_missing_locations(assign)
            self.visit_Assign(assign)

        return tmp_name

    def _is_newaxis(self, node):
        """True if a subscript index is np.newaxis / None (inserts a size-1 axis)."""
        if isinstance(node, ast.Constant) and node.value is None:
            return True
        if isinstance(node, ast.Attribute) and node.attr == "newaxis":
            return True
        return False

    def _handle_newaxis_subscript(self, value_str, indices_nodes):
        """Handle np.newaxis / None indexing (arr[:, None], arr[None, :], ...).

        Produces a view of the source array with size-1 axes inserted at the
        newaxis positions. Only full slices may accompany the new axes.
        """
        tensor = self.tensor_table[value_str]

        new_axis_positions = []
        for i, idx in enumerate(indices_nodes):
            if self._is_newaxis(idx):
                new_axis_positions.append(i)
                continue
            is_full_slice = (
                isinstance(idx, ast.Slice)
                and idx.lower is None
                and idx.upper is None
                and (
                    idx.step is None
                    or (isinstance(idx.step, ast.Constant) and idx.step.value == 1)
                )
            )
            if not is_full_slice:
                raise NotImplementedError(
                    "np.newaxis indexing is only supported combined with full slices"
                )

        # Insert size-1 axes (in increasing position order so each insertion
        # accounts for the previously inserted axes).
        new_tensor = tensor
        for pos in sorted(new_axis_positions):
            new_tensor = new_tensor.newaxis(pos)

        tmp_name = self.builder.find_new_name("_newaxis_")
        ptr_type = Pointer(new_tensor.element_type)
        self.builder.add_container(tmp_name, ptr_type, False)
        self.container_table[tmp_name] = ptr_type
        self.tensor_table[tmp_name] = new_tensor

        block = self.builder.add_block()
        t_src = self.builder.add_access(block, value_str)
        t_dst = self.builder.add_access(block, tmp_name)
        self.builder.add_reference_memlet(block, t_src, t_dst, "0", ptr_type)
        return tmp_name

    def _is_array_member_attribute(self, node):
        """Return True if node is `obj.attr` where attr is an array (pointer) member."""
        if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)):
            return False
        obj_type = self.container_table.get(node.value.id)
        if not (isinstance(obj_type, Pointer) and obj_type.has_pointee_type()):
            return False
        pointee = obj_type.pointee_type
        if not isinstance(pointee, Structure):
            return False
        member = self.structure_member_info.get(pointee.name, {}).get(node.attr)
        return member is not None and isinstance(member[1], Pointer)

    def _struct_scalar_member(self, node):
        """If `node` is `obj.attr` where attr is a SCALAR struct member (not an
        array/pointer member), return (obj_name, member_index, member_type);
        otherwise None."""
        if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)):
            return None
        obj_name = node.value.id
        obj_type = self.container_table.get(obj_name)
        if not (isinstance(obj_type, Pointer) and obj_type.has_pointee_type()):
            return None
        pointee = obj_type.pointee_type
        if not isinstance(pointee, Structure):
            return None
        member = self.structure_member_info.get(pointee.name, {}).get(node.attr)
        if member is None or isinstance(member[1], Pointer):
            return None
        member_index, member_type, _member_shape = member
        return obj_name, member_index, member_type

    def _store_scalar_member(
        self, obj_name, member_index, member_type, value_node, node
    ):
        """Store a scalar value into a struct member: `obj.attr = value` becomes
        a write to obj[0, member_index] so it propagates through the struct."""
        debug_info = get_debug_info(node, self.filename, self.function_name)
        rhs = self.visit(value_node)
        subset = "0," + str(member_index)

        block = self.builder.add_block(debug_info)
        t_task = self.builder.add_tasklet(
            block, TaskletCode.assign, ["_in"], ["_out"], debug_info
        )
        t_src, src_sub = self._add_read(block, rhs, debug_info)
        self.builder.add_memlet(
            block, t_src, "void", t_task, "_in", src_sub, None, debug_info
        )
        obj_access = self.builder.add_access(block, obj_name, debug_info)
        self.builder.add_memlet(
            block, t_task, "_out", obj_access, "", subset, None, debug_info
        )

    def _is_boolean_mask_index(self, node):
        """Check if a subscript index is a boolean mask (e.g. arr[arr <= 0.1])."""
        # A boolean array variable used directly as a mask.
        if isinstance(node, ast.Name) and node.id in self.tensor_table:
            element_type = self.tensor_table[node.id].element_type
            return element_type.primitive_type == PrimitiveType.Bool
        # A whole-array comparison (e.g. arr <= 0.1, x > y) yields a boolean mask.
        if isinstance(node, ast.Compare):
            operands = [node.left] + list(node.comparators)
            return any(
                isinstance(o, ast.Name) and o.id in self.tensor_table for o in operands
            )
        return False

    def _handle_masked_assignment(
        self, target, mask_node, value_node, target_name, orig_node
    ):
        """Handle boolean-mask assignment: target[mask] = value.

        Rewritten as ``target[:] = np.where(mask, value, target)`` which has
        identical NumPy semantics (elements where the mask is False keep their
        original value) and reuses the existing np.where lowering.
        """
        ndim = len(self.tensor_table[target_name].shape)

        full_slice = ast.Slice(lower=None, upper=None, step=None)
        if ndim > 1:
            slice_arg = ast.Tuple(
                elts=[
                    ast.Slice(lower=None, upper=None, step=None) for _ in range(ndim)
                ],
                ctx=ast.Load(),
            )
        else:
            slice_arg = full_slice

        new_target = ast.Subscript(
            value=copy.deepcopy(target.value), slice=slice_arg, ctx=ast.Store()
        )

        where_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()), attr="where", ctx=ast.Load()
            ),
            args=[
                copy.deepcopy(mask_node),
                copy.deepcopy(value_node),
                ast.Name(id=target_name, ctx=ast.Load()),
            ],
            keywords=[],
        )

        new_assign = ast.Assign(targets=[new_target], value=where_call)
        ast.copy_location(new_assign, orig_node)
        ast.fix_missing_locations(new_assign)
        self.visit_Assign(new_assign)

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

        if idx_ndim == 0:
            raise NotImplementedError("Scalar index not supported for gather")

        # Gather over an N-D index array: the result has the same shape as the
        # index array and is computed elementwise on the flattened arrays:
        #   result_flat[k] = value[idx_flat[k]]
        # Both the (contiguous) index array and result are addressed with a
        # single flat loop variable.
        result_shapes = [
            idx_shapes[d] if d < len(idx_shapes) else f"_{idx_array_name}_shape_{d}"
            for d in range(idx_ndim)
        ]
        total_count = str(result_shapes[0])
        for s in result_shapes[1:]:
            total_count = f"({total_count} * {s})"

        # For runtime evaluation, prefer shapes_runtime_info if available
        # This ensures we use expressions that can be evaluated at runtime
        if idx_array_name in self.shapes_runtime_info:
            result_shapes_runtime = list(self.shapes_runtime_info[idx_array_name])
        else:
            result_shapes_runtime = list(result_shapes)

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
        total_size = f"({total_count} * {element_size})"

        ptr_type = Pointer(dtype)
        self.builder.add_container(tmp_name, ptr_type, False)
        self.container_table[tmp_name] = ptr_type
        self.tensor_table[tmp_name] = Tensor(dtype, list(result_shapes))
        # Store runtime evaluable shape for this gather result
        self.shapes_runtime_info[tmp_name] = result_shapes_runtime

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

        self.builder.begin_for(loop_var, "0", str(total_count), "1", debug_info)

        block_load_idx = self.builder.add_block(debug_info)
        idx_arr_access = self.builder.add_access(
            block_load_idx, idx_array_name, debug_info
        )
        idx_var_access = self.builder.add_access(block_load_idx, idx_var, debug_info)
        tasklet_load = self.builder.add_tasklet(
            block_load_idx, TaskletCode.assign, ["_in"], ["_out"], debug_info
        )
        # For a 1-D index array pass its Tensor as the memlet base type so a
        # strided index (e.g. a column view `nodelist[:, i]`) is linearized with
        # its stride/offset; a flat (None) base type would ignore the stride and
        # read out of bounds. N-D index arrays are assumed contiguous (flat).
        idx_base_type = self.tensor_table[idx_array_name] if idx_ndim == 1 else None
        self.builder.add_memlet(
            block_load_idx,
            idx_arr_access,
            "void",
            tasklet_load,
            "_in",
            loop_var,
            idx_base_type,
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

    def _handle_scatter_assignment(
        self, target_name, index_node, value_node, accumulate, debug_info=None
    ):
        """Handle scatter (indexed) assignment: arr[idx] = vals or arr[idx] += vals.

        ``idx`` is an integer index array; the assignment is lowered to a
        sequential loop ``for k: arr[idx[k]] (=|+=) vals[k]``. Tensor base types
        are used on the memlets so strided views (e.g. a column slice) are
        linearized correctly.
        """
        if debug_info is None:
            debug_info = DebugInfo()

        idx_name = self.visit(index_node)
        if idx_name not in self.tensor_table:
            raise NotImplementedError("Scatter index must be an array")

        arr_tensor = self.tensor_table[target_name]
        elem_type = arr_tensor.element_type
        is_int = elem_type.primitive_type in (
            PrimitiveType.Int64,
            PrimitiveType.Int32,
            PrimitiveType.Int16,
            PrimitiveType.Int8,
            PrimitiveType.UInt64,
            PrimitiveType.UInt32,
            PrimitiveType.UInt16,
            PrimitiveType.UInt8,
        )
        add_code = TaskletCode.int_add if is_int else TaskletCode.fp_add

        idx_tensor = self.tensor_table[idx_name]
        idx_shapes = idx_tensor.shape
        total_count = str(idx_shapes[0]) if idx_shapes else "1"
        for s in idx_shapes[1:]:
            total_count = f"({total_count} * {s})"

        idx_dtype = Scalar(PrimitiveType.Int64)
        if idx_name in self.container_table:
            t = self.container_table[idx_name]
            if isinstance(t, Pointer) and t.has_pointee_type():
                idx_dtype = t.pointee_type

        # The RHS values: an array (indexed per element) or a broadcast scalar.
        vals_name = self.visit(value_node)
        vals_is_array = vals_name in self.tensor_table

        loop_var = self.builder.find_new_name("_scatter_i_")
        self.builder.add_container(loop_var, Scalar(PrimitiveType.Int64), False)
        self.container_table[loop_var] = Scalar(PrimitiveType.Int64)

        idx_var = self.builder.find_new_name("_scatter_idx_")
        self.builder.add_container(idx_var, idx_dtype, False)
        self.container_table[idx_var] = idx_dtype

        self.builder.begin_for(loop_var, "0", total_count, "1", debug_info)

        # Load the destination index: idx_var = idx[loop_var]
        block_idx = self.builder.add_block(debug_info)
        idx_acc = self.builder.add_access(block_idx, idx_name, debug_info)
        idx_var_acc = self.builder.add_access(block_idx, idx_var, debug_info)
        t_load = self.builder.add_tasklet(
            block_idx, TaskletCode.assign, ["_in"], ["_out"], debug_info
        )
        self.builder.add_memlet(
            block_idx, idx_acc, "void", t_load, "_in", loop_var, idx_tensor, debug_info
        )
        self.builder.add_memlet(
            block_idx, t_load, "_out", idx_var_acc, "void", "", None, debug_info
        )

        # Scatter the value into arr[idx_var].
        block = self.builder.add_block(debug_info)
        if accumulate:
            arr_in = self.builder.add_access(block, target_name, debug_info)
            arr_out = self.builder.add_access(block, target_name, debug_info)
            task = self.builder.add_tasklet(
                block, add_code, ["_in1", "_in2"], ["_out"], debug_info
            )
            self.builder.add_memlet(
                block, arr_in, "void", task, "_in1", idx_var, arr_tensor, debug_info
            )
            if vals_is_array:
                vals_acc = self.builder.add_access(block, vals_name, debug_info)
                self.builder.add_memlet(
                    block,
                    vals_acc,
                    "void",
                    task,
                    "_in2",
                    loop_var,
                    self.tensor_table[vals_name],
                    debug_info,
                )
            elif self.builder.exists(vals_name):
                vals_acc = self.builder.add_access(block, vals_name, debug_info)
                self.builder.add_memlet(
                    block, vals_acc, "void", task, "_in2", "", None, debug_info
                )
            else:
                vals_const = self.builder.add_constant(
                    block, vals_name, elem_type, debug_info
                )
                self.builder.add_memlet(
                    block, vals_const, "void", task, "_in2", "", None, debug_info
                )
            self.builder.add_memlet(
                block, task, "_out", arr_out, "void", idx_var, arr_tensor, debug_info
            )
        else:
            arr_out = self.builder.add_access(block, target_name, debug_info)
            task = self.builder.add_tasklet(
                block, TaskletCode.assign, ["_in"], ["_out"], debug_info
            )
            if vals_is_array:
                vals_acc = self.builder.add_access(block, vals_name, debug_info)
                self.builder.add_memlet(
                    block,
                    vals_acc,
                    "void",
                    task,
                    "_in",
                    loop_var,
                    self.tensor_table[vals_name],
                    debug_info,
                )
            elif self.builder.exists(vals_name):
                vals_acc = self.builder.add_access(block, vals_name, debug_info)
                self.builder.add_memlet(
                    block, vals_acc, "void", task, "_in", "", None, debug_info
                )
            else:
                vals_const = self.builder.add_constant(
                    block, vals_name, elem_type, debug_info
                )
                self.builder.add_memlet(
                    block, vals_const, "void", task, "_in", "", None, debug_info
                )
            self.builder.add_memlet(
                block, task, "_out", arr_out, "void", idx_var, arr_tensor, debug_info
            )

        self.builder.end_for()

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
        self,
        target,
        materialized_rhs,
        target_name,
        indices,
        target_ndim,
        value_ndim,
        debug_info,
    ):
        """Handle slice assignment with broadcasting (e.g., 2D[:,:] = 1D[:]).

        materialized_rhs is the already-evaluated RHS array name (not AST node).
        """
        broadcast_dims = target_ndim - value_ndim
        shapes = self.tensor_table[target_name].shape
        rhs_tensor = self.tensor_table.get(materialized_rhs)
        rhs_shapes = rhs_tensor.shape if rhs_tensor else []

        # Create outer loops for broadcast dimensions
        outer_loop_vars = []
        for i in range(broadcast_dims):
            loop_var = self.builder.find_new_name(f"_bcast_iter_{i}_")
            outer_loop_vars.append(loop_var)

            if not self.builder.exists(loop_var):
                self.builder.add_container(loop_var, Scalar(PrimitiveType.Int64), False)
                self.container_table[loop_var] = Scalar(PrimitiveType.Int64)

            dim_size = shapes[i] if i < len(shapes) else f"_{target_name}_shape_{i}"
            self.builder.begin_for(loop_var, "0", str(dim_size), "1", debug_info)

        # Create inner loops for value dimensions
        inner_loop_vars = []
        for i in range(value_ndim):
            loop_var = self.builder.find_new_name(f"_inner_iter_{i}_")
            inner_loop_vars.append(loop_var)

            if not self.builder.exists(loop_var):
                self.builder.add_container(loop_var, Scalar(PrimitiveType.Int64), False)
                self.container_table[loop_var] = Scalar(PrimitiveType.Int64)

            # Use RHS shape for inner dimension bounds
            dim_size = (
                rhs_shapes[i] if i < len(rhs_shapes) else shapes[broadcast_dims + i]
            )
            self.builder.begin_for(loop_var, "0", str(dim_size), "1", debug_info)

        # Create assignment block: target[outer_vars, inner_vars] = rhs[inner_vars]
        block = self.builder.add_block(debug_info)
        t_src = self.builder.add_access(block, materialized_rhs, debug_info)
        t_dst = self.builder.add_access(block, target_name, debug_info)
        t_task = self.builder.add_tasklet(
            block, TaskletCode.assign, ["_in"], ["_out"], debug_info
        )

        # Source index: just inner loop vars
        src_index = ",".join(inner_loop_vars) if inner_loop_vars else "0"

        # Target index: outer_vars + inner_vars combined
        all_target_vars = outer_loop_vars + inner_loop_vars
        target_index = ",".join(all_target_vars) if all_target_vars else "0"

        self.builder.add_memlet(
            block, t_src, "void", t_task, "_in", src_index, rhs_tensor, debug_info
        )

        tensor_dst = self.tensor_table[target_name]
        self.builder.add_memlet(
            block, t_task, "_out", t_dst, "void", target_index, tensor_dst, debug_info
        )

        # Close all loops (inner first, then outer)
        for _ in inner_loop_vars:
            self.builder.end_for()
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

        # ALWAYS evaluate RHS first (NumPy semantics) - before any loops
        materialized_rhs = self.visit(value)

        if (
            target_slice_ndim > 0
            and value_max_ndim > 0
            and target_slice_ndim > value_max_ndim
        ):
            # Broadcasting case: use row-by-row approach with reference memlets
            self._handle_broadcast_slice_assignment(
                target,
                materialized_rhs,
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
                # intermediate computations are placed outside the loops
                idx_str = self.visit(normalized_idx)
                new_target_indices.append(ast.Name(id=idx_str, ctx=ast.Load()))

        rewriter = SliceRewriter(loop_vars, self.tensor_table, self)
        new_value = rewriter.visit(copy.deepcopy(value))

        new_target = copy.deepcopy(target)
        if len(new_target_indices) == 1:
            new_target.slice = new_target_indices[0]
        else:
            new_target.slice = ast.Tuple(elts=new_target_indices, ctx=ast.Load())

        rhs_memlet_type = None
        rhs_indexed_subset = ""
        if materialized_rhs in self.tensor_table:
            rhs_tensor = self.tensor_table[materialized_rhs]
            rhs_ndim = len(rhs_tensor.shape)
            if rhs_ndim > 0 and rhs_ndim == len(loop_vars):
                # RHS is an array matching the slice dimensions - index it with loop vars
                rhs_indexed_subset = ",".join(loop_vars)
                rhs_memlet_type = rhs_tensor

        block = self.builder.add_block(debug_info)
        t_task = self.builder.add_tasklet(
            block, TaskletCode.assign, ["_in"], ["_out"], debug_info
        )

        t_src, src_sub = self._add_read(block, materialized_rhs, debug_info)
        # Use indexed subset if RHS is an array that needs indexing
        actual_src_sub = rhs_indexed_subset if rhs_indexed_subset else src_sub
        self.builder.add_memlet(
            block,
            t_src,
            "void",
            t_task,
            "_in",
            actual_src_sub,
            rhs_memlet_type,
            debug_info,
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
                # Visit the index NOW before any loops are opened to ensure
                # intermediate computations are placed outside the loops
                idx_str = self.visit(normalized_idx)
                new_target_indices.append(ast.Name(id=idx_str, ctx=ast.Load()))

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

    def _get_memlet_type_for_access(self, expr_str, subset):
        """Get the Tensor type for an indexed array access expression.

        When accessing an array like "arr(i,j)" with a multi-dimensional subset,
        we need to pass the Tensor type to add_memlet for correct type inference.
        If the expression is a simple scalar variable or constant, returns None.
        """
        if not subset:
            return None

        # Check if expr_str is an indexed array access like "arr(i,j)"
        if "(" in expr_str and expr_str.endswith(")"):
            name = expr_str.split("(")[0]
            if name in self.tensor_table:
                return self.tensor_table[name]

        # Check if expr_str is a simple array name with a non-empty subset from _add_read
        if expr_str in self.tensor_table:
            return self.tensor_table[expr_str]

        return None

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
