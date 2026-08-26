"""
GraphParser modules for parsing BLAS and LAPACK operations.
"""

import torch.fx
from torch.fx.node import Argument

from docc.sdfg import StructuredSDFGBuilder, Tensor, DebugInfo

from docc.pytorch.graph_parser.utils import (
    TensorInfo,
    TensorConstant,
    TensorMetadata,
    GraphParserError,
    GraphParserModule,
    register_module,
)


class MatmulParserBase(GraphParserModule):
    def _last_dims_non_transposed(self, tensor: Tensor) -> bool:
        return tensor.is_contiguous()

    def _last_dims_transposed(self, tensor: Tensor) -> bool:
        if len(tensor.shape) < 2:
            return False
        test_shape: list[str] = tensor.shape
        test_shape[-1] = tensor.shape[-2]
        test_shape[-2] = tensor.shape[-1]
        test_strides: list[str] = tensor.strides
        test_strides[-1] = tensor.strides[-2]
        test_strides[-2] = tensor.strides[-1]
        test_tensor: Tensor = Tensor(tensor.element_type, test_shape)
        return test_tensor.strides == test_strides

    def _copy_if_needed(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
        info: TensorInfo,
        debug_info: DebugInfo,
    ) -> TensorInfo:
        if self._last_dims_non_transposed(
            info.sdfg_tensor_type()
        ) or self._last_dims_transposed(info.sdfg_tensor_type()):
            return info  # Not needed

        intermediate_tensor: Tensor = Tensor(info.element_type(), info.shape())
        intermediate_info: TensorInfo = self.create_intermediate_tensor_info(
            node, builder, metadata, info.sdfg_type(), intermediate_tensor, [info]
        )
        builder.add_copy_op(
            info.container(),
            info.sdfg_tensor_type(),
            intermediate_info.container(),
            intermediate_tensor,
            debug_info,
        )
        return intermediate_info

    def add_matmul_op(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
        A_info: TensorInfo,
        B_info: TensorInfo,
        Y_info: TensorInfo,
        debug_info: DebugInfo,
    ) -> None:
        A_info_: TensorInfo = self._copy_if_needed(
            node, builder, metadata, A_info, debug_info
        )
        B_info_: TensorInfo = self._copy_if_needed(
            node, builder, metadata, B_info, debug_info
        )
        builder.add_matmul_op(
            A_info_.container(),
            A_info_.sdfg_tensor_type(),
            B_info_.container(),
            B_info_.sdfg_tensor_type(),
            Y_info.container(),
            Y_info.sdfg_tensor_type(),
            debug_info,
        )


class MMParser(MatmulParserBase):
    """Formula is: result = self @ mat2"""

    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
    ) -> None:
        if len(node.args) != 2:
            raise GraphParserError(
                self,
                node,
                "Expected exactly 2 arguments but got " + str(len(node.args)),
            )
        if len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )
        self_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        mat2_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 1)
        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        debug_info: DebugInfo = self.get_debug_info(node)
        self.add_matmul_op(
            node,
            builder,
            metadata,
            self_info,
            mat2_info,
            result_info,
            debug_info,
        )


register_module("aten.mm.default", MMParser())
register_module("aten.bmm.default", MMParser())


class AddMMParser(MatmulParserBase):
    """Formula is: result = beta * self + alpha * (mat1 @ mat2)"""

    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
    ) -> None:
        if len(node.args) != 3:
            raise GraphParserError(
                self,
                node,
                "Expected exactly 3 arguments but got " + str(len(node.args)),
            )
        alpha_arg: Argument = None
        beta_arg: Argument = None
        for key in node.kwargs:
            if key == "alpha":
                alpha_arg: Argument = node.kwargs[key]
            elif key == "beta":
                beta_arg: Argument = node.kwargs[key]
            else:
                raise GraphParserError(
                    self, node, "Unsupported kwargs: " + str(node.kwargs)
                )

        self_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        mat1_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 1)
        mat2_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 2)
        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        debug_info: DebugInfo = self.get_debug_info(node)

        # matmul = mat1 @ mat2
        matmul_info: TensorInfo = self.create_intermediate_tensor_info(
            node,
            builder,
            metadata,
            result_info.sdfg_type(),
            result_info.sdfg_tensor_type(),
            [mat1_info, mat2_info],
        )
        self.add_matmul_op(
            node,
            builder,
            metadata,
            mat1_info,
            mat2_info,
            matmul_info,
            debug_info,
        )

        # mul1 = alpha * matmul
        if alpha_arg is None:
            mul1_info: TensorInfo = matmul_info
        else:
            alpha_info_or_const: TensorInfo | TensorConstant = (
                self.convert_arg_to_tensor_info_or_constant(
                    node,
                    metadata,
                    alpha_arg,
                    align_constant_type=matmul_info.element_type(),
                )
            )
            mul1_info: TensorInfo = self.create_intermediate_tensor_info(
                node,
                builder,
                metadata,
                matmul_info.sdfg_type(),
                matmul_info.sdfg_tensor_type(),
                [alpha_info_or_const, matmul_info],
            )
            builder.add_elementwise_op(
                "mul",
                alpha_info_or_const.container(),
                alpha_info_or_const.sdfg_tensor_type(),
                matmul_info.container(),
                matmul_info.sdfg_tensor_type(),
                mul1_info.container(),
                mul1_info.sdfg_tensor_type(),
                debug_info,
            )

        # mul2 = beta * self
        if beta_arg is None:
            mul2_info: TensorInfo = self_info
        else:
            beta_info_or_const: TensorInfo | TensorConstant = (
                self.convert_arg_to_tensor_info_or_constant(
                    node,
                    metadata,
                    beta_arg,
                    align_constant_type=self_info.element_type(),
                )
            )
            mul2_info: TensorInfo = self.create_intermediate_tensor_info(
                node,
                builder,
                metadata,
                self_info.sdfg_type(),
                self_info.sdfg_tensor_type(),
                [beta_info_or_const, self_info],
            )
            builder.add_elementwise_op(
                "mul",
                beta_info_or_const.container(),
                beta_info_or_const.sdfg_tensor_type(),
                self_info.container(),
                self_info.sdfg_tensor_type(),
                mul2_info.container(),
                mul2_info.sdfg_tensor_type(),
                debug_info,
            )

        # Broadcast mul2 shape to mul1 shape if needed
        mul1_tensor: Tensor = mul1_info.sdfg_tensor_type()
        mul2_tensor: Tensor = mul2_info.sdfg_tensor_type()
        if len(mul1_info.shape()) != len(mul2_info.shape()):
            mul1_tensor, mul2_tensor = self.align_elementwise_tensors(
                node, mul1_tensor, mul2_tensor
            )

        # result = mul2 + mul1
        builder.add_elementwise_op(
            "add",
            mul2_info.container(),
            mul2_tensor,
            mul1_info.container(),
            mul1_tensor,
            result_info.container(),
            result_info.sdfg_tensor_type(),
            debug_info,
        )


register_module("aten.addmm.default", AddMMParser())
