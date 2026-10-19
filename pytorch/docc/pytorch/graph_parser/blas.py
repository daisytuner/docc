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


class MMParser(GraphParserModule):
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
        builder.add_matmul_op(
            self_info.container(),
            self_info.sdfg_tensor_type(),
            mat2_info.container(),
            mat2_info.sdfg_tensor_type(),
            result_info.container(),
            result_info.sdfg_tensor_type(),
            debug_info,
        )


register_module("aten.mm.default", MMParser())
register_module("aten.bmm.default", MMParser())


class AddMMParser(GraphParserModule):
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
        builder.add_matmul_op(
            mat1_info.container(),
            mat1_info.sdfg_tensor_type(),
            mat2_info.container(),
            mat2_info.sdfg_tensor_type(),
            matmul_info.container(),
            matmul_info.sdfg_tensor_type(),
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
