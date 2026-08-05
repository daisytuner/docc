"""
GraphParser modules for parsing BLAS and LAPACK operations.
"""

import torch.fx
from torch.fx.node import Argument

from docc.sdfg import StructuredSDFGBuilder, Tensor, DebugInfo, Scalar, Type

from docc.pytorch.graph_parser.utils import (
    ContainerInfo,
    ContainerInfos,
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

    def _copy_container_if_needed(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
        mat_container: str,
        mat_tensor: Tensor,
        debug_info: DebugInfo,
    ) -> tuple[str, Tensor]:
        if self._last_dims_non_transposed(mat_tensor) or self._last_dims_transposed(
            mat_tensor
        ):
            return mat_container, mat_tensor  # Not needed

        mat_type: Type = container_info[mat_container].sdfg_type()
        intermediate_tensor: Tensor = Tensor(mat_tensor.element_type, mat_tensor.shape)
        intermediate_container: str = self.create_intermediate_container(
            node, builder, container_info, mat_type, intermediate_tensor
        )
        builder.add_copy_op(
            mat_container,
            mat_tensor,
            intermediate_container,
            intermediate_tensor,
            debug_info,
        )
        return intermediate_container, intermediate_tensor

    def add_matmul_op(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
        A_container: str,
        A_tensor: Tensor,
        B_container: str,
        B_tensor: Tensor,
        Y_container: str,
        Y_tensor: Tensor,
        debug_info: DebugInfo,
    ) -> None:
        A_container, A_tensor = self._copy_container_if_needed(
            node, builder, container_info, A_container, A_tensor, debug_info
        )
        B_container, B_tensor = self._copy_container_if_needed(
            node, builder, container_info, B_container, B_tensor, debug_info
        )
        builder.add_matmul_op(
            A_container,
            A_tensor,
            B_container,
            B_tensor,
            Y_container,
            Y_tensor,
            debug_info,
        )


class MMParser(MatmulParserBase):
    """Formula is: result = self @ mat2"""

    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
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
        self_container: str = self.get_arg_container(node, container_info, 0)
        self_tensor: Tensor = self.get_tensor_type(node, container_info, self_container)
        mat2_container: str = self.get_arg_container(node, container_info, 1)
        mat2_tensor: Tensor = self.get_tensor_type(node, container_info, mat2_container)
        result_container: str = self.get_result_container(node, builder, container_info)
        result_tensor: Tensor = self.get_tensor_type(
            node, container_info, result_container
        )
        debug_info: DebugInfo = self.get_debug_info(node)
        self.add_matmul_op(
            node,
            builder,
            container_info,
            self_container,
            self_tensor,
            mat2_container,
            mat2_tensor,
            result_container,
            result_tensor,
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
        container_info: ContainerInfos,
    ) -> None:
        if len(node.args) != 3:
            raise GraphParserError(
                self,
                node,
                "Expected exactly 3 arguments but got " + str(len(node.args)),
            )
        alpha_arg: Argument | None = None
        beta_arg: Argument | None = None
        for key in node.kwargs:
            if key == "alpha":
                alpha_arg: Argument | None = node.kwargs[key]
            elif key == "beta":
                beta_arg: Argument | None = node.kwargs[key]
            else:
                raise GraphParserError(
                    self, node, "Unsupported kwargs: " + str(node.kwargs)
                )

        self_container: str = self.get_arg_container(node, container_info, 0)
        self_tensor: Tensor = self.get_tensor_type(node, container_info, self_container)
        mat1_container: str = self.get_arg_container(node, container_info, 1)
        mat1_tensor: Tensor = self.get_tensor_type(node, container_info, mat1_container)
        mat2_container: str = self.get_arg_container(node, container_info, 2)
        mat2_tensor: Tensor = self.get_tensor_type(node, container_info, mat2_container)
        result_container: str = self.get_result_container(node, builder, container_info)
        result_tensor: Tensor = self.get_tensor_type(
            node, container_info, result_container
        )

        # matmul = mat1 @ mat2
        matmul_container: str = self.create_intermediate_container(
            node,
            builder,
            container_info,
            container_info[result_container].sdfg_type(),
            result_tensor,
        )
        matmul_tensor: Tensor = self.get_tensor_type(
            node, container_info, matmul_container
        )
        debug_info: DebugInfo = self.get_debug_info(node)
        self.add_matmul_op(
            node,
            builder,
            container_info,
            mat1_container,
            mat1_tensor,
            mat2_container,
            mat2_tensor,
            matmul_container,
            matmul_tensor,
            debug_info,
        )

        # mul1 = alpha * matmul
        if alpha_arg is None:
            mul1_container: str = matmul_container
            mul1_tensor: Tensor = matmul_tensor
        else:
            alpha: str | tuple[str, Scalar] = self.convert_arg_to_sdfg_value(
                node, container_info, alpha_arg
            )
            if isinstance(alpha, str):
                alpha_container: str = alpha
                alpha_tensor: Tensor = self.get_tensor_type(
                    node, container_info, alpha_container
                )
            else:
                alpha_container: str = alpha[0]
                alpha_tensor: Tensor = Tensor(
                    self.align_constant_type(node, alpha, matmul_tensor.element_type),
                    [],
                )
            mul1_container: str = self.create_intermediate_container(
                node,
                builder,
                container_info,
                container_info[matmul_container].sdfg_type(),
                matmul_tensor,
            )
            mul1_tensor: Tensor = self.get_tensor_type(
                node, container_info, mul1_container
            )
            builder.add_elementwise_op(
                "mul",
                alpha_container,
                alpha_tensor,
                matmul_container,
                matmul_tensor,
                mul1_container,
                mul1_tensor,
                debug_info,
            )

        # mul2 = beta * self
        if beta_arg is None:
            mul2_container: str = self_container
            mul2_tensor: Tensor = self_tensor
        else:
            beta: str | tuple[str, Scalar] = self.convert_arg_to_sdfg_value(
                node, container_info, beta_arg
            )
            if isinstance(beta, str):
                beta_container: str = beta
                beta_tensor: Tensor = self.get_tensor_type(
                    node, container_info, beta_container
                )
            else:
                beta_container: str = beta[0]
                beta_tensor: Tensor = Tensor(
                    self.align_constant_type(node, beta, self_tensor.element_type), []
                )
            mul2_container: str = self.create_intermediate_container(
                node,
                builder,
                container_info,
                container_info[self_container].sdfg_type(),
                self_tensor,
            )
            mul2_tensor: Tensor = self.get_tensor_type(
                node, container_info, mul2_container
            )
            builder.add_elementwise_op(
                "mul",
                beta_container,
                beta_tensor,
                self_container,
                self_tensor,
                mul2_container,
                mul2_tensor,
                debug_info,
            )

        # Broadcast mul2 shape to mul1 shape: broadcast = Broadcast(mul2)
        if mul2_tensor.shape == mul1_tensor.shape:
            broadcast_container: str = mul2_container
            broadcast_tensor: Tensor = mul2_tensor
        else:
            broadcast_container: str = self.create_intermediate_container(
                node,
                builder,
                container_info,
                container_info[mul1_container].sdfg_type(),
                mul1_tensor,
            )
            broadcast_tensor: Tensor = self.get_tensor_type(
                node, container_info, broadcast_container
            )
            builder.add_broadcast_op(
                mul2_container,
                mul2_tensor,
                broadcast_container,
                broadcast_tensor,
                mul2_tensor.shape,
                broadcast_tensor.shape,
                debug_info,
            )

        # result = broadcast + mul1
        builder.add_elementwise_op(
            "add",
            broadcast_container,
            broadcast_tensor,
            mul1_container,
            mul1_tensor,
            result_container,
            result_tensor,
            debug_info,
        )


register_module("aten.addmm.default", AddMMParser())
