"""
GraphParser modules for parsing normalization layers.
"""

import torch.fx

from docc.sdfg import StructuredSDFGBuilder, Tensor, Scalar, DebugInfo

from docc.pytorch.graph_parser.utils import (
    GraphParserModule,
    ContainerInfos,
    GraphParserError,
    register_module,
)


class BatchNormNoTrainingParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
    ) -> None:
        if len(node.args) != 7:
            raise GraphParserError(
                self,
                node,
                "Expected exactly 7 arguments but got " + str(len(node.args)),
            )
        if len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )
        input_container: str = self.get_arg_container(node, container_info, 0)
        input_tensor: Tensor = self.get_tensor_type(
            node, container_info, input_container
        )
        if node.args[1] is None:
            raise GraphParserError(
                self, node, "Currently the weight argument is required but got none"
            )
        weight_container: str = self.get_arg_container(node, container_info, 1)
        weight_tensor: Tensor = self.get_tensor_type(
            node, container_info, weight_container
        )
        if node.args[2] is None:
            raise GraphParserError(
                self, node, "Currently the bias argument is required but got none"
            )
        bias_container: str = self.get_arg_container(node, container_info, 2)
        bias_tensor: Tensor = self.get_tensor_type(node, container_info, bias_container)
        running_mean_container: str = self.get_arg_container(node, container_info, 3)
        running_mean_tensor: Tensor = self.get_tensor_type(
            node, container_info, running_mean_container
        )
        running_var_container: str = self.get_arg_container(node, container_info, 4)
        running_var_tensor: Tensor = self.get_tensor_type(
            node, container_info, running_var_container
        )
        # We just ignore momentum for now (argument 5)
        eps: str | tuple[str, Scalar] = self.get_arg_sdfg_value(node, container_info, 6)
        if isinstance(eps, str):
            eps_container: str = eps
            eps_type: Scalar = self.get_scalar_type(node, container_info, eps_container)
        else:
            eps_container: str = eps[0]
            eps_type: Scalar = self.align_constant_type(
                node, eps, input_tensor.element_type
            )
        result_containers: tuple[str, ...] = self.get_result_containers(
            3, node, builder, container_info
        )
        result_container: str = result_containers[0]
        result_tensor: Tensor = self.get_tensor_type(
            node, container_info, result_container
        )
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_batchnorm_with_bias(
            input_container,
            input_tensor,
            running_var_container,
            running_var_tensor,
            running_mean_container,
            running_mean_tensor,
            weight_container,
            weight_tensor,
            bias_container,
            bias_tensor,
            eps_container,
            eps_type,
            result_container,
            result_tensor,
            debug_info,
        )


register_module(
    "aten._native_batch_norm_legit_no_training.default", BatchNormNoTrainingParser()
)
