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


class LayerNormNoTrainingParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
    ) -> None:
        if len(node.args) != 5:
            raise GraphParserError(
                self,
                node,
                "Expected exactly 5 arguments but got " + str(len(node.args)),
            )
        if len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )
        input_container: str = self.get_arg_container(node, container_info, 0)
        input_tensor: Tensor = self.get_tensor_type(
            node, container_info, input_container
        )
        # normalized_shape is a list of ints; its length is the number of trailing
        # dimensions that are normalized over.
        num_normalized_dims: int = len(self.get_arg_multi_expr(node, 1))
        # weight (Gamma) is optional: elementwise_affine=False -> None
        if node.args[2] is None:
            weight_container: str = ""
            weight_tensor: Tensor = input_tensor
        else:
            weight_container = self.get_arg_container(node, container_info, 2)
            weight_tensor = self.get_tensor_type(node, container_info, weight_container)
        # bias (Beta) is optional: bias=False or elementwise_affine=False -> None
        if node.args[3] is None:
            bias_container: str = ""
            bias_tensor: Tensor = input_tensor
        else:
            bias_container = self.get_arg_container(node, container_info, 3)
            bias_tensor = self.get_tensor_type(node, container_info, bias_container)
        eps: str | tuple[str, Scalar] = self.get_arg_sdfg_value(node, container_info, 4)
        if isinstance(eps, str):
            eps_container: str = eps
            eps_type: Scalar = self.get_scalar_type(node, container_info, eps_container)
        else:
            eps_container: str = eps[0]
            eps_type: Scalar = self.align_constant_type(
                node, eps, input_tensor.element_type
            )
        # native_layer_norm returns a tuple of three results: (output, mean, rstd).
        # Only the first (the normalized output) is consumed here; the mean and rstd
        # result containers are requested to satisfy the metadata but left unwritten,
        # analogous to the BatchNorm parser above.
        result_containers: tuple[str, ...] = self.get_result_containers(
            3, node, builder, container_info
        )
        result_container: str = result_containers[0]
        result_tensor: Tensor = self.get_tensor_type(
            node, container_info, result_container
        )
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_layernorm_with_bias(
            input_container,
            input_tensor,
            weight_container,
            weight_tensor,
            bias_container,
            bias_tensor,
            eps_container,
            eps_type,
            result_container,
            result_tensor,
            num_normalized_dims,
            debug_info,
        )


register_module("aten.native_layer_norm.default", LayerNormNoTrainingParser())
