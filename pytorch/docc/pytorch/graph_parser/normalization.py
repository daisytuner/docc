"""
GraphParser modules for parsing normalization layers.
"""

import torch.fx

from docc.sdfg import StructuredSDFGBuilder, Tensor, DebugInfo

from docc.pytorch.graph_parser.utils import (
    TensorInfo,
    TensorConstant,
    TensorMetadata,
    GraphParserModule,
    GraphParserError,
    register_module,
)


class BatchNormNoTrainingParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
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

        input_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        if node.args[1] is None:
            raise GraphParserError(
                self, node, "Currently the weight argument is required but got None"
            )
        weight_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 1)
        if node.args[2] is None:
            raise GraphParserError(
                self, node, "Currently the bias argument is required but got None"
            )
        bias_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 2)
        running_mean_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 3)
        running_var_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 4)
        # We just ignore momentum for now (argument 5)
        eps_info_or_const: TensorInfo | TensorConstant = (
            self.get_arg_tensor_info_or_constant(
                node, metadata, 6, align_constant_type=input_info.element_type()
            )
        )
        result_infos: tuple[TensorInfo, ...] = self.get_result_tensor_infos(
            3, node, builder, metadata
        )
        result_info: TensorInfo = result_infos[0]
        debug_info: DebugInfo = self.get_debug_info(node)

        builder.add_batchnorm_with_bias(
            input_info.container(),
            input_info.sdfg_tensor_type(),
            running_var_info.container(),
            running_var_info.sdfg_tensor_type(),
            running_mean_info.container(),
            running_mean_info.sdfg_tensor_type(),
            weight_info.container(),
            weight_info.sdfg_tensor_type(),
            bias_info.container(),
            bias_info.sdfg_tensor_type(),
            eps_info_or_const.container(),
            eps_info_or_const.element_type(),
            result_info.container(),
            result_info.sdfg_tensor_type(),
            debug_info,
        )


register_module(
    "aten._native_batch_norm_legit_no_training.default", BatchNormNoTrainingParser()
)


class LayerNormParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
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

        input_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)

        # normalized_shape is a list of ints; its length is the number of trailing
        # dimensions that are normalized over.
        normalized_shape: list[str] = self.get_arg_multi_expr(node, 1)

        # weight (Gamma) is optional: elementwise_affine=False -> None
        if node.args[2] is None:
            weight_info: TensorInfo | None = None
        else:
            weight_info: TensorInfo | None = self.get_arg_tensor_info(node, metadata, 2)

        # bias (Beta) is optional: bias=False or elementwise_affine=False -> None
        if node.args[3] is None:
            bias_info: TensorInfo | None = None
        else:
            bias_info: TensorInfo | None = self.get_arg_tensor_info(node, metadata, 3)

        eps_info_or_const: TensorInfo | TensorConstant = (
            self.get_arg_tensor_info_or_constant(
                node, metadata, 4, align_constant_type=input_info.element_type()
            )
        )

        # native_layer_norm returns a tuple of three results: (output, mean, rstd).
        result_infos: tuple[TensorInfo, ...] = self.get_result_tensor_infos(
            3, node, builder, metadata
        )
        output_info: TensorInfo = result_infos[0]
        mean_info: TensorInfo = result_infos[1]
        rstd_info: TensorInfo = result_infos[2]
        mean_tensor: Tensor = mean_info.sdfg_tensor_type()
        rstd_tensor: Tensor = rstd_info.sdfg_tensor_type()
        dims: int = len(mean_tensor.shape)
        for i in range(dims - 1, dims - len(normalized_shape) - 1, -1):
            mean_tensor: Tensor = mean_tensor.squeeze(i)
            rstd_tensor: Tensor = rstd_tensor.squeeze(i)
        if mean_tensor.shape == []:
            mean_tensor: Tensor = mean_tensor.unsqueeze(0)
            rstd_tensor: Tensor = rstd_tensor.unsqueeze(0)

        debug_info: DebugInfo = self.get_debug_info(node)
        if weight_info is None:
            if bias_info is None:
                builder.add_layernorm(
                    input_info.container(),
                    input_info.sdfg_tensor_type(),
                    eps_info_or_const.container(),
                    eps_info_or_const.element_type(),
                    output_info.container(),
                    output_info.sdfg_tensor_type(),
                    mean_info.container(),
                    mean_tensor,
                    rstd_info.container(),
                    rstd_tensor,
                    normalized_shape,
                    debug_info,
                )
            else:
                raise GraphParserError(self, node, "weight is None but bias is set")
        else:
            if bias_info is None:
                builder.add_layernorm_affine(
                    input_info.container(),
                    input_info.sdfg_tensor_type(),
                    eps_info_or_const.container(),
                    eps_info_or_const.element_type(),
                    weight_info.container(),
                    weight_info.sdfg_tensor_type(),
                    output_info.container(),
                    output_info.sdfg_tensor_type(),
                    mean_info.container(),
                    mean_tensor,
                    rstd_info.container(),
                    rstd_tensor,
                    normalized_shape,
                    debug_info,
                )
            else:
                builder.add_layernorm_affine_with_bias(
                    input_info.container(),
                    input_info.sdfg_tensor_type(),
                    eps_info_or_const.container(),
                    eps_info_or_const.element_type(),
                    weight_info.container(),
                    weight_info.sdfg_tensor_type(),
                    bias_info.container(),
                    bias_info.sdfg_tensor_type(),
                    output_info.container(),
                    output_info.sdfg_tensor_type(),
                    mean_info.container(),
                    mean_tensor,
                    rstd_info.container(),
                    rstd_tensor,
                    normalized_shape,
                    debug_info,
                )


register_module("aten.native_layer_norm.default", LayerNormParser())
