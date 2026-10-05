#include "sdfg/targets/gpu/math/tensor/embedding.h"

namespace sdfg::gpu::tensor {

void GPUEmbeddingDispatcher::dispatch_device(
    codegen::CodegenOutput& out,
    const std::vector<TensorOperand>& operands,
    const std::unordered_map<std::string, std::string>& device_ptrs
) {
    auto& node = static_cast<const math::tensor::EmbeddingNode&>(node_);
    auto& output = find_operand(operands, "Y");
    auto value_t = copy_type(find_operand(operands, "W").element_type);
    auto index_t = index_type(find_operand(operands, "I").element_type);
    auto name = kernel_name("embedding_kernel");

    std::string params = value_t + "* __restrict__ out, const " + value_t + "* __restrict__ weight, const " + index_t +
                         "* __restrict__ indices, long long total, long long dim";
    out.globals_stream << "__global__ void " << name << "(" << params << ");" << std::endl;

    auto& ks = require_kernel_source(out, name);
    ks << "__global__ void " << name << "(" << params << ") {" << std::endl;
    ks.setIndent(ks.indent() + 4);
    ks << "for (long long e = (long long) blockIdx.x * blockDim.x + threadIdx.x; e < total; "
          "e += (long long) gridDim.x * blockDim.x) {"
       << std::endl;
    ks.setIndent(ks.indent() + 4);
    ks << "long long row = e / dim;" << std::endl;
    ks << "long long col = e - row * dim;" << std::endl;
    ks << "out[e] = weight[(long long) indices[row] * dim + col];" << std::endl;
    ks.setIndent(ks.indent() - 4);
    ks << "}" << std::endl;
    ks.setIndent(ks.indent() - 4);
    ks << "}" << std::endl;

    auto total = "(long long) (" + language_extension_.expression(output.num_elements) + ")";
    auto dim = "(long long) (" + language_extension_.expression(node.weight_shape().at(1)) + ")";
    dispatch_launch(
        out,
        name,
        total,
        {"(" + value_t + "*) " + device_ptrs.at("Y"),
         "(const " + value_t + "*) " + device_ptrs.at("W"),
         "(const " + index_t + "*) " + device_ptrs.at("I"),
         total,
         dim}
    );
}

} // namespace sdfg::gpu::tensor
