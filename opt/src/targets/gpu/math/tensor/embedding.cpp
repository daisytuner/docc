#include "sdfg/targets/gpu/math/tensor/embedding.h"

#include "sdfg/exceptions.h"

namespace sdfg::gpu::tensor {

static constexpr int BLOCK_SIZE = 256;
static constexpr int MAX_GRID_SIZE = 65535;

static types::PrimitiveType connector_type(
    const math::tensor::EmbeddingNode& node, const data_flow::DataFlowGraph& graph, const std::string& connector
) {
    auto* edge = graph.in_edge_for_connector(node, connector);
    if (edge == nullptr) {
        throw InvalidSDFGException("EmbeddingNode: missing connector " + connector);
    }
    return edge->base_type().primitive_type();
}

static symbolic::Expression product(const std::vector<symbolic::Expression>& dims) {
    symbolic::Expression result = symbolic::one();
    for (auto& dim : dims) {
        result = symbolic::mul(result, dim);
    }
    return result;
}

static symbolic::Expression embedding_elements(const math::tensor::EmbeddingNode& node, const std::string& connector) {
    if (connector == "W") {
        return product(node.weight_shape());
    }
    if (connector == "I") {
        return product(node.index_shape());
    }
    return symbolic::mul(product(node.index_shape()), node.weight_shape().at(1));
}

symbolic::Expression embedding_bytes(
    const math::tensor::EmbeddingNode& node, const data_flow::DataFlowGraph& graph, const std::string& connector
) {
    auto element_bytes = types::bit_width(connector_type(node, graph, connector)) / 8;
    return symbolic::mul(embedding_elements(node, connector), symbolic::integer(element_bytes));
}

// Elements are only moved, so an unsigned integer of the same width copies them bit-exactly.
static std::string copy_type(types::PrimitiveType type) {
    switch (types::bit_width(type)) {
        case 8:
            return "uint8_t";
        case 16:
            return "uint16_t";
        case 32:
            return "uint32_t";
        case 64:
            return "uint64_t";
        default:
            throw InvalidSDFGException(
                "GPUEmbeddingDispatcher: unsupported weight type " + std::string(types::primitive_type_to_string(type))
            );
    }
}

static std::string index_type(types::PrimitiveType type) {
    switch (type) {
        case types::PrimitiveType::Int32:
            return "int32_t";
        case types::PrimitiveType::Int64:
            return "int64_t";
        default:
            throw InvalidSDFGException(
                "GPUEmbeddingDispatcher: unsupported index type " + std::string(types::primitive_type_to_string(type))
            );
    }
}

GPUEmbeddingDispatcher::GPUEmbeddingDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const data_flow::LibraryNode& node,
    std::unique_ptr<GPUOffloadDispatcherStrategy> strategy,
    bool with_transfers
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node),
      strategy_(std::move(strategy)), with_transfers_(with_transfers) {
}

void GPUEmbeddingDispatcher::
    dispatch_kernel(codegen::CodegenOutput& out, const std::string& y, const std::string& w, const std::string& i) {
    auto& node = static_cast<const math::tensor::EmbeddingNode&>(node_);
    auto value_t = copy_type(connector_type(node, data_flow_graph_, "W"));
    auto index_t = index_type(connector_type(node, data_flow_graph_, "I"));
    auto name = "embedding_kernel_" + function_.name() + "_" + std::to_string(node_.element_id());

    std::string params = value_t + "* __restrict__ out, const " + value_t + "* __restrict__ weight, const " + index_t +
                         "* __restrict__ indices, long long total, long long dim";
    out.globals_stream << "__global__ void " << name << "(" << params << ");" << std::endl;

    auto& factory = out.library_snippet_factory;
    auto& kernel = factory.require(name, strategy_->kernel_file_extension(), true);
    auto& header = factory.require(name + "_inc", strategy_->kernel_header_file_extension(), true);
    header.stream() << "#include " << factory.header_path().filename() << std::endl;
    strategy_->emit_target_header_declarations(header.stream());

    auto& ks = kernel.stream();
    ks << "#include " << header.filename() << std::endl << std::endl;
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

    auto total = "(long long) (" + language_extension_.expression(embedding_elements(node, "Y")) + ")";
    auto dim = "(long long) (" + language_extension_.expression(node.weight_shape().at(1)) + ")";
    std::vector<std::string> arguments = {
        "(" + value_t + "*) " + y, "(const " + value_t + "*) " + w, "(const " + index_t + "*) " + i, total, dim
    };

    auto& stream = out.stream;
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << "long long __daisy_work_items = " << total << ";" << std::endl;
    stream << "if (__daisy_work_items > 0) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << "long long __daisy_num_blocks = (__daisy_work_items + " << BLOCK_SIZE << " - 1) / " << BLOCK_SIZE << ";"
           << std::endl;
    stream << "if (__daisy_num_blocks > " << MAX_GRID_SIZE << ") {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << "__daisy_num_blocks = " << MAX_GRID_SIZE << ";" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;

    symbolic::Expression num_blocks = symbolic::symbol("__daisy_num_blocks");
    symbolic::Expression grid_y = symbolic::one();
    symbolic::Expression grid_z = symbolic::one();
    symbolic::Expression block_x = symbolic::integer(BLOCK_SIZE);
    symbolic::Expression block_y = symbolic::one();
    symbolic::Expression block_z = symbolic::one();
    strategy_->dispatch_kernel_call(
        stream, name, language_extension_, num_blocks, grid_y, grid_z, block_x, block_y, block_z, arguments
    );

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

void GPUEmbeddingDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& node = static_cast<const math::tensor::EmbeddingNode&>(node_);
    out.library_snippet_factory.add_global("#include " + strategy_->runtime_header());
    out.library_snippet_factory.add_global("#include <cstdint>");

    auto& y = inputs.at(math::tensor::EmbeddingNode::RESULT_PTR_IDX).expr;
    auto& w = inputs.at(math::tensor::EmbeddingNode::W_INPUT_IDX).expr;
    auto& i = inputs.at(math::tensor::EmbeddingNode::INDEX_IDX).expr;
    if (!with_transfers_) {
        dispatch_kernel(out, y, w, i);
        return;
    }

    auto api = strategy_->runtime_api_prefix();
    auto& stream = out.stream;
    auto check = [&]() {
        strategy_->dispatch_runtime_error_check(stream, language_extension_, "__daisy_status");
    };
    auto bytes = [&](const std::string& connector) {
        return "(size_t) (" + language_extension_.expression(embedding_bytes(node, data_flow_graph_, connector)) + ")";
    };

    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << api << "Error_t __daisy_status;" << std::endl;
    for (auto* connector : {"W", "I", "Y"}) {
        stream << "void* __daisy_dev_" << connector << " = nullptr;" << std::endl;
        stream << "__daisy_status = " << api << "Malloc(&__daisy_dev_" << connector << ", " << bytes(connector) << ");"
               << std::endl;
        check();
    }
    // Y is fully overwritten, so only W and I are copied in.
    stream << "__daisy_status = " << api << "Memcpy(__daisy_dev_W, " << w << ", " << bytes("W") << ", " << api
           << "MemcpyHostToDevice);" << std::endl;
    check();
    stream << "__daisy_status = " << api << "Memcpy(__daisy_dev_I, " << i << ", " << bytes("I") << ", " << api
           << "MemcpyHostToDevice);" << std::endl;
    check();

    dispatch_kernel(out, "__daisy_dev_Y", "__daisy_dev_W", "__daisy_dev_I");

    stream << "__daisy_status = " << api << "Memcpy(" << y << ", __daisy_dev_Y, " << bytes("Y") << ", " << api
           << "MemcpyDeviceToHost);" << std::endl;
    check();
    for (auto* connector : {"W", "I", "Y"}) {
        stream << "__daisy_status = " << api << "Free(__daisy_dev_" << connector << ");" << std::endl;
        check();
    }
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

codegen::InstrumentationInfo GPUEmbeddingDispatcher::instrumentation_info() const {
    return {
        node_.element_id(),
        std::string(node_.element_type()) + ":::" + node_.code().value(),
        strategy_->get_instrumentation_kernel_target_type(),
        codegen::InstrumentationEventType::CUDA,
        analysis::LoopInfo{},
        {}
    };
}

} // namespace sdfg::gpu::tensor
