#include "sdfg/targets/gpu/math/tensor/gpu_tensor_dispatcher.h"

#include "sdfg/exceptions.h"

namespace sdfg::gpu::tensor {

std::string copy_type(types::PrimitiveType type) {
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
                "GPU tensor dispatcher: unsupported element type " + std::string(types::primitive_type_to_string(type))
            );
    }
}

std::string index_type(types::PrimitiveType type) {
    switch (type) {
        case types::PrimitiveType::Int32:
            return "int32_t";
        case types::PrimitiveType::Int64:
            return "int64_t";
        default:
            throw InvalidSDFGException(
                "GPU tensor dispatcher: unsupported index type " + std::string(types::primitive_type_to_string(type))
            );
    }
}

GPUTensorNodeDispatcher::GPUTensorNodeDispatcher(
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

std::string GPUTensorNodeDispatcher::kernel_name(const std::string& prefix) const {
    return prefix + "_" + function_.name() + "_" + std::to_string(node_.element_id());
}

codegen::PrettyPrinter& GPUTensorNodeDispatcher::
    require_kernel_source(codegen::CodegenOutput& out, const std::string& kernel_name) {
    auto& factory = out.library_snippet_factory;
    auto& kernel = factory.require(kernel_name, strategy_->kernel_file_extension(), true);
    auto& header = factory.require(kernel_name + "_inc", strategy_->kernel_header_file_extension(), true);

    header.stream() << "#include " << factory.header_path().filename() << std::endl;
    strategy_->emit_target_header_declarations(header.stream());

    kernel.stream() << "#include " << header.filename() << std::endl << std::endl;
    return kernel.stream();
}

void GPUTensorNodeDispatcher::dispatch_launch(
    codegen::CodegenOutput& out,
    const std::string& kernel_name,
    const std::string& num_work_items,
    std::vector<std::string> arguments
) {
    auto& stream = out.stream;
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << "long long __daisy_work_items = (long long) (" << num_work_items << ");" << std::endl;
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
    symbolic::Expression block_size = symbolic::integer(BLOCK_SIZE);
    symbolic::Expression one_y = symbolic::one();
    symbolic::Expression one_z = symbolic::one();
    symbolic::Expression block_y = symbolic::one();
    symbolic::Expression block_z = symbolic::one();
    strategy_->dispatch_kernel_call(
        stream, kernel_name, language_extension_, num_blocks, one_y, one_z, block_size, block_y, block_z, arguments
    );

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

void GPUTensorNodeDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto operands = tensor_operands(node_, data_flow_graph_);
    if (!operands) {
        throw InvalidSDFGException("GPU tensor dispatcher: no tensor operands for " + node_.code().value());
    }

    out.library_snippet_factory.add_global("#include " + strategy_->runtime_header());
    out.library_snippet_factory.add_global("#include <cstdint>");

    std::unordered_map<std::string, std::string> pointers;
    for (size_t i = 0; i < inputs.size(); ++i) {
        pointers.emplace(node_.inputs().at(i), inputs.at(i).expr);
    }

    if (!with_transfers_) {
        dispatch_device(out, *operands, pointers);
        return;
    }

    auto api = strategy_->runtime_api_prefix();
    auto& stream = out.stream;
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << api << "Error_t __daisy_status;" << std::endl;

    auto buffers = tensor_buffers(*operands, pointers);
    std::unordered_map<std::string, std::string> device_ptrs;
    for (size_t i = 0; i < buffers.size(); ++i) {
        auto& buffer = buffers[i];
        auto device_ptr = "__daisy_tensor_dev_" + std::to_string(i);
        auto bytes = "(size_t) (" + language_extension_.expression(buffer.num_bytes) + ")";
        stream << "void* " << device_ptr << " = nullptr;" << std::endl;
        stream << "__daisy_status = " << api << "Malloc(&" << device_ptr << ", " << bytes << ");" << std::endl;
        strategy_->dispatch_runtime_error_check(stream, language_extension_, "__daisy_status");
        if (buffer.copy_to_device) {
            stream << "__daisy_status = " << api << "Memcpy(" << device_ptr << ", " << buffer.host << ", " << bytes
                   << ", " << api << "MemcpyHostToDevice);" << std::endl;
            strategy_->dispatch_runtime_error_check(stream, language_extension_, "__daisy_status");
        }
        for (auto& connector : buffer.connectors) {
            device_ptrs.emplace(connector, device_ptr);
        }
    }

    dispatch_device(out, *operands, device_ptrs);

    for (size_t i = 0; i < buffers.size(); ++i) {
        auto& buffer = buffers[i];
        auto device_ptr = "__daisy_tensor_dev_" + std::to_string(i);
        if (buffer.copy_to_host) {
            auto bytes = "(size_t) (" + language_extension_.expression(buffer.num_bytes) + ")";
            stream << "__daisy_status = " << api << "Memcpy(" << buffer.host << ", " << device_ptr << ", " << bytes
                   << ", " << api << "MemcpyDeviceToHost);" << std::endl;
            strategy_->dispatch_runtime_error_check(stream, language_extension_, "__daisy_status");
        }
        stream << "__daisy_status = " << api << "Free(" << device_ptr << ");" << std::endl;
        strategy_->dispatch_runtime_error_check(stream, language_extension_, "__daisy_status");
    }

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

codegen::InstrumentationInfo GPUTensorNodeDispatcher::instrumentation_info() const {
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
