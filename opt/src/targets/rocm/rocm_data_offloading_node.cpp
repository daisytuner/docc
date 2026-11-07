#include "sdfg/targets/rocm/rocm_data_offloading_node.h"

#include <cstddef>
#include <memory>

#include <nlohmann/json_fwd.hpp>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/function.h"
#include "sdfg/graph/graph.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/targets/rocm/rocm.h"

namespace sdfg {
namespace rocm {

ROCMDataOffloadingNode::ROCMDataOffloadingNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    offloading::DataTransferDirection transfer_direction,
    offloading::BufferLifecycle buffer_lifecycle,
    symbolic::Expression size,
    symbolic::Expression device_id
)
    : offloading::DataOffloadingNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_ROCM_Offloading,
          transfer_direction,
          buffer_lifecycle,
          size
      ),
      device_id_(device_id) {
}

void ROCMDataOffloadingNode::validate(const Function& function) const {
    // Prevent copy-in and free
    if (this->is_h2d() && this->is_free()) {
        throw InvalidSDFGException("ROCMDataOffloadingNode: Combination copy-in and free is not allowed");
    }

    // Prevent copy-out and alloc
    if (this->is_d2h() && this->is_alloc()) {
        throw InvalidSDFGException("ROCMDataOffloadingNode: Combination copy-out and alloc is not allowed");
    }
}

const symbolic::Expression ROCMDataOffloadingNode::device_id() const {
    return this->device_id_;
}

std::unique_ptr<data_flow::DataFlowNode> ROCMDataOffloadingNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<ROCMDataOffloadingNode>(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        this->transfer_direction(),
        this->buffer_lifecycle(),
        this->size(),
        this->device_id()
    );
}

symbolic::SymbolSet ROCMDataOffloadingNode::symbols() const {
    if (this->device_id().is_null()) {
        return offloading::DataOffloadingNode::symbols();
    }
    auto symbols = offloading::DataOffloadingNode::symbols();
    auto device_id_atoms = symbolic::atoms(this->device_id());
    symbols.insert(device_id_atoms.begin(), device_id_atoms.end());
    return symbols;
}

void ROCMDataOffloadingNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    offloading::DataOffloadingNode::replace(old_expression, new_expression);
    this->device_id_ = symbolic::subs(this->device_id_, old_expression, new_expression);
}

bool ROCMDataOffloadingNode::blocking() const {
    return true;
}

bool ROCMDataOffloadingNode::redundant_with(const offloading::DataOffloadingNode& other) const {
    if (!offloading::DataOffloadingNode::redundant_with(other)) {
        return false;
    }

    auto& other_node = static_cast<const ROCMDataOffloadingNode&>(other);
    if (!symbolic::null_safe_eq(this->device_id(), other_node.device_id())) {
        return false;
    }

    return true;
}

bool ROCMDataOffloadingNode::equal_with(const offloading::DataOffloadingNode& other) const {
    if (!offloading::DataOffloadingNode::equal_with(other)) {
        return false;
    }

    auto& other_node = static_cast<const ROCMDataOffloadingNode&>(other);
    if (!symbolic::null_safe_eq(this->device_id(), other_node.device_id())) {
        return false;
    }

    return true;
}

bool ROCMDataOffloadingNode::is_same_target(const DataOffloadingNode& other) const {
    return dynamic_cast<const ROCMDataOffloadingNode*>(&other) != nullptr;
    // TODO check device id
}

ROCMDataOffloadingNodeDispatcher::ROCMDataOffloadingNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const data_flow::LibraryNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {
}

void ROCMDataOffloadingNodeDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& offloading_node = static_cast<const ROCMDataOffloadingNode&>(this->node_);

    out.stream << "hipError_t err;" << std::endl;

    // Selected by the "rocm_async_alloc" SDFG metadata (DoccOptions.rocm_async_alloc).
    // Default (absent / "0"): synchronous hipMalloc/hipFree. "1": stream-ordered
    // hipMallocAsync/hipFreeAsync on __daisy_stream.
    const std::string* async_alloc_md = this->function_.metadata_if_exists("rocm_async_alloc");
    const bool async_alloc = async_alloc_md != nullptr && *async_alloc_md == "1";

    std::string dev_ptr;

    if (offloading_node.is_alloc()) {
        auto& ptr = outputs.at(0);
        pre_allocate_output(out, ptr, offloading_node.output(0));
        if (async_alloc) {
            const std::string scratch = "__daisy_scratch_" + std::to_string(offloading_node.element_id());
            const std::string scratch_size = scratch + "_size";
            const std::string size_str = this->language_extension_.expression(offloading_node.size());
            out.stream << "static void *" << scratch << " = nullptr;" << std::endl
                       << "static int64_t " << scratch_size << " = 0;" << std::endl
                       << "if (" << scratch_size << " < " << size_str << ") {" << std::endl;
            out.stream.changeIndent(+4);
            out.stream << "if (" << scratch << " != nullptr) {" << std::endl;
            out.stream.changeIndent(+4);
            dispatch_hip_free(out.stream, this->language_extension_, scratch, "err", true);
            out.stream.changeIndent(-4);
            out.stream << "}" << std::endl;
            dispatch_hip_malloc(out.stream, this->language_extension_, scratch, offloading_node.size(), "err", true);
            out.stream << scratch_size << " = " << size_str << ";" << std::endl;
            out.stream.changeIndent(-4);
            out.stream << "}" << std::endl
                       << *ptr.local_name << " = (decltype(" << *ptr.local_name << "))" << scratch << ";" << std::endl;
        } else {
            dispatch_hip_malloc(out.stream, this->language_extension_, *ptr.local_name, offloading_node.size(), "err");
        }
        dev_ptr = *ptr.local_name;
    } else {
        dev_ptr = inputs.at(offloading_node.dev_ptr_input_idx()).expr;
    }

    if (offloading_node.is_h2d()) {
        dispatch_hip_memcpy(
            out.stream,
            this->language_extension_,
            dev_ptr,
            inputs.at(offloading_node.host_ptr_input_idx()).expr,
            offloading_node.size(),
            "hipMemcpyHostToDevice",
            "err",
            async_alloc
        );
    } else if (offloading_node.is_d2h()) {
        dispatch_hip_memcpy(
            out.stream,
            this->language_extension_,
            inputs.at(offloading_node.host_ptr_input_idx()).expr,
            dev_ptr,
            offloading_node.size(),
            "hipMemcpyDeviceToHost",
            "err",
            async_alloc
        );
    }

    if (offloading_node.is_free()) {
        if (async_alloc) {
            out.stream << "// No free in graph capture mode" << std::endl;
        } else {
            dispatch_hip_free(out.stream, this->language_extension_, dev_ptr, "err");
        }
    }
}

codegen::InstrumentationInfo ROCMDataOffloadingNodeDispatcher::instrumentation_info() const {
    auto& rocm_node = static_cast<const ROCMDataOffloadingNode&>(node_);
    if (rocm_node.is_d2h()) {
        return codegen::InstrumentationInfo(
            node_.element_id(),
            std::string(node_.element_type()) + ":::" + node_.code().value(),
            TargetType_ROCM,
            codegen::InstrumentationEventType::NONE,
            analysis::LoopInfo{},
            {{"pcie_bytes", language_extension_.expression(rocm_node.size())}}
        );
    } else if (rocm_node.is_h2d()) {
        return codegen::InstrumentationInfo(
            node_.element_id(),
            std::string(node_.element_type()) + ":::" + node_.code().value(),
            TargetType_ROCM,
            codegen::InstrumentationEventType::NONE,
            analysis::LoopInfo{},
            {{"pcie_bytes", language_extension_.expression(rocm_node.size())}}
        );
    } else {
        return codegen::InstrumentationInfo(
            node_.element_id(),
            std::string(node_.element_type()) + ":::" + node_.code().value(),
            TargetType_ROCM,
            codegen::InstrumentationEventType::NONE,
            analysis::LoopInfo{},
            {}
        );
    }
}

nlohmann::json ROCMDataOffloadingNodeSerializer::serialize(const sdfg::data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const ROCMDataOffloadingNode&>(library_node);
    auto j = offloading::DataOffloadingNodeSerializer::serialize(library_node);

    // Offloading node properties
    serializer::JSONSerializer serializer;
    j["device_id"] = serializer.expression(node.device_id());

    return j;
}

data_flow::LibraryNode& ROCMDataOffloadingNodeSerializer::deserialize(
    const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_ROCM_Offloading.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    SymEngine::Expression device_id(j.at("device_id"));

    return offloading::DataOffloadingNodeSerializer::deserialize_generic_offload<
        ROCMDataOffloadingNode>(j, builder, parent, device_id);
}

void dispatch_hip_malloc(
    codegen::PrettyPrinter& stream,
    codegen::LanguageExtension& language_extension,
    const std::string& var_name,
    const symbolic::Expression& size,
    const std::string& err_var_name,
    bool async
) {
    stream << err_var_name << " = hipMalloc";
    if (async) {
        stream << "Async";
    }
    stream << "(&" << var_name << ", " << language_extension.expression(size);
    if (async) {
        stream << ", __daisy_stream";
    }
    stream << ");" << std::endl;
    rocm_error_checking(stream, language_extension, err_var_name);
}

void dispatch_hip_free(
    codegen::PrettyPrinter& stream,
    codegen::LanguageExtension& language_extension,
    const std::string& var_name,
    const std::string& err_var_name,
    bool async
) {
    stream << err_var_name << " = hipFree";
    if (async) {
        stream << "Async";
    }
    stream << "(" << var_name;
    if (async) {
        stream << ", __daisy_stream";
    }
    stream << ");" << std::endl;
    rocm_error_checking(stream, language_extension, err_var_name);
}

void dispatch_hip_memcpy(
    codegen::PrettyPrinter& stream,
    codegen::LanguageExtension& language_extension,
    const std::string& src_var_name,
    const std::string& dst_var_name,
    const symbolic::Expression& size,
    const std::string& direction,
    const std::string& err_var_name,
    bool async
) {
    stream << err_var_name << " = hipMemcpy";
    if (async) {
        stream << "Async";
    }
    stream << "(" << src_var_name << ", " << dst_var_name << ", " << language_extension.expression(size) << ", "
           << direction;
    if (async) {
        stream << ", __daisy_stream";
    }
    stream << ");" << std::endl;
    rocm_error_checking(stream, language_extension, err_var_name);
}

} // namespace rocm
} // namespace sdfg
