#include "sdfg/targets/rocm/blas/dot.h"

#include <algorithm>
#include <cctype>

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/library_nodes/math/blas/blas_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/dot_node.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/rocm/blas/utils.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_data_offloading_node.h"

namespace sdfg::rocm::blas {

DotNodeDispatcher_ROCMBLASWithTransfers::DotNodeDispatcher_ROCMBLASWithTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::DotNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {
}

void DotNodeDispatcher_ROCMBLASWithTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& dot_node = static_cast<const math::blas::DotNode&>(this->node_);

    library_snippet_factory.add_global("#include <hip/hip_runtime.h>");
    library_snippet_factory.add_global("#include <hipblas/hipblas.h>");

    std::string precision_str(math::blas::BLAS_Precision_to_string(dot_node.precision()));
    std::transform(precision_str.begin(), precision_str.end(), precision_str.begin(), ::toupper);

    types::Scalar base_type(math::blas::BLAS_Precision_to_primitive_type(dot_node.precision()));
    const symbolic::Expression x_size = symbolic::mul(
        symbolic::add(symbolic::mul(symbolic::sub(dot_node.n(), symbolic::one()), dot_node.incx()), symbolic::one()),
        symbolic::size_of_type(base_type)
    );
    const symbolic::Expression y_size = symbolic::mul(
        symbolic::add(symbolic::mul(symbolic::sub(dot_node.n(), symbolic::one()), dot_node.incy()), symbolic::one()),
        symbolic::size_of_type(base_type)
    );

    const std::string* async_alloc_md = this->function_.metadata_if_exists("rocm_async_alloc");
    const bool async_alloc = async_alloc_md != nullptr && *async_alloc_md == "1";

    stream << "hipError_t err_hip;" << std::endl;
    stream << this->language_extension_.primitive_type(base_type.primitive_type()) << " *dx, *dy;" << std::endl;

    dispatch_hip_malloc(stream, this->language_extension_, "dx", x_size, "err_hip", async_alloc);
    dispatch_hip_malloc(stream, this->language_extension_, "dy", y_size, "err_hip", async_alloc);

    dispatch_hip_memcpy(
        stream, this->language_extension_, "dx", "__x", x_size, "hipMemcpyHostToDevice", "err_hip", async_alloc
    );
    dispatch_hip_memcpy(
        stream, this->language_extension_, "dy", "__y", y_size, "hipMemcpyHostToDevice", "err_hip", async_alloc
    );

    setup_blas_handle(library_snippet_factory, this->language_extension_, async_alloc);

    generate_kernel_dot(stream, this->language_extension_, dot_node);

    dispatch_hip_free(stream, this->language_extension_, "dx", "err_hip", async_alloc);
    dispatch_hip_free(stream, this->language_extension_, "dy", "err_hip", async_alloc);
}

codegen::InstrumentationInfo DotNodeDispatcher_ROCMBLASWithTransfers::instrumentation_info() const {
    return {
        node_.element_id(),
        std::string(node_.element_type()) + ":::" + node_.code().value(),
        TargetType_ROCM,
        codegen::InstrumentationEventType::CUDA,
        analysis::LoopInfo{},
        {}
    };
}

DotNodeDispatcher_ROCMBLASWithoutTransfers::DotNodeDispatcher_ROCMBLASWithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::DotNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {
}

void DotNodeDispatcher_ROCMBLASWithoutTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& dot_node = static_cast<const math::blas::DotNode&>(this->node_);
    library_snippet_factory.add_global("#include <hip/hip_runtime.h>");
    library_snippet_factory.add_global("#include <hipblas/hipblas.h>");

    const std::string* async_alloc_md = this->function_.metadata_if_exists("rocm_async_alloc");
    const bool async_alloc = async_alloc_md != nullptr && *async_alloc_md == "1";

    setup_blas_handle(library_snippet_factory, this->language_extension_, async_alloc);

    generate_kernel_dot(stream, this->language_extension_, dot_node);
}

codegen::InstrumentationInfo DotNodeDispatcher_ROCMBLASWithoutTransfers::instrumentation_info() const {
    return {
        node_.element_id(),
        std::string(node_.element_type()) + ":::" + node_.code().value(),
        TargetType_ROCM,
        codegen::InstrumentationEventType::CUDA,
        analysis::LoopInfo{},
        {}
    };
}

void generate_kernel_dot(
    codegen::PrettyPrinter& stream, codegen::LanguageExtension& language_extension, const math::blas::DotNode& dot_node
) {
    std::string precision_str(math::blas::BLAS_Precision_to_string(dot_node.precision()));
    std::transform(precision_str.begin(), precision_str.end(), precision_str.begin(), ::toupper);

    const std::string prefix = dot_node.implementation_type() == rocm::ImplementationType_ROCMWithTransfers ? "d"
                                                                                                            : "__";
    const std::string x_arg = prefix + "x";
    const std::string y_arg = prefix + "y";

    stream << "hipblasStatus_t err;" << std::endl
           << "err = hipblas" << precision_str << "dot(handle, " << language_extension.expression(dot_node.n()) << ", "
           << x_arg << ", " << language_extension.expression(dot_node.incx()) << ", " << y_arg << ", "
           << language_extension.expression(dot_node.incy()) << ", &__out);" << std::endl;
    rocmblas_error_checking(stream, language_extension, "err");
    check_rocm_kernel_launch_errors(stream, language_extension);
}

} // namespace sdfg::rocm::blas
