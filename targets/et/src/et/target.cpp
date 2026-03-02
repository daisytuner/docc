#include <docc/target/et/et_lib_node_mapper.h>


#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/data_flow/library_nodes/math/blas/gemm_node.h>
#include <sdfg/passes/targets/target_mapping_pass.h>
#include <sdfg/plugins/plugins.h>

#include "../../include/docc/target/et/target.h"

#include <sdfg/codegen/docc_paths.h>

#include "docc/target/et/blas/gemm.h"

namespace docc::target::et {

using namespace sdfg;

static std::filesystem::path ET_INSTALL_PATH = "/opt/et";

void register_plugin(plugins::Context& context) {
    auto& libNodeDispatcherRegistry = context.library_node_dispatcher_registry;
    auto& libNodeSerRegistry = context.library_node_serializer_registry;

    libNodeDispatcherRegistry.register_library_node_dispatcher(
        math::blas::LibraryNodeType_GEMM.value() + "::" + ImplementationType_ETSOC_WithTransfers.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<blas::GEMMNodeDispatcher_ETSOC_WithoutTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const sdfg::math::blas::GEMMNode&>(node)
            );
        }
    );
}

void et_scheduling_passes(
    sdfg::builder::StructuredSDFGBuilder& builder,
    sdfg::analysis::AnalysisManager& analysis_manager,
    const std::string& category
) {
    std::vector<std::shared_ptr<plugins::TargetMapper>> mappers{std::make_shared<EtLibNodeMapper>()};
    sdfg::passes::TargetMappingPass mappingPass(mappers);
    mappingPass.run_pass(builder, analysis_manager);
}

std::string
et_get_host_additional_compile_args(const StructuredSDFG& sdfg, const codegen::CodeSnippetFactory& snippet_factory) {
    return "-I" + (ET_INSTALL_PATH / "include").string() + " -I" + (ET_INSTALL_PATH / "include/esperanto").string();
}

std::string
et_get_host_additional_link_args(const StructuredSDFG& sdfg, const codegen::CodeSnippetFactory& snippet_factory) {
    return "-L" + (ET_INSTALL_PATH / "lib").string() + " -Wl,-rpath," + (ET_INSTALL_PATH / "lib").string() +
           " -ldocc-rt-et -ldebug -llogging -lg3log -letrt -ldeviceLayer -lsw-sysemu -lglog";
}

std::string et_build_kernel(
    const StructuredSDFG& sdfg,
    const codegen::CodeSnippetFactory& snippet_factory,
    const std::filesystem::path& kernel_src,
    const codegen::DoccPaths& paths
) {}

} // namespace docc::target::et
