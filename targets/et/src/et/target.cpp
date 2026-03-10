#include <docc/target/et/et_lib_node_mapper.h>


#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/data_flow/library_nodes/math/blas/gemm_node.h>
#include <sdfg/passes/targets/target_mapping_pass.h>
#include <sdfg/plugins/plugins.h>

#include "docc/target/et/target.h"

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
            return std::make_unique<blas::GEMMNodeDispatcher_ETSOC_WithTransfers>(
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
    DEBUG_PRINTLN("Running etsoc passes...");
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


std::filesystem::path et_build_kernel(
    const StructuredSDFG& sdfg,
    const codegen::CodeSnippetFactory& snippet_factory,
    const std::filesystem::path& kernel_src,
    const EtBuildArgs& paths
) {
    auto compiler = ET_INSTALL_PATH / "bin" / "riscv64-unknown-elf-g++";
    auto src_file_name = kernel_src.filename().string();
    auto file_name_end = src_file_name.rfind(ETSOC_KERNEL_FILE_EXT);
    if (file_name_end == std::string::npos) {
        throw std::runtime_error("ET kernel source file must end with .et.c: " + kernel_src.string());
    }
    auto elf_file_name = src_file_name.replace(file_name_end, ETSOC_KERNEL_FILE_EXT.size(), "elf");
    auto bin_file = paths.build_dir / elf_file_name;

    auto dev_dir = paths.plugin_rt_dir / "libexec" / "docc" / "et" / "etsoc1-dev";

    std::stringstream cmd;
    cmd << compiler.string() << " ";
    cmd << " --specs=nano.specs -mcmodel=medany -march=rv64imfc -mabi=lp64f -mno-strict-align -mno-riscv-attribute";
    cmd << " -fstack-usage -Wall -Wextra -Wdouble-promotion -Wformat -Wnull-dereference -Wswitch-enum -Wshadow";
    cmd << " -Wstack-protector -Wpointer-arith -Wundef -Wbad-function-cast -Wcast-qual -Wcast-align -Wconversion";
    cmd << " -Wlogical-op -Wstrict-prototypes -Wmissing-prototypes -Wmissing-declarations -Wno-main";
    cmd << " -I " << (dev_dir / "include");
    cmd << " -isystem " << (ET_INSTALL_PATH / "cm-umode/include") << " -isystem "
        << (ET_INSTALL_PATH / "include/esperanto");
    cmd << " -O3 -DNDEBUG -flto=auto -fno-fat-lto-objects -nostdlib -nostartfiles -Wl,--gc-sections -T "
        << (dev_dir / "sections.ld");
    cmd << " " << kernel_src << " " << (dev_dir / "crt.S") << " -o " << bin_file;
    cmd << " -lc -lm -lgcc " << (ET_INSTALL_PATH / "cm-umode/lib/libcm-umode.a");

    int ret = std::system(cmd.str().c_str());
    if (ret != 0) {
        throw std::runtime_error("Compilation of ET kernel failed: " + cmd.str());
    }
    return bin_file;
}

} // namespace docc::target::et
