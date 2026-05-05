#include <docc/target/et/et_lib_node_mapper.h>


#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/data_flow/library_nodes/math/blas/gemm_node.h>
#include <sdfg/passes/targets/target_mapping_pass.h>
#include <sdfg/plugins/plugins.h>

#include "docc/target/et/target.h"

#include "docc/compile/src_file_compiler_builder.h"
#include "docc/target/et/blas/gemm.h"

namespace docc::target::et {

using namespace sdfg;

static std::filesystem::path ET_INSTALL_PATH = "/opt/et";

DoccTarget et_target = {
    .short_name = "etsoc",
    .apply_additional_compile_options = [](compile::SrcFileCompilerBuilder& builder) -> bool {
        builder.add_include_path(ET_INSTALL_PATH / "include");
        builder.add_include_path(ET_INSTALL_PATH / "include/esperanto");
        builder.add_library_path(ET_INSTALL_PATH / "lib");
        builder.add_link_option(" -Wl,-rpath," + (ET_INSTALL_PATH / "lib").string());
        builder.add_link_option("-ldocc-rt-et");
        builder.add_link_option("-ldebug");
        builder.add_link_option("-llogging");
        builder.add_link_option("-lg3log");
        builder.add_link_option("-letrt");
        builder.add_link_option("-ldeviceLayer");
        builder.add_link_option("-lsw-sysemu");
        builder.add_link_option("-lglog");

        auto& docc_paths = builder.docc_paths();
        if (auto rt_inc_dir = docc_paths.get_builtin_target_plugin_rt_include_path("et")) {
            builder.add_include_path(rt_inc_dir.value());
        }
        if (auto rt_lib_dir = docc_paths.get_builtin_target_plugin_rt_lib_path("et")) {
            builder.add_library_path(rt_lib_dir.value());
        }

        compile::SrcFileCompilerBuilder etBuilder;
        etBuilder.inherit(builder, false);
        etBuilder.set_src_extension(ETSOC_KERNEL_FILE_EXT);
        etBuilder.set_bin_extension("elf");
        etBuilder.set_compiler(ET_INSTALL_PATH / "bin" / "riscv64-unknown-elf-g++");
        etBuilder.set_link_immediately(true);
        etBuilder.add_common_option("--specs=nano.specs");
        etBuilder.add_common_option("-mcmodel=medany");
        etBuilder.add_common_option("-march=rv64imfc");
        etBuilder.add_common_option("-mabi=lp64f");
        etBuilder.add_common_option("-mno-strict-align");
        etBuilder.add_common_option("-mno-riscv-attribute");
        etBuilder.add_common_option("-fstack-usage");
        etBuilder.add_compile_option("-Wall");
        etBuilder.add_compile_option("-Wextra");
        etBuilder.add_compile_option("-Wdouble-promotion");
        etBuilder.add_compile_option("-Wformat");
        etBuilder.add_compile_option("-Wnull-dereference");
        etBuilder.add_compile_option("-Wswitch-enum");
        etBuilder.add_compile_option("-Wshadow");
        etBuilder.add_compile_option("-Wstack-protector");
        etBuilder.add_compile_option("-Wpointer-arith");
        etBuilder.add_compile_option("-Wundef");
        etBuilder.add_compile_option("-Wcast-qual");
        etBuilder.add_compile_option("-Wcast-align");
        etBuilder.add_compile_option("-Wconversion");
        etBuilder.add_compile_option("-Wlogical-op");
        etBuilder.add_compile_option("-Wmissing-declarations");
        etBuilder.add_compile_option("-Wno-main");

        if (auto libexec_src_dir = docc_paths.get_builtin_target_plugin_rt_libexec_src_path("et", "etsoc1-dev")) {
            auto dir = libexec_src_dir.value();
            etBuilder.add_include_path(dir / "include");
            etBuilder.add_link_option(dir / "crt.S");
            etBuilder.add_link_option("-T " + (dir / "sections.ld").string());
        }

        etBuilder.add_include_path(ET_INSTALL_PATH / "cm-umode" / "include"); // was -isystem
        etBuilder.add_include_path(ET_INSTALL_PATH / "include" / "esperanto"); // was -isystem
        etBuilder.add_compile_option("-O3");
        etBuilder.add_compile_option("-DNDEBUG");
        etBuilder.add_compile_option("-flto=auto");
        etBuilder.add_compile_option("-fno-fat-lto-objects");
        etBuilder.add_link_option("-nostdlib");
        etBuilder.add_link_option("-nostartfiles");
        etBuilder.add_link_option("-Wl,--gc-sections");
        etBuilder.add_link_option("-lc");
        etBuilder.add_link_option("-lm");
        etBuilder.add_link_option("-lgcc");
        etBuilder.add_link_option(ET_INSTALL_PATH / "cm-umode/lib/libcm-umode.a");

        builder.redirect_snippet(ETSOC_KERNEL_FILE_EXT, std::move(etBuilder));
        return true;
    }
};

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

    context.add_target(&et_target);
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
