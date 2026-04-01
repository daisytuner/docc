#pragma once

#include "codegen.h"
#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {
namespace tenstorrent {

struct TTKernelConfig {
    bool is_external_file_path;
    std::string kernel;
    std::string core_config_var;
    TTKernelTarget target;
    std::vector<std::string> compile_args;
    std::vector<std::string> run_args;
};

enum class TTMovementKernelType { Read = 0x1, Write = 0x2, Combined = 0x3 };

class TenstorrentMapDispatcher : public codegen::NodeDispatcher {
private:
    structured_control_flow::Map& map_;
    static constexpr bool debug_scalar_args_ = false;
    static constexpr bool dev_profile_ = false;
    static constexpr bool tracy_tags_ = false;

    void emit_kernel_local_decls(codegen::PrettyPrinter& stream, const std::unordered_set<std::string>& locals);

    void emit_inner_kernel(
        codegen::PrettyPrinter& stream,
        codegen::CodeSnippetFactory& snippets,
        const std::string& kernel_id,
        const structured_control_flow::Map& inner_map,
        symbolic::Integer stride
    );

    std::string build_kernel_file_name(const std::string& kernel_id, const std::string& type);

public:
    TenstorrentMapDispatcher(
        codegen::LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Map& node,
        codegen::InstrumentationPlan& instrumentation_plan,
        codegen::ArgCapturePlan& arg_capture_plan
    );

    void generate_movement_kernel(
        TTKernelManagementCodegen& codegen,
        codegen::CodeSnippetFactory& snippets,
        const std::string& kernel_id,
        const std::string& cores_var,
        const symbolic::Expression num_tiles,
        const symbolic::Expression stride,
        const std::vector<std::pair<std::string, std::string>>& sorted_args,
        const std::unordered_map<std::string, analysis::RegionArgument>& arg_meta,
        TTMovementKernelType type
    );

    void generate_combined_kernel(
        TTKernelManagementCodegen& codegen,
        codegen::CodeSnippetFactory& snippets,
        const structured_control_flow::Map& inner_map,
        const std::string& kernel_id,
        const std::vector<std::tuple<std::string, std::string>>& scalar_args,
        const std::unordered_set<std::string>& locals
    );

    TTKernelConfig generate_compute_kernel(
        codegen::CodeSnippetFactory& snippets,
        const std::string& kernel_id,
        const std::string& cores_var,
        const symbolic::Expression bound,
        const symbolic::Expression stride,
        const std::vector<std::pair<std::string, std::string>>& mem_args,
        const std::vector<std::pair<std::string, std::string>>& compute_args,
        const std::unordered_map<std::string, analysis::RegionArgument>& arg_meta
    );
    void add_with_cast_if_needed(std::vector<std::tuple<std::string, std::string>>& mem_args, std::string sym_name);

    bool begin_node(codegen::PrettyPrinter& stream) override;

    void end_node(codegen::PrettyPrinter& stream, bool has_declaration) override;

    void dispatch_node(
        codegen::PrettyPrinter& main_stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

    codegen::InstrumentationInfo instrumentation_info() const override;
};

} // namespace tenstorrent
} // namespace sdfg
