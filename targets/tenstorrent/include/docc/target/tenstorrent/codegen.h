#pragma once

#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg::tenstorrent {

enum class TTKernelTarget { DatMovRd, DatMovWr, Compute };

void emit_tt_device_ready(
    const StructuredSDFG& sdfg,
    codegen::PrettyPrinter& main_stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory,
    const std::string& handle,
    int device_id
);

void emit_tilized_padded_copy_helper(
    const codegen::LanguageExtension& language_extension,
    codegen::PrettyPrinter& stream,
    codegen::CodeSnippetFactory& code_snippet_factory
);

void emit_untilized_unpadded_copy_helper(
    const codegen::LanguageExtension& language_extension,
    codegen::PrettyPrinter& stream,
    codegen::CodeSnippetFactory& code_snippet_factory
);

void emit_h2d_transfer_helper_once(
    const codegen::LanguageExtension& language_extension,
    codegen::PrettyPrinter& stream,
    codegen::CodeSnippetFactory& code_snippet_factory
);

void emit_d2h_transfer_helper_once(
    const codegen::LanguageExtension& language_extension,
    codegen::PrettyPrinter& stream,
    codegen::CodeSnippetFactory& code_snippet_factory
);

void emit_tt_includes_once(codegen::PrettyPrinter& stream, codegen::CodeSnippetFactory& code_snippet_factory);

struct TTDataMovementConfig {
    std::string org_name;
    types::PrimitiveType primitive_type;
    std::string device_name;
    symbolic::Expression tile_elems;
    symbolic::Expression num_tiles;
    bool is_input;
    bool is_output;

    TTDataMovementConfig(
        const std::string& org_name,
        const std::string& device_name,
        types::PrimitiveType primitive_type,
        const symbolic::Expression tile_elems,
        const symbolic::Expression num_tiles,
        bool is_input = true,
        bool is_output = false
    )
        : org_name(org_name), device_name(device_name), primitive_type(primitive_type), tile_elems(tile_elems),
          num_tiles(num_tiles), is_input(is_input), is_output(is_output) {}

    std::string get_device_address_var() const;
};


enum class MemArgType {
    CB_ID,
    ADDR,
    NUM_TILES,
};

// Memory argument: refers to a property of a TT memory buffer
struct MemArg {
    int idx;
    MemArgType property;
};

// Literal argument: a constant value, stored as string for flexibility
struct LiteralArg {
    std::string value; // e.g., "1"
};

struct ExprArg {
    symbolic::Expression expr;
};

struct LateArg {
    std::string arg_name;
};

// Argument description: can be either a Memory argument or a Literal
using ArgDesc = std::variant<MemArg, LiteralArg, ExprArg, LateArg>;

struct KernelConfig {
    int id;
    bool is_external_file_path;
    std::string kernel;
    TTKernelTarget target;
    std::vector<ArgDesc> compile_args;
    std::vector<ArgDesc> run_args;
    std::string compile_args_vector_var;
    std::string rt_common_args_vector_var;
    bool mem_addr_in_rt_common = false;
    bool force_32bit_fpu = false;

    KernelConfig(
        int id,
        bool is_external_file_path,
        std::string kernel,
        TTKernelTarget target,
        std::vector<ArgDesc> compile_args,
        std::vector<ArgDesc> run_args
    )
        : id(id), is_external_file_path(is_external_file_path), kernel(kernel), target(target),
          compile_args(compile_args), run_args(run_args) {}

    std::string handle_var(int sub, const std::string& instance_suffix) const;
};

typedef int kernel_id_t;
typedef int kernel_instance_id_t;

class TTKernelManagementCodegen {
private:
    codegen::PrettyPrinter& stream_;
    codegen::CodeSnippetFactory& snippets_;
    codegen::LanguageExtension& language_extension_;
    std::string instance_suffix_;
    std::string dev_handle_var_;

    std::vector<TTDataMovementConfig> managed_memory_buffers_;
    std::vector<std::pair<std::string, std::string>> mem_args_;
    std::vector<std::pair<std::string, std::string>> comp_args_;

    std::string available_cores_var_;
    std::string num_available_cores_var_;
    std::string cores_var_;
    std::string num_used_cores_var_;
    std::string prog_var_;
    std::vector<KernelConfig> kernels_;
    std::optional<std::pair<std::string, std::string>> main_workload_;
    std::optional<std::pair<std::string, std::string>> rem_workload_;

    std::vector<kernel_id_t> instantiations_to_kernel_;

public:
    TTKernelManagementCodegen(
        codegen::PrettyPrinter& stream,
        codegen::CodeSnippetFactory& snippets,
        codegen::LanguageExtension& language_extension,
        const std::string& dev_handle_var,
        const Element& elem,
        const std::vector<TTDataMovementConfig>& managed_memory_buffers,
        const std::optional<std::string> preallocated_program = std::nullopt
    );

    kernel_id_t add_kernel(
        const codegen::CodeSnippet& snippet,
        TTKernelTarget type,
        const std::vector<ArgDesc>& compile_args,
        const std::vector<ArgDesc>& run_args,
        bool force_32bit_fpu = true
    );

    void emit_default_size_distribution(symbolic::Expression work_units);

    std::tuple<std::string, std::string> get_default_distribution_vars() const {
        return {"units_done", "units_on_core"};
    }

    void emit_per_core_config(std::function<void()> per_coord);

    const std::string& get_main_cores() const { return main_workload_.value().first; }
    const std::string& get_units_per_main() const { return main_workload_.value().second; }
    const std::string& get_rem_cores() const { return rem_workload_.value().first; }
    const std::string& get_units_per_rem() const { return rem_workload_.value().second; }
    const std::string& get_used_cores() const { return cores_var_; }
    const std::string& get_available_cores() const { return available_cores_var_; }
    const std::string& get_num_used_cores() const { return num_used_cores_var_; }
    const std::string& get_num_avail_cores() const { return num_available_cores_var_; }

    const TTDataMovementConfig& get_managed_membuf(int idx) const { return managed_memory_buffers_[idx]; }

    int get_num_managed_membufs() const { return managed_memory_buffers_.size(); }

    void for_each_memBuf(std::function<void(int idx, const TTDataMovementConfig& memBuf)> callback) const {
        for (int i = 0; i < managed_memory_buffers_.size(); ++i) {
            callback(i, get_managed_membuf(i));
        }
    }

    void emit_buffer_setup_code();

    kernel_instance_id_t emit_kernel_load(
        kernel_id_t id,
        const std::string& cores,
        const std::unordered_map<std::string, std::string>* late_args,
        bool runtime_args = true
    );
    kernel_instance_id_t emit_kernel_load(
        kernel_id_t id,
        const std::string& cores,
        const std::unordered_map<std::string, std::string>& late_args,
        bool runtime_args = true
    ) {
        return emit_kernel_load(id, cores, &late_args, runtime_args);
    }


    void emit_kernel_set_runtime_args(
        kernel_instance_id_t id, std::string cores, const std::unordered_map<std::string, std::string>* late_args
    ) const;
    void emit_launch(bool blocking);

    void emit_kernel_set_runtime_args(
        kernel_instance_id_t id, std::string cores, const std::unordered_map<std::string, std::string>& late_args
    ) const {
        return emit_kernel_set_runtime_args(id, cores, &late_args);
    }

    void emit_set_runtime_args(const std::string& cores, const std::unordered_map<std::string, std::string>& late_args)
        const;

    void emit_resolved_args(const std::vector<ArgDesc>& args, const std::unordered_map<std::string, std::string>* late_args)
        const;

    void emit_kernel_tensor_addr_args(kernel_id_t i);

    void emit_kernel_set_common_runtime_args(kernel_instance_id_t id) const;
    template<typename T>
    const T& get_num_used_cores();

    const codegen::CodeSnippet& emit_predefined_kernel(const std::string& name, const std::string& code);
};

} // namespace sdfg::tenstorrent
