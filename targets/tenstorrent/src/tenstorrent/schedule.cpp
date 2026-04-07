#include "docc/target/tenstorrent/schedule.h"

#include "docc/target/tenstorrent/codegen.h"
#include "docc/target/tenstorrent/plugin.h"
#include "docc/target/tenstorrent/tenstorrent_offloading_expansion.h"
#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/type_analysis.h"
#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"

bool docc_debug_tt = true;

#define DOCC_DEBUG(X) \
    if (docc_debug_tt) X

namespace sdfg {
namespace tenstorrent {

static constexpr const char* TT_FIRST_UNIT = "first_unit";
static constexpr const char* TT_WORK_UNITS = "work_units";
static constexpr const char* TT_IDX_WITHIN_BLOCK = "tt_idx_within_block";

TenstorrentMapDispatcher::TenstorrentMapDispatcher(
    codegen::LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    sdfg::analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Map& node,
    codegen::InstrumentationPlan& instrumentation_plan,
    codegen::ArgCapturePlan& arg_capture_plan
)
    : codegen::NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan),
      map_(node) {

      };
void emit_tt_create_kernel(
    codegen::PrettyPrinter& out,
    std::string p_name,
    std::string core_spec,
    std::string instance_suffix,
    TTKernelConfig kernel_config
) {
    if (kernel_config.is_external_file_path) {
        std::string k_handle;
        switch (kernel_config.target) {
            case TTKernelTarget::DatMovRd:
                k_handle = "daisy_tt_movRd_kid_" + instance_suffix;
                break;
            case TTKernelTarget::DatMovWr:
                k_handle = "daisy_tt_movWr_kid_" + instance_suffix;
                break;
            case TTKernelTarget::Compute:
                k_handle = "daisy_tt_comp_kid_" + instance_suffix;
                break;
            default:
                throw std::runtime_error("Invalid kernel target " + std::to_string(static_cast<int>(kernel_config.target)));
        }

        out << "auto " << k_handle << " = tt::tt_metal::CreateKernel(" << p_name << ", "
            << "\"" << kernel_config.kernel << "\", " << kernel_config.core_config_var << ", ";

        switch (kernel_config.target) {
            case TTKernelTarget::DatMovRd:
                out << "tt::tt_metal::ReaderDataMovementConfig";
                out << "(";
                out << "{";
                out << helpers::join(kernel_config.compile_args, ", ");
                out << "}, {})";
                break;
            case TTKernelTarget::DatMovWr:
                out << "tt::tt_metal::WriterDataMovementConfig";
                out << "(";
                out << "{";
                out << helpers::join(kernel_config.compile_args, ", ");
                out << "}, {})";
                break;
            case TTKernelTarget::Compute:
                out << "tt::tt_metal::ComputeConfig";
                out << " { ";
                out << ".compile_args = {";
                out << helpers::join(kernel_config.compile_args, ", ");
                out << "} }";
                break;
            default:
                throw std::runtime_error("Invalid kernel target " + std::to_string(static_cast<int>(kernel_config.target)));
        }

        out << ");" << std::endl;

        out << "tt::tt_metal::SetRuntimeArgs(" << p_name << ", " << k_handle << ", " << core_spec << ", ";
        out << "{";
        out << helpers::join(kernel_config.run_args, ", ");
        out << "});" << std::endl;
    } else {
        throw std::runtime_error("Not implemented");
    }
}

std::string build_movement_kernel_addr_name(const std::string& var_name, int arg_idx) {
    auto name = TENSTORRENT_DEVICE_VAR_PREFIX + var_name + "_addr";
    return name;
}

std::string build_movement_kernel_addr_gen_name(const std::string& var_name, int arg_idx) {
    auto name = "__daisy_addr_gen_" + std::to_string(arg_idx);
    return name;
}

bool is_output_allowed(TTMovementKernelType type) {
    return type == TTMovementKernelType::Write || type == TTMovementKernelType::Combined;
}

bool is_input_allowed(TTMovementKernelType type) {
    return type == TTMovementKernelType::Read || type == TTMovementKernelType::Combined;
}


void TenstorrentMapDispatcher::generate_movement_kernel(
    TTKernelManagementCodegen& codegen,
    codegen::CodeSnippetFactory& snippets,
    const std::string& kernel_id,
    const std::string& cores_var,
    const symbolic::Expression num_tiles,
    const symbolic::Expression stride,
    const std::vector<std::pair<std::string, std::string>>& sorted_args,
    const std::unordered_map<std::string, analysis::RegionArgument>& arg_meta,
    TTMovementKernelType type
) {
    auto filename = "tenstorrent_kernel_" + kernel_id + (type == TTMovementKernelType::Read ? ".movRd" : ".movWr");
    auto& snippet = snippets.require(filename, "cpp", true);
    auto& stream = snippet.stream();

    std::vector<ArgDesc> selected_args;

    stream << "#include <cstdint>" << std::endl;
    stream << "#include <dataflow_api.h>" << std::endl;
    stream << "#include <debug/dprint.h>" << std::endl;
    stream << std::endl;

    stream << "void kernel_main() {" << std::endl;
    stream.setIndent(4);

    TTKernelConfig kernel_config{
        true,
        snippets.output_path() / (filename + ".cpp"),
        cores_var,
        type == TTMovementKernelType::Read ? TTKernelTarget::DatMovRd : TTKernelTarget::DatMovWr
    };

    int mem_arg_idx = 0;
    int arg_idx = 0;
    for (const auto& [arg_name, org] : sorted_args) {
        bool was_used = false;
        auto& meta = arg_meta.at(org);
        if (meta.is_ptr) {
            auto name = build_movement_kernel_addr_name(org, arg_idx);
            stream << "uint32_t " << name << " = get_arg_val<uint32_t>(" << arg_idx << "); // " << org << std::endl;

            uint8_t todo = (meta.is_output ? 0x2 : 0) | (meta.is_input ? 0x1 : 0);
            for (uint8_t mask = 0x1; mask <= 0x2; mask <<= 1) {
                if (todo & mask) {
                    if (mask & static_cast<uint8_t>(type)) {
                        auto addr_gen_name = build_movement_kernel_addr_gen_name(org, mem_arg_idx);
                        stream << "const InterleavedAddrGenFast<true> " << addr_gen_name << " = {" << std::endl;
                        stream << "\t.bank_base_address = " << name << "," << std::endl;
                        stream << "\t.page_size = get_tile_size(" << mem_arg_idx << ")," << std::endl;
                        stream << "\t.data_format = get_dataformat(" << mem_arg_idx << ")" << std::endl;
                        stream << "};" << std::endl;
                        was_used = true;
                    }
                    ++mem_arg_idx;
                }
            }
        } else if (meta.is_scalar) {
            stream << "uint32_t" << org << " = get_arg_val<uint32_t>(" << mem_arg_idx << ");" << std::endl;
            was_used = true;
        }
        if (was_used) {
            kernel_config.run_args.push_back(arg_name);
        }
        ++arg_idx;
    }

    stream << "DPRINT << \"Hello world from " << (type == TTMovementKernelType::Read ? "movRd" : "movWr") << " tt_"
           << kernel_id << "\" << ENDL();" << std::endl;

    auto& indvar = map_.indvar();
    auto& indvar_name = indvar->get_name();
    auto& indvar_type = sdfg_.type(indvar_name);
    stream << language_extension_.declaration(indvar_name, indvar_type) << ";" << std::endl;

    stream << std::endl;
    stream << "uint32_t tile_idx = 0;" << std::endl;
    int pipeline_depth = 0;
    stream << std::endl;

    // push 1 block of read-data so compute can process it
    // TODO repeat pipeline_depth times
    // mem_arg_idx = 0;
    // for (const auto& org: sorted_args | std::views::values) {
    //     auto& meta = arg_meta.at(org);
    //     if (meta.is_ptr) {
    //         if (meta.is_input) {
    //             if ( is_input_allowed(type)) {
    //                 auto addr_gen_name = build_movement_kernel_addr_gen_name(org, mem_arg_idx);
    //                 stream << "cb_reserve_back(" << mem_arg_idx << ", 1);" << std::endl;
    //                 stream << "uint32_t l1_write_addr_" << mem_arg_idx << " = get_write_ptr(" << mem_arg_idx << ");"
    //                 << std::endl; stream << "noc_async_read_tile(tile_idx, " << addr_gen_name << ", l1_write_addr_"
    //                 << mem_arg_idx << ");" << std::endl; stream << "noc_async_read_barrier();" << std::endl; stream
    //                 << "cb_push_back(" << mem_arg_idx << ", 1);" << std::endl;
    //             }
    //             ++mem_arg_idx;
    //         }
    //         if (meta.is_output) {
    //             ++mem_arg_idx;
    //         }
    //     }
    // }

    stream << std::endl;

    stream << "for (tile_idx = " << pipeline_depth << "; tile_idx < " << language_extension_.expression(num_tiles)
           << "; ++tile_idx) {" << std::endl;
    stream.setIndent(8);
    symbolic::Expression org_indvar = symbolic::add(symbolic::mul(symbolic::symbol("tile_idx"), stride), map_.init());
    stream << indvar_name << " = " << language_extension_.expression(org_indvar) << ";" << std::endl;

    bool postfix_math = true;

    mem_arg_idx = 0;
    for (const auto& org : sorted_args | std::views::values) {
        auto& meta = arg_meta.at(org);
        if (meta.is_ptr) {
            if (meta.is_input) {
                if (is_input_allowed(type)) {
                    auto addr_gen_name = build_movement_kernel_addr_gen_name(org, mem_arg_idx);
                    stream << "cb_reserve_back(" << mem_arg_idx << ", 1);" << std::endl;
                    stream << "uint32_t l1_write_addr_" << mem_arg_idx << " = get_write_ptr(" << mem_arg_idx << ");"
                           << std::endl;
                    stream << "noc_async_read_tile(tile_idx, " << addr_gen_name << ", l1_write_addr_" << mem_arg_idx
                           << ");" << std::endl;
                }
                ++mem_arg_idx;
            }
            if (meta.is_output) {
                if (is_output_allowed(type)) {
                    auto addr_gen_name = build_movement_kernel_addr_gen_name(org, mem_arg_idx);
                    stream << "cb_wait_front(" << mem_arg_idx << ", 1);" << std::endl;
                    auto tile_addr_var = "l1_read_addr_" + std::to_string(mem_arg_idx);
                    stream << "uint32_t " << tile_addr_var << " = get_read_ptr(" << mem_arg_idx << ");" << std::endl;

                    if (postfix_math) {
                        auto tile_end_addr_var = "tile_end_addr_" + std::to_string(mem_arg_idx);
                        stream << "float* " << tile_end_addr_var << " = reinterpret_cast<float*>(" << tile_addr_var
                               << ") + get_tile_size(" << mem_arg_idx << ") / sizeof(float);" << std::endl;
                        // stream << "for (float* ptr = reinterpret_cast<float*>(" << tile_addr_var << "); ";
                        // stream << "ptr < " << tile_end_addr_var << "; ++ptr) {" << std::endl;
                        // stream.setIndent(12);
                        // stream << "*ptr = *ptr + 2.0f;" << std::endl; // Example operation, replace with actual
                        // computation
                        //
                        // stream.setIndent(8);
                        // stream << "}" << std::endl;
                    }

                    stream << "noc_async_write_tile(tile_idx - " << pipeline_depth << ", " << addr_gen_name
                           << ", l1_read_addr_" << mem_arg_idx << ");" << std::endl;
                }
                ++mem_arg_idx;
            }
        }
    }

    mem_arg_idx = 0;
    for (const auto& org : sorted_args | std::views::values) {
        auto& meta = arg_meta.at(org);
        if (meta.is_ptr) {
            if (meta.is_input) {
                if (is_input_allowed(type)) {
                    stream << "noc_async_read_barrier();" << std::endl;
                    stream << "if (tile_idx == 0) {" << std::endl;
                    stream << "    uint32_t* addr = reinterpret_cast<uint32_t*>(l1_write_addr_0);" << std::endl;
                    stream << "    for (uint32_t i = 0; i < 1024; ++i) {" << std::endl;
                    stream << "        DPRINT << i << \": 0x\" << HEX() << addr[i] << DEC() << ENDL();" << std::endl;
                    stream << "    }" << std::endl;
                    stream << "}" << std::endl;
                    stream << "cb_push_back(" << mem_arg_idx << ", 1);" << std::endl;
                }
                ++mem_arg_idx;
            }
            if (meta.is_output) {
                if (is_output_allowed(type)) {
                    stream << "noc_async_write_barrier();" << std::endl;
                    stream << "cb_pop_front(" << mem_arg_idx << ", 1);" << std::endl;
                }
                ++mem_arg_idx;
            }
        }
    }

    stream.setIndent(4);
    stream << "}" << std::endl;

    stream << std::endl;

    // pull last block of write-data
    // mem_arg_idx = 0;
    // for (const auto& org: sorted_args | std::views::values) {
    //     auto& meta = arg_meta.at(org);
    //     if (meta.is_ptr) {
    //         if (meta.is_input) {
    //             ++mem_arg_idx;
    //         }
    //         if (meta.is_output) {
    //             if (is_output_allowed(type)) {
    //                 auto addr_gen_name = build_movement_kernel_addr_gen_name(org, mem_arg_idx);
    //                 stream << "cb_wait_front(" << mem_arg_idx << ", 1);" << std::endl;
    //                 stream << "uint32_t l1_read_addr_" << mem_arg_idx << " = get_read_ptr(" << mem_arg_idx << ");" <<
    //                 std::endl; stream << "noc_async_write_tile(tile_idx - " << pipeline_depth << ", " <<
    //                 addr_gen_name << ", l1_read_addr_" << mem_arg_idx << ");" << std::endl; stream <<
    //                 "noc_async_write_barrier();" << std::endl; stream << "cb_pop_front(" << mem_arg_idx << ", 1);" <<
    //                 std::endl;
    //             }
    //             ++mem_arg_idx;
    //         }
    //     }
    // }

    stream << std::endl;

    stream.setIndent(0);
    stream << "}";

    auto fw_id = codegen.add_kernel(snippet, TTKernelTarget::DatMovRd, {}, selected_args);

    codegen.emit_kernel_load(fw_id, codegen.get_used_cores(), nullptr, false);
}

std::string TenstorrentMapDispatcher::build_kernel_file_name(const std::string& kernel_id, const std::string& type) {
    std::filesystem::path sdfg_file = this->sdfg_.metadata("sdfg_file");
    auto sdfg_name = sdfg_file.filename().replace_extension().string();
    auto filename = sdfg_name + "_tenstorrent_kernel_" + kernel_id + "." + type;
    return filename;
}

void TenstorrentMapDispatcher::generate_combined_kernel(
    TTKernelManagementCodegen& codegen,
    codegen::CodeSnippetFactory& snippets,
    const structured_control_flow::Map& inner_map,
    const std::string& kernel_id,
    const std::vector<std::tuple<std::string, std::string>>& scalar_args, // [input_via, org_name]
    const std::unordered_set<std::string>& locals
) {
    auto filename = build_kernel_file_name(kernel_id, "combined");
    auto& snippet = snippets.require(filename, "cpp", true);
    auto& stream = snippet.stream();

    stream << "#include <cstdint>" << std::endl;
    stream << "#include <dataflow_api.h>" << std::endl;
    stream << "#include <debug/dprint.h>" << std::endl;
    stream << "#include <tools/profiler/kernel_profiler.hpp>" << std::endl;
    stream << std::endl;
    stream << "#define __daisy_min(a, b) ((a) < (b) ? (a) : (b))" << std::endl;
    stream << "#define __daisy_max(a, b) ((a) > (b) ? (a) : (b))" << std::endl;
    stream << "#define __daisy_fma(a, b, c) a * b + c" << std::endl;
    stream << std::endl;

    stream << "void kernel_main() {" << std::endl;
    stream.changeIndent(+4);

    std::vector<std::tuple<const TTDataMovementConfig&, std::string, std::string, int>> bufs;
    std::vector<ArgDesc> selected_args{LateArg(TT_FIRST_UNIT), LateArg(TT_WORK_UNITS)};

    stream << "uint32_t tt_first_unit = get_arg_val<uint32_t>(0);" << std::endl;
    stream << "uint32_t tt_work_units = get_arg_val<uint32_t>(1);" << std::endl;

    int arg_idx = selected_args.size();

    for (const auto& [use_as, org] : scalar_args) {
        auto& type = sdfg_.type(org);
        stream << language_extension_.declaration(org, type) << " = get_arg_val<"
               << language_extension_.declaration("", type) << ">(" << arg_idx << ");" << std::endl;
        selected_args.emplace_back(LiteralArg{use_as});
        ++arg_idx;
    }

    for (int mem_arg_idx = 0; mem_arg_idx < codegen.get_num_managed_membufs(); ++mem_arg_idx) {
        auto& bufDesc = codegen.get_managed_membuf(mem_arg_idx);

        if (true) { // filter input / output in split kernels

            auto addr_name = "tt_tensor_addr_" + std::to_string(mem_arg_idx);
            auto tensor_name = "tt_tensor_" + std::to_string(mem_arg_idx);
            stream << "uint32_t " << addr_name << " = get_arg_val<uint32_t>(" << arg_idx << "); // base addr of "
                   << bufDesc.device_name << std::endl;

            selected_args.emplace_back(MemArg{mem_arg_idx, MemArgType::ADDR});

            bufs.emplace_back(bufDesc, addr_name, tensor_name, mem_arg_idx);

            arg_idx++;
        }
    }

    stream << std::endl;
    std::string next_comp_arg_idx = "0";
    std::string next_common_rt_arg_idx = "0";

    for (auto& [bufDesc, addr_name, tensor_name, cb_id] : bufs) {
        auto tensor_args_name = "tt_tensor_args_" + std::to_string(cb_id);
        stream << "constexpr auto " << tensor_args_name << " = TensorAccessorArgs<" << next_comp_arg_idx << ", "
               << next_common_rt_arg_idx << ">();" << std::endl;
        stream << "const auto " << tensor_name << " = TensorAccessor(" << tensor_args_name << ", " << addr_name << ", "
               << language_extension_.expression(bufDesc.tile_elems) << "*"
               << types::bit_width(bufDesc.primitive_type) / 8 << ");" << std::endl;

        next_comp_arg_idx = tensor_args_name + ".next_compile_time_args_offset()";
        next_common_rt_arg_idx = tensor_args_name + ".next_common_runtime_args_offset()";
    }

    stream << std::endl;

    emit_kernel_local_decls(stream, locals);

    stream << std::endl;

    stream << "DPRINT << \"Up " << filename << "\" << ENDL();" << std::endl;

    if (debug_scalar_args_) {
        for (const auto& [use_as, org] : scalar_args) {
            stream << "DPRINT << \" scalar arg " << org << " = \" << " << org << " << ENDL();" << std::endl;
        }
    }

    stream << std::endl;

    for (auto& [bufDesc, addr_name, tensor_name, cb_id] : bufs) { // CB make space (we do not actually stream through
                                                                  // them, so this is
        // just scratch space
        stream << "cb_reserve_back(" << cb_id << ", 1);" << std::endl;
    }

    auto& indvar_name = map_.indvar()->get_name();
    auto& indvar_type = sdfg_.type(map_.indvar()->get_name());
    auto stride = map_.stride();

    stream << "uint32_t tt_last_unit = tt_first_unit + tt_work_units;" << std::endl;
    stream << indvar_name << " = tt_first_unit*" << language_extension_.expression(stride) << ";" << std::endl;
    stream << std::endl;

    stream << "for";
    stream << "(uint32_t tt_tile_idx = tt_first_unit; tt_tile_idx < tt_last_unit; ++tt_tile_idx) {" << std::endl;
    stream.changeIndent(+4);
    stream << std::endl;

    for (auto& [bufDesc, addr_name, tensor_name, cb_id] : bufs) { // allocate addresses outside of scope;
        if (bufDesc.is_input) {
            auto ptr_name = "l1_write_addr_" + std::to_string(cb_id);
            stream << "uint32_t " << ptr_name << " = get_write_ptr(" << cb_id << ");" << std::endl;
        }
    }

    stream << std::endl;
    stream << "{" << std::endl;
    stream.changeIndent(+4);
    stream << "DeviceZoneScopedN(\"fetch\")" << std::endl << std::endl;

    for (auto& [bufDesc, addr_name, tensor_name, cb_id] : bufs) { // read data
        if (bufDesc.is_input) { // fill input-buffer with data from DRAM
            auto ptr_name = "l1_write_addr_" + std::to_string(cb_id);
            stream << "noc_async_read_page(tt_tile_idx, " << tensor_name << ", " << ptr_name << ");" << std::endl;
        }
    }

    stream << std::endl;
    stream << "noc_async_read_barrier();" << std::endl;
    stream.changeIndent(-4);
    stream << "}" << std::endl;
    stream << std::endl;

    stream << "{" << std::endl;
    stream.changeIndent(+4);
    stream << "DeviceZoneScopedN(\"compute\")" << std::endl;
    stream << std::endl;

    for (auto& [bufDesc, addr_name, tensor_name, cb_id] : bufs) { // provide ptrs under org-name

        if (bufDesc.is_input) {
            stream << "void* " << bufDesc.device_name << " = reinterpret_cast<void*>(l1_write_addr_" << cb_id << ");"
                   << std::endl;
        } else if (bufDesc.is_output) { //
            stream << "void* " << bufDesc.device_name << " = reinterpret_cast<void*>(get_write_ptr(" << cb_id << "));"
                   << std::endl;
        }
    }

    stream << std::endl;

    // main kernel

    emit_inner_kernel(stream, snippets, kernel_id, inner_map, stride);

    stream.changeIndent(-4);
    stream << "}" << std::endl;

    // end main kernel

    stream << "{" << std::endl;
    stream.changeIndent(+4);
    stream << "DeviceZoneScopedN(\"wb\")" << std::endl;
    stream << std::endl;

    for (auto& [bufDesc, addr_name, tensor_name, cb_id] : bufs) { // write data back to DRAM
        if (bufDesc.is_output) {
            auto ptr_name = "l1_read_addr_" + std::to_string(cb_id);
            stream << "uint32_t " << ptr_name << " = get_write_ptr(" << cb_id << ");" << std::endl;
            stream << "noc_async_write_page(tt_tile_idx, " << tensor_name << ", " << ptr_name << ");" << std::endl;
        }
    }

    stream << std::endl;
    stream << "noc_async_writes_flushed();" << std::endl;

    stream.changeIndent(-4);
    stream << "}" << std::endl;

    stream << indvar_name << " += " << language_extension_.expression(stride) << ";" << std::endl;
    stream.changeIndent(-4);
    stream << "}" << std::endl;

    stream << std::endl;
    stream << "noc_async_write_barrier();" << std::endl;

    stream.changeIndent(-4);
    stream << "}" << std::endl;

    auto fw_id = codegen.add_kernel(snippet, TTKernelTarget::DatMovRd, {}, selected_args);

    codegen.emit_kernel_tensor_addr_args(fw_id);

    auto k_id = codegen.emit_kernel_load(fw_id, codegen.get_used_cores(), nullptr, false);

    codegen.emit_kernel_set_common_runtime_args(k_id);
}

void TenstorrentMapDispatcher::
    emit_kernel_local_decls(codegen::PrettyPrinter& stream, const std::unordered_set<std::string>& locals) {
    for (auto& container : locals) {
        auto& type = sdfg_.type(container);

        //        assert(type.storage_type().value() == StorageType_Tenstorrent_Local && "kernel-local var does not have
        //        TT type");
        // TODO types never set, also not for cuda

        std::string val = this->language_extension_.declaration(container, sdfg_.type(container), false, true);
        if (!val.empty()) {
            stream << val;
            stream << ";" << std::endl;
        }
    }
}

void TenstorrentMapDispatcher::emit_inner_kernel(
    codegen::PrettyPrinter& stream,
    codegen::CodeSnippetFactory& snippets,
    const std::string& kernel_id,
    const Map& inner_map,
    symbolic::Integer stride
) {
    auto& kernel_globals = snippets.require("tt_globals_" + kernel_id, "", false);

    stream << "uint32_t " << TT_IDX_WITHIN_BLOCK << ";" << std::endl;
    stream << std::endl;

    stream << "// Map" << std::endl;
    stream << "for";
    stream << "(" << TT_IDX_WITHIN_BLOCK << " = 0";
    stream << ", ";
    stream << inner_map.indvar()->get_name() << " = " << this->language_extension_.expression(inner_map.init());
    stream << "; ";
    stream << this->language_extension_.expression(inner_map.condition());
    stream << "; ";
    stream << TT_IDX_WITHIN_BLOCK << " += 1";
    stream << ", ";
    stream << inner_map.indvar()->get_name();
    stream << " = ";
    stream << this->language_extension_.expression(inner_map.update());
    stream << ")" << std::endl;
    stream << "{" << std::endl;
    stream.changeIndent(+4);

    auto no_instrument = codegen::InstrumentationPlan::none(this->sdfg_);
    auto no_capture = codegen::ArgCapturePlan::none(this->sdfg_);

    codegen::SequenceDispatcher dispatcher(
        this->language_extension_, this->sdfg_, this->analysis_manager_, inner_map.root(), *no_instrument, *no_capture
    );
    dispatcher.dispatch(stream, kernel_globals.stream(), snippets);

    stream.changeIndent(-4);
    stream << "}" << std::endl;
}

TTKernelConfig TenstorrentMapDispatcher::generate_compute_kernel(
    codegen::CodeSnippetFactory& snippets,
    const std::string& kernel_id,
    const std::string& cores_var,
    const symbolic::Expression num_tiles,
    const symbolic::Expression stride,
    const std::vector<std::pair<std::string, std::string>>& mem_args,
    const std::vector<std::pair<std::string, std::string>>& compute_args,
    const std::unordered_map<std::string, analysis::RegionArgument>& arg_meta
) {
    auto filename = "tenstorrent_kernel_" + kernel_id + ".comp";
    auto& stream = snippets.require(filename, "cpp", true).stream();

    stream << "#include <cstdint>" << std::endl;
    stream << "#include <compute_kernel_api.h>" << std::endl;
    stream << "#include <compute_kernel_api/common.h>" << std::endl;
    stream << "#include <compute_kernel_api/cb_api.h>" << std::endl;
    stream << "#include <compute_kernel_api/pack.h>" << std::endl;
    stream << "#include <compute_kernel_api/tile_move_copy.h>" << std::endl;
    stream << "#include <compute_kernel_api/eltwise_unary/eltwise_unary.h>" << std::endl;
    stream << "#include <debug/dprint.h>" << std::endl;
    stream << std::endl;
    stream << "namespace NAMESPACE {" << std::endl;

    stream << "void MAIN {" << std::endl;
    stream.setIndent(4);

    TTKernelConfig kernel_config{
        true, snippets.output_path() / (filename + ".cpp"), cores_var, TTKernelTarget::Compute
    };

    int arg_idx = 0;
    for (const auto& [outer, org] : compute_args) {
        auto& meta = arg_meta.at(org);
        auto& type = sdfg_.type(org);
        if (meta.is_scalar) {
            stream << language_extension_.declaration(org, type) << " = get_arg_val<"
                   << language_extension_.declaration("", type) << ">(" << arg_idx << ");" << std::endl;
            kernel_config.run_args.push_back(outer);
        }
        ++arg_idx;
    }
    int mem_arg_idx = 0;
    for (const auto& org : mem_args | std::views::values) {
        auto& meta = arg_meta.at(org);
        if (meta.is_ptr) {
            if (meta.is_input) {
                stream << "// Input CB " << mem_arg_idx << ": " << org << std::endl;
                ++mem_arg_idx;
            }
            if (meta.is_output) {
                stream << "// Output CB " << mem_arg_idx << ": " << org << std::endl;
                ++mem_arg_idx;
            }
        }
        ++arg_idx;
    }

    stream << "DPRINT_MATH(DPRINT << \"Hello world from comp tt_" << kernel_id << "\" << ENDL());" << std::endl;

    stream << std::endl;

    stream << "init_sfpu(0, 1);"; // TODO hardcoded 0 -> 1

    stream << std::endl;

    auto& indvar = map_.indvar();
    auto& indvar_name = indvar->get_name();
    auto& indvar_type = sdfg_.type(indvar_name);
    stream << language_extension_.declaration(indvar_name, indvar_type) << ";" << std::endl;

    stream << "for (uint32_t tile_idx = 0; tile_idx < " << language_extension_.expression(num_tiles)
           << "; ++tile_idx) {" << std::endl;
    stream.setIndent(8);
    symbolic::Expression org_indvar = symbolic::add(symbolic::mul(symbolic::symbol("tile_idx"), stride), map_.init());
    stream << indvar_name << " = " << language_extension_.expression(org_indvar) << ";" << std::endl;

    mem_arg_idx = 0;
    for (const auto& org : mem_args | std::views::values) {
        auto& meta = arg_meta.at(org);
        if (meta.is_ptr) {
            if (meta.is_input) {
                stream << "cb_wait_front(" << mem_arg_idx << ", 1);" << std::endl;
                ++mem_arg_idx;
            }
            if (meta.is_output) {
                ++mem_arg_idx;
            }
        }
    }

    stream << std::endl;

    bool log_progress = false;

    if (log_progress) {
        stream << "DPRINT_MATH(DPRINT << \"tile\" << tile_idx << ENDL());" << std::endl;
    }
    stream << "tile_regs_acquire();" << std::endl;

    stream << std::endl;

    // read inputs from CB to DST

    stream << "copy_tile(0, 0, 0);" << std::endl; // TODO hardcoded copy from 0 to DST

    stream << std::endl;

    // done with math
    stream << "tile_regs_commit();" << std::endl;

    stream << std::endl;

    mem_arg_idx = 0;
    for (const auto& org : mem_args | std::views::values) {
        auto& meta = arg_meta.at(org);
        if (meta.is_ptr) {
            if (meta.is_input) {
                stream << "cb_pop_front(" << mem_arg_idx << ", 1);" << std::endl;
                ++mem_arg_idx;
            }
            if (meta.is_output) {
                stream << "cb_reserve_back(" << mem_arg_idx << ", 1);" << std::endl;
                ++mem_arg_idx;
            }
        }
    }

    stream << std::endl;
    stream << "tile_regs_wait();" << std::endl;

    stream << std::endl;

    // output to cb
    stream << "pack_tile(0, 1, 0);" << std::endl; // TODO hardcoded copy from DST to 1

    stream << std::endl;

    // donw with output
    stream << "tile_regs_release();" << std::endl;

    stream << std::endl;

    mem_arg_idx = 0;
    for (const auto& org : mem_args | std::views::values) {
        auto& meta = arg_meta.at(org);
        if (meta.is_ptr) {
            if (meta.is_input) {
                ++mem_arg_idx;
            }
            if (meta.is_output) {
                stream << "cb_push_back(" << mem_arg_idx << ", 1);" << std::endl;
                ++mem_arg_idx;
            }
        }
    }

    stream.setIndent(4);
    stream << "}" << std::endl;

    stream.setIndent(0);
    stream << "}" << std::endl;

    stream << std::endl;
    stream << "} // namespace NAMESPACE" << std::endl;

    return kernel_config;
}

constexpr std::string TT_SCALAR_COPY_PREFIX = "tt_scalar_arg";

void TenstorrentMapDispatcher::
    add_with_cast_if_needed(std::vector<std::tuple<std::string, std::string>>& kernel_args, std::string name) {
    if (std::ranges::none_of(kernel_args, [&](const std::tuple<std::string, std::string>& arg) {
            return std::get<1>(arg) == name;
        })) {
        auto& type = sdfg_.type(name);
        std::string use = name;
        if (auto scalar = dynamic_cast<const types::Scalar*>(&type)) {
            types::PrimitiveType primitiveType = scalar->primitive_type();
            if (primitiveType == types::Float) {
                use = "reinterpret_cast<uint32_t&>(" + name + ")";
            } else if (primitiveType == types::Int32 || primitiveType == types::UInt16 ||
                       primitiveType == types::Int16 || primitiveType == types::UInt8 || primitiveType == types::Int8) {
                use = "static_cast<uint32_t>(" + name + ")";
            } else if (primitiveType == types::UInt64 || primitiveType == types::Int64) {
                DEBUG_PRINTLN("WARNING: Downgrading 64 bit arg " << name << " to 32 bit for TT!");
                use = "static_cast<uint32_t>(" + name + ")";
            }
        }
        kernel_args.emplace_back(use, name);
    }
}

void TenstorrentMapDispatcher::dispatch_node(
    codegen::PrettyPrinter& main_stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& tt_schedType = map_.schedule_type();

    emit_tt_includes_once(globals_stream, library_snippet_factory);

    std::string dev_handle_var = "tt_device";

    emit_tt_device_ready(sdfg_, main_stream, globals_stream, library_snippet_factory, dev_handle_var, 0);

    analysis::AnalysisManager ana_mgr(sdfg_);

    auto& arguments_analysis = ana_mgr.get<analysis::ArgumentsAnalysis>();

    auto& arguments = arguments_analysis.arguments(ana_mgr, map_);

    std::vector<std::string> sorted_arguments;
    sorted_arguments.reserve(arguments.size());
    std::ranges::transform(arguments, std::back_inserter(sorted_arguments), [](const auto& pair) {
        return pair.first;
    });
    std::ranges::sort(sorted_arguments);

    auto init = map_.init();
    if (!symbolic::eq(init, symbolic::zero())) {
        throw std::runtime_error("Init is not zero");
    }
    auto stride = map_.stride();
    auto its_per_work_unit = stride;
    auto workUnits = map_.num_iterations();
    if (workUnits.is_null()) {
        throw std::runtime_error("Could not compute number of iterations for TT Map " + std::to_string(map_.element_id()));
    }
    DEBUG_PRINTLN("TT Work Units: " << workUnits->__str__());

    std::string instance_suffix = std::to_string(map_.element_id());

    structured_control_flow::Map* inner_map_ptr = nullptr;
    auto& outer_map_body = map_.root();
    if (outer_map_body.size() > 1) {
        throw std::runtime_error("TT Map " + instance_suffix + " has more than one child!");
    }
    for (int ci = 0; ci < outer_map_body.size(); ++ci) {
        auto child = outer_map_body.at(ci);
        if (auto* map_candidate = dynamic_cast<structured_control_flow::Map*>(&child.first)) {
            if (map_candidate->schedule_type().value() == ScheduleType_Tenstorrent_Kernel::value()) {
                inner_map_ptr = map_candidate;
            }
        }
    }
    if (inner_map_ptr == nullptr) {
        throw std::runtime_error("TT Map " + instance_suffix + " has no inner map!");
    }
    auto& inner_map = *inner_map_ptr;

    auto& type_analysis = ana_mgr.get<analysis::TypeAnalysis>();

    std::vector<TTDataMovementConfig> managed_memory_buffers;
    std::vector<std::tuple<std::string, std::string>> scalar_args;

    for (auto& name : sorted_arguments) {
        auto& meta = arguments.at(name);
        std::string devName;
        bool is_buf = false;

        const types::IType* type;

        if (meta.is_ptr) {
            type = type_analysis.get_outer_type(name);
            is_buf = true;
            devName = name;
        }
        if (is_buf) {
            auto tile_work_factor = symbolic::integer(1);
            managed_memory_buffers.emplace_back(
                name,
                devName,
                type->primitive_type(),
                symbolic::mul(its_per_work_unit, tile_work_factor), // for when tiles have different sizes (then
                                                                    // its_per work is expected to be the least
                                                                    // amount of its)
                workUnits, // if we reuse and do patterns than that
                meta.is_input,
                meta.is_output
            );
        } else if (meta.is_scalar) {
            add_with_cast_if_needed(scalar_args, name);
        } else if (!meta.is_ptr) {
            DEBUG_PRINTLN("TT Map " << instance_suffix << " has unrecognized arg " << name);
        }
    }

    // Remap the offsets onto the managed memory buffers (need to local to each tile)
    auto& users = analysis_manager_.get<analysis::Users>();
    analysis::UsersView scope_users(users, inner_map);
    auto& inner_seq = inner_map.root();
    auto& indvar = inner_map.indvar();
    auto local_indvar = symbolic::symbol(TT_IDX_WITHIN_BLOCK);

    for (auto& buf : managed_memory_buffers) {
        for (auto* user : scope_users.uses(buf.device_name)) {
            auto elem = user->element();

            auto* a_node = dynamic_cast<data_flow::AccessNode*>(elem);
            if (a_node) {
                auto& g = a_node->get_parent();
                for (auto& rdMemlet : g.out_edges(*a_node)) {
                    auto t = rdMemlet.type();
                    rdMemlet.replace(indvar, local_indvar);
                }
                for (auto& wrMemlet : g.in_edges(*a_node)) {
                    auto t = wrMemlet.type();
                    wrMemlet.replace(indvar, local_indvar);
                }
            }
        }
    }

    // in case
    //    symbolic::Expression num_cb_tiles = symbolic::divide_ceil(workUnits, tile_size_entries);
    //    SymEngine::set_basic num_cb_tiles_params = SymEngine::free_symbols(*num_cb_tiles);
    //    if (!num_cb_tiles_params.empty()) { // add inputs that are needed to compute the size of the buffer to
    //                                        // kernel arguments
    //        for (auto& param : num_cb_tiles_params) {
    //            auto sym_name = SymEngine::rcp_static_cast<const SymEngine::Symbol>(param)->get_name();
    //            add_with_cast_if_needed(mem_args, sym_name);
    //        }
    //    }
    //    main_stream << "static std::once_flag flag;" << std::endl;
    //    main_stream << "static tt::tt_metal::program tt_program;" << std::endl;
    //    main_stream << "std::call_once(flag, [&] {" << std::endl;
    //    main_stream.changeIndent(+4);

    TTKernelManagementCodegen
        codegen(main_stream, library_snippet_factory, language_extension_, dev_handle_var, map_, managed_memory_buffers);

    codegen.emit_default_size_distribution(workUnits);

    codegen.emit_buffer_setup_code();

    std::vector<std::pair<std::string, std::string>> mem_args;

    auto& locals = arguments_analysis.locals(ana_mgr, map_);
    generate_combined_kernel(codegen, library_snippet_factory, inner_map, instance_suffix, scalar_args, locals);

    main_stream << std::endl;

    for (const auto& [use_as, org] : scalar_args) {
        if (debug_scalar_args_) {
            main_stream << "std::cout << +\" arg " << org << " = \" << " << org << " << \";\" << std::endl;"
                        << std::endl;
        }
    }

    auto [units_done_var, units_on_core_var] = codegen.get_default_distribution_vars();

    std::unordered_map<std::string, std::string> core_args{// late arg-matching for the following set-args loop
                                                           {TT_FIRST_UNIT, units_done_var},
                                                           {TT_WORK_UNITS, units_on_core_var}
    };

    codegen.emit_per_core_config([&]() {
        codegen.emit_set_runtime_args("core", core_args); // for all known cores. Depends on how they were registered
                                                          // which they actually use in which order!
    });

    bool blocking = ScheduleType_Tenstorrent_Device::is_blocking(tt_schedType);
    codegen.emit_launch(blocking);

    if (blocking && dev_profile_) {
        main_stream << "tt::tt_metal::detail::ReadDeviceProfilerResults(" << dev_handle_var
                    << ", tt::tt_metal::ProfilerReadState::LAST_FD_READ);" << std::endl;
    }

    main_stream << std::endl;
    main_stream << "double tt_num_cores_used" << " = static_cast<double>(" << codegen.get_num_used_cores() << ");"
                << std::endl;
    if (tt_emit_full_metrics) {
        main_stream << "double tt_num_cores_used_primary" << " = static_cast<double>(" << codegen.get_main_cores()
                    << ".num_cores());" << std::endl;
        main_stream << "double tt_work_units_primary" << " = static_cast<double>(" << codegen.get_units_per_main()
                    << ");" << std::endl;
    }
    main_stream << "double tt_num_cores_available" << " = static_cast<double>(" << codegen.get_num_avail_cores() << ");"
                << std::endl;
    main_stream << "double tt_cores_used_rel" << " = tt_num_cores_used"
                << " / tt_num_cores_available" << ";" << std::endl;
    main_stream << "double tt_work_units_per_core" << " = static_cast<double>(units_done"
                << ") / tt_num_cores_available" << ";" << std::endl;

    main_stream << std::endl;
}

codegen::InstrumentationInfo TenstorrentMapDispatcher::instrumentation_info() const {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
    analysis::LoopInfo loop_info = loop_analysis.loop_info(&map_);

    // Perform FlopAnalysis
    std::unordered_map<std::string, std::string> metrics;
    auto& flop_analysis = analysis_manager_.get<analysis::FlopAnalysis>();
    auto flops = flop_analysis.get_if_available_for_codegen(&map_);
    if (!flops.is_null()) {
        std::string flop_str = language_extension_.expression(flops);
        metrics.insert({"flop", flop_str});
    }

    metrics.insert({"tt_cores_used_rel", "tt_cores_used_rel_"});
    metrics.insert({"tt_work_units_per_core", "tt_work_units_per_core_"});
    if (tt_emit_full_metrics) {
        metrics.insert({"tt_num_cores_used", "tt_num_cores_used_"});
        metrics.insert({"tt_num_cores_used_primary", "tt_num_cores_used_primary_"});
        metrics.insert({"tt_work_units_primary", "tt_work_units_primary_"});
        metrics.insert({"tt_num_cores_available", "tt_num_cores_available_"});
    }

    return codegen::
        InstrumentationInfo(map_.element_id(), codegen::ElementType_Map, TargetType_Tenstorrent, loop_info, metrics);
}

bool TenstorrentMapDispatcher::begin_node(codegen::PrettyPrinter& stream) {
    stream << "{" << std::endl;
    stream.changeIndent(+4);
    return true;
}

void TenstorrentMapDispatcher::end_node(codegen::PrettyPrinter& stream, bool has_declaration) {
    if (has_declaration) {
        stream.changeIndent(-4);
        stream << "}" << std::endl;
    }
};

} // namespace tenstorrent
} // namespace sdfg
