#include "docc/target/tenstorrent/codegen.h"

#include "docc/target/tenstorrent/tenstorrent_transform.h"

namespace sdfg::tenstorrent {

void emit_tt_device_ready(
    const StructuredSDFG& sdfg,
    codegen::PrettyPrinter& main_stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory,
    const std::string& handle, // WARNING: neds to be "tt_device" for now, because if another user already created it
                               // under a different name, we are not smart enough to create a new one under the new name
    int device_id
) {
    std::unordered_map<int, std::unique_ptr<codegen::CodeSnippet>> device_ready_snippets;
    auto& entry = device_ready_snippets[0];

    if (!sdfg.exists(handle)) {
        auto tt_global_marker = "tenstorrent_global_init";
        if (library_snippet_factory.find(tt_global_marker) == library_snippet_factory.snippets().end()) {
            library_snippet_factory.require(tt_global_marker, "", false);

            emit_tt_includes_once(globals_stream, library_snippet_factory);

            globals_stream << std::endl;

            auto& init_stream = library_snippet_factory.require(codegen::CODE_SNIPPET_INIT_ONCE, "cpp", false).stream();
            if (false) {
                init_stream << "ZoneScopedN(\"" << sdfg.name() << "\");" << std::endl;
            }
            init_stream << "tt::tt_metal::IDevice* " << handle << " = " << "daisy::tenstorrent::daisy_get_tt_device("
                        << device_id << ");" << std::endl;
            //            init_stream << "std::shared_ptr<tt::tt_metal::IDevice> tt_device0 = "
            //                           "std::shared_ptr<tt::tt_metal::IDevice>(tt::tt_metal::CreateDevice("
            //                        << device_id << "), __daisy_DeviceDeleter());" << std::endl;

            //            auto& deinit_stream =
            //                library_snippet_factory.require(codegen::CODE_SNIPPET_DEINIT_ONCE, "cpp", false).stream();
            //            deinit_stream << "tt::tt_metal::CloseDevice(tt_device0);" << std::endl;
        }
    }
}

static const std::string tilized_padded_copy_code = R"a(
template <typename T>
static void tilized_padded_copy(
    const void* src,
    int in_h,
    int in_w,
    int in_line_stride,
    void* dst,
    int h,
    int w,
    int tile_h,
    int tile_w,
    int face_h,
    int face_w
) {

    bool transpose_face = false;
    bool transpose_face_order = false;

    uint32_t row_tiles = h / tile_h;
    uint32_t col_tiles = w / tile_w;

    assert(h > 0 && w > 0 && "None of the input size, H, nor W can be 0");
    assert((h % tile_h == 0) && (w % tile_w == 0) && "H and W must be divisible by {} and {}" && tile_h && tile_w);

    T* write_ptr = reinterpret_cast<T*>(dst);
    const T* src_ptr = reinterpret_cast<const T*>(src);

    auto write_face = [&](uint32_t read_face_offset, int face_x_begin, int face_y_begin) {
        T* face_dst = write_ptr;
        const T* face_src = src_ptr + read_face_offset;

        int in_x_end = std::min(face_x_begin + face_w, in_w);
        uint32_t in_face_w = std::max(0, in_x_end - face_x_begin);
        int in_y_end = std::min(face_y_begin + face_h, in_h);
        uint32_t in_face_h = std::max(0, in_y_end - face_y_begin);

        if (!transpose_face) {
            for (uint32_t row = 0; row < face_h; row++) {
                const T* row_src = face_src + row * in_line_stride;
                if (row < in_face_h) {
                    std::memcpy(face_dst, row_src, in_face_w * sizeof(T));
                } else { // add 0-padded rows
                    std::memset(face_dst, 0, in_face_w * sizeof(T));
                }
                if (in_face_w < face_w) {
                    std::memset(face_dst + in_face_w, 0, (face_w - in_face_w) * sizeof(T)); // zero pad cols
                }
                face_dst += face_w;
            }
        } else {
            for (uint32_t row = 0; row < face_h; row++) {
                for (uint32_t col = 0; col < face_w; col++) {
                    T value;
                    if (row < in_face_h && col < in_face_w) {
                        value = face_src[row * in_line_stride + col];
                    } else {
                        value = 0; // zero pad
                    }
                    face_dst[col * face_h + row] = value;
                }
            }
        }
        write_ptr += face_h * face_w;
    };

    for (uint32_t row_tile = 0; row_tile < row_tiles; row_tile++) {
        uint32_t in_row_y = row_tile * tile_h;
        for (uint32_t col_tile = 0; col_tile < col_tiles; col_tile++) {
            uint32_t in_col_x = col_tile * tile_w;
            uint32_t tile_begin = in_row_y * in_line_stride + in_col_x;
            if (!transpose_face_order) {
                for (int face_h_index = 0; face_h_index < static_cast<int>(tile_h / face_h); face_h_index++) {
                    for (int face_w_index = 0; face_w_index < static_cast<int>(tile_w / face_w); face_w_index++) {
                        uint32_t y_offset = face_h_index * face_h;
                        uint32_t row_offset = y_offset * in_line_stride;
                        uint32_t x_offset = face_w_index * face_w;
                        uint32_t src_idx = tile_begin + x_offset + row_offset;
                        write_face(src_idx, in_col_x + x_offset, in_row_y + y_offset);
                    }
                }
            } else {
                for (int face_w_index = 0; face_w_index < static_cast<int>(tile_w / face_w); face_w_index++) {
                    for (int face_h_index = 0; face_h_index < static_cast<int>(tile_h / face_h); face_h_index++) {
                        uint32_t y_offset = face_h_index * face_h;
                        uint32_t row_offset = y_offset * in_line_stride;
                        uint32_t x_offset = face_w_index * face_w;
                        uint32_t src_idx = tile_begin + x_offset + row_offset;
                        write_face(src_idx, in_col_x + x_offset, in_row_y + y_offset);
                    }
                }
            }
        }
    }
}
)a";

static const std::string untilized_unpadded_copy_code = R"a(
template <typename T>
static void untilized_unpadded_copy(
    void* dst,
    int out_h,
    int out_w,
    int out_line_stride,
    const void* src,
    int h,
    int w,
    int tile_h,
    int tile_w,
    int face_h,
    int face_w
) {

    bool transpose_face = false;
    bool transpose_face_order = false;

    uint32_t row_faces = tile_h / face_h;
    uint32_t col_faces = tile_w / face_w;
    uint32_t tile_rows = h / tile_h;
    uint32_t tile_cols = w / tile_w;

    // We don't transpose face order if we have only one face in the row or column
    transpose_face_order = transpose_face_order && row_faces > 1 && col_faces > 1;

    assert(h > 0 && w > 0 && "None of the input size, H, nor W can be 0");
    assert((h % tile_h == 0) && (w % tile_w == 0) && "H and W must be divisible by {} and {}" && tile_h && tile_w);

    T* dst_ptr = reinterpret_cast<T*>(dst);
    const T* src_ptr = reinterpret_cast<const T*>(src);

    auto write_face =
        [&](uint32_t in_face_start, uint32_t out_face_start, int face_x_begin, int face_y_begin) {
            int out_x_end = std::min(face_x_begin + face_w, out_w);
            uint32_t out_face_w = std::max(0, out_x_end - face_x_begin);
            int out_y_end = std::min(face_y_begin + face_h, out_h);
            uint32_t out_face_h = std::max(0, out_y_end - face_y_begin);

            if (!transpose_face) {
                for (uint32_t row = 0; row < face_h; row++) {
                    uint32_t src_idx = in_face_start + row * face_w;
                    uint32_t dst_idx = out_face_start + row * out_line_stride;
                    if (row < out_face_h && out_face_w > 0) {
                        std::memcpy(&dst_ptr[dst_idx], &src_ptr[src_idx], out_face_w * sizeof(T));
                    }
                }
            } else {
                for (uint32_t row = 0; row < face_h; row++) {
                    for (uint32_t col = 0; col < face_w; col++) {
                        size_t dst_idx = out_face_start + col * out_line_stride + row;
                        T value;
                        if (row < out_face_h && col < out_face_w) {
                            size_t src_idx = in_face_start + row * face_w + col;
                            value = src_ptr[src_idx];
                        } else {
                            value = 0; // zero pad
                        }

                        dst_ptr[dst_idx] = value;
                    }
                }
            }
        };

    for (size_t tile_row = 0; tile_row < tile_rows; tile_row++) {
        for (size_t tile_col = 0; tile_col < tile_cols; tile_col++) {
            size_t in_tile_start = (tile_row * tile_h * w) + (tile_col * tile_h * tile_w);
            int out_tile_x_begin = transpose_face ? tile_row * tile_h : tile_col * tile_w;
            int out_tile_y_begin = transpose_face ? tile_col * tile_w : tile_row * tile_h;
            size_t out_tile_start = transpose_face ? (tile_row * tile_w * out_line_stride) +
                                                         tile_col * tile_h
                                                   : (tile_row * tile_h * out_line_stride) + tile_col * tile_w;

            for (size_t face_h_idx = 0; face_h_idx < row_faces; face_h_idx++) {
                for (size_t face_w_idx = 0; face_w_idx < col_faces; face_w_idx++) {
                    size_t in_face_start =
                        in_tile_start + face_h_idx * face_h * tile_w + face_w_idx * face_h * face_w;

                    auto face_h_idx_dst = transpose_face_order ? face_w_idx : face_h_idx;
                    auto face_w_idx_dst = transpose_face_order ? face_h_idx : face_w_idx;
                    int face_x_offset = transpose_face ? face_h_idx_dst * face_h : face_w_idx_dst * face_w;
                    int face_y_offset = transpose_face ? face_w_idx_dst * face_w : face_h_idx_dst * face_h;
                    int face_x_begin = out_tile_x_begin + face_x_offset;
                    int face_y_begin = out_tile_y_begin + face_y_offset;
                    size_t out_face_start =
                        transpose_face ? out_tile_start + face_h_idx_dst * face_h * out_line_stride + face_w_idx_dst * face_h
                                       : out_tile_start + face_h_idx_dst * face_h * out_line_stride + face_w_idx_dst * face_w;
                    write_face(in_face_start, out_face_start, face_x_begin, face_y_begin);
                }
            }
        }
    }
}
)a";

void emit_tilized_padded_copy_helper(
    const codegen::LanguageExtension& language_extension,
    codegen::PrettyPrinter& stream,
    codegen::CodeSnippetFactory& code_snippet_factory
) {
    auto tilized_padded_marker = "tilized_padded_copy";
    if (code_snippet_factory.find(tilized_padded_marker) == code_snippet_factory.snippets().end()) {
        code_snippet_factory.require(tilized_padded_marker, "", false);
        stream << tilized_padded_copy_code << std::endl;
    }
}

void emit_untilized_unpadded_copy_helper(
    const codegen::LanguageExtension& language_extension,
    codegen::PrettyPrinter& stream,
    codegen::CodeSnippetFactory& code_snippet_factory
) {
    auto untilized_unpadded_marker = "untilized_unpadded_copy";
    if (code_snippet_factory.find(untilized_unpadded_marker) == code_snippet_factory.snippets().end()) {
        code_snippet_factory.require(untilized_unpadded_marker, "", false);
        stream << untilized_unpadded_copy_code << std::endl;
    }
}

static const std::string tt_h2d_transfer_helper_code = R"a(void __daisy_tt_h2d_transfer(
    tt::tt_metal::IDevice* device,
    int cq_no,
    std::shared_ptr<tt::tt_metal::Buffer> buffer,
    const void* src_ptr,
    size_t size,
    bool blocking
) {
    ZoneScopedN("daisy_tt_h2d_transfer");
    auto page_size = buffer->page_size();

    auto safe_read_bytes = tt::round_down(size, page_size);
    auto left_bytes = size - safe_read_bytes;

    if (safe_read_bytes > 0) {
        tt::tt_metal::EnqueueWriteSubBuffer(
            device->command_queue(cq_no),
            buffer,
            src_ptr,
            {0, safe_read_bytes},
            false
        );
    }

    if (safe_read_bytes < size) {

        uint8_t* temp = new uint8_t[page_size]{};

        const uint8_t* byte_src = reinterpret_cast<const uint8_t*>(src_ptr);

        memcpy(temp, byte_src+safe_read_bytes, left_bytes);

        tt::tt_metal::EnqueueWriteSubBuffer(
            device->command_queue(cq_no),
            buffer,
            temp,
            {safe_read_bytes, page_size},
            true
        );

        delete[] temp;
    }
}
)a";

static const std::string tt_d2h_transfer_helper_code = R"a(void __daisy_tt_d2h_transfer(
    tt::tt_metal::IDevice* device,
    int cq_no,
    std::shared_ptr<tt::tt_metal::Buffer> buffer,
    void* dst_ptr,
    size_t size,
    bool blocking
) {
    ZoneScopedN("daisy_tt_d2h_transfer");
    auto page_size = buffer->page_size();

    auto safe_write_bytes = tt::round_down(size, page_size);
    auto left_bytes = size - safe_write_bytes;

    if (safe_write_bytes > 0) {
        tt::tt_metal::EnqueueReadSubBuffer(
            device->command_queue(cq_no),
            buffer,
            dst_ptr,
            {0, safe_write_bytes},
            left_bytes == 0
        );
    }

    if (safe_write_bytes < size) {

        uint8_t* temp = new uint8_t[page_size];

        tt::tt_metal::EnqueueReadSubBuffer(
            device->command_queue(cq_no),
            buffer,
            temp,
            {safe_write_bytes, page_size},
            true
        );

        uint8_t* byte_dst = reinterpret_cast<uint8_t*>(dst_ptr);

        memcpy(byte_dst+safe_write_bytes, temp, left_bytes);

        delete[] temp;
    }
}
)a";


void emit_h2d_transfer_helper_once(
    const codegen::LanguageExtension& language_extension,
    codegen::PrettyPrinter& stream,
    codegen::CodeSnippetFactory& code_snippet_factory
) {
    auto marker = "tt_h2d_transfer_once";
    if (code_snippet_factory.find(marker) == code_snippet_factory.snippets().end()) {
        code_snippet_factory.require(marker, "", false);

        emit_tt_includes_once(stream, code_snippet_factory);

        // hacky way to only emit to global once
        stream << std::endl << "static " << tt_h2d_transfer_helper_code << std::endl;
    }
}

void emit_d2h_transfer_helper_once(
    const codegen::LanguageExtension& language_extension,
    codegen::PrettyPrinter& stream,
    codegen::CodeSnippetFactory& code_snippet_factory
) {
    auto marker = "tt_d2h_transfer_once";
    if (code_snippet_factory.find(marker) == code_snippet_factory.snippets().end()) {
        code_snippet_factory.require(marker, "", false);

        emit_tt_includes_once(stream, code_snippet_factory);

        // hacky way to only emit to global once
        stream << std::endl << "static " << tt_d2h_transfer_helper_code << std::endl;
    }
}

void emit_tt_includes_once(codegen::PrettyPrinter& stream, codegen::CodeSnippetFactory& code_snippet_factory) {
    auto marker = "tt_includes_once";

    if (code_snippet_factory.find(marker) == code_snippet_factory.snippets().end()) {
        code_snippet_factory.require(marker, "", false);
        // hacky way to only emit to global once
        stream << "#include <cstdint>" << std::endl;
        stream << "#include <cstring>" << std::endl;
        stream << "#include <cassert>" << std::endl;
        stream << "#include <memory>" << std::endl;
        stream << "#include <vector>" << std::endl;
        stream << "#include <tt-metalium/host_api.hpp>" << std::endl;
        stream << "#include <tt-metalium/work_split.hpp>" << std::endl;
        stream << "#include <tt-metalium/tensor_accessor_args.hpp>" << std::endl;
        stream << "#include <daisy_rtl/global_tenstorrent_init.h>" << std::endl;

        stream << "#include <tracy/Tracy.hpp>" << std::endl;
        stream << "#include <tt-metalium/tt_metal_profiler.hpp>" << std::endl;

        stream << std::endl;
    }
}

std::string TTDataMovementConfig::get_device_address_var() const { return device_name + "->address()"; }

std::string KernelConfig::handle_var(int sub, const std::string& instance_suffix) const {
    std::string base;
    switch (target) {
        case TTKernelTarget::DatMovRd:
            base = "kernel_movRd_";
            break;
        case TTKernelTarget::DatMovWr:
            base = "kernel_movWr_";
            break;
        case TTKernelTarget::Compute:
            base = "kernel_compute_";
            break;
        default:
            throw std::runtime_error("Invalid kernel target " + std::to_string(static_cast<int>(this->target)));
    }
    return base + std::to_string(id) + "_" + std::to_string(sub);
}


TTKernelManagementCodegen::TTKernelManagementCodegen(
    codegen::PrettyPrinter& stream,
    codegen::CodeSnippetFactory& snippets,
    codegen::LanguageExtension& language_extension,
    const std::string& dev_handle_var,
    const Element& elem,
    const std::vector<TTDataMovementConfig>& managed_memory_buffers,
    const std::optional<std::string> preallocated_program
)
    : stream_(stream), snippets_(snippets), language_extension_(language_extension),
      instance_suffix_(std::to_string(elem.element_id())), dev_handle_var_(dev_handle_var),
      managed_memory_buffers_(managed_memory_buffers) {
    if (preallocated_program.has_value()) {
        prog_var_ = *preallocated_program;
    } else {
        prog_var_ = "tt_program";
        stream_ << "tt::tt_metal::Program " << prog_var_ << ";" << std::endl;
    }

    available_cores_var_ = "tt_full_grid";
    num_available_cores_var_ = "(" + available_cores_var_ + ".x * " + available_cores_var_ + ".y)";
    cores_var_ = available_cores_var_;
    stream_ << "tt::tt_metal::CoreCoord " << cores_var_ << " = " << dev_handle_var_
            << "->compute_with_storage_grid_size();" << std::endl;
    num_used_cores_var_ = available_cores_var_ + ".num_cores()";
}

int TTKernelManagementCodegen::add_kernel(
    const codegen::CodeSnippet& snippet,
    TTKernelTarget type,
    const std::vector<ArgDesc>& compile_args,
    const std::vector<ArgDesc>& run_args,
    bool force_32bit_fpu
) {
    int idx = kernels_.size();

    std::string kernel_str;
    if (snippet.is_as_file()) {
        kernel_str = snippets_.output_path() / (snippet.name() + ".cpp");
    } else {
        kernel_str = snippet.stream().str();
    }

    auto& ref = kernels_.emplace_back(idx, snippet.is_as_file(), kernel_str, type, compile_args, run_args);

    if (type == TTKernelTarget::Compute && force_32bit_fpu) {
        ref.force_32bit_fpu = true;
    }

    return idx;
}

void TTKernelManagementCodegen::emit_default_size_distribution(symbolic::Expression work_units) {
    cores_var_ = "all_cores";
    main_workload_ = {"main_cores", "units_per_main"};
    rem_workload_ = {"remainder_cores", "units_per_remainder"};
    num_used_cores_var_ = "num_cores";

    stream_ << "auto [" << num_used_cores_var_ << ", " << cores_var_ << ", " << main_workload_->first << ", "
            << rem_workload_->first << ", " << main_workload_->second << ", " << rem_workload_->second
            << "] = "
               "tt::tt_metal::split_work_to_cores("
            << available_cores_var_ << ", " << language_extension_.expression(work_units) << ");" << std::endl;
}

void TTKernelManagementCodegen::emit_per_core_config(std::function<void()> per_coord) {
    auto [units_done_var, units_on_core_var] = get_default_distribution_vars();

    auto range_var = "tt_range";
    stream_ << "uint32_t " << units_done_var << " = 0;" << std::endl;
    stream_ << "for (auto& " << range_var << " : " << get_used_cores() << ".ranges()) {" << std::endl;
    stream_.changeIndent(+4);
    stream_ << "for (auto& core : " << range_var << ") {" << std::endl;

    stream_.changeIndent(+4);
    stream_ << "uint32_t " << units_on_core_var << " = 0;" << std::endl;
    stream_ << "if (" << get_main_cores() << ".contains(core)) {" << std::endl;
    stream_ << "\t" << units_on_core_var << " = " << get_units_per_main() << ";" << std::endl;
    stream_ << "} else if (" << get_rem_cores() << ".contains(core)) {" << std::endl;
    stream_ << "\t" << units_on_core_var << " = " << get_units_per_rem() << ";" << std::endl;
    stream_ << "} else {" << std::endl;
    stream_ << "\tTT_ASSERT(false, \"Core not in specified core ranges\");" << std::endl;
    stream_ << "}" << std::endl;

    per_coord();

    stream_ << units_done_var << " += " << units_on_core_var << ";" << std::endl;
    stream_.changeIndent(-4);
    stream_ << "}" << std::endl;
    stream_.changeIndent(-4);
    stream_ << "}" << std::endl;
    stream_ << std::endl;
}

void TTKernelManagementCodegen::emit_buffer_setup_code() {
    int mem_arg_idx = 0;
    for (auto& meta : managed_memory_buffers_) {
        auto elem_size = symbolic::integer(types::bit_width(meta.primitive_type) / 8);
        symbolic::Expression tile_size_bytes = SymEngine::mul(meta.tile_elems, elem_size);

        auto config_name = "tt_cb_" + std::to_string(mem_arg_idx) + "_config";
        std::string arg_format;
        switch (meta.primitive_type) {
            case types::Int32:
                arg_format = "tt::DataFormat::Int32";
                break;
            case types::UInt32:
                arg_format = "tt::DataFormat::UInt32";
                break;
            case types::Float:
                arg_format = "tt::DataFormat::Float32";
                break;
            default:
                throw std::runtime_error(
                    "Unsupported primitive type: " + std::string(types::primitive_type_to_string(meta.primitive_type))
                );
        }
        auto handle_name = "tt_cb_" + std::to_string(mem_arg_idx);

        stream_ << "auto " << config_name << " = tt::tt_metal::CircularBufferConfig(";
        stream_ << language_extension_.expression(SymEngine::mul(tile_size_bytes, symbolic::integer(2))) << ", ";
        stream_ << "{{" << mem_arg_idx << ", " << arg_format << "}})" << std::endl;
        stream_ << "\t.set_page_size(" << mem_arg_idx << ", " << language_extension_.expression(tile_size_bytes) << ");"
                << std::endl;
        stream_ << "auto " << handle_name << " = tt::tt_metal::CreateCircularBuffer(";
        stream_ << prog_var_ << ", ";
        stream_ << cores_var_ << ", ";
        stream_ << config_name << ");" << std::endl;

        ++mem_arg_idx;
    }
}

int TTKernelManagementCodegen::emit_kernel_load(
    int id, const std::string& cores, const std::unordered_map<std::string, std::string>* late_args, bool runtime_args
) {
    auto& kernel = kernels_[id];

    auto instance_id = instantiations_to_kernel_.size();
    instantiations_to_kernel_.push_back(id);

    if (kernel.is_external_file_path) {
        std::string k_handle = kernel.handle_var(instance_id, instance_suffix_);

        if (!kernel.compile_args_vector_var.empty()) {
            stream_ << kernel.compile_args_vector_var << ".insert(" << kernel.compile_args_vector_var << ".end(), {";
            emit_resolved_args(kernel.compile_args, late_args);
            stream_ << "});" << std::endl;
        }

        stream_ << "auto " << k_handle << " = tt::tt_metal::CreateKernel(" << prog_var_ << ", "
                << "\"" << kernel.kernel << "\", " << cores << ", ";

        switch (kernel.target) {
            case TTKernelTarget::DatMovRd:
                stream_ << "tt::tt_metal::ReaderDataMovementConfig";
                stream_ << "(";
                break;
            case TTKernelTarget::DatMovWr:
                stream_ << "tt::tt_metal::WriterDataMovementConfig";
                stream_ << "(";
                break;
            case TTKernelTarget::Compute:
                stream_ << "tt::tt_metal::ComputeConfig";
                stream_ << " { " << std::endl;
                stream_.changeIndent(+4);
                stream_ << ".fp32_dest_acc_en = " << kernel.force_32bit_fpu << "," << std::endl;
                stream_ << ".compile_args = ";
                break;
            default:
                throw std::runtime_error("Invalid kernel target " + std::to_string(static_cast<int>(kernel.target)));
        }

        if (!kernel.compile_args_vector_var.empty()) {
            stream_ << kernel.compile_args_vector_var;
        } else {
            stream_ << "{";
            emit_resolved_args(kernel.compile_args, late_args);
            stream_ << "}";
        }

        switch (kernel.target) {
            case TTKernelTarget::DatMovRd:
                stream_ << ", {})";
                break;
            case TTKernelTarget::DatMovWr:
                stream_ << ", {})";
                break;
            case TTKernelTarget::Compute:
                stream_ << std::endl;
                stream_.changeIndent(-4);
                stream_ << "}";
                break;
            default:
                throw std::runtime_error("Invalid kernel target " + std::to_string(static_cast<int>(kernel.target)));
        }

        stream_ << ");" << std::endl;


    } else {
        throw std::runtime_error("Not implemented");
    }

    if (!kernel.rt_common_args_vector_var.empty()) {
        //        stream_ << kernel.rt_common_args_vector_var << ".insert(" << kernel.rt_common_args_vector_var <<
        //        ".begin()), {" << std::endl; stream_ << "});" << std::endl;
    }

    if (runtime_args) {
        if (std::ranges::any_of(kernel.run_args, [](const auto& arg) {
                return std::holds_alternative<LateArg>(arg);
            })) {
            emit_kernel_set_runtime_args(instance_id, cores, late_args);
        }
    }

    return instance_id;
}

void TTKernelManagementCodegen::emit_kernel_set_common_runtime_args(kernel_instance_id_t id) const {
    auto kernel_id = instantiations_to_kernel_[id];
    auto& kernel = kernels_[kernel_id];

    // resolve args here (currently only tensorArgs...)

    if (!kernel.rt_common_args_vector_var.empty()) {
        stream_ << "tt::tt_metal::SetCommonRuntimeArgs(" << prog_var_ << ", " << kernel.handle_var(id, instance_suffix_)
                << ", " << kernel.rt_common_args_vector_var << ");" << std::endl;
    }
}

void TTKernelManagementCodegen::emit_kernel_set_runtime_args(
    kernel_instance_id_t id, std::string cores, const std::unordered_map<std::string, std::string>* late_args
) const {
    auto kernel_id = instantiations_to_kernel_[id];
    auto& kernel = kernels_[kernel_id];

    stream_ << "tt::tt_metal::SetRuntimeArgs(" << prog_var_ << ", " << kernel.handle_var(id, instance_suffix_) << ", "
            << cores << ", ";
    stream_ << "{";
    emit_resolved_args(kernel.run_args, late_args);
    stream_ << "});" << std::endl;
}

void TTKernelManagementCodegen::emit_set_runtime_args(
    const std::string& cores, const std::unordered_map<std::string, std::string>& late_args
) const {
    for (auto& instance_id : instantiations_to_kernel_) {
        emit_kernel_set_runtime_args(instance_id, cores, &late_args);
    }
}

void TTKernelManagementCodegen::emit_launch(bool blocking) {
    stream_ << "tt::tt_metal::EnqueueProgram(" << dev_handle_var_ << "->command_queue(0), " << prog_var_ << ", "
            << blocking << ");" << std::endl;
}

void TTKernelManagementCodegen::emit_resolved_args(
    const std::vector<ArgDesc>& args, const std::unordered_map<std::string, std::string>* late_args
) const {
    auto resolve_arg = [this, &late_args](const ArgDesc& arg) -> std::string {
        if (const auto* mem = std::get_if<MemArg>(&arg)) {
            auto& buf = managed_memory_buffers_.at(mem->idx);
            switch (mem->property) {
                case MemArgType::CB_ID:
                    return std::to_string(mem->idx); // for now we just reuse the mem_arg_idx directly
                case MemArgType::ADDR:
                    return buf.get_device_address_var();
                case MemArgType::NUM_TILES:
                    return language_extension_.expression(buf.num_tiles);
                default:
                    throw std::runtime_error("Unknown memory argument property");
            }
        } else if (const auto* expr = std::get_if<ExprArg>(&arg)) {
            auto expr_str = language_extension_.expression(expr->expr);
            if (SymEngine::is_a<SymEngine::Integer>(*expr->expr)) {
                return expr_str;
            } else {
                return "static_cast<uint32_t>(" + expr_str + ")";
            }
        } else if (const auto* lit = std::get_if<LiteralArg>(&arg)) {
            return lit->value;
        } else if (const auto* late = std::get_if<LateArg>(&arg)) {
            if (late_args) {
                auto it = late_args->find(late->arg_name);
                if (it == late_args->end()) {
                    throw std::runtime_error("Late argument not found: " + late->arg_name);
                }
                return it->second;
            } else {
                throw std::runtime_error("Late arg, but no resolution info provided: " + late->arg_name);
            }
        } else {
            throw std::runtime_error("Unknown argument type");
        }
    };

    bool first = true;
    for (auto& arg : args) {
        auto resolved = resolve_arg(arg);
        if (!first) {
            stream_ << ", ";
        }
        stream_ << resolved;
        first = false;
    }
}

void TTKernelManagementCodegen::emit_kernel_tensor_addr_args(kernel_id_t i) {
    auto& kernel = kernels_[i];

    auto comp_args_vec_name = "compile_args_k" + std::to_string(i);
    auto rt_common_args_vec_name = "rt_common_args_k" + std::to_string(i);

    stream_ << "std::vector<uint32_t> " << comp_args_vec_name << ", " << rt_common_args_vec_name << ";" << std::endl;

    for (const auto& item : kernel.run_args) {
        if (auto* mem = std::get_if<MemArg>(&item)) {
            if (mem->property == MemArgType::ADDR) {
                stream_ << "tt::tt_metal::TensorAccessorArgs(" << managed_memory_buffers_.at(mem->idx).device_name
                        << ").append_to(" << comp_args_vec_name << ", " << rt_common_args_vec_name << ");" << std::endl;
            }
        }
    }

    kernel.compile_args_vector_var = std::move(comp_args_vec_name);
    kernel.rt_common_args_vector_var = std::move(rt_common_args_vec_name);
    kernel.mem_addr_in_rt_common = true;
}

const codegen::CodeSnippet& TTKernelManagementCodegen::
    emit_predefined_kernel(const std::string& name, const std::string& code) {
    auto it = snippets_.find(name);
    if (it == snippets_.snippets().end()) {
        auto& snippet = snippets_.require(name, "cpp");
        snippet.stream() << code << std::endl;
        return snippet;
    } else {
        return it->second;
    }
}


} // namespace sdfg::tenstorrent
