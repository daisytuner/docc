#include "sdfg/targets/tenstorrent/blas/dot.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/targets/tenstorrent/codegen.h"
#include "sdfg/targets/tenstorrent/plugin.h"

#include "sdfg/targets/tenstorrent/tenstorrent_offloading_node.h"

namespace sdfg::tenstorrent::blas {

static const std::string reader_kernel_2_tile_streams = R"a(
#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_y = 1;

    uint32_t addr_x = get_arg_val<uint32_t>(0);
    uint32_t addr_y = get_arg_val<uint32_t>(1);
    uint32_t begin_tile = get_arg_val<uint32_t>(2);
    uint32_t input_tiles = get_arg_val<uint32_t>(3);

    constexpr auto x_tensor_args = TensorAccessorArgs<0, 0>();
    const auto x_tensor = TensorAccessor(x_tensor_args, addr_x, get_tile_size(cb_x));
    constexpr auto y_tensor_args = TensorAccessorArgs<x_tensor_args.next_compile_time_args_offset(), x_tensor_args.next_common_runtime_args_offset()>();
    const auto y_tensor = TensorAccessor(y_tensor_args, addr_y, get_tile_size(cb_y));

    DPRINT << "reader up" << ENDL();

    uint32_t end_tile = begin_tile + input_tiles;
    for (uint32_t tile = begin_tile; tile < end_tile; ++tile) {
        {
            DeviceZoneScopedN("WaitingForCbSpace");
            cb_reserve_back(cb_x, 1);
            cb_reserve_back(cb_y, 1);
        }

        {
            DeviceZoneScopedN("FetchingTiles");
            noc_async_read_page(tile, x_tensor, get_write_ptr(cb_x));
            noc_async_read_page(tile, y_tensor, get_write_ptr(cb_y));

            noc_async_read_barrier();
            cb_push_back(cb_x, 1);
            cb_push_back(cb_y, 1);
        }

        DPRINT << "read tiles " << tile << ENDL();
    }
}
)a";

static const std::string compute_kernel = R"a(
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include <debug/dprint.h>
#include <tools/profiler/kernel_profiler.hpp>

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpi.h"

inline void calculate_daisy_sum() {
    vFloat sum = 0.0f;
    for (int rBlock = 0; rBlock < 8; rBlock += 2) { // walks vec-blocks row-wise (each is 4 rows) across all faces at the same time
        sum += dst_reg[rBlock];
        sum += dst_reg[rBlock+1];
        sum += dst_reg[rBlock+8]; // face1
        sum += dst_reg[rBlock+9]; // face1
        sum += dst_reg[rBlock+16]; // face2
        sum += dst_reg[rBlock+17]; // face2
        sum += dst_reg[rBlock+24]; // face3
        sum += dst_reg[rBlock+25]; // face3
    }

    vFloat rotated = sum;

    for (int i = 0; i < 7; ++i) {
        rotated = subvec_shflror1(rotated);
        sum += rotated;
    }

    dst_reg[0] = sum;
}
#endif

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_y = 1;
    constexpr uint32_t cb_res = 2;

    uint32_t input_tiles = get_arg_val<uint32_t>(0);

    init_sfpu(cb_x, cb_res);

    DPRINT << "dot up" << ENDL();

    tile_regs_acquire();

    for (uint32_t tile = 0; tile < input_tiles; ++tile) {
        {
            UNPACK(DeviceZoneScopedN("WaitingForTiles"));
            cb_wait_front(cb_x, 1);
            cb_wait_front(cb_y, 1);
        }

        UNPACK(DPRINT << "got tiles " << tile << ENDL();)

        copy_tile(cb_x, 0, 0);
        copy_tile(cb_y, 0, 1);

        cb_pop_front(cb_x, 1);
        cb_pop_front(cb_y, 1);

        if (tile == 0) {
            mul_binary_tile(0, 1, 2);
        } else {
            mul_binary_tile(0, 1, 0);
            add_binary_tile(0, 2, 2);
        }
    }

    {
        MATH(DeviceZoneScopedN("CustomVecMath"));
        MATH(_llk_math_eltwise_unary_sfpu_params_<APPROX>(calculate_daisy_sum, 2, (int)VectorMode::RC_custom));
    }

    tile_regs_commit();

    {
        UNPACK(DeviceZoneScopedN("WritingResult"));
        tile_regs_wait();
        cb_reserve_back(cb_res, 1);

        //PACK((llk_pack_reduce_mask_config<false /*untilize*/, Reduce::RC>())); // only output first value we output 1 sFPU vector
        pack_tile(2, cb_res);

        cb_push_back(cb_res, 1);

        tile_regs_release();
    }
}
}  // namespace NAMESPACE
)a";

inline const std::string writer_kernel_1_scalar = R"a(
#include "dataflow_api.h"
#include <debug/dprint.h>

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t core_idx = get_arg_val<uint32_t>(1);
    uint32_t bytes_per_page = get_arg_val<uint32_t>(2);

    uint32_t cb_res = 2;

    constexpr auto out_tensor_args = TensorAccessorArgs<0,0>();
    const auto out_tensor = TensorAccessor(out_tensor_args, dst_addr, bytes_per_page);

    DPRINT << "writer up" << ENDL();

    {
        DeviceZoneScopedN("WaitingForData");
        cb_wait_front(cb_res, 1);
    }

    DPRINT << "got data" << ENDL();
    auto cb_rd_ptr = get_read_ptr(cb_res);
    float* val_ptr = reinterpret_cast<float*>(cb_rd_ptr);

    float sum = 0.0f;
    {
        DeviceZoneScopedN("SoftFloat");
        int t = 0;
        for (int i = 0; i < 64; i += 16) { // steps over first 4x8 (even cols) vector in first face
            float val = val_ptr[i];
            val_ptr[t++] = val;
            DPRINT << "val" << t << ": " << val << ENDL();
        }
    }


    noc_async_write_page(0, out_tensor, cb_rd_ptr, 16, core_idx * sizeof(float)*4);
    DPRINT << "wrote result: " << sum << ENDL();

    noc_async_writes_flushed();
    cb_pop_front(cb_res, 1);

    noc_async_write_barrier();
}
)a";

DotNodeDispatcher_Tenstorrent::DotNodeDispatcher_Tenstorrent(
    sdfg::codegen::LanguageExtension& language_extension,
    const sdfg::Function& function,
    const sdfg::data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::math::blas::DotNode& node
)
    : LibraryNodeDispatcherBase(language_extension, function, data_flow_graph, node) {}

std::vector<const data_flow::AccessNode*>
find_dot_access_nodes(const data_flow::DataFlowGraph& dfg, const math::blas::DotNode& node) {
    std::vector<const data_flow::AccessNode*> access_nodes(3);

    auto in_edges = dfg.in_edges(node);
    auto in_edges_it = in_edges.begin();

    while (in_edges_it != in_edges.end()) {
        auto& edge = *in_edges_it;
        auto dst_conn = edge.dst_conn();
        if (dst_conn == "__x") {
            access_nodes[0] = dynamic_cast<const data_flow::AccessNode*>(&edge.src());
        } else if (dst_conn == "__y") {
            access_nodes[1] = dynamic_cast<const data_flow::AccessNode*>(&edge.src());
        } else {
            throw InvalidSDFGException("DotNode has unexpected input: " + dst_conn);
        }
        ++in_edges_it;
    }

    auto& out_edge = *dfg.out_edges(node).begin();
    assert(out_edge.src_conn() == "__out");
    access_nodes[2] = dynamic_cast<const data_flow::AccessNode*>(&out_edge.dst());

    return access_nodes;
}

void DotNodeDispatcher_Tenstorrent::dispatch(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    emit_tt_includes_once(globals_stream, library_snippet_factory);

    auto& dot_node = static_cast<const math::blas::DotNode&>(this->node_);

    auto prim = dot_node.scalar_primitive();
    if (prim != types::Float) {
        throw std::runtime_error(
            "Tenstorrent only supports float for now. Attempted type: " +
            std::string{types::primitive_type_to_string(prim)} + "."
        );
    }
    types::Scalar base_type(prim);
    auto prim_bytes = symbolic::integer(types::bit_width(prim) / 8);

    auto& sdfg = dynamic_cast<const StructuredSDFG&>(this->function_);

    stream << " // TT Dot goes here" << std::endl;

    auto& dflow = dot_node.get_parent();
    auto access_nodes = find_dot_access_nodes(dflow, dot_node);

    std::string dev_handle_var = "tt_device";

    emit_tt_device_ready(sdfg, stream, globals_stream, library_snippet_factory, dev_handle_var, 0);
    emit_h2d_transfer_helper_once(language_extension_, globals_stream, library_snippet_factory);
    emit_d2h_transfer_helper_once(language_extension_, globals_stream, library_snippet_factory);

    if (!symbolic::eq(dot_node.incx(), symbolic::one())) {
        throw std::runtime_error("Tenstorrent only supports Dot incx=1 for now.");
    } else if (!symbolic::eq(dot_node.incy(), symbolic::one())) {
        throw std::runtime_error("Tenstorrent only supports Dot incy=1 for now.");
    }

    types::Pointer ptr_type(base_type);

    auto tile_dim = symbolic::integer(32);
    auto tile_elems = symbolic::mul(tile_dim, tile_dim);
    auto tile_bytes = symbolic::mul(tile_elems, prim_bytes);
    auto one_value = symbolic::integer(1);

    auto input_tiles = symbolic::divide_ceil(dot_node.n(), tile_elems);
    auto output_tiles = symbolic::symbol("num_cores");

    TTKernelManagementCodegen codegen(
        stream,
        library_snippet_factory,
        language_extension_,
        dev_handle_var,
        dot_node,
        {
            {access_nodes[0]->data(), access_nodes[0]->data(), prim, tile_elems, input_tiles, true},
            {access_nodes[1]->data(), access_nodes[1]->data(), prim, tile_elems, input_tiles, true},
            {"res_tt_host", "res_tt", prim, tile_elems, one_value, false, true},
        }
    );

    auto& compute_snippet = codegen.emit_predefined_kernel("tt_compute_dot", compute_kernel);
    auto fw_compute = codegen.add_kernel(compute_snippet, TTKernelTarget::Compute, {}, {LateArg{"input_tiles"}});

    auto& reader_snippet = codegen.emit_predefined_kernel("tt_reader_2_tile_streams", reader_kernel_2_tile_streams);
    auto fw_reader = codegen.add_kernel(
        reader_snippet,
        TTKernelTarget::DatMovRd,
        {LiteralArg("")},
        {
            MemArg{0, MemArgType::ADDR},
            MemArg{1, MemArgType::ADDR},
            LateArg{"begin_tile"}, // start tile offset
            LateArg{"input_tiles"} // tile count
        }
    );

    auto& writer_snippet = codegen.emit_predefined_kernel("tt_writer_dotcollect", writer_kernel_1_scalar);
    auto fw_writer = codegen.add_kernel(
        writer_snippet,
        TTKernelTarget::DatMovWr,
        {},
        {MemArg{2, MemArgType::ADDR}, LateArg{"core_idx"}, LateArg{"bytes_per_page"}}
    );

    codegen.emit_default_size_distribution(input_tiles);

    auto num_results = symbolic::mul(symbolic::symbol(codegen.get_num_used_cores()), symbolic::integer(4));

    auto resBytes = symbolic::mul(num_results, prim_bytes);
    auto intType = types::Scalar(types::PrimitiveType::UInt32);
    stream << language_extension_.declaration("resBytes", intType) << " = " << language_extension_.expression(resBytes)
           << ";" << std::endl;
    stream << "std::shared_ptr<tt::tt_metal::Buffer> " << "res_tt;" << std::endl;
    TTDataOffloadingNodeDispatcher::dispatch_allocate(
        stream, globals_stream, library_snippet_factory, language_extension_, "res_tt", dev_handle_var, resBytes, resBytes
    );

    codegen.emit_buffer_setup_code();

    stream << std::endl;
    stream << std::endl;

    codegen.emit_kernel_tensor_addr_args(fw_reader);
    auto k_reader = codegen.emit_kernel_load(fw_reader, codegen.get_used_cores(), nullptr, false);
    codegen.emit_kernel_set_common_runtime_args(k_reader);

    codegen.emit_kernel_tensor_addr_args(fw_writer);
    auto k_writer = codegen.emit_kernel_load(fw_writer, codegen.get_used_cores(), nullptr, false);
    codegen.emit_kernel_set_common_runtime_args(k_writer);

    auto k_compute = codegen.emit_kernel_load(fw_compute, codegen.get_used_cores(), nullptr, false);

    codegen.emit_kernel_set_runtime_args(
        k_compute, codegen.get_main_cores(), {{"input_tiles", codegen.get_units_per_main()}}
    );

    stream << "if (!" << codegen.get_rem_cores() << ".ranges().empty()) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    codegen
        .emit_kernel_set_runtime_args(k_compute, codegen.get_rem_cores(), {{"input_tiles", codegen.get_units_per_rem()}});

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;

#ifdef RUNTIME_DEBUG_OUT
    stream << "std::cout << \"layout: \" << " << codegen.get_units_per_main() << " << \" on \" << "
           << codegen.get_main_cores() << ".str() << \", \" << " << codegen.get_units_per_rem() << " << \" on \" << "
           << codegen.get_rem_cores() << ".str() << std::endl;" << std::endl;
#endif

    auto [tile_consumed_var, tiles_on_core_var] = codegen.get_default_distribution_vars();
    auto core_idx_var = "core_idx";

    std::unordered_map<std::string, std::string> core_args{
        {"begin_tile", tile_consumed_var},
        {"input_tiles", tiles_on_core_var},
        {"core_idx", core_idx_var},
        {"bytes_per_page", "resBytes"}
    };

    stream << "uint32_t " << core_idx_var << " = 0;" << std::endl;

    codegen.emit_per_core_config([&]() {
#ifdef RUNTIME_DEBUG_OUT
        stream << "std::cout << \"core (\" << core.x << \",\" << core.y << \"): \" << tiles_consumed << \", \" << "
                  "tiles_on_core << "
                  "std::endl;"
               << std::endl;
#endif

        codegen.emit_kernel_set_runtime_args(k_reader, "core", core_args);
        codegen.emit_kernel_set_runtime_args(k_writer, "core", core_args);
        stream << "++" << core_idx_var << ";" << std::endl;
    });

    codegen.emit_launch(false);

    stream << std::endl;

    auto arrType = types::Array(base_type, num_results);
    stream << language_extension_.declaration("res_tt_host", ptr_type) << " = new "
           << language_extension_.declaration("", arrType) << ";" << std::endl;

    TTDataOffloadingNodeDispatcher::dispatch_enqueue_read_full(
        stream, globals_stream, library_snippet_factory, "res_tt", "res_tt_host", dev_handle_var, true
    );
    stream << language_extension_.declaration("res_tt_acc", base_type) << " = 0.0;" << std::endl;
    stream << "for (int i = 0; i < " << language_extension_.expression(num_results) << "; ++i) {" << std::endl;
    stream.changeIndent(+4);
#ifdef RUNTIME_DEBUG_OUT
    stream << "std::cout << \"res_tt_host[\" << i << \"] = \" << res_tt_host[i] << std::endl;" << std::endl;
#endif
    stream << "res_tt_acc += res_tt_host[i];" << std::endl;

    stream.changeIndent(-4);
    stream << "}" << std::endl;
    stream << std::endl;

    auto& out_edge = *dflow.out_edges(dot_node).begin();
    data_flow::MemletType destType = out_edge.type();
    if (destType == data_flow::MemletType::Computational) {
        stream << access_nodes[2]->data();
    } else if (destType == data_flow::MemletType::Dereference_Dst) {
        stream << "*" << access_nodes[2]->data();
    }
    stream << " = res_tt_acc;" << std::endl;
    stream << std::endl;
    stream << "delete[] res_tt_host;" << std::endl;

    stream << std::endl;
    stream << "double tt_num_cores_used = static_cast<double>(" << codegen.get_num_used_cores() << ");" << std::endl;
    if (tt_emit_full_metrics) {
        stream << "double tt_num_cores_used_primary = static_cast<double>(" << codegen.get_main_cores()
               << ".num_cores());" << std::endl;
        stream << "double tt_work_units_primary = static_cast<double>(" << codegen.get_units_per_main() << ");"
               << std::endl;
    }
    stream << "double tt_num_cores_available = static_cast<double>(" << codegen.get_num_avail_cores() << ");"
           << std::endl;
    stream << "double tt_cores_used_rel = tt_num_cores_used / tt_num_cores_available;" << std::endl;
    stream << "double tt_work_units_per_core = static_cast<double>(" << tile_consumed_var
           << ") / tt_num_cores_available;" << std::endl;

    stream << std::endl;
    if (tt_force_close_devices_after_kernel) { // WARNING: This breaks with more than 1 kernel per function. We do not
                                               // currently have
        // sth. to issue each exit
        stream << "daisy::tenstorrent::daisy_force_close_tt_device(tt_device);" << std::endl;
        stream << std::endl;
    }
}

codegen::InstrumentationInfo DotNodeDispatcher_Tenstorrent::instrumentation_info() const {
    std::unordered_map<std::string, std::string> metrics;

    auto flops = analysis::FlopAnalysis::get_flops_if_valid_for_codegen(node_);
    if (!flops.is_null()) {
        metrics.insert({"flop", language_extension_.expression(flops)});
    }

    metrics.insert({"tt_cores_used_rel", "tt_cores_used_rel"});
    metrics.insert({"tt_work_units_per_core", "tt_work_units_per_core"});
    if (tt_emit_full_metrics) {
        metrics.insert({"tt_num_cores_used", "tt_num_cores_used"});
        metrics.insert({"tt_num_cores_used_primary", "tt_num_cores_used_primary"});
        metrics.insert({"tt_work_units_primary", "tt_work_units_primary"});
        metrics.insert({"tt_num_cores_available", "tt_num_cores_available"});
    }
    return {node_.element_id(), codegen::ElementType_Math, TargetType_Tenstorrent, analysis::LoopInfo{}, metrics};
}
} // namespace sdfg::tenstorrent::blas
