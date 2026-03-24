#include "sdfg/targets/tenstorrent/blas/gemm.h"

#include "sdfg/targets/tenstorrent/codegen.h"
#include "sdfg/targets/tenstorrent/kernels/generic_writer_unary_interleaved.h"
#include "sdfg/targets/tenstorrent/schedule.h"
#include "sdfg/targets/tenstorrent/tenstorrent_offloading_node.h"
#include "sdfg/transformations/offloading/offload_transform.h"

namespace sdfg::tenstorrent::blas {

GEMMNodeDispatcher_Tenstorrent::GEMMNodeDispatcher_Tenstorrent(
    sdfg::codegen::LanguageExtension& language_extension,
    const sdfg::Function& function,
    const sdfg::data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::math::blas::GEMMNode& node
)
    : LibraryNodeDispatcherBase(language_extension, function, data_flow_graph, node) {}

std::vector<const data_flow::AccessNode*>
find_gemm_access_nodes(const data_flow::DataFlowGraph& dfg, const math::blas::GEMMNode& node) {
    std::vector<const data_flow::AccessNode*> access_nodes(6);

    auto in_edges = dfg.in_edges(node);
    auto in_edges_it = in_edges.begin();

    while (in_edges_it != in_edges.end()) {
        auto& edge = *in_edges_it;
        auto dst_conn = edge.dst_conn();
        if (dst_conn == "__A") {
            access_nodes[0] = dynamic_cast<const data_flow::AccessNode*>(&edge.src());
        } else if (dst_conn == "__B") {
            access_nodes[1] = dynamic_cast<const data_flow::AccessNode*>(&edge.src());
        } else if (dst_conn == "__C") {
            access_nodes[2] = dynamic_cast<const data_flow::AccessNode*>(&edge.src());
        } else if (dst_conn == "__alpha") {
            access_nodes[3] = dynamic_cast<const data_flow::AccessNode*>(&edge.src());
        } else if (dst_conn == "__beta") {
            access_nodes[4] = dynamic_cast<const data_flow::AccessNode*>(&edge.src());
        } else {
            throw InvalidSDFGException("GEMMNode has unexpected input: " + dst_conn);
        }
        ++in_edges_it;
    }

    auto& out_edge = *dfg.out_edges(node).begin();
    access_nodes[5] = dynamic_cast<const data_flow::AccessNode*>(&out_edge.dst());

    return access_nodes;
}


std::pair<symbolic::Expression, symbolic::Expression> GEMMNodeDispatcher_Tenstorrent::emit_padded_size(
    codegen::PrettyPrinter& stream, const std::string& var_name, const symbolic::Expression size, int pad_to_mul
) const {
    auto p = symbolic::integer(pad_to_mul);
    auto count = symbolic::div(symbolic::add(size, symbolic::sub(p, symbolic::one())), p);
    auto new_total = symbolic::mul(count, p);
    auto add_padding = symbolic::sub(new_total, size);
    stream << "uint32_t " << var_name << " = " << language_extension_.expression(new_total) << ";" << std::endl;
    return {new_total, add_padding};
}

static const std::string reader_kernel = R"a(
#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    // same arg indices as in reader_binary_diff_lenghts for compat
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t MtKt = get_arg_val<uint32_t>(5);  // if 0
    uint32_t KtNt = get_arg_val<uint32_t>(6);
    uint32_t batch = get_arg_val<uint32_t>(7);
    uint32_t bcast_B = get_arg_val<uint32_t>(8);  // if 1 we broadcast B to batch
    uint32_t output_tile_start_id = get_arg_val<uint32_t>(9);
    uint32_t num_output_tiles = get_arg_val<uint32_t>(10);
    uint32_t MtNt = get_arg_val<uint32_t>(11);
    uint32_t c_in_addr = get_arg_val<uint32_t>(12);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool src1_is_dram = get_compile_time_arg_val(1) == 1;

    // DPRINT << "Mt=" << Mt << " Kt=" << Kt << " Nt=" << Nt << " Caddr=" << HEX() << c_in_addr << DEC() << ENDL();
    // DPRINT << "batch=" << batch << << " out_start=" << output_tile_start_id << " out_num=" << num_output_tiles << ENDL();

    constexpr uint8_t cb_id_a = 0;
    constexpr uint8_t cb_id_b = 1;
    constexpr uint8_t cb_id_in_c = 2;

    constexpr uint32_t onetile = 1;
    const uint32_t in0_tile_bytes = get_tile_size(cb_id_a);
    const DataFormat in0_data_format = get_dataformat(cb_id_a);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_b);
    const DataFormat in1_data_format = get_dataformat(cb_id_b);
    const uint32_t c_in_tile_bytes = get_tile_size(cb_id_in_c);
    const DataFormat c_in_data_format = get_dataformat(cb_id_in_c);

    uint32_t itileA = output_tile_start_id / Nt * Kt;  // input0 row = output row * input0 width

    // Keep track of end of output row and end of output batch
    uint32_t outbatch = output_tile_start_id % MtNt;
    uint32_t itileB_batch = output_tile_start_id % Nt;
    uint32_t itileB = itileB_batch;  // input1 col = output col if we are bcasting
    if (bcast_B == 0) {
        itileB += output_tile_start_id / MtNt * KtNt;  // offset into correct batch if not bcasting
    }

    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr, .page_size = in0_tile_bytes, .data_format = in0_data_format
    };

    const InterleavedAddrGenFast<src1_is_dram> s1 = {
        .bank_base_address = src1_addr, .page_size = in1_tile_bytes, .data_format = in1_data_format
    };

    const InterleavedAddrGenFast<src1_is_dram> c_addr_gen = {
        .bank_base_address = c_in_addr, .page_size = c_in_tile_bytes, .data_format = c_in_data_format
    };

    for (uint32_t n = 0; n < num_output_tiles; n++) {
        if (c_in_addr) {
            cb_reserve_back(cb_id_in_c, onetile);
            uint32_t l1_write_addr_in_c = get_write_ptr(cb_id_in_c);
            noc_async_read_tile(output_tile_start_id + n, c_addr_gen, l1_write_addr_in_c);
        }
        for (uint32_t kt = 0; kt < Kt; kt++) {
            {  // Read A's tile at (mt, kt)
                cb_reserve_back(cb_id_a, onetile);
                uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_a);
                noc_async_read_tile(itileA, s0, l1_write_addr_in0);
            }

            {  // Read B's tile at (kt, nt)
                cb_reserve_back(cb_id_b, onetile);
                uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_b);
                noc_async_read_tile(itileB, s1, l1_write_addr_in1);
            }

            noc_async_read_barrier();
            cb_push_back(cb_id_a, onetile);
            cb_push_back(cb_id_b, onetile);

            // DPRINT << "Pushed itileA=" << itileA << " itileB=" << itileB << ENDL();

            itileA += 1;   // A is MK
            itileB += Nt;  // B is KN, so to get k++ we stride by Nt
        }  // Kt loop

        if (c_in_addr) {
            // we already had multiple rd barriers above
            cb_push_back(cb_id_in_c, onetile);
        }

        outbatch += 1;
        itileB_batch += 1;
        itileB -= KtNt;  // revert B to previous state before the K loop (to avoid multiplies)
        itileB += 1;     // Move to next B col

        if (itileB_batch == Nt) {
            itileB_batch = 0;
            itileB -= Nt;  // Go back to first column in batch
            if (outbatch == MtNt) {
                if (bcast_B == 0) {
                    itileB += KtNt;  // Move B to start of next batch
                }
                outbatch = 0;
            }
        } else {
            itileA -= Kt;  // resets tileA to kt=0, keep the same mt
        }
    }  // batch loop
}
)a";

static const std::string compute_kernel = R"a(
#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/tile_move_copy.h"
#include <debug/dprint.h>

using std::uint32_t;

namespace NAMESPACE {
void MAIN {
    constexpr int onetile = 1;

    constexpr int cb_a = 0;
    constexpr int cb_b = 1;
    constexpr int cb_in_c = 2;
    constexpr int cb_out_c = 3;

    uint32_t batch = get_arg_val<uint32_t>(0);
    uint32_t Mt = get_arg_val<uint32_t>(1);
    uint32_t Kt = get_arg_val<uint32_t>(2);
    uint32_t Nt = get_arg_val<uint32_t>(3);
    uint32_t alpha = get_arg_val<uint32_t>(4);
    uint32_t beta = get_arg_val<uint32_t>(5);


    int dst_tile_index = 0;
    int in0_block_tile_index = 0;

    mm_init(cb_a, cb_b, cb_out_c);

    binop_with_scalar_tile_init();

    // the simplest possible version of outer product blocked matmul
    // the reader is expected to read the A's and B's tile rows and tile columns for each output tile
    for (uint32_t nb = 0; nb < batch; nb++) {
        for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C) {    // output tile of C
            for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C)  // output tile index of C
            {
                tile_regs_acquire();
                tile_regs_wait();

                for (uint32_t kt = 0; kt < Kt; kt++) {
                    cb_wait_front(cb_a, onetile);
                    cb_wait_front(cb_b, onetile);

                    matmul_tiles(cb_a, cb_b, 0, 0, 0, false);

                    cb_pop_front(cb_a, onetile);
                    cb_pop_front(cb_b, onetile);
                }

                if (alpha != 0x3f800000) { // != 1.0f
                    mul_unary_tile(0, alpha);
                }

                if (beta != 0x0) { // != 0.0f
                    copy_tile_init(cb_in_c);
                    cb_wait_front(cb_in_c, onetile);

                    copy_tile(cb_in_c, 0, 2);

                    if (beta != 0x3f800000) {
                        mul_unary_tile(2, beta);
                    }

                    // binary and unary sfpu ops only statically configure the FPU and the data format, this does not change!
                    //add_binary_tile_init();
                    // this is SFPU add. So higher precision, but also inefficient, especially because we move the inputs through the FPU Src regs.
                    // There is FPU ELWMUL that can mul SrcA and SrcB and accumulate in Dst, so ideally, we would put a factor-matrix into one src and stream the in_c in, while multiplying and adding to DST in one go. But there is no premade streaming config for this, they all stream A as well.
                    add_binary_tile(0, 2, 0);

                    cb_pop_front(cb_in_c, onetile);

                    // reset config back to matmul
                    mm_init(cb_a, cb_b, cb_out_c);

                    //binop_with_scalar_tile_init();
                }

                tile_regs_commit();

                cb_reserve_back(cb_out_c, onetile);
                pack_tile(0, cb_out_c);
                cb_push_back(cb_out_c, onetile);

                tile_regs_release();
            }
        }
    }
}
}  // namespace NAMESPACE
)a";

void GEMMNodeDispatcher_Tenstorrent::dispatch(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    emit_tt_includes_once(globals_stream, library_snippet_factory);

    auto& gemm_node = static_cast<const math::blas::GEMMNode&>(this->node_);

    auto prim = gemm_node.scalar_primitive();
    if (prim != types::Float) {
        throw std::runtime_error(
            "Tenstorrent only supports float for now. Attempted type: " +
            std::string{types::primitive_type_to_string(prim)} + "."
        );
    }
    types::Scalar base_type(prim);
    auto prim_bytes = symbolic::integer(types::bit_width(prim) / 8);

    auto& sdfg = dynamic_cast<const StructuredSDFG&>(this->function_);

    auto& dflow = gemm_node.get_parent();
    auto access_nodes = find_gemm_access_nodes(dflow, gemm_node);

    std::string dev_handle_var = "tt_device";

    emit_tt_device_ready(sdfg, stream, globals_stream, library_snippet_factory, dev_handle_var, 0);

    std::string M_tt_var = "m_tt";
    auto [M_padded, M_padding] = emit_padded_size(stream, M_tt_var, gemm_node.m(), 32);
    std::string N_tt_var = "n_tt";
    auto [N_padded, N_padding] = emit_padded_size(stream, N_tt_var, gemm_node.n(), 32);
    std::string K_tt_var = "k_tt";
    auto [K_padded, K_padding] = emit_padded_size(stream, K_tt_var, gemm_node.k(), 32);

    auto alpha_var = require_param_as_var_equivalent(stream, access_nodes[3], "alpha");
    auto beta_var = require_param_as_var_equivalent(stream, access_nodes[4], "beta");

    bool beta_is_zero_static = false;
    std::string beta_is_zero_var;
    if (auto const_node = dynamic_cast<const data_flow::ConstantNode*>(access_nodes[4])) {
        if (const_node->data() == "0.0") { // beta is statically known 0
            beta_is_zero_static = true;
        }
    }
    if (!beta_is_zero_static) {
        beta_is_zero_var = "beta_is_zero";
        stream << "bool " << beta_is_zero_var << " = " << beta_var << " == static_cast<"
               << language_extension_.declaration("", base_type) << ">(0.0);" << std::endl;
    }

    types::Pointer ptr_type(base_type);

    // TT requires different data layout (tilized, in 32x32 sub matrixes as to split tiles across memory banks with max.
    // locality within each) we need to reorder data on the host, these are buffers where to store the reorganized data.
    // We also 0-pad it to multiples of 32x32 in the same process
    // TODO once we model data movement, we could use a global buffer of same or larger size, which is no longer used by
    // an async/bg transfer these buffers are unused after the transfer into device memory completed.
    auto A_elems = symbolic::mul(symbolic::symbol(M_tt_var), symbolic::symbol(K_tt_var));
    stream << language_extension_.declaration("A_tt_tilized", ptr_type) << " = " << "new "
           << language_extension_.declaration("", types::Array(base_type, A_elems)) << ";" << std::endl;
    auto B_elems = symbolic::mul(symbolic::symbol(K_tt_var), symbolic::symbol(N_tt_var));
    stream << language_extension_.declaration("B_tt_tilized", ptr_type) << " = " << "new "
           << language_extension_.declaration("", types::Array(base_type, B_elems)) << ";" << std::endl;
    auto C_elems = symbolic::mul(symbolic::symbol(M_tt_var), symbolic::symbol(N_tt_var));
    stream << language_extension_.declaration("C_tt_tilized", ptr_type) << " = " << "new "
           << language_extension_.declaration("", types::Array(base_type, C_elems)) << ";" << std::endl;
    // shared between input and output as currently not exposed as only tempory for data conversion

    emit_tilized_padded_copy_helper(language_extension_, globals_stream, library_snippet_factory);
    emit_untilized_unpadded_copy_helper(language_extension_, globals_stream, library_snippet_factory);


    auto tile_dim = symbolic::integer(32);
    auto tile_elems = symbolic::mul(tile_dim, tile_dim);

    stream << "tilized_padded_copy<" << language_extension_.primitive_type(prim) << ">(" << access_nodes[0]->data()
           << ", " << language_extension_.expression(gemm_node.m()) << ", "
           << language_extension_.expression(gemm_node.k()) << ", " << language_extension_.expression(gemm_node.lda())
           << ", " << "A_tt_tilized, " << M_tt_var << ", " << K_tt_var << ", 32, 32, 16, 16);" << std::endl;

    auto tile_size = symbolic::mul(tile_elems, prim_bytes);

    stream << "std::shared_ptr<tt::tt_metal::Buffer> " << "A_tt;" << std::endl;
    TTDataOffloadingNodeDispatcher::dispatch_allocate(
        stream,
        globals_stream,
        library_snippet_factory,
        language_extension_,
        "A_tt",
        dev_handle_var,
        symbolic::mul(B_elems, prim_bytes),
        tile_size
    );
    TTDataOffloadingNodeDispatcher::dispatch_enqueue_write_full(
        stream, globals_stream, library_snippet_factory, "A_tt_tilized", "A_tt", dev_handle_var, false
    );
    stream << "tilized_padded_copy<" << language_extension_.primitive_type(prim) << ">(" << access_nodes[1]->data()
           << ", " << language_extension_.expression(gemm_node.k()) << ", "
           << language_extension_.expression(gemm_node.n()) << ", " << language_extension_.expression(gemm_node.ldb())
           << ", " << "B_tt_tilized, " << K_tt_var << ", " << N_tt_var << ", 32, 32, 16, 16);" << std::endl;
    stream << "std::shared_ptr<tt::tt_metal::Buffer> " << "B_tt;" << std::endl;
    TTDataOffloadingNodeDispatcher::dispatch_allocate(
        stream,
        globals_stream,
        library_snippet_factory,
        language_extension_,
        "B_tt",
        dev_handle_var,
        symbolic::mul(A_elems, prim_bytes),
        tile_size
    );
    TTDataOffloadingNodeDispatcher::dispatch_enqueue_write_full(
        stream, globals_stream, library_snippet_factory, "B_tt_tilized", "B_tt", dev_handle_var, false
    );

    stream << "std::shared_ptr<tt::tt_metal::Buffer> " << "C_in_tt;" << std::endl;
    if (!beta_is_zero_static) {
        stream << "if (!" << beta_is_zero_var << ") {" << std::endl;
        stream.setIndent(stream.indent() + 4);
        // TODO combine in_C and out_C if the same to save memory. If we can pre-initialize DST from in_C, we can do
        // this almost for free, as TT hardware always adds the matrix-mul sum to DST and HAS to zero DST first
        // otherwise
        stream << "tilized_padded_copy<" << language_extension_.primitive_type(prim) << ">(" << access_nodes[2]->data()
               << ", " << language_extension_.expression(gemm_node.m()) << ", "
               << language_extension_.expression(gemm_node.n()) << ", "
               << language_extension_.expression(gemm_node.ldc()) << ", " << "C_tt_tilized, " << M_tt_var << ", "
               << N_tt_var << ", 32, 32, 16, 16);" << std::endl;
        TTDataOffloadingNodeDispatcher::dispatch_allocate(
            stream,
            globals_stream,
            library_snippet_factory,
            language_extension_,
            "C_in_tt",
            dev_handle_var,
            symbolic::mul(C_elems, prim_bytes),
            tile_size
        );
        TTDataOffloadingNodeDispatcher::dispatch_enqueue_write_full(
            stream, globals_stream, library_snippet_factory, "C_tt_tilized", "C_in_tt", dev_handle_var, false
        );

        stream.setIndent(stream.indent() - 4);
        stream << "}" << std::endl;
    }

    stream << "std::shared_ptr<tt::tt_metal::Buffer> " << "C_out_tt;" << std::endl;
    TTDataOffloadingNodeDispatcher::dispatch_allocate(
        stream,
        globals_stream,
        library_snippet_factory,
        language_extension_,
        "C_out_tt",
        dev_handle_var,
        symbolic::mul(C_elems, prim_bytes),
        tile_size
    );

    auto Mt = symbolic::div(M_padded, tile_dim);
    auto Nt = symbolic::div(N_padded, tile_dim);
    auto Kt = symbolic::div(K_padded, tile_dim);
    auto mt_nt_exp = symbolic::mul(Mt, Nt);
    auto mt_kt_exp = symbolic::mul(Mt, Kt);
    auto kt_nt_exp = symbolic::mul(Kt, Nt);

    TTKernelManagementCodegen codegen(
        stream,
        library_snippet_factory,
        language_extension_,
        dev_handle_var,
        gemm_node,
        {
            {access_nodes[0]->data(), "A_tt", prim, tile_elems, symbolic::mul(Mt, Kt), true},
            {access_nodes[1]->data(), "B_tt", prim, tile_elems, symbolic::mul(Kt, Nt), true},
            {access_nodes[2]->data(), "C_in_tt", prim, tile_elems, symbolic::mul(Mt, Nt), true},
            {access_nodes[5]->data(), "C_out_tt", prim, tile_elems, symbolic::mul(Mt, Nt), false},
        }
    );

    auto& compute_snippet = codegen.emit_predefined_kernel("tt_compute_gemm_simple", compute_kernel);
    auto fw_compute = codegen.add_kernel(
        compute_snippet,
        TTKernelTarget::Compute,
        {},
        {LiteralArg{"1"},
         LiteralArg{"1"},
         ExprArg{Kt},
         LateArg{"work_units"},
         LiteralArg{"*reinterpret_cast<uint32_t*>(&" + alpha_var + ")"},
         LiteralArg{"*reinterpret_cast<uint32_t*>(&" + beta_var + ")"}}
    );

    auto& reader_snippet = codegen.emit_predefined_kernel("tt_reader_gemm_simple", reader_kernel);
    auto fw_reader = codegen.add_kernel(
        reader_snippet,
        TTKernelTarget::DatMovRd,
        {LiteralArg{"1"}, LiteralArg{"1"}},
        {
            MemArg{0, MemArgType::ADDR},
            MemArg{1, MemArgType::ADDR},
            ExprArg{Mt},
            ExprArg{Kt},
            ExprArg{Nt},
            ExprArg{mt_kt_exp},
            ExprArg{kt_nt_exp},
            LiteralArg{"1"}, // batch count
            LiteralArg{"0"}, // bcast
            LateArg{"begin_tile"}, // start tile offset
            LateArg{"tile_count"}, // tile count
            ExprArg{mt_nt_exp},
            LateArg{"c_in_addr"},
        }
    );

    auto& writer_snippet = codegen.emit_predefined_kernel("tt_writer_mat", generic_tt_writer_unary_interleaved);
    auto fw_writer = codegen.add_kernel(
        writer_snippet,
        TTKernelTarget::DatMovWr,
        {MemArg{3, MemArgType::CB_ID}, LiteralArg{"1"}},
        {MemArg{3, MemArgType::ADDR}, LateArg{"tile_count"}, LateArg{"begin_tile"}}
    );

    codegen.emit_default_size_distribution(mt_nt_exp);

    codegen.emit_buffer_setup_code();

    stream << std::endl;
    stream << std::endl;

    auto k_reader = codegen.emit_kernel_load(fw_reader, codegen.get_used_cores(), nullptr, false);
    auto k_writer = codegen.emit_kernel_load(fw_writer, codegen.get_used_cores(), nullptr, false);
    auto k_compute = codegen.emit_kernel_load(fw_compute, codegen.get_used_cores(), nullptr, false);

    codegen
        .emit_kernel_set_runtime_args(k_compute, codegen.get_main_cores(), {{"work_units", codegen.get_units_per_main()}});

    stream << "if (!" << codegen.get_rem_cores() << ".ranges().empty()) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    codegen
        .emit_kernel_set_runtime_args(k_compute, codegen.get_rem_cores(), {{"work_units", codegen.get_units_per_rem()}});

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;

#ifdef RUNTIME_DEBUG_OUT
    stream << "std::cout << \"layout: \" << " << codegen.get_units_per_main() << " << \" on \" << "
           << codegen.get_main_cores() << ".str() << \", \" << " << codegen.get_units_per_rem() << " << \" on \" << "
           << codegen.get_rem_cores() << ".str() << std::endl;" << std::endl;
#endif

    auto [tile_consumed_var, tiles_on_core_var] = codegen.get_default_distribution_vars();

    codegen.emit_per_core_config([&]() {
#ifdef RUNTIME_DEBUG_OUT
        stream << "std::cout << \"core \" << core.str() << \": \" << num_tiles_written << \", \" << tiles_on_core << "
                  "std::endl;"
               << std::endl;
#endif

        auto c_in_addr_var = beta_is_zero_static ? "0"
                                                 : (beta_is_zero_var +
                                                    "? 0 : " + codegen.get_managed_membuf(2).get_device_address_var());
        std::unordered_map<std::string, std::string> core_args{
            {"begin_tile", tile_consumed_var}, {"tile_count", tiles_on_core_var}, {"c_in_addr", c_in_addr_var}
        };
        codegen.emit_kernel_set_runtime_args(k_reader, "core", core_args);
        codegen.emit_kernel_set_runtime_args(k_writer, "core", core_args);
    });

    codegen.emit_launch(false);

    TTDataOffloadingNodeDispatcher::dispatch_enqueue_read_full(
        stream, globals_stream, library_snippet_factory, "C_out_tt", "C_tt_tilized", dev_handle_var, true
    );
    stream << "untilized_unpadded_copy<" << language_extension_.primitive_type(prim) << ">(" << access_nodes[5]->data()
           << ", " << language_extension_.expression(gemm_node.m()) << ", "
           << language_extension_.expression(gemm_node.n()) << ", " << language_extension_.expression(gemm_node.ldc())
           << ", " << "C_tt_tilized, " << M_tt_var << ", " << N_tt_var << ", 32, 32, 16, 16);" << std::endl;
}

} // namespace sdfg::tenstorrent::blas
