#include "docc/target/et/blas/gemm.h"

#include "docc/target/et/target.h"

namespace docc::target::et::blas {

static const std::string kernel_stream_setup = R"a(
        auto& et = docc::rt::et::EtRuntimeWrapper::get_instance();
        auto& et_rt = et.get_runtime();
        auto et_dev = et.get_device();
        auto et_stream = et_rt.createStream(et_dev);
)a";

static const std::string kernel_launch_template_blocking = R"a(
        auto* traceDevBuf = et.alloc_trace_buffer(et_dev);

        ::rt::KernelLaunchOptions launchOpts;
        //launchOpts.setShireMask(et_shire_mask);
        launchOpts.setBarrier(true);
        launchOpts.setUserTracing(
            reinterpret_cast<uint64_t>(traceDevBuf),
            static_cast<uint32_t>(et.DEFAULT_TRACE_BUFFER_SIZE),
            0,                              // threshold
            0xFFFFFFFFUL,                     // trace shireMask
            0xFFFFFFFFFFFFFFFFULL,          // threadMask — all threads
            0xFFFFFFFFU,                    // eventMask — all events
            0xFFFFFFFFU                     // filterMask — all levels
        );

        auto k_launch = et_rt.kernelLaunch(et_stream, et_k, reinterpret_cast<std::byte*>(k_args.data()), (k_args.size()*sizeof(decltype(k_args)::value_type)), launchOpts);
        et_rt.waitForStream(et_stream);

        et.dump_trace_outputs(et_dev, et_stream, traceDevBuf);
)a";

static const std::string kernel_cleanup = R"a(
        et_rt.unloadCode(et_k);
        et_rt.destroyStream(et_stream);
)a";

static const std::string matul_fp32_kernel_simple = R"a(
#include <etsoc/common/utils.h>
#include <stdint.h>

#include "et_tensor.hpp"

constexpr et::scp_region<0,  16, et::fp32> scp_a;
constexpr et::scp_region<16, 16, et::fp32> scp_b;
constexpr unsigned TILE = 16;
constexpr unsigned NUM_HARTS = 1024;

typedef struct {
    float* a_ptr;
    float* b_ptr;
    float* c_ptr;
    float alpha;
    float beta;
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t lda;
    int32_t ldb;
    int32_t ldc;
} Parameters;

extern "C"
int entry_point(const Parameters* params) {
    uint64_t hart_id = get_hart_id();
    if (hart_id & 1) return 0;
    uint64_t gid = ((hart_id >> 6) << 5) + ((hart_id >> 1) & 0x1F);

    const int64_t K = params->k;
    const int64_t M = params->m;
    const int64_t N = params->n;
    const int64_t ne2_0 = 1, ne3_0 = 1;
    const int64_t ne2_1 = 1, ne3_1 = 1;

    // Byte strides for batch dimensions, converted to float offsets
    const int64_t bs2_0 = M * params->lda;
    const int64_t bs3_0 = bs2_0;
    const int64_t bs2_1 = K * params->ldb;
    const int64_t bs3_1 = bs2_1;
    const int64_t bs2_d = M * params->ldc;
    const int64_t bs3_d = bs2_d;

    // Row strides in bytes — for tensor load/store hardware
    const uint64_t stride_s0 = (uint64_t)params->lda * sizeof(float);
    const uint64_t stride_s1 = (uint64_t)params->ldb * sizeof(float);
    const uint64_t stride_d  = (uint64_t)params->ldc * sizeof(float);

    // Row strides in floats — for pointer arithmetic
    const int64_t rs0 = (int64_t)(stride_s0 / sizeof(float));
    const int64_t rs1 = (int64_t)(stride_s1 / sizeof(float));
    const int64_t rd  = (int64_t)(stride_d  / sizeof(float));

    const float* src0_base = (const float*)params->a_ptr;
    const float* src1_base = (const float*)params->b_ptr;
    float*       dst_base  = (float*)params->c_ptr;

    et::setup_l1scp();
    et::clear_tensor_error();

    const int64_t m_tiles = M / TILE;
    const int64_t n_tiles = (N + TILE - 1) / TILE;
    const int64_t tpb     = m_tiles * n_tiles;
    const int64_t batches = ne2_1 * ne3_1;
    const int64_t total   = batches * tpb;
    const int64_t r2 = ne2_1 / ne2_0, r3 = ne3_1 / ne3_0;

    et::matmul_result<et::fp32> c;

    for (int64_t tile = gid; tile < total; tile += NUM_HARTS) {
        const int64_t bi = tile / tpb, ti = tile % tpb;
        const int64_t ni = ti / m_tiles, mi = ti % m_tiles;
        const int64_t i3 = bi / ne2_1, i2 = bi % ne2_1;

        const float* s0 = src0_base + (i3/r3)*bs3_0 + (i2/r2)*bs2_0;
        const float* s1 = src1_base + i3*bs3_1 + i2*bs2_1;
        float*       d  = dst_base  + i3*bs3_d + i2*bs2_d;

        const int64_t mb = mi * TILE, nb = ni * TILE;
        const unsigned n_cur = (nb + TILE <= N) ? TILE : (unsigned)(N - nb);

        for (int64_t kb = 0; kb < K; kb += TILE) {
            bool first = kb == 0;
            const unsigned k_cur = (kb + TILE <= K) ? TILE : (unsigned)(K - kb);

            // There are 2 load ports - we use 0 for matrix A and 1 for matrix B
            // these data are typed so loading the wrong format is impossible
            auto la = scp_a.load<0>(&s0[mb * rs0 + kb], k_cur, stride_s0);
            auto lb = scp_b.load<1>(&s1[kb * rs1 + nb], n_cur, stride_s1);
            // Loads are async so we wait for completion
            la.wait();
            lb.wait();

            // Perform matrix multiplication
            c = et::matmul(scp_a, scp_b, n_cur, k_cur, first);
            // Wait for completion
            c.wait();
        }

        // Write back to memory
        c.store(&d[mb * rd + nb], n_cur, stride_d).wait();
    }

    if (gid < total) {
        et_printf("working\n");
    }

    auto error = et::get_tensor_error();
    if (error) {
        et_printf("Tensor error: %016lx\n", error.raw);
        return 1;
    }

    et::fence();
    return 0;
}
)a";

static const std::string gemm_s_kernel_single = R"a(
#include <stdint.h>
#include "etsoc/isa/hart.h"
#include "etsoc/isa/atomic.h"
#include "etsoc/common/utils.h"

typedef struct {
    float* a_ptr;
    float* b_ptr;
    float* c_ptr;
    float alpha;
    float beta;
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t lda;
    int32_t ldb;
    int32_t ldc;
} Parameters;

extern "C" int64_t entry_point(const Parameters* param);

int64_t entry_point(const Parameters* const param) {
    int h = get_hart_id();

    float* a_ptr = param->a_ptr;
    float* b_ptr = param->b_ptr;
    float* c_ptr = param->c_ptr;
    int a_line_w = param->lda;
    int b_line_w = param->ldb;
    int c_line_w = param->ldc;
    float alpha = param->alpha;
    float beta = param->beta;

    if (h < 1) {
        int m_start = 0;
        int m_end = param->m;
        et_printf("%d..%d x %d..%d: %d %d\n", m_start, m_end, 0, param->n, a_line_w, c_line_w);
        for (int m = m_start; m < m_end; ++m) {
            for (int n = 0; n < param->n; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < param->k; ++k) {
                    acc += a_ptr[m * a_line_w + k] * b_ptr[k * b_line_w + n];
                }
                float res = alpha * acc; // + beta * c_ptr
                atomic_store_global_32(reinterpret_cast<uint32_t*>(&c_ptr[m * c_line_w + n]), res);
                //c_ptr[m * c_line_w + n] = res;
            }
        }
    } else {
        et_printf("idling");
    }

    return 0;
}
)a";

std::vector<const data_flow::AccessNode*> static find_gemm_access_nodes(
    const data_flow::DataFlowGraph& dfg, const math::blas::GEMMNode& node
) {
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

GEMMNodeDispatcher_ETSOC_WithTransfers::GEMMNodeDispatcher_ETSOC_WithTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::GEMMNode& node
)
    : LibraryNodeDispatcherBase(language_extension, function, data_flow_graph, node) {}

void GEMMNodeDispatcher_ETSOC_WithTransfers::dispatch(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    library_snippet_factory.add_global("#include <docc/rt/et/et.h>");
    library_snippet_factory.add_global("#include <filesystem>");
    library_snippet_factory.add_global("#include <iostream>");

    stream << kernel_stream_setup << std::endl;

    auto& gemm_node = dynamic_cast<const math::blas::GEMMNode&>(this->node_);
    auto& dflow = gemm_node.get_parent();
    auto access_nodes = find_gemm_access_nodes(dflow, gemm_node);

    std::string kernel_name = this->function_.name() + "_et_kernel_" + std::to_string(this->node_.element_id());
    auto& snippet = library_snippet_factory.require(kernel_name, ETSOC_KERNEL_FILE_EXT, true);
    auto& kstream = snippet.stream();
    kstream << matul_fp32_kernel_simple << std::endl;

    stream << "auto et_k = et.load_kernel_binary_blocking(et_stream, \""
           << library_snippet_factory.output_path().string() << "\", \"" << kernel_name << "\");" << std::endl;

    // fill args, transfer input data over
    auto alpha_var = require_param_as_var_equivalent(stream, access_nodes[3], "alpha");
    auto beta_var = require_param_as_var_equivalent(stream, access_nodes[4], "beta");

    stream << "auto A_size = sizeof(float)*" << language_extension_.expression(gemm_node.m()) << "*"
           << language_extension_.expression(gemm_node.lda()) << ";" << std::endl;
    stream << "std::byte* et_A = et_rt.mallocDevice(et_dev, A_size);" << std::endl;
    stream << "auto B_size = sizeof(float)*" << language_extension_.expression(gemm_node.k()) << "*"
           << language_extension_.expression(gemm_node.ldb()) << ";" << std::endl;
    stream << "std::byte* et_B = et_rt.mallocDevice(et_dev, B_size);" << std::endl;
    stream << "auto C_size = sizeof(float)*" << language_extension_.expression(gemm_node.m()) << "*"
           << language_extension_.expression(gemm_node.ldc()) << ";" << std::endl;
    stream << "std::byte* et_C = et_rt.mallocDevice(et_dev, C_size);" << std::endl;

    stream << "et_rt.memcpyHostToDevice(et_stream, reinterpret_cast<std::byte*>(" << access_nodes[0]->data()
           << "), et_A, A_size);" << std::endl;
    stream << "et_rt.memcpyHostToDevice(et_stream, reinterpret_cast<std::byte*>(" << access_nodes[1]->data()
           << "), et_B, B_size);" << std::endl;
    stream << "et_rt.memcpyHostToDevice(et_stream, reinterpret_cast<std::byte*>(" << access_nodes[2]->data()
           << "), et_C, C_size);" << std::endl;


    stream << "std::vector<uint32_t> k_args(14);" << std::endl;
    stream << "*reinterpret_cast<std::byte**>(k_args.data()+0) = et_A;" << std::endl;
    stream << "*reinterpret_cast<std::byte**>(k_args.data()+2) = et_B;" << std::endl;
    stream << "*reinterpret_cast<std::byte**>(k_args.data()+4) = et_C;" << std::endl;
    stream << "*reinterpret_cast<float*>(k_args.data()+6) = " << alpha_var << ";" << std::endl;
    stream << "*reinterpret_cast<float*>(k_args.data()+7) = " << beta_var << ";" << std::endl;
    stream << "*reinterpret_cast<int32_t*>(k_args.data()+8) = " << language_extension_.expression(gemm_node.m()) << ";"
           << std::endl;
    stream << "*reinterpret_cast<int32_t*>(k_args.data()+9) = " << language_extension_.expression(gemm_node.n()) << ";"
           << std::endl;
    stream << "*reinterpret_cast<int32_t*>(k_args.data()+10) = " << language_extension_.expression(gemm_node.k()) << ";"
           << std::endl;
    stream << "*reinterpret_cast<int32_t*>(k_args.data()+11) = " << language_extension_.expression(gemm_node.lda())
           << ";" << std::endl;
    stream << "*reinterpret_cast<int32_t*>(k_args.data()+12) = " << language_extension_.expression(gemm_node.ldb())
           << ";" << std::endl;
    stream << "*reinterpret_cast<int32_t*>(k_args.data()+13) = " << language_extension_.expression(gemm_node.ldc())
           << ";" << std::endl;
    stream << "uint64_t et_shire_mask = 0xFFFFFFFFUL;" << std::endl;
    stream << kernel_launch_template_blocking << std::endl;

    // pull out data here
    stream << "et_rt.memcpyDeviceToHost(et_stream, et_C, reinterpret_cast<std::byte*>(" << access_nodes[2]->data()
           << "), C_size);" << std::endl;
    stream << "et_rt.waitForStream(et_stream);" << std::endl;


    stream << kernel_cleanup << std::endl;
}

} // namespace docc::target::et::blas
