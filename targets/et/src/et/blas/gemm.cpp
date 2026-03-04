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
        launchOpts.setShireMask(et_shire_mask);
        launchOpts.setBarrier(true);
        launchOpts.setUserTracing(
            reinterpret_cast<uint64_t>(traceDevBuf),
            static_cast<uint32_t>(et.DEFAULT_TRACE_BUFFER_SIZE),
            0,                              // threshold
            0x1       ,                     // trace shireMask
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

static const std::string gemm_s_kernel = R"a(
#include <stdint.h>
#include "etsoc/isa/hart.h"
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

    if (h == 0) {
        et_printf("%d..%d x %d..%d: %d %d\n", 0, param->m, 0, param->n, a_line_w, c_line_w);
        for (int m = 0; m < param->m; ++m) {
            for (int n = 0; n < param->n; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < param->k; ++k) {
                    acc += a_ptr[m * a_line_w + k] * b_ptr[k * b_line_w + n];
                }
                c_ptr[m * c_line_w + n] = alpha * acc + beta * c_ptr[m * c_line_w + n];
            }
        }
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
    globals_stream << "#include <docc/rt/et/et.h>" << std::endl;
    globals_stream << "#include <filesystem>" << std::endl;
    globals_stream << "#include <iostream>" << std::endl;

    stream << kernel_stream_setup << std::endl;

    auto& gemm_node = dynamic_cast<const math::blas::GEMMNode&>(this->node_);
    auto& dflow = gemm_node.get_parent();
    auto access_nodes = find_gemm_access_nodes(dflow, gemm_node);

    std::string kernel_name = this->function_.name() + "_et_kernel_" + std::to_string(this->node_.element_id());
    auto& snippet = library_snippet_factory.require(kernel_name, ETSOC_KERNEL_FILE_EXT, true);
    auto& kstream = snippet.stream();
    kstream << gemm_s_kernel << std::endl;

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
    stream << "uint64_t et_shire_mask = 0x1;" << std::endl;
    stream << kernel_launch_template_blocking << std::endl;

    // pull out data here
    stream << "et_rt.memcpyDeviceToHost(et_stream, et_C, reinterpret_cast<std::byte*>(" << access_nodes[2]->data()
           << "), C_size);" << std::endl;
    stream << "et_rt.waitForStream(et_stream);" << std::endl;


    stream << kernel_cleanup << std::endl;
}

} // namespace docc::target::et::blas
