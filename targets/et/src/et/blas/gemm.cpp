#include "docc/target/et/blas/gemm.h"

namespace docc::target::et::blas {

static const std::string kernel_stream_setup = R"a(
        auto& et = docc::rt::et::EtRuntimeWrapper::get_instance();
        auto& et_rt = et.get_runtime();
        auto et_dev = et.get_device();
        auto et_stream = et_rt.createStream(et_dev);
)a";

static const std::string kernel_launch_template = R"a(
        auto et_k = et.load_kernel_binary_blocking(et_stream, et_k_file);
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

GEMMNodeDispatcher_ETSOC_WithoutTransfers::GEMMNodeDispatcher_ETSOC_WithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::GEMMNode& node
)
    : LibraryNodeDispatcherBase(language_extension, function, data_flow_graph, node) {}

void GEMMNodeDispatcher_ETSOC_WithoutTransfers::dispatch(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    globals_stream << "#include <docc/rt/et/et.h>" << std::endl;
    globals_stream << "#include <filesystem>" << std::endl;

    stream << kernel_stream_setup << std::endl;
    stream << "std::filesystem::path et_k_file = \"hello.elf\";" << std::endl;
    // fill args, transfer input data over
    stream << "std::vector<uint32_t> k_args;" << std::endl;
    stream << "uint64_t et_shire_mask = 0x1;" << std::endl;
    stream << kernel_launch_template << std::endl;
    // pull out data here
    stream << kernel_cleanup << std::endl;
}

} // namespace docc::target::et::blas
