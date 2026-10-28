#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/code_generators/cpp_code_generator.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

namespace sdfg::gpu {

namespace {

namespace hip = ::sdfg::rocm;

constexpr int kExtent = 64;

/// A chain of ROCm kernels `src -> statics[0] -> ... -> statics[n-1] -> dst`; each kernel is one X_GRID map
/// copying its input buffer into its output buffer, so every static buffer is used by exactly two kernels.
struct StaticChain {
    builder::StructuredSDFGBuilder builder{"static_sdfg", FunctionType_CPU};
    std::vector<std::string> statics;
    std::vector<structured_control_flow::Map*> kernels;

    explicit StaticChain(size_t num_statics) {
        auto& root = builder.subject().root();
        types::Scalar i32(types::PrimitiveType::Int32);
        types::Scalar f32(types::PrimitiveType::Float);
        types::Pointer dev_ptr(types::StorageType("AMD_Generic"), 0, "", f32);
        types::Array static_array(types::StorageType::GPU_Static(), 16, "", f32, symbolic::integer(kExtent));

        builder.add_container("__daisy_hip_src", dev_ptr, true);
        builder.add_container("__daisy_hip_dst", dev_ptr, true);
        for (size_t s = 0; s < num_statics; ++s) {
            statics.push_back("ws" + std::to_string(s));
            builder.add_container(statics.back(), static_array);
        }

        std::vector<std::string> chain{"__daisy_hip_src"};
        chain.insert(chain.end(), statics.begin(), statics.end());
        chain.push_back("__daisy_hip_dst");
        for (size_t k = 0; k + 1 < chain.size(); ++k) {
            auto indvar = symbolic::symbol("j" + std::to_string(k));
            builder.add_container(indvar->get_name(), i32);
            auto sched = hip::ScheduleType_ROCM_Offload::create<
                hip::ScheduleType_ROCM_Offload>(TargetLevel::X_GRID, symbolic::integer(kExtent));
            auto& map = builder.add_map(
                root,
                indvar,
                symbolic::Lt(indvar, symbolic::integer(kExtent)),
                symbolic::zero(),
                symbolic::add(indvar, symbolic::one()),
                sched
            );
            kernels.push_back(&map);

            auto& block = builder.add_block(map.root());
            auto& in = builder.add_access(block, chain[k]);
            auto& out = builder.add_access(block, chain[k + 1]);
            auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
            const types::IType& in_type = builder.subject().type(chain[k]);
            const types::IType& out_type = builder.subject().type(chain[k + 1]);
            builder.add_computational_memlet(block, in, tasklet, "_in", {indvar}, in_type);
            builder.add_computational_memlet(block, tasklet, "_out", out, {indvar}, out_type);
        }
    }

    struct Code {
        std::string host;
        std::vector<std::string> kernels;
    };

    Code generate() {
        auto& sdfg = builder.subject();
        analysis::AnalysisManager analysis_manager(sdfg);
        auto instrumentation = codegen::InstrumentationPlan::none(sdfg);
        auto arg_capture = codegen::ArgCapturePlan::none(sdfg);
        auto snippets = std::make_shared<codegen::CodeSnippetFactory>();
        codegen::CPPCodeGenerator generator(sdfg, analysis_manager, *instrumentation, *arg_capture, snippets);
        generator.generate();
        std::stringstream host;
        generator.append_function_source(host);

        Code code{host.str(), {}};
        for (auto* kernel : kernels) {
            auto name = "kernel_static_sdfg_" + std::to_string(kernel->element_id());
            code.kernels.push_back(generator.library_snippets().at(name).stream().str());
        }
        return code;
    }
};

size_t count(const std::string& text, const std::string& pattern) {
    size_t n = 0;
    for (size_t pos = text.find(pattern); pos != std::string::npos; pos = text.find(pattern, pos + 1)) {
        ++n;
    }
    return n;
}

std::string declaration(const std::string& name) {
    return "static __device__ float " + name + "[" + std::to_string(kExtent) + "] __attribute__((aligned(16)));";
}

std::string kernel_signature(const std::string& kernel) {
    auto begin = kernel.find("__global__");
    return kernel.substr(begin, kernel.find('{', begin) - begin);
}

} // namespace

TEST(GpuStaticStorageTest, StorageTypeIdentity) {
    auto storage = types::StorageType::GPU_Static();
    EXPECT_TRUE(storage.is_gpu_static());
    EXPECT_TRUE(storage.is_device());
    EXPECT_FALSE(types::StorageType::NV_Global().is_gpu_static());
    EXPECT_FALSE(types::StorageType("AMD_Generic").is_gpu_static());
}

/// Base: a single static buffer is defined exactly once at file scope of each kernel using it, before the
/// kernel, and is neither a kernel parameter nor a launch argument nor a host variable.
TEST(GpuStaticStorageTest, BaseSingleBuffer) {
    StaticChain chain(1);
    auto code = chain.generate();
    ASSERT_EQ(code.kernels.size(), 2u);

    for (auto& kernel : code.kernels) {
        EXPECT_EQ(count(kernel, declaration("ws0")), 1u) << kernel;
        EXPECT_LT(kernel.find(declaration("ws0")), kernel.find("__global__")) << kernel;
        EXPECT_EQ(count(kernel_signature(kernel), "ws0"), 0u) << kernel;
    }
    EXPECT_EQ(count(code.host, "ws0"), 0u) << code.host;
}

/// Step: adding a static buffer to the chain adds exactly its definition to the two kernels using it and
/// leaves the parameters of every kernel restricted to device pointers.
TEST(GpuStaticStorageTest, StepAdditionalBuffers) {
    for (size_t n = 2; n <= 3; ++n) {
        StaticChain chain(n);
        auto code = chain.generate();
        ASSERT_EQ(code.kernels.size(), n + 1);

        for (size_t k = 0; k < code.kernels.size(); ++k) {
            auto& kernel = code.kernels[k];
            for (size_t s = 0; s < n; ++s) {
                bool used = (s + 1 == k) || (s == k);
                EXPECT_EQ(count(kernel, declaration(chain.statics[s])), used ? 1u : 0u) << kernel;
                EXPECT_EQ(count(kernel_signature(kernel), chain.statics[s]), 0u) << kernel;
            }
        }
        for (auto& name : chain.statics) {
            EXPECT_EQ(count(code.host, name), 0u) << code.host;
        }
    }
}

} // namespace sdfg::gpu
