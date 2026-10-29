#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/code_generators/cpp_code_generator.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/gpu/gpu_tile_arrival_node.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

namespace sdfg::gpu {

namespace {

namespace hip = ::sdfg::rocm;

constexpr int kBlocks = 8;
constexpr int kItems = 4;

/// One ROCm kernel over `kBlocks` blocks in which every block arrives at `num_counters` counter arrays in
/// sequence; arrival `a` targets item `b / (kBlocks / kItems)` of counter array `cnt<a>` and expects
/// `kBlocks / kItems` arrivals. The election result of arrival `a` is written to `out[a * kBlocks + b]`.
struct ArrivalKernel {
    builder::StructuredSDFGBuilder builder{"arrival_sdfg", FunctionType_CPU};
    std::vector<GpuTileArrivalNode*> arrivals;
    structured_control_flow::Map* grid = nullptr;

    explicit ArrivalKernel(size_t num_counters) {
        auto& root = builder.subject().root();
        types::Scalar i32(types::PrimitiveType::Int32);
        types::Pointer dev_ptr(types::StorageType("AMD_Generic"), 0, "", i32);
        types::Array counter(types::StorageType::GPU_Static(), 4, "", i32, symbolic::integer(kItems));
        builder.add_container("b", i32);
        builder.add_container("__daisy_hip_out", dev_ptr, true);

        auto b = symbolic::symbol("b");
        grid = &builder.add_map(
            root,
            b,
            symbolic::Lt(b, symbolic::integer(kBlocks)),
            symbolic::zero(),
            symbolic::add(b, symbolic::one()),
            hip::ScheduleType_ROCM_Offload::create<
                hip::ScheduleType_ROCM_Offload>(TargetLevel::X_GRID, symbolic::integer(kBlocks))
        );

        const auto per_item = symbolic::integer(kBlocks / kItems);
        for (size_t a = 0; a < num_counters; ++a) {
            auto cnt = "cnt" + std::to_string(a);
            auto last = "last" + std::to_string(a);
            builder.add_container(cnt, counter);
            builder.add_container(last, i32);

            auto& block = builder.add_block(grid->root());
            auto& cnt_access = builder.add_access(block, cnt);
            auto& last_access = builder.add_access(block, last);
            auto& arrival =
                builder.add_library_node<GpuTileArrivalNode>(block, DebugInfo(), symbolic::div(b, per_item), per_item);
            builder.add_computational_memlet(block, cnt_access, arrival, "counter", {}, counter);
            builder.add_computational_memlet(block, arrival, "_last", last_access, {}, i32);
            arrivals.push_back(static_cast<GpuTileArrivalNode*>(&arrival));

            auto& store = builder.add_block(grid->root());
            auto& in = builder.add_access(store, last);
            auto& out = builder.add_access(store, "__daisy_hip_out");
            auto& tasklet = builder.add_tasklet(store, data_flow::TaskletCode::assign, "_out", {"_in"});
            builder.add_computational_memlet(store, in, tasklet, "_in", {}, i32);
            auto idx = symbolic::add(symbolic::integer(static_cast<int>(a) * kBlocks), b);
            builder.add_computational_memlet(store, tasklet, "_out", out, {idx}, dev_ptr);
        }
    }

    std::string kernel() {
        auto& sdfg = builder.subject();
        analysis::AnalysisManager analysis_manager(sdfg);
        auto instrumentation = codegen::InstrumentationPlan::none(sdfg);
        auto arg_capture = codegen::ArgCapturePlan::none(sdfg);
        auto snippets = std::make_shared<codegen::CodeSnippetFactory>();
        codegen::CPPCodeGenerator generator(sdfg, analysis_manager, *instrumentation, *arg_capture, snippets);
        generator.generate();
        return generator.library_snippets()
            .at("kernel_arrival_sdfg_" + std::to_string(grid->element_id()))
            .stream()
            .str();
    }
};

size_t count(const std::string& text, const std::string& pattern) {
    size_t n = 0;
    for (size_t pos = text.find(pattern); pos != std::string::npos; pos = text.find(pattern, pos + 1)) {
        ++n;
    }
    return n;
}

/// The protocol lines of one arrival, in emission order.
std::vector<std::string> protocol(const GpuTileArrivalNode& node, const std::string& counter) {
    auto flag = "__daisy_arrival_" + std::to_string(node.element_id());
    auto slot = "(" + counter + ")[";
    return {
        "__threadfence();",
        "__syncthreads();",
        "__shared__ int " + flag + ";",
        "if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {",
        flag + " = atomicAdd(&" + slot,
        "== (2) - 1;",
        "if (" + flag + ") {",
        slot,
        "] = 0;",
        "__syncthreads();",
        "if (" + flag + ") {",
        "__threadfence();",
    };
}

void expect_in_order(const std::string& text, const std::vector<std::string>& lines, size_t& pos) {
    for (auto& line : lines) {
        auto next = text.find(line, pos);
        ASSERT_NE(next, std::string::npos) << "missing '" << line << "' after offset " << pos << " in\n" << text;
        pos = next + line.size();
    }
}

} // namespace

/// Base: a single arrival emits the fence/barrier/elect/broadcast protocol on its counter slot and binds the
/// election to its output.
TEST(GpuTileArrivalTest, BaseSingleArrival) {
    ArrivalKernel kernel(1);
    EXPECT_NO_THROW(kernel.builder.subject().validate());
    auto code = kernel.kernel();

    size_t pos = 0;
    expect_in_order(code, protocol(*kernel.arrivals[0], "cnt0"), pos);
    EXPECT_NE(code.find("last0 = __daisy_arrival_" + std::to_string(kernel.arrivals[0]->element_id())), std::string::npos)
        << code;
    EXPECT_EQ(count(code, "atomicAdd("), 1u) << code;
}

/// Step: each further arrival appends its own complete protocol after the previous one, with its own flag
/// and counter, leaving the earlier protocols unchanged.
TEST(GpuTileArrivalTest, StepSequentialArrivals) {
    for (size_t n = 2; n <= 3; ++n) {
        ArrivalKernel kernel(n);
        EXPECT_NO_THROW(kernel.builder.subject().validate());
        auto code = kernel.kernel();

        size_t pos = 0;
        for (size_t a = 0; a < n; ++a) {
            expect_in_order(code, protocol(*kernel.arrivals[a], "cnt" + std::to_string(a)), pos);
        }
        EXPECT_EQ(count(code, "atomicAdd("), n) << code;
        EXPECT_EQ(count(code, "__shared__ int __daisy_arrival_"), n) << code;
    }
}

/// Symbol substitution and serialization preserve the arrival's index and expected count.
TEST(GpuTileArrivalTest, ReplaceAndSerializationPreserveParameters) {
    ArrivalKernel kernel(1);
    auto& node = *kernel.arrivals[0];
    auto b = symbolic::symbol("b");
    auto c = symbolic::symbol("c");

    node.replace(b, c);
    EXPECT_TRUE(symbolic::eq(node.index(), symbolic::div(c, symbolic::integer(2))));
    EXPECT_TRUE(symbolic::eq(node.expected(), symbolic::integer(2)));
    EXPECT_TRUE(node.symbols().count(c));
    EXPECT_FALSE(node.symbols().count(b));

    GpuTileArrivalNodeSerializer arrival_serializer;
    auto j = arrival_serializer.serialize(node);
    serializer::JSONSerializer json;
    json.debug_info_to_json(j["debug_info"], node.debug_info());
    auto& block = kernel.builder.add_block(kernel.builder.subject().root());
    auto& restored = static_cast<GpuTileArrivalNode&>(arrival_serializer.deserialize(j, kernel.builder, block));
    EXPECT_TRUE(symbolic::eq(restored.index(), node.index()));
    EXPECT_TRUE(symbolic::eq(restored.expected(), node.expected()));
    EXPECT_TRUE(restored.side_effect());
}

} // namespace sdfg::gpu
