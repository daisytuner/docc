#include "sdfg/tiles/analysis/reduction_analysis.h"
#include "sdfg/passes/offloading/reduction_shared_memory_delinearization.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/tiles/analysis/reduction_buffer_analysis.h"
#include "sdfg/transformations/loop_interchange.h"
#include "sdfg/transformations/loop_tiling.h"
#include "sdfg/transformations/multi_level_tiling.h"

#include <gtest/gtest.h>
#include <type_traits>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

static void check_schedule_estimates(
    builder::StructuredSDFGBuilder& source,
    analysis::AnalysisManager& manager,
    const std::vector<structured_control_flow::Reduce*>& reductions
) {
    auto& buffers = manager.get<tiles::ReductionBufferAnalysis>();
    serializer::JSONSerializer serializer;
    const auto unchanged = serializer.serialize(source.subject());
    for (auto* target : reductions) {
        const auto footprint = buffers.require(*target, "acc");
        EXPECT_TRUE(buffers.supports_schedule(*target, target->schedule_type()));
        auto wider = target->schedule_type();
        gpu::ScheduleType_GPU_Offload::parallel_size(wider, symbolic::integer(16));
        EXPECT_EQ(buffers.supports_schedule(*target, wider), !footprint.materialized);
        auto invalid = target->schedule_type();
        gpu::ScheduleType_GPU_Offload::partial_storage(invalid, gpu::ReduceStrategy::Register);
        EXPECT_FALSE(buffers.supports_schedule(*target, invalid));
        EXPECT_EQ(
            buffers.estimate_schedule(*target, "acc", footprint, {*target, invalid}).status,
            tiles::ReductionBufferStatus::Unsupported
        );
        const auto estimate = buffers.estimate_schedule(*target, "acc", footprint, {*target, wider});
        ASSERT_EQ(estimate.status, tiles::ReductionBufferStatus::Exact) << estimate.diagnostic;
        EXPECT_FALSE(estimate.materialized);
        EXPECT_EQ(estimate.layout->extent, 1);
        EXPECT_TRUE(symbolic::eq(estimate.layout->base, symbolic::zero()));
        EXPECT_EQ(estimate.shared_owner, reductions.front()->element_id());
        const auto width =
            SymEngine::rcp_static_cast<
                const SymEngine::Integer>(gpu::ScheduleType_GPU_Offload::parallel_size(target->schedule_type()))
                ->as_int();
        EXPECT_EQ(estimate.shared_bytes, *footprint.shared_bytes / width * 16);
        EXPECT_EQ(estimate.private_bytes, footprint.private_bytes);
        EXPECT_EQ(serializer.serialize(source.subject()), unchanged);
    }
}

TEST(ReductionBufferAnalysisTest, SchedulePreviewIsLocalToTargetNest) {
    builder::StructuredSDFGBuilder builder("local_reduction_preview", FunctionType_CPU);
    types::Scalar integer_type(types::PrimitiveType::Int32);
    types::Pointer pointer{types::Scalar(types::PrimitiveType::Float)};
    auto schedule = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(8));
    auto add_reduction = [&](const std::string& name) -> structured_control_flow::Reduce& {
        builder.add_container(name, integer_type);
        builder.add_container(name + "_acc", pointer);
        auto indvar = symbolic::symbol(name);
        auto& reduction = builder.add_reduce(
            builder.subject().root(),
            indvar,
            symbolic::Lt(indvar, symbolic::integer(8)),
            symbolic::zero(),
            symbolic::add(indvar, symbolic::one()),
            {{structured_control_flow::ReductionOperation::Add, name + "_acc"}},
            schedule
        );
        auto& block = builder.add_block(reduction.root());
        auto& input = builder.add_access(block, name + "_acc");
        auto& output = builder.add_access(block, name + "_acc");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "out", {"left", "right"});
        builder.add_computational_memlet(block, input, tasklet, "left", {symbolic::zero()}, pointer);
        builder.add_computational_memlet(block, input, tasklet, "right", {symbolic::zero()}, pointer);
        builder.add_computational_memlet(block, tasklet, "out", output, {symbolic::zero()}, pointer);
        return reduction;
    };
    auto& target = add_reduction("target");
    auto& unrelated = add_reduction("unrelated");
    analysis::AnalysisManager manager(builder.subject());
    serializer::JSONSerializer serializer;
    const auto before = serializer.serialize(builder.subject());
    auto& buffers = manager.get<tiles::ReductionBufferAnalysis>();
    const auto wider = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(16));
    const auto estimate =
        buffers.estimate_schedule(target, "target_acc", buffers.require(target, "target_acc"), {target, wider});
    EXPECT_EQ(estimate.status, tiles::ReductionBufferStatus::Exact);
    EXPECT_EQ(estimate.shared_bytes, 64);
    EXPECT_EQ(estimate.private_bytes, 4);
    EXPECT_EQ(estimate.shared_owner, target.element_id());
    EXPECT_EQ(buffers.require(unrelated, "unrelated_acc").shared_bytes, 32);
    EXPECT_TRUE(buffers.supports_schedule(target, wider));
    EXPECT_EQ(serializer.serialize(builder.subject()), before);
    passes::ReductionSharedMemoryDelinearization packing;
    ASSERT_TRUE(packing.run_pass(builder, manager));
    const auto materialized = serializer.serialize(builder.subject());
    auto& packed_buffers = manager.get<tiles::ReductionBufferAnalysis>();
    EXPECT_TRUE(packed_buffers.supports_schedule(target, target.schedule_type()));
    EXPECT_FALSE(packed_buffers.supports_schedule(target, wider));
    EXPECT_NE(
        packed_buffers.require(target, "target_acc").shared_buffer,
        packed_buffers.require(unrelated, "unrelated_acc").shared_buffer
    );
    EXPECT_EQ(serializer.serialize(builder.subject()), materialized);
}

TEST(ReductionBufferAnalysisTest, FootprintGrowsOneAffineAxisAtATime) {
    builder::StructuredSDFGBuilder builder("inductive_footprint", FunctionType_CPU);
    types::Scalar integer_type(types::PrimitiveType::Int32);
    types::Pointer pointer{types::Scalar(types::PrimitiveType::Float)};
    builder.add_container("step", integer_type);
    builder.add_container("acc", pointer);
    auto step = symbolic::symbol("step");
    auto& reduction = builder.add_reduce(
        builder.subject().root(),
        step,
        symbolic::Lt(step, symbolic::integer(8)),
        symbolic::zero(),
        symbolic::add(step, symbolic::one()),
        {{structured_control_flow::ReductionOperation::Add, "acc"}},
        gpu::ScheduleType_GPU_Offload::create<
            cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(8))
    );
    auto* body = &reduction.root();
    auto& block = builder.add_block(*body);
    auto& input = builder.add_access(block, "acc");
    auto& output = builder.add_access(block, "acc");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "out", {"left", "right"});
    symbolic::Expression index = symbolic::integer(7);
    std::vector<data_flow::Memlet*> accesses{
        &builder.add_computational_memlet(block, input, tasklet, "left", {index}, pointer),
        &builder.add_computational_memlet(block, input, tasklet, "right", {index}, pointer),
        &builder.add_computational_memlet(block, tasklet, "out", output, {index}, pointer)
    };
    struct Axis {
        int64_t count;
        int64_t init;
        int64_t step;
        int64_t coefficient;
    };
    const std::vector<Axis> axes{{2, 5, 2, 1}, {1, 3, 1, 8}, {3, 2, 1, 8}, {2, 1, 3, 16}};
    std::vector<int64_t> addresses{7};
    std::vector<gpu::ReductionLayout::Dimension> dimensions;
    analysis::AnalysisManager manager(builder.subject());
    for (size_t depth = 0; depth <= axes.size(); ++depth) {
        SCOPED_TRACE(depth);
        const auto info = manager.get<tiles::ReductionBufferAnalysis>().require(reduction, "acc");
        ASSERT_TRUE(info.layout);
        EXPECT_EQ(info.layout->extent, addresses.size());
        EXPECT_TRUE(symbolic::eq(info.layout->base, symbolic::integer(addresses.front())));
        EXPECT_TRUE(symbolic::eq(info.accumulator_index, index));
        EXPECT_EQ(info.multi_output, depth != 0);
        EXPECT_EQ(info.element_bytes, 4);
        EXPECT_EQ(info.private_bytes, 4 * addresses.size());
        EXPECT_EQ(info.shared_bytes, 8 * 4 * addresses.size());
        EXPECT_EQ(info.shared_owner, reduction.element_id());
        EXPECT_FALSE(info.materialized);
        ASSERT_EQ(info.layout->dimensions.size(), dimensions.size());
        for (size_t axis = 0; axis < dimensions.size(); ++axis) {
            EXPECT_EQ(info.layout->dimensions[axis].stride, dimensions[axis].stride);
            EXPECT_EQ(info.layout->dimensions[axis].count, dimensions[axis].count);
        }
        for (size_t slot = 0; slot < addresses.size(); ++slot) {
            EXPECT_TRUE(symbolic::eq(info.layout->unpack(symbolic::integer(slot)), symbolic::integer(addresses[slot])));
            EXPECT_TRUE(symbolic::eq(info.layout->pack(symbolic::integer(addresses[slot])), symbolic::integer(slot)));
        }
        if (depth == axes.size()) {
            break;
        }
        const auto& axis = axes[depth];
        auto name = "axis_" + std::to_string(depth);
        builder.add_container(name, integer_type);
        auto variable = symbolic::symbol(name);
        auto& loop = builder.add_for(
            *body,
            variable,
            symbolic::Lt(variable, symbolic::integer(axis.init + axis.count * axis.step)),
            symbolic::integer(axis.init),
            symbolic::add(variable, symbolic::integer(axis.step))
        );
        builder.move_child(*body, 0, loop.root());
        body = &loop.root();
        index = symbolic::add(index, symbolic::mul(symbolic::integer(axis.coefficient), variable));
        for (auto* access : accesses) {
            access->set_subset({index});
        }
        // Enumerate the new axis over the already-verified addresses, independently of pack/unpack.
        std::vector<int64_t> expanded;
        for (int64_t coordinate = 0; coordinate < axis.count; ++coordinate) {
            for (auto address : addresses) {
                expanded.push_back(address + axis.coefficient * (axis.init + coordinate * axis.step));
            }
        }
        addresses = std::move(expanded);
        if (axis.count != 1) {
            dimensions.push_back({axis.coefficient * axis.step, axis.count});
        }
        manager.invalidate_all();
    }
    const auto before = manager.get<tiles::ReductionBufferAnalysis>().require(reduction, "acc");
    builder.add_container("unused", integer_type);
    auto unused = symbolic::symbol("unused");
    auto& extra = builder.add_for(
        *body,
        unused,
        symbolic::Lt(unused, symbolic::integer(13)),
        symbolic::zero(),
        symbolic::add(unused, symbolic::one())
    );
    builder.move_child(*body, 0, extra.root());
    manager.invalidate_all();
    const auto& after = manager.get<tiles::ReductionBufferAnalysis>().require(reduction, "acc");
    EXPECT_EQ(after.layout->extent, before.layout->extent);
    EXPECT_EQ(after.private_bytes, before.private_bytes);
    EXPECT_EQ(after.shared_bytes, before.shared_bytes);
    EXPECT_TRUE(symbolic::eq(after.layout->base, before.layout->base));

    builder.add_container("overlap", integer_type);
    auto overlap = symbolic::symbol("overlap");
    auto& invalid_axis = builder.add_for(
        extra.root(),
        overlap,
        symbolic::Lt(overlap, symbolic::integer(2)),
        symbolic::zero(),
        symbolic::add(overlap, symbolic::one())
    );
    builder.move_child(extra.root(), 0, invalid_axis.root());
    for (auto* access : accesses) {
        access->set_subset({symbolic::add(index, symbolic::mul(symbolic::integer(2), overlap))});
    }
    manager.invalidate_all();
    const auto& unsupported = manager.get<tiles::ReductionBufferAnalysis>().buffer(reduction, "acc");
    EXPECT_EQ(unsupported.status, tiles::ReductionBufferStatus::Unsupported);
    EXPECT_FALSE(unsupported.layout);
    EXPECT_FALSE(unsupported.private_bytes);
    EXPECT_FALSE(unsupported.shared_bytes);
    EXPECT_NE(unsupported.diagnostic.find("overlapping output dimensions"), std::string::npos);
    EXPECT_THROW(manager.get<tiles::ReductionBufferAnalysis>().require(reduction, "acc"), InvalidSDFGException);
}

TEST(ReductionBufferAnalysisTest, FootprintUnfoldsDependentTileOriginsInductively) {
    builder::StructuredSDFGBuilder builder("inductive_tile_origins", FunctionType_CPU);
    types::Scalar integer_type(types::PrimitiveType::Int32);
    types::Pointer pointer{types::Scalar(types::PrimitiveType::Double)};
    builder.add_container("step", integer_type);
    builder.add_container("point", integer_type);
    builder.add_container("acc", pointer);
    auto step = symbolic::symbol("step");
    auto point = symbolic::symbol("point");
    auto& reduction = builder.add_reduce(
        builder.subject().root(),
        step,
        symbolic::Lt(step, symbolic::integer(8)),
        symbolic::zero(),
        symbolic::add(step, symbolic::one()),
        {{structured_control_flow::ReductionOperation::Add, "acc"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& points = builder.add_for(
        reduction.root(),
        point,
        symbolic::Lt(point, symbolic::integer(3)),
        symbolic::one(),
        symbolic::add(point, symbolic::one())
    );
    auto& block = builder.add_block(points.root());
    auto& input = builder.add_access(block, "acc");
    auto& output = builder.add_access(block, "acc");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "out", {"left", "right"});
    auto index = symbolic::add(symbolic::integer(5), point);
    builder.add_computational_memlet(block, input, tasklet, "left", {index}, pointer);
    builder.add_computational_memlet(block, input, tasklet, "right", {index}, pointer);
    builder.add_computational_memlet(block, tasklet, "out", output, {index}, pointer);
    analysis::AnalysisManager manager(builder.subject());
    structured_control_flow::StructuredLoop* outermost = &points;
    const std::vector<int64_t> tile_counts{3, 2};
    int64_t outer_count = 2;
    int64_t extent = 2;
    int64_t origin = 6;
    for (size_t depth = 0; depth <= tile_counts.size(); ++depth) {
        SCOPED_TRACE(depth);
        const auto info = manager.get<tiles::ReductionBufferAnalysis>().require(reduction, "acc");
        ASSERT_TRUE(info.layout);
        EXPECT_EQ(info.layout->extent, extent);
        EXPECT_TRUE(symbolic::eq(info.layout->base, symbolic::integer(origin)));
        EXPECT_TRUE(symbolic::eq(info.accumulator_index, index));
        EXPECT_EQ(info.private_bytes, extent * 8);
        EXPECT_EQ(info.shared_bytes, 0);
        EXPECT_FALSE(info.shared_owner);
        for (int64_t slot = 0; slot < extent; ++slot) {
            EXPECT_TRUE(symbolic::eq(info.layout->unpack(symbolic::integer(slot)), symbolic::integer(origin + slot)));
            EXPECT_TRUE(symbolic::eq(info.layout->pack(symbolic::integer(origin + slot)), symbolic::integer(slot)));
        }
        if (depth == tile_counts.size()) {
            break;
        }
        auto name = "tile_" + std::to_string(depth);
        builder.add_container(name, integer_type);
        auto tile = symbolic::symbol(name);
        auto& outer = builder.add_for(
            reduction.root(),
            tile,
            symbolic::Lt(tile, symbolic::integer(1 + tile_counts[depth])),
            symbolic::one(),
            symbolic::add(tile, symbolic::one())
        );
        builder.move_child(reduction.root(), 0, outer.root());
        auto init = symbolic::add(symbolic::mul(symbolic::integer(outer_count), tile), symbolic::one());
        builder.update_loop(
            *outermost,
            outermost->indvar(),
            symbolic::Lt(outermost->indvar(), symbolic::add(init, symbolic::integer(outer_count))),
            init,
            symbolic::add(outermost->indvar(), symbolic::one())
        );
        // A new tile starting at one skips exactly one previous footprint before its first output.
        origin += extent;
        extent *= tile_counts[depth];
        outer_count = tile_counts[depth];
        outermost = &outer;
        manager.invalidate_all();
    }
}

TEST(ReductionBufferAnalysisTest, NestedSharedOwnersGrowWithoutDoubleCounting) {
    builder::StructuredSDFGBuilder builder("inductive_shared_owners", FunctionType_CPU);
    types::Scalar integer_type(types::PrimitiveType::Int32);
    types::Pointer pointer{types::Scalar(types::PrimitiveType::Float)};
    builder.add_container("acc", pointer);
    const std::vector<gpu::TargetLevel> levels{
        gpu::TargetLevel::X_BLOCK, gpu::TargetLevel::Y_BLOCK, gpu::TargetLevel::Z_BLOCK
    };
    const std::vector<int64_t> widths{8, 3, 2};
    std::vector<structured_control_flow::Reduce*> reductions;
    auto* body = &builder.subject().root();
    auto& block = builder.add_block(*body);
    auto& input = builder.add_access(block, "acc");
    auto& output = builder.add_access(block, "acc");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "out", {"left", "right"});
    builder.add_computational_memlet(block, input, tasklet, "left", {symbolic::zero()}, pointer);
    builder.add_computational_memlet(block, input, tasklet, "right", {symbolic::zero()}, pointer);
    builder.add_computational_memlet(block, tasklet, "out", output, {symbolic::zero()}, pointer);
    analysis::AnalysisManager manager(builder.subject());
    int64_t threads = 1;
    for (size_t depth = 0; depth < levels.size(); ++depth) {
        SCOPED_TRACE(depth);
        auto name = "reduce_" + std::to_string(depth);
        builder.add_container(name, integer_type);
        auto variable = symbolic::symbol(name);
        auto& reduction = builder.add_reduce(
            *body,
            variable,
            symbolic::Lt(variable, symbolic::integer(widths[depth])),
            symbolic::zero(),
            symbolic::add(variable, symbolic::one()),
            {{structured_control_flow::ReductionOperation::Add, "acc"}},
            gpu::ScheduleType_GPU_Offload::create<
                cuda::ScheduleType_CUDA_Offload>(levels[depth], symbolic::integer(widths[depth]))
        );
        builder.move_child(*body, 0, reduction.root());
        body = &reduction.root();
        reductions.push_back(&reduction);
        threads *= widths[depth];
        manager.invalidate_all();
        auto& buffers = manager.get<tiles::ReductionBufferAnalysis>();
        for (auto* nested : reductions) {
            const auto& info = buffers.require(*nested, "acc");
            EXPECT_EQ(info.layout->extent, 1);
            EXPECT_EQ(info.shared_owner, reductions.front()->element_id());
            EXPECT_EQ(info.shared_bytes, nested == reductions.front() ? threads * 4 : 0);
            EXPECT_EQ(info.private_bytes, depth == 0 ? 4 : 0);
        }
        const auto& owner = buffers.require(*reductions.front(), "acc");
        auto last_thread = SymEngine::subs(
            owner.linear_thread_index,
            {{symbolic::threadIdx_x(), symbolic::integer(widths[0] - 1)},
             {symbolic::threadIdx_y(), symbolic::integer(depth >= 1 ? widths[1] - 1 : 0)},
             {symbolic::threadIdx_z(), symbolic::integer(depth >= 2 ? widths[2] - 1 : 0)}}
        );
        EXPECT_TRUE(symbolic::eq(last_thread, symbolic::integer(threads - 1)));
        const auto kernel = buffers.kernel(*reductions.front());
        EXPECT_EQ(kernel.status, tiles::ReductionBufferStatus::Exact);
        EXPECT_EQ(kernel.shared_bytes, threads * 4);
        check_schedule_estimates(builder, manager, reductions);
        if (reductions.size() > 1) {
            auto& outer = *reductions.front();
            auto& inner = *reductions.at(1);
            transformations::LoopInterchange interchange(outer, inner);
            const auto proposal = interchange.proposal();
            serializer::JSONSerializer serializer;
            const auto unchanged = serializer.serialize(builder.subject());
            for (auto* nested : reductions) {
                const auto estimate =
                    buffers.estimate_interchange(*nested, "acc", buffers.require(*nested, "acc"), proposal);
                EXPECT_EQ(estimate.status, tiles::ReductionBufferStatus::Exact);
                EXPECT_EQ(estimate.private_bytes, 0);
                EXPECT_EQ(estimate.shared_bytes, nested == &inner ? threads * 4 : 0);
                EXPECT_EQ(estimate.shared_owner, inner.element_id());
                EXPECT_FALSE(estimate.materialized);
            }
            EXPECT_TRUE(buffers.supports_interchange(proposal));
            EXPECT_EQ(serializer.serialize(builder.subject()), unchanged);
        }
    }
    passes::ReductionSharedMemoryDelinearization packing;
    ASSERT_TRUE(packing.run_pass(builder, manager));
    auto& packed_buffers = manager.get<tiles::ReductionBufferAnalysis>();
    const auto shared_name = packed_buffers.require(*reductions.front(), "acc").shared_buffer;
    ASSERT_FALSE(shared_name.empty());
    for (auto* nested : reductions) {
        const auto info = packed_buffers.require(*nested, "acc");
        EXPECT_TRUE(info.materialized);
        EXPECT_EQ(info.shared_owner, reductions.front()->element_id());
        EXPECT_EQ(info.shared_buffer, shared_name);
        EXPECT_EQ(info.shared_bytes, nested == reductions.front() ? threads * 4 : 0);
    }
    check_schedule_estimates(builder, manager, reductions);
    EXPECT_FALSE(packing.run_pass(builder, manager));
}

TEST(ReductionBufferAnalysisTest, SymbolicHeadersPreserveAffineCounts) {
    auto point = symbolic::symbol("point");
    auto origin = symbolic::symbol("origin");
    auto limit = symbolic::symbol("limit");
    const auto domain = tiles::ReductionLoopDomain::from_header(
        point,
        {origin,
         symbolic::
             Le(symbolic::add(symbolic::mul(symbolic::integer(2), point), symbolic::one()),
                symbolic::add(symbolic::mul(symbolic::integer(2), origin), symbolic::integer(8))),
         symbolic::add(point, symbolic::integer(2))}
    );
    EXPECT_TRUE(symbolic::eq(domain.init, origin));
    ASSERT_FALSE(domain.count.is_null());
    EXPECT_TRUE(symbolic::eq(domain.count, symbolic::integer(2)));
    EXPECT_TRUE(symbolic::eq(domain.stride, symbolic::integer(2)));
    const auto guarded = tiles::ReductionLoopDomain::from_header(
        point,
        {origin,
         symbolic::And(symbolic::Lt(point, limit), symbolic::Lt(point, symbolic::add(origin, symbolic::integer(4)))),
         symbolic::add(point, symbolic::one())}
    );
    ASSERT_FALSE(guarded.count.is_null());
    EXPECT_TRUE(symbolic::
                    eq(SymEngine::subs(guarded.count, {{origin, symbolic::integer(3)}, {limit, symbolic::integer(5)}}),
                       symbolic::integer(2)));
    EXPECT_TRUE(symbolic::
                    eq(SymEngine::subs(guarded.count, {{origin, symbolic::integer(3)}, {limit, symbolic::integer(20)}}),
                       symbolic::integer(4)));
    const auto nonlinear = tiles::ReductionLoopDomain::from_header(
        point, {origin, symbolic::Lt(symbolic::mul(point, point), limit), symbolic::add(point, symbolic::one())}
    );
    EXPECT_TRUE(nonlinear.count.is_null());
}

TEST(ReductionBufferAnalysisTest, DependentInterchangePreservesScalarFootprint) {
    builder::StructuredSDFGBuilder builder("dependent_reduction_interchange", FunctionType_CPU);
    types::Scalar integer_type(types::PrimitiveType::Int32);
    types::Pointer pointer{types::Scalar(types::PrimitiveType::Float)};
    builder.add_container("row", integer_type);
    builder.add_container("step", integer_type);
    builder.add_container("acc", pointer);
    auto row = symbolic::symbol("row");
    auto step = symbolic::symbol("step");
    auto& outer = builder.add_for(
        builder.subject().root(),
        row,
        symbolic::Lt(row, symbolic::integer(4)),
        symbolic::zero(),
        symbolic::add(row, symbolic::one())
    );
    auto& reduction = builder.add_reduce(
        outer.root(),
        step,
        symbolic::Lt(step, symbolic::add(row, symbolic::integer(8))),
        row,
        symbolic::add(step, symbolic::one()),
        {{structured_control_flow::ReductionOperation::Add, "acc"}},
        gpu::ScheduleType_GPU_Offload::create<
            cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(8))
    );
    auto& block = builder.add_block(reduction.root());
    auto& input = builder.add_access(block, "acc");
    auto& output = builder.add_access(block, "acc");
    auto& addition = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "out", {"left", "right"});
    builder.add_computational_memlet(block, input, addition, "left", {symbolic::zero()}, pointer);
    builder.add_computational_memlet(block, input, addition, "right", {symbolic::zero()}, pointer);
    builder.add_computational_memlet(block, addition, "out", output, {symbolic::zero()}, pointer);
    analysis::AnalysisManager manager(builder.subject());
    serializer::JSONSerializer serializer;
    const auto unchanged = serializer.serialize(builder.subject());
    transformations::LoopInterchange interchange(outer, reduction);
    ASSERT_TRUE(interchange.can_be_applied(builder, manager));
    EXPECT_EQ(serializer.serialize(builder.subject()), unchanged);
    interchange.apply(builder, manager);
    auto* rewritten = dyn_cast<structured_control_flow::Reduce*>(interchange.new_outer_loop());
    ASSERT_NE(rewritten, nullptr);
    const auto footprint = manager.get<tiles::ReductionBufferAnalysis>().require(*rewritten, "acc");
    EXPECT_EQ(footprint.layout->extent, 1);
    EXPECT_TRUE(symbolic::eq(footprint.layout->base, symbolic::zero()));
    EXPECT_EQ(footprint.private_bytes, 4);
    EXPECT_EQ(footprint.shared_bytes, 32);
    EXPECT_EQ(footprint.shared_owner, rewritten->element_id());
}

TEST(ReductionBufferAnalysisTest, TilingPreservesFootprintAddresses) {
    for (bool simplify : {false, true}) {
        for (bool use_rocm : {false, true}) {
            for (bool mapped : {false, true}) {
                SCOPED_TRACE(use_rocm);
                SCOPED_TRACE(mapped);
                builder::StructuredSDFGBuilder builder("tiling_projection", FunctionType_CPU);
                types::Scalar integer_type(types::PrimitiveType::Int32);
                types::Pointer pointer{types::Scalar(types::PrimitiveType::Float)};
                builder.add_container("step", integer_type);
                builder.add_container("row", integer_type);
                builder.add_container("acc", pointer);
                auto make_schedule = [&](gpu::TargetLevel level, int64_t width) {
                    if (use_rocm) {
                        return gpu::ScheduleType_GPU_Offload::create<
                            rocm::ScheduleType_ROCM_Offload>(level, symbolic::integer(width));
                    }
                    return gpu::ScheduleType_GPU_Offload::create<
                        cuda::ScheduleType_CUDA_Offload>(level, symbolic::integer(width));
                };
                auto step = symbolic::symbol("step");
                auto row = symbolic::symbol("row");
                auto& reduction = builder.add_reduce(
                    builder.subject().root(),
                    step,
                    symbolic::Lt(step, symbolic::integer(8)),
                    symbolic::zero(),
                    symbolic::add(step, symbolic::one()),
                    {{structured_control_flow::ReductionOperation::Add, "acc"}},
                    make_schedule(gpu::TargetLevel::X_BLOCK, 8)
                );
                structured_control_flow::StructuredLoop* output_loop = nullptr;
                if (mapped) {
                    output_loop = &builder.add_map(
                        reduction.root(),
                        row,
                        symbolic::Lt(row, symbolic::integer(19)),
                        symbolic::integer(3),
                        symbolic::add(row, symbolic::one()),
                        make_schedule(gpu::TargetLevel::Y_BLOCK, 4)
                    );
                } else {
                    output_loop = &builder.add_for(
                        reduction.root(),
                        row,
                        symbolic::Lt(row, symbolic::integer(19)),
                        symbolic::integer(3),
                        symbolic::add(row, symbolic::one())
                    );
                }
                auto& block = builder.add_block(output_loop->root());
                auto& input = builder.add_access(block, "acc");
                auto& output = builder.add_access(block, "acc");
                auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "out", {"left", "right"});
                auto index = symbolic::add(symbolic::integer(7), symbolic::mul(symbolic::integer(2), row));
                builder.add_computational_memlet(block, input, tasklet, "left", {index}, pointer);
                builder.add_computational_memlet(block, input, tasklet, "right", {index}, pointer);
                builder.add_computational_memlet(block, tasklet, "out", output, {index}, pointer);
                analysis::AnalysisManager manager(builder.subject());
                auto& buffers = manager.get<tiles::ReductionBufferAnalysis>();
                const auto source = buffers.require(reduction, "acc");
                auto tile = symbolic::symbol("tile");
                auto subtile = symbolic::symbol("subtile");
                const std::vector<tiles::ReductionLoopDomain> domains{
                    {row, subtile, symbolic::integer(2), symbolic::one()},
                    {subtile, tile, symbolic::integer(2), symbolic::integer(2)},
                    {tile, symbolic::integer(3), symbolic::integer(4), symbolic::integer(4)}
                };
                serializer::JSONSerializer serializer;
                const auto unchanged = serializer.serialize(builder.subject());
                const auto estimate = buffers.estimate_geometry(reduction, "acc", source, domains);
                ASSERT_EQ(estimate.status, tiles::ReductionBufferStatus::Exact) << estimate.diagnostic;
                EXPECT_FALSE(estimate.materialized);
                EXPECT_EQ(estimate.layout->extent, 16);
                EXPECT_TRUE(symbolic::eq(estimate.layout->base, symbolic::integer(13)));
                EXPECT_EQ(estimate.private_bytes, 64);
                EXPECT_EQ(estimate.shared_bytes, mapped ? 2048 : 512);
                transformations::LoopTiling ragged_tiling(*output_loop, 5, true);
                EXPECT_FALSE(ragged_tiling.can_be_applied(builder, manager));
                transformations::MultiLevelTiling tiling(*output_loop, 4, 2, simplify);
                ASSERT_TRUE(tiling.can_be_applied(builder, manager));
                EXPECT_EQ(serializer.serialize(builder.subject()), unchanged);
                tiling.apply(builder, manager);
                const auto actual = manager.get<tiles::ReductionBufferAnalysis>().require(reduction, "acc");
                EXPECT_EQ(actual.layout->extent, 16);
                EXPECT_EQ(actual.private_bytes, 64);
                EXPECT_EQ(actual.shared_bytes, mapped ? 2048 : 512);
                for (int64_t slot = 0; slot < 16; ++slot) {
                    auto address = symbolic::integer(13 + 2 * slot);
                    EXPECT_TRUE(symbolic::eq(estimate.layout->unpack(symbolic::integer(slot)), address));
                    EXPECT_TRUE(symbolic::eq(actual.layout->unpack(symbolic::integer(slot)), address));
                    EXPECT_TRUE(symbolic::eq(actual.layout->pack(address), symbolic::integer(slot)));
                }
                transformations::LoopTiling reduction_tiling(reduction, 3, false);
                const auto before_reduction_tiling = serializer.serialize(builder.subject());
                ASSERT_TRUE(reduction_tiling.can_be_applied(builder, manager));
                EXPECT_EQ(serializer.serialize(builder.subject()), before_reduction_tiling);
                reduction_tiling.apply(builder, manager);
                auto* owner = dyn_cast<structured_control_flow::Reduce*>(reduction_tiling.outer_loop());
                ASSERT_NE(owner, nullptr);
                auto& tiled_buffers = manager.get<tiles::ReductionBufferAnalysis>();
                const auto owner_info = tiled_buffers.require(*owner, "acc");
                const auto inner_info = tiled_buffers.require(reduction, "acc");
                EXPECT_EQ(owner_info.layout->extent, 16);
                EXPECT_EQ(owner_info.private_bytes, 0);
                EXPECT_EQ(owner_info.shared_bytes, mapped ? 2048 : 512);
                EXPECT_EQ(inner_info.shared_bytes, 0);
                EXPECT_EQ(inner_info.shared_owner, owner->element_id());
                EXPECT_EQ(tiled_buffers.kernel(*owner).shared_bytes, mapped ? 2048 : 512);
            }
        }
    }
}

TEST(ReductionBufferAnalysisTest, OriginalTargetSurvivesCloneAndReplacement) {
    builder::StructuredSDFGBuilder builder("writeback_relation", FunctionType_CPU);
    builder.add_container("step", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("offset", types::Scalar(types::PrimitiveType::Int32), true);
    builder.add_container("acc", types::Pointer(types::Scalar(types::PrimitiveType::Float)), true);
    auto step = symbolic::symbol("step");
    auto& reduction = builder.add_reduce(
        builder.subject().root(),
        step,
        symbolic::Lt(step, symbolic::integer(8)),
        symbolic::zero(),
        symbolic::add(step, symbolic::one()),
        {{structured_control_flow::ReductionOperation::Add, "acc"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );
    reduction.original_index("acc", symbolic::symbol("offset"));
    auto clone = builder.subject().clone();
    auto& cloned = static_cast<structured_control_flow::Reduce&>(clone->root().at(0));
    EXPECT_TRUE(symbolic::eq(cloned.reductions().front().original_index, symbolic::symbol("offset")));
    analysis::AnalysisManager manager(*clone);
    auto& users = manager.get<analysis::Users>();
    EXPECT_EQ(users.reads("acc").size(), 1);
    EXPECT_EQ(users.writes("acc").size(), 1);
    EXPECT_TRUE(symbolic::eq(users.reads("acc").front()->subsets().front().front(), symbolic::symbol("offset")));
    cloned.replace(symbolic::symbol("offset"), symbolic::integer(7));
    EXPECT_TRUE(symbolic::eq(cloned.reductions().front().original_index, symbolic::integer(7)));
    EXPECT_NO_THROW(cloned.validate(*clone));
    builder::StructuredSDFGBuilder cloned_builder(*clone);
    auto& following = cloned_builder.add_reduce(
        clone->root(),
        step,
        symbolic::Lt(step, symbolic::integer(8)),
        symbolic::zero(),
        symbolic::add(step, symbolic::one()),
        {{structured_control_flow::ReductionOperation::Add, "acc"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );
    following.original_index("acc", symbolic::integer(7));
    manager.invalidate_all();
    auto& current_users = manager.get<analysis::Users>();
    auto* readback = current_users.get_user("acc", &following, analysis::Use::READ);
    auto* writeback = current_users.get_user("acc", &cloned, analysis::Use::WRITE);
    EXPECT_TRUE(manager.get<analysis::DataDependencyAnalysis>().defined_by(*readback).contains(writeback));
    cloned.original_index("acc", symbolic::symbol("missing"));
    EXPECT_THROW(cloned.validate(*clone), InvalidSDFGException);
    cloned.original_index("acc", step);
    EXPECT_THROW(cloned.validate(*clone), InvalidSDFGException);
}

TEST(ReductionBufferAnalysisTest, DenseFootprintAndInvalidation) {
    builder::StructuredSDFGBuilder builder("reduction_buffer", FunctionType_CPU);
    types::Scalar integer_type(types::PrimitiveType::Int32);
    types::Pointer pointer{types::Scalar(types::PrimitiveType::Float)};
    for (const auto* name : {"row", "col", "reduce", "origin"}) {
        builder.add_container(name, integer_type);
    }
    builder.add_container("acc", pointer);
    auto reduction_var = symbolic::symbol("reduce");
    auto row = symbolic::symbol("row");
    auto col = symbolic::symbol("col");
    auto origin = symbolic::symbol("origin");
    auto& reduction = builder.add_reduce(
        builder.subject().root(),
        reduction_var,
        symbolic::Lt(reduction_var, symbolic::integer(8)),
        symbolic::zero(),
        symbolic::add(reduction_var, symbolic::one()),
        {{structured_control_flow::ReductionOperation::Add, "acc"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& rows = builder.add_for(
        reduction.root(),
        row,
        symbolic::Lt(row, symbolic::integer(4)),
        symbolic::zero(),
        symbolic::add(row, symbolic::integer(2))
    );
    auto& cols = builder.add_for(
        rows.root(), col, symbolic::Lt(col, symbolic::integer(2)), symbolic::zero(), symbolic::add(col, symbolic::one())
    );
    auto& block = builder.add_block(cols.root());
    auto& input = builder.add_access(block, "acc");
    auto& output = builder.add_access(block, "acc");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "out", {"left", "right"});
    auto index = symbolic::add(origin, symbolic::add(symbolic::mul(symbolic::integer(16), row), col));
    builder.add_computational_memlet(block, input, tasklet, "left", {index}, pointer);
    builder.add_computational_memlet(block, input, tasklet, "right", {index}, pointer);
    builder.add_computational_memlet(block, tasklet, "out", output, {index}, pointer);
    analysis::AnalysisManager manager(builder.subject());
    const auto result = manager.get<tiles::ReductionBufferAnalysis>().require(reduction, "acc");
    EXPECT_EQ(result.status, tiles::ReductionBufferStatus::Exact);
    ASSERT_TRUE(result.layout);
    EXPECT_EQ(result.layout->extent, 4);
    EXPECT_TRUE(symbolic::eq(result.layout->base, origin));
    EXPECT_EQ(result.element_bytes, 4);
    EXPECT_EQ(result.private_bytes, 16);
    EXPECT_EQ(result.shared_bytes, 0);
    EXPECT_TRUE(symbolic::eq(result.layout->unpack(symbolic::integer(3)), symbolic::add(origin, symbolic::integer(33)))
    );
    auto independent = manager.get<tiles::ReductionBufferAnalysis>().buffer(reduction, "acc");
    static_assert(std::is_same_v<
                  decltype(manager.get<tiles::ReductionBufferAnalysis>().buffer(reduction, "acc")),
                  tiles::ReductionBufferInfo>);
    static_assert(std::is_same_v<
                  decltype(manager.get<tiles::ReductionBufferAnalysis>().require(reduction, "acc")),
                  tiles::ReductionBufferInfo>);
    independent.layout->extent = 99;
    independent.private_bytes = 0;
    const auto fresh = manager.get<tiles::ReductionBufferAnalysis>().buffer(reduction, "acc");
    EXPECT_EQ(fresh.layout->extent, 4);
    EXPECT_EQ(fresh.private_bytes, 16);
    EXPECT_THROW(manager.get<tiles::ReductionBufferAnalysis>().require(reduction, "missing"), InvalidSDFGException);
    rows.replace(symbolic::integer(4), symbolic::integer(8));
    manager.invalidate_all();
    const auto& updated = manager.get<tiles::ReductionBufferAnalysis>().require(reduction, "acc");
    EXPECT_EQ(updated.layout->extent, 8);
    EXPECT_EQ(result.layout->extent, 4);
    auto schedule = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(8));
    builder.update_schedule_type(reduction, schedule);
    manager.invalidate_all();
    auto shared = manager.get<tiles::ReductionBufferAnalysis>().require(reduction, "acc");
    EXPECT_EQ(shared.private_bytes, 32);
    EXPECT_EQ(shared.shared_bytes, 256);
    EXPECT_EQ(shared.shared_owner, reduction.element_id());
    auto larger_schedule = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(16));
    serializer::JSONSerializer schedule_serializer;
    const auto before_schedule_preview = schedule_serializer.serialize(builder.subject());
    EXPECT_TRUE(manager.get<tiles::ReductionBufferAnalysis>().supports_schedule(reduction, larger_schedule));
    EXPECT_EQ(schedule_serializer.serialize(builder.subject()), before_schedule_preview);
    transformations::LoopTiling column_tiling(cols, 2, true);
    EXPECT_TRUE(column_tiling.can_be_applied(builder, manager));
    transformations::LoopInterchange interchange(reduction, rows);
    serializer::JSONSerializer serializer;
    const auto before_preview = serializer.serialize(builder.subject());
    const auto interchanged = manager.get<tiles::ReductionBufferAnalysis>()
                                  .estimate_interchange(reduction, "acc", shared, interchange.proposal());
    ASSERT_EQ(interchanged.status, tiles::ReductionBufferStatus::Exact);
    EXPECT_EQ(interchanged.private_bytes, 8);
    EXPECT_EQ(interchanged.shared_bytes, 64);
    EXPECT_TRUE(manager.get<tiles::ReductionBufferAnalysis>().supports_interchange(interchange.proposal()));
    EXPECT_EQ(serializer.serialize(builder.subject()), before_preview);
    auto expected_origin = symbolic::add(origin, symbolic::mul(symbolic::integer(16), row));
    gpu::ScheduleType_GPU_Offload::partial_storage(schedule, gpu::ReduceStrategy::Global);
    builder.update_schedule_type(reduction, schedule);
    manager.invalidate_all();
    const auto& global = manager.get<tiles::ReductionBufferAnalysis>().require(reduction, "acc");
    EXPECT_EQ(global.private_bytes, 32);
    EXPECT_EQ(global.shared_bytes, 0);
    EXPECT_FALSE(global.shared_owner);
    builder.update_schedule_type(
        reduction,
        gpu::ScheduleType_GPU_Offload::create<
            cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(8))
    );
    manager.invalidate_all();
    interchange.apply(builder, manager);
    auto* new_reduction = dyn_cast<structured_control_flow::Reduce*>(interchange.new_inner_loop());
    ASSERT_NE(new_reduction, nullptr);
    const auto& actual = manager.get<tiles::ReductionBufferAnalysis>().require(*new_reduction, "acc");
    EXPECT_EQ(actual.private_bytes, 8);
    EXPECT_EQ(actual.shared_bytes, 64);
    EXPECT_EQ(actual.layout->extent, 2);
    EXPECT_TRUE(symbolic::eq(actual.layout->base, expected_origin));
    EXPECT_TRUE(symbolic::eq(actual.layout->base, interchanged.layout->base));
    transformations::LoopInterchange reverse(*interchange.new_outer_loop(), *interchange.new_inner_loop());
    EXPECT_TRUE(manager.get<tiles::ReductionBufferAnalysis>().supports_interchange(reverse.proposal()));
    builder.add_container("unknown_bound", integer_type, true);
    auto& new_rows = *interchange.new_outer_loop();
    builder.update_loop(
        new_rows,
        new_rows.indvar(),
        symbolic::Lt(row, symbolic::symbol("unknown_bound")),
        symbolic::zero(),
        symbolic::add(row, symbolic::integer(2))
    );
    manager.invalidate_all();
    EXPECT_FALSE(manager.get<tiles::ReductionBufferAnalysis>().supports_interchange(reverse.proposal()));
    const auto before_reject = serializer.serialize(builder.subject());
    EXPECT_FALSE(reverse.can_be_applied(builder, manager));
    EXPECT_THROW(reverse.apply(builder, manager), InvalidSDFGException);
    EXPECT_EQ(before_reject, serializer.serialize(builder.subject()));
    builder.update_loop(
        new_rows,
        new_rows.indvar(),
        symbolic::Lt(row, symbolic::integer(8)),
        symbolic::zero(),
        symbolic::add(row, symbolic::integer(2))
    );
    manager.invalidate_all();
    builder.update_loop(
        cols,
        col,
        symbolic::Lt(col, symbolic::add(row, symbolic::one())),
        symbolic::zero(),
        symbolic::add(col, symbolic::one())
    );
    manager.invalidate_all();
    const auto bounded = manager.get<tiles::ReductionBufferAnalysis>().buffer(*new_reduction, "acc");
    EXPECT_EQ(bounded.status, tiles::ReductionBufferStatus::ConservativeBound) << bounded.diagnostic;
    EXPECT_FALSE(bounded.layout);
    ASSERT_TRUE(bounded.shared_bytes);
    EXPECT_GT(*bounded.shared_bytes, 64);
    EXPECT_THROW(manager.get<tiles::ReductionBufferAnalysis>().require(*new_reduction, "acc"), InvalidSDFGException);
    EXPECT_EQ(
        manager.get<tiles::ReductionBufferAnalysis>().kernel(new_rows).status,
        tiles::ReductionBufferStatus::ConservativeBound
    );
    builder.update_loop(
        cols, col, symbolic::Lt(col, symbolic::integer(2)), symbolic::zero(), symbolic::add(col, symbolic::one())
    );
    manager.invalidate_all();
    passes::ReductionSharedMemoryDelinearization pass;
    EXPECT_TRUE(pass.run_pass(builder, manager));
    const auto& packed = manager.get<tiles::ReductionBufferAnalysis>().require(*new_reduction, "acc");
    EXPECT_TRUE(packed.materialized);
    EXPECT_EQ(packed.private_bytes, 8);
    EXPECT_EQ(packed.shared_bytes, 64);
    EXPECT_FALSE(manager.get<tiles::ReductionBufferAnalysis>().supports_schedule(*new_reduction, larger_schedule));
    EXPECT_TRUE(manager.get<tiles::ReductionBufferAnalysis>()
                    .supports_schedule(*new_reduction, new_reduction->schedule_type()));
    EXPECT_EQ(input.data(), packed.private_buffer);
    EXPECT_EQ(output.data(), packed.private_buffer);
    const auto after_pass = serializer.serialize(builder.subject());
    EXPECT_FALSE(pass.run_pass(builder, manager));
    EXPECT_EQ(after_pass, serializer.serialize(builder.subject()));
    auto cloned = builder.subject().clone();
    analysis::AnalysisManager cloned_manager(*cloned);
    builder::StructuredSDFGBuilder cloned_builder(*cloned);
    EXPECT_FALSE(pass.run_pass(cloned_builder, cloned_manager));
    EXPECT_FALSE(reverse.can_be_applied(builder, manager));
    EXPECT_THROW(reverse.apply(builder, manager), InvalidSDFGException);
    auto input_edges = block.dataflow().out_edges(input);
    const auto before_tiling = serializer.serialize(builder.subject());
    transformations::LoopTiling packed_tiling(cols, 2, true);
    transformations::MultiLevelTiling packed_multilevel(new_rows, 4, 2, true);
    EXPECT_FALSE(packed_tiling.can_be_applied(builder, manager));
    EXPECT_THROW(packed_tiling.apply(builder, manager), InvalidSDFGException);
    EXPECT_FALSE(packed_multilevel.can_be_applied(builder, manager));
    EXPECT_THROW(packed_multilevel.apply(builder, manager), InvalidSDFGException);
    EXPECT_EQ(before_tiling, serializer.serialize(builder.subject()));
    auto& input_edge = *input_edges.begin();
    auto valid_subset = input_edge.subset();
    input_edge.set_subset({symbolic::integer(100)});
    manager.invalidate_all();
    EXPECT_THROW(manager.get<tiles::ReductionBufferAnalysis>().require(*new_reduction, "acc"), InvalidSDFGException);
    input_edge.set_subset(valid_subset);
    manager.invalidate_all();
    EXPECT_NO_THROW(manager.get<tiles::ReductionBufferAnalysis>().require(*new_reduction, "acc"));
}

TEST(ReductionBufferAnalysisTest, MaterializationReservesExistingOwnersBeforeMutation) {
    builder::StructuredSDFGBuilder builder("reduction_owners", FunctionType_CPU);
    types::Scalar scalar(types::PrimitiveType::Float);
    types::Pointer pointer(scalar);
    builder.add_container("step", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("acc", pointer);
    auto step = symbolic::symbol("step");
    auto add_reduction = [&](const std::string& partial) -> structured_control_flow::Reduce& {
        auto schedule = gpu::ScheduleType_GPU_Offload::create<
            cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(8));
        gpu::ScheduleType_GPU_Offload::partial_container(schedule, partial);
        auto& reduction = builder.add_reduce(
            builder.subject().root(),
            step,
            symbolic::Lt(step, symbolic::integer(8)),
            symbolic::zero(),
            symbolic::add(step, symbolic::one()),
            {{structured_control_flow::ReductionOperation::Add, "acc"}},
            schedule
        );
        auto& block = builder.add_block(reduction.root());
        auto& input = builder.add_access(block, "acc");
        auto& output = builder.add_access(block, "acc");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "out", {"left", "right"});
        builder.add_computational_memlet(block, input, tasklet, "left", {symbolic::zero()}, pointer);
        builder.add_computational_memlet(block, input, tasklet, "right", {symbolic::zero()}, pointer);
        builder.add_computational_memlet(block, tasklet, "out", output, {symbolic::zero()}, pointer);
        return reduction;
    };
    auto& first = add_reduction("placed_shared");
    analysis::AnalysisManager manager(builder.subject());
    passes::ReductionSharedMemoryDelinearization packing;
    ASSERT_TRUE(packing.run_pass(builder, manager));
    const auto original = manager.get<tiles::ReductionBufferAnalysis>().require(first, "acc");
    auto& second = add_reduction("placed_shared");
    manager.invalidate_all();
    serializer::JSONSerializer serializer;
    const auto before = serializer.serialize(builder.subject());
    EXPECT_THROW(packing.run_pass(builder, manager), InvalidSDFGException);
    EXPECT_EQ(serializer.serialize(builder.subject()), before);
    auto schedule = second.schedule_type();
    gpu::ScheduleType_GPU_Offload::partial_container(schedule, "");
    builder.update_schedule_type(second, schedule);
    manager.invalidate_all();
    ASSERT_TRUE(packing.run_pass(builder, manager));
    const auto added = manager.get<tiles::ReductionBufferAnalysis>().require(second, "acc");
    EXPECT_NE(added.private_buffer, original.private_buffer);
    EXPECT_NE(added.shared_buffer, original.shared_buffer);
    EXPECT_FALSE(packing.run_pass(builder, manager));

    auto& third = add_reduction("");
    auto& fourth = add_reduction("");
    auto prefix = "__daisy_reduce_reg_acc_" + std::to_string(third.element_id()) + "_";
    auto occupied_name = builder.find_new_name(prefix);
    builder.add_container(occupied_name, scalar);
    auto expected_name = builder.find_new_name(prefix);
    manager.invalidate_all();
    ASSERT_TRUE(packing.run_pass(builder, manager));
    const auto third_info = manager.get<tiles::ReductionBufferAnalysis>().require(third, "acc");
    const auto fourth_info = manager.get<tiles::ReductionBufferAnalysis>().require(fourth, "acc");
    EXPECT_EQ(third_info.private_buffer, expected_name);
    EXPECT_EQ(builder.subject().type(occupied_name), scalar);
    EXPECT_NE(third_info.private_buffer, fourth_info.private_buffer);
    EXPECT_NE(third_info.shared_buffer, fourth_info.shared_buffer);
    EXPECT_FALSE(packing.run_pass(builder, manager));
}

// is_reduction_accumulator detects it whether the Reduce is the loop itself, an
// ancestor, or a descendant.
TEST(ReductionAnalysisTest, IsReductionAccumulator_EnclosingAndNested) {
    builder::StructuredSDFGBuilder builder("ls_reduce_acc", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", types::Scalar(types::PrimitiveType::Float));
    auto j = symbolic::symbol("j");
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("j", loop_var);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);
    builder.add_container("acc", ptr);
    builder.add_container("other", ptr);

    auto& for_j =
        builder.add_for(seq, j, symbolic::Lt(j, N), symbolic::integer(0), symbolic::add(j, symbolic::integer(1)));
    auto& reduce_i = builder.add_reduce(
        for_j.root(),
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& loop_k =
        builder
            .add_for(reduce_i.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    EXPECT_TRUE(tiles::is_reduction_accumulator(reduce_i, "acc", am)); // the reduce itself
    EXPECT_TRUE(tiles::is_reduction_accumulator(loop_k, "acc", am)); // ancestor reduce
    EXPECT_TRUE(tiles::is_reduction_accumulator(for_j, "acc", am)); // descendant reduce
    EXPECT_FALSE(tiles::is_reduction_accumulator(loop_k, "other", am));
}

// collect_reduction_owners: a sequential (non-cooperative) Reduce at the localized
// loop is privatizable — it is returned so apply() can retarget its descriptor.
TEST(ReductionAnalysisTest, CollectReductionOwners_SequentialAccepts) {
    builder::StructuredSDFGBuilder builder("ls_collect_seq", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", types::Scalar(types::PrimitiveType::Float));
    auto j = symbolic::symbol("j");
    auto N = symbolic::symbol("N");
    builder.add_container("N", loop_var, true);
    builder.add_container("j", loop_var);
    builder.add_container("acc", ptr);

    auto& reduce_j = builder.add_reduce(
        seq,
        j,
        symbolic::Lt(j, N),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );

    analysis::AnalysisManager am(builder.subject());
    std::vector<structured_control_flow::Reduce*> owners;
    EXPECT_TRUE(tiles::collect_reduction_owners(reduce_j, "acc", am, owners));
    ASSERT_EQ(owners.size(), 1u);
    EXPECT_EQ(owners.front(), &reduce_j);
}

// collect_reduction_owners: a GPU-offloaded (cooperatively combined) Reduce is
// owned by the reduce dispatcher — reject.
TEST(ReductionAnalysisTest, CollectReductionOwners_CooperativeRejects) {
    builder::StructuredSDFGBuilder builder("ls_collect_coop", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", types::Scalar(types::PrimitiveType::Float));
    auto j = symbolic::symbol("j");
    auto N = symbolic::symbol("N");
    builder.add_container("N", loop_var, true);
    builder.add_container("j", loop_var);
    builder.add_container("acc", ptr);

    auto block = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(32));
    auto& reduce_j = builder.add_reduce(
        seq,
        j,
        symbolic::Lt(j, N),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        block
    );

    analysis::AnalysisManager am(builder.subject());
    std::vector<structured_control_flow::Reduce*> owners;
    EXPECT_FALSE(tiles::collect_reduction_owners(reduce_j, "acc", am, owners));
}

// collect_reduction_owners: a *sequential* ancestor Reduce is accepted — its outer
// iterations are barrier-separated, so a read-modify-write copy-in/out around the
// localized loop carries the accumulation through the global container each
// iteration (classical BLIS pc loop). The ancestor is not retargeted (owners empty).
TEST(ReductionAnalysisTest, CollectReductionOwners_SequentialAncestorAccepts) {
    builder::StructuredSDFGBuilder builder("ls_collect_ancestor_seq", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", types::Scalar(types::PrimitiveType::Float));
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("j", loop_var);
    builder.add_container("k", loop_var);
    builder.add_container("acc", ptr);

    auto& reduce_j = builder.add_reduce(
        seq,
        j,
        symbolic::Lt(j, N),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& loop_k =
        builder
            .add_for(reduce_j.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    std::vector<structured_control_flow::Reduce*> owners;
    EXPECT_TRUE(tiles::collect_reduction_owners(loop_k, "acc", am, owners));
    EXPECT_TRUE(owners.empty());
}

// collect_reduction_owners: a GPU block-cooperative ancestor Reduce is combined by
// the reduce dispatcher (not an atomic-merge grid reduce), so localizing its
// accumulator at an inner loop is rejected.
TEST(ReductionAnalysisTest, CollectReductionOwners_GpuBlockAncestorRejects) {
    builder::StructuredSDFGBuilder builder("ls_collect_ancestor_block", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", types::Scalar(types::PrimitiveType::Float));
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("j", loop_var);
    builder.add_container("k", loop_var);
    builder.add_container("acc", ptr);

    auto block = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(32));
    auto& reduce_j = builder.add_reduce(
        seq,
        j,
        symbolic::Lt(j, N),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        block
    );
    auto& loop_k =
        builder
            .add_for(reduce_j.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    std::vector<structured_control_flow::Reduce*> owners;
    EXPECT_FALSE(tiles::collect_reduction_owners(loop_k, "acc", am, owners));
}

// collect_reduction_owners: a grid-parallel (split-K Z_GRID) ancestor Reduce merges
// per-block partials via an atomic writeback, so privatizing the per-block partial
// into a register tile at an inner loop is permitted — and the reduce is NOT
// retargeted (the cross-block merge is taken over separately).
TEST(ReductionAnalysisTest, CollectReductionOwners_GridParallelAncestorAccepts) {
    builder::StructuredSDFGBuilder builder("ls_collect_grid_ancestor", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", types::Scalar(types::PrimitiveType::Float));
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("j", loop_var);
    builder.add_container("k", loop_var);
    builder.add_container("acc", ptr);

    auto grid = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::Z_GRID, symbolic::integer(16));
    auto& reduce_j = builder.add_reduce(
        seq,
        j,
        symbolic::Lt(j, N),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        grid
    );
    auto& loop_k =
        builder
            .add_for(reduce_j.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    std::vector<structured_control_flow::Reduce*> owners;
    EXPECT_TRUE(tiles::collect_reduction_owners(loop_k, "acc", am, owners));
    EXPECT_TRUE(owners.empty());
}
