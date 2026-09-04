#include "sdfg/passes/dataflow/memlet_simplification.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"
#include "sdfg_debug_dump.h"
#include "symengine/add.h"

using namespace sdfg;

namespace {

structured_control_flow::Map& add_normal_map(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& parent,
    symbolic::Symbol iv,
    symbolic::Expression bound
) {
    return builder.add_map(
        parent,
        iv,
        symbolic::Lt(iv, bound),
        symbolic::integer(0),
        symbolic::add(iv, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
}

} // namespace

// 2D: 56*(idx/56) + (idx%56) == idx
TEST(MemletSimplification, SimplifyTwoDimensional) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar scalar(types::PrimitiveType::Float);
    types::Pointer ptr_type(scalar);
    types::Pointer opaque_ptr;

    builder.add_container("source", opaque_ptr);
    builder.add_container("arr", opaque_ptr);
    builder.add_container("idx", types::Scalar(types::PrimitiveType::UInt64));

    auto idx = symbolic::symbol("idx");
    auto& root = builder.subject().root();

    auto& map = add_normal_map(builder, root, idx, symbolic::integer(3136));
    auto& block = builder.add_block(map.root());

    auto& access_source = builder.add_access(block, "source");
    auto& access_arr = builder.add_access(block, "arr");

    // 56*(idx/56) + (idx%56)
    auto term1 = symbolic::mul(symbolic::integer(56), symbolic::div(idx, symbolic::integer(56)));
    auto term2 = symbolic::mod(idx, symbolic::integer(56));
    auto index_expr = symbolic::add(term1, term2);

    auto& memlet = builder.add_reference_memlet(block, access_source, access_arr, {index_expr}, ptr_type);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::MemletSimplificationPass pass;
    EXPECT_TRUE(pass.run(builder_opt, analysis_manager));

    auto& subset = memlet.subset();
    EXPECT_EQ(subset.size(), 1);
    EXPECT_TRUE(symbolic::eq(subset[0], idx));
}

// 3D: 3136*(idx/3136) + 56*((idx/56)%56) + (idx%56) == idx
TEST(MemletSimplification, SimplifyThreeDimensional) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar scalar(types::PrimitiveType::Float);
    types::Pointer ptr_type(scalar);
    types::Pointer opaque_ptr;

    builder.add_container("source", opaque_ptr);
    builder.add_container("arr", opaque_ptr);
    builder.add_container("idx", types::Scalar(types::PrimitiveType::UInt64));

    auto idx = symbolic::symbol("idx");
    auto& root = builder.subject().root();

    auto& map = add_normal_map(builder, root, idx, symbolic::integer(802816));
    auto& block = builder.add_block(map.root());

    auto& access_source = builder.add_access(block, "source");
    auto& access_arr = builder.add_access(block, "arr");

    // 3136*(idx/3136) + 56*((idx/56)%56) + (idx%56)
    auto term1 = symbolic::mul(symbolic::integer(3136), symbolic::div(idx, symbolic::integer(3136)));
    auto term2 = symbolic::
        mul(symbolic::integer(56), symbolic::mod(symbolic::div(idx, symbolic::integer(56)), symbolic::integer(56)));
    auto term3 = symbolic::mod(idx, symbolic::integer(56));
    auto index_expr = symbolic::add(symbolic::add(term1, term2), term3);

    auto& memlet = builder.add_reference_memlet(block, access_source, access_arr, {index_expr}, ptr_type);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::MemletSimplificationPass pass;
    EXPECT_TRUE(pass.run(builder_opt, analysis_manager));

    auto& subset = memlet.subset();
    EXPECT_EQ(subset.size(), 1);
    EXPECT_TRUE(symbolic::eq(subset[0], idx));
}

// 4D ReLU pattern [64, 256, 56, 56]:
// 802816*(idx/802816) + 3136*((idx/3136)%256) + 56*((idx/56)%56) + (idx%56) == idx
TEST(MemletSimplification, SimplifyFourDimensionalReLU) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar scalar(types::PrimitiveType::Float);
    types::Pointer ptr_type(scalar);
    types::Pointer opaque_ptr;

    builder.add_container("source", opaque_ptr);
    builder.add_container("arr", opaque_ptr);
    builder.add_container("idx", types::Scalar(types::PrimitiveType::UInt64));

    auto idx = symbolic::symbol("idx");
    auto& root = builder.subject().root();

    // 64*256*56*56 = 51380224
    auto& map = add_normal_map(builder, root, idx, symbolic::integer(51380224));
    auto& block = builder.add_block(map.root());

    auto& access_source = builder.add_access(block, "source");
    auto& access_arr = builder.add_access(block, "arr");

    // 802816*(idx/802816) + 3136*((idx/3136)%256) + 56*((idx/56)%56) + (idx%56)
    auto term1 = symbolic::mul(symbolic::integer(802816), symbolic::div(idx, symbolic::integer(802816)));
    auto term2 = symbolic::
        mul(symbolic::integer(3136),
            symbolic::mod(symbolic::div(idx, symbolic::integer(3136)), symbolic::integer(256)));
    auto term3 = symbolic::
        mul(symbolic::integer(56), symbolic::mod(symbolic::div(idx, symbolic::integer(56)), symbolic::integer(56)));
    auto term4 = symbolic::mod(idx, symbolic::integer(56));
    auto index_expr = symbolic::add(symbolic::add(symbolic::add(term1, term2), term3), term4);

    auto& memlet = builder.add_reference_memlet(block, access_source, access_arr, {index_expr}, ptr_type);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::MemletSimplificationPass pass;
    EXPECT_TRUE(pass.run(builder_opt, analysis_manager));

    auto& subset = memlet.subset();
    EXPECT_EQ(subset.size(), 1);
    EXPECT_TRUE(symbolic::eq(subset[0], idx));
}

TEST(MemletSimplification, SimplifyCollapsedIndexing) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("args_0", desc, true);
    builder.add_container("args_1", index_desc, true);
    builder.add_container("index", desc, true);
    builder.add_container("_i0_collapsed0", sym_desc);
    builder.add_container("_i3", sym_desc);

    auto i0 = symbolic::symbol("_i0_collapsed0");
    auto i3 = symbolic::symbol("_i3");
    auto two = symbolic::integer(2);
    auto four = symbolic::integer(4);
    data_flow::Subset args_1_subset = {
        symbolic::add(symbolic::mod(i0, two), symbolic::mul(two, symbolic::mod(symbolic::div(i0, two), two)))
    };
    data_flow::Subset args_0_subset = {symbolic::add(i3, symbolic::mul(four, symbolic::div(i0, four)))};
    data_flow::Subset index_subset = {SymEngine::add(
        {symbolic::mul(four, symbolic::div(i0, four)),
         symbolic::mod(i0, two),
         symbolic::mul(two, symbolic::mod(symbolic::div(i0, two), two))}
    )};

    auto& map = builder.add_map(
        root,
        i0,
        symbolic::Lt(i0, symbolic::integer(20)),
        symbolic::zero(),
        symbolic::add(i0, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto& index_block = builder.add_block(map.root());
    auto& args_1_access = builder.add_access(index_block, "args_1");
    auto& i3_access = builder.add_access(index_block, "_i3");
    auto& index_tasklet = builder.add_tasklet(index_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& args_1_memlet =
        builder.add_computational_memlet(index_block, args_1_access, index_tasklet, "_in", args_1_subset);
    builder.add_computational_memlet(index_block, index_tasklet, "_out", i3_access, {});

    auto& copy_block = builder.add_block(map.root());
    auto& args_0_access = builder.add_access(copy_block, "args_0");
    auto& index_access = builder.add_access(copy_block, "index");
    auto& copy_tasklet = builder.add_tasklet(copy_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(copy_block, args_0_access, copy_tasklet, "_in", args_0_subset);
    builder.add_computational_memlet(copy_block, copy_tasklet, "_out", index_access, index_subset);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::MemletSimplificationPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");

    ASSERT_EQ(args_1_memlet.subset().size(), 1);
    EXPECT_TRUE(symbolic::eq(args_1_memlet.subset()[0], args_1_subset[0]));
}
