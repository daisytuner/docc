#include "sdfg/passes/dataflow/memlet_simplification.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

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
