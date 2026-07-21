#include "sdfg/passes/zero_fill_to_memset_pass.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

namespace {

// Element type used throughout: 8-byte double.
static const types::Scalar kElement(types::PrimitiveType::Double);
static constexpr int kElementBytes = 8;

// Helper wrapper that builds normalized [0, bound) unit-stride Map nests.
class NestFixture {
public:
    builder::StructuredSDFGBuilder& builder;
    structured_control_flow::ScheduleType sched = structured_control_flow::ScheduleType_Sequential::create();

    explicit NestFixture(builder::StructuredSDFGBuilder& builder) : builder(builder) {}

    // Registers a size symbol as an integer container (idempotent per name).
    symbolic::Symbol size(const std::string& name) {
        if (!builder.subject().exists(name)) {
            builder.add_container(name, types::Scalar(types::PrimitiveType::Int64), true);
            // Array/loop extents are positive; make that explicit for the layout analysis.
            builder.subject().assumption(symbolic::symbol(name)).add_lower_bound(symbolic::one());
        }
        return symbolic::symbol(name);
    }

    structured_control_flow::Map&
    add_map(structured_control_flow::Sequence& parent, const std::string& iv, const symbolic::Expression& bound) {
        builder.add_container(iv, types::Scalar(types::PrimitiveType::Int64));
        auto sym = symbolic::symbol(iv);
        return builder
            .add_map(parent, sym, symbolic::Lt(sym, bound), symbolic::zero(), symbolic::add(sym, symbolic::one()), sched);
    }

    // Builds a perfect nest of maps [iv_0, bound_0], ... and returns the innermost body.
    structured_control_flow::Sequence& build_nest(
        structured_control_flow::Sequence& root,
        const std::vector<std::string>& ivs,
        const std::vector<symbolic::Expression>& bounds
    ) {
        structured_control_flow::Sequence* cur = &root;
        for (size_t i = 0; i < ivs.size(); i++) {
            auto& map = add_map(*cur, ivs[i], bounds[i]);
            cur = &map.root();
        }
        return *cur;
    }

    // Emits a single "array[subset] = <value>" assignment into a fresh block.
    void write_scalar(
        structured_control_flow::Sequence& body,
        const std::string& array,
        const data_flow::Subset& subset,
        const types::IType& base_type,
        const std::string& value
    ) {
        auto& block = builder.add_block(body);
        auto& constant = builder.add_constant(block, value, kElement);
        auto& access = builder.add_access(block, array);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, constant, tasklet, "_in", {});
        builder.add_computational_memlet(block, tasklet, "_out", access, subset, base_type);
    }

    // Emits a single "dst[subset] = src[subset]" load-and-store into a fresh block.
    void copy_scalar(
        structured_control_flow::Sequence& body,
        const std::string& src,
        const std::string& dst,
        const data_flow::Subset& subset,
        const types::IType& base_type
    ) {
        auto& block = builder.add_block(body);
        auto& src_access = builder.add_access(block, src);
        auto& dst_access = builder.add_access(block, dst);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, src_access, tasklet, "_in", subset, base_type);
        builder.add_computational_memlet(block, tasklet, "_out", dst_access, subset, base_type);
    }
};

// Collects every MemsetNode reachable at the top level of the SDFG root.
std::vector<const stdlib::MemsetNode*> collect_memsets(structured_control_flow::Sequence& root) {
    std::vector<const stdlib::MemsetNode*> result;
    for (size_t i = 0; i < root.size(); i++) {
        auto* block = dynamic_cast<structured_control_flow::Block*>(&root.at(i).first);
        if (block == nullptr) {
            continue;
        }
        for (auto* node : block->dataflow().library_nodes()) {
            if (auto* memset = dynamic_cast<const stdlib::MemsetNode*>(node)) {
                result.push_back(memset);
            }
        }
    }
    return result;
}

// Returns the name of the container the memset writes to (its "_ptr" input).
std::string memset_target(structured_control_flow::Sequence& root, const stdlib::MemsetNode& memset) {
    for (size_t i = 0; i < root.size(); i++) {
        auto* block = dynamic_cast<structured_control_flow::Block*>(&root.at(i).first);
        if (block == nullptr) {
            continue;
        }
        auto& dfg = block->dataflow();
        // Only inspect the block that actually owns this memset node; calling
        // in_edges() for a node from a different graph is undefined.
        bool owns = false;
        for (auto* node : dfg.library_nodes()) {
            if (static_cast<const void*>(node) == static_cast<const void*>(&memset)) {
                owns = true;
                break;
            }
        }
        if (!owns) {
            continue;
        }
        for (auto& edge : dfg.in_edges(memset)) {
            if (auto* access = dynamic_cast<const data_flow::AccessNode*>(&edge.src())) {
                return access->data();
            }
        }
    }
    return "";
}

bool run(builder::StructuredSDFGBuilder& builder) {
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ZeroFillToMemsetPass pass;
    return pass.run_pass(builder, analysis_manager);
}

} // namespace

// ---------------------------------------------------------------------------
// Positive cases: a full zero-fill nest is rewritten into a single memset.
// ---------------------------------------------------------------------------

TEST(ZeroFillToMemsetPassTest, FullAccess_1DArray) {
    builder::StructuredSDFGBuilder builder("sdfg_1d_array", FunctionType_CPU);
    NestFixture fx(builder);

    auto n = fx.size("N");
    types::Array array(kElement, n);
    builder.add_container("A", array);

    auto& body = fx.build_nest(builder.subject().root(), {"i"}, {n});
    fx.write_scalar(body, "A", {symbolic::symbol("i")}, array, "0");

    dump_sdfg(builder.subject(), "0.init");
    EXPECT_TRUE(run(builder));

    auto& root = builder.subject().root();
    auto memsets = collect_memsets(root);
    ASSERT_EQ(memsets.size(), 1);
    EXPECT_EQ(memset_target(root, *memsets[0]), "A");
    EXPECT_TRUE(symbolic::eq(memsets[0]->value(), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(memsets[0]->num(), symbolic::mul(symbolic::integer(kElementBytes), n)));
}

TEST(ZeroFillToMemsetPassTest, FullAccess_2DArray) {
    builder::StructuredSDFGBuilder builder("sdfg_2d_array", FunctionType_CPU);
    NestFixture fx(builder);

    auto n = fx.size("N");
    auto m = fx.size("M");
    types::Array inner(kElement, m);
    types::Array array(inner, n);
    builder.add_container("A", array);

    auto& body = fx.build_nest(builder.subject().root(), {"i", "j"}, {n, m});
    fx.write_scalar(body, "A", {symbolic::symbol("i"), symbolic::symbol("j")}, array, "0");

    dump_sdfg(builder.subject(), "0.init");
    EXPECT_TRUE(run(builder));

    auto& root = builder.subject().root();
    auto memsets = collect_memsets(root);
    ASSERT_EQ(memsets.size(), 1);
    EXPECT_EQ(memset_target(root, *memsets[0]), "A");
    EXPECT_TRUE(symbolic::eq(memsets[0]->num(), symbolic::mul(symbolic::mul(symbolic::integer(kElementBytes), n), m)));
}

TEST(ZeroFillToMemsetPassTest, FullAccess_1DPointer) {
    builder::StructuredSDFGBuilder builder("sdfg_1d_ptr", FunctionType_CPU);
    NestFixture fx(builder);

    auto n = fx.size("N");
    types::Pointer ptr(kElement);
    builder.add_container("A", ptr, true);

    auto& body = fx.build_nest(builder.subject().root(), {"i"}, {n});
    fx.write_scalar(body, "A", {symbolic::symbol("i")}, ptr, "0");

    dump_sdfg(builder.subject(), "0.init");
    EXPECT_TRUE(run(builder));

    auto& root = builder.subject().root();
    auto memsets = collect_memsets(root);
    ASSERT_EQ(memsets.size(), 1);
    EXPECT_EQ(memset_target(root, *memsets[0]), "A");
    EXPECT_TRUE(symbolic::eq(memsets[0]->num(), symbolic::mul(symbolic::integer(kElementBytes), n)));
}

TEST(ZeroFillToMemsetPassTest, FullAccess_2DPointer) {
    builder::StructuredSDFGBuilder builder("sdfg_2d_ptr", FunctionType_CPU);
    NestFixture fx(builder);

    auto n = fx.size("N");
    auto m = fx.size("M");
    types::Array inner(kElement, m);
    types::Pointer ptr(inner);
    builder.add_container("A", ptr, true);

    auto& body = fx.build_nest(builder.subject().root(), {"i", "j"}, {n, m});
    fx.write_scalar(body, "A", {symbolic::symbol("i"), symbolic::symbol("j")}, ptr, "0");

    dump_sdfg(builder.subject(), "0.init");
    EXPECT_TRUE(run(builder));

    auto& root = builder.subject().root();
    auto memsets = collect_memsets(root);
    ASSERT_EQ(memsets.size(), 1);
    EXPECT_EQ(memset_target(root, *memsets[0]), "A");
    EXPECT_TRUE(symbolic::eq(memsets[0]->num(), symbolic::mul(symbolic::mul(symbolic::integer(kElementBytes), n), m)));
}

TEST(ZeroFillToMemsetPassTest, FullAccess_3DPointer) {
    builder::StructuredSDFGBuilder builder("sdfg_3d_ptr", FunctionType_CPU);
    NestFixture fx(builder);

    auto n = fx.size("N");
    auto m = fx.size("M");
    auto k = fx.size("K");
    types::Array inner(kElement, k);
    types::Array middle(inner, m);
    types::Pointer ptr(middle);
    builder.add_container("A", ptr, true);

    auto& body = fx.build_nest(builder.subject().root(), {"i", "j", "l"}, {n, m, k});
    fx.write_scalar(body, "A", {symbolic::symbol("i"), symbolic::symbol("j"), symbolic::symbol("l")}, ptr, "0");

    dump_sdfg(builder.subject(), "0.init");
    EXPECT_TRUE(run(builder));

    auto& root = builder.subject().root();
    auto memsets = collect_memsets(root);
    ASSERT_EQ(memsets.size(), 1);
    EXPECT_EQ(memset_target(root, *memsets[0]), "A");
    EXPECT_TRUE(symbolic::
                    eq(memsets[0]->num(),
                       symbolic::mul(symbolic::mul(symbolic::mul(symbolic::integer(kElementBytes), n), m), k)));
}

TEST(ZeroFillToMemsetPassTest, FullAccess_TwoIndependent1DPointers) {
    builder::StructuredSDFGBuilder builder("sdfg_two_1d_ptr", FunctionType_CPU);
    NestFixture fx(builder);

    auto n = fx.size("N");
    types::Pointer ptr(kElement);
    builder.add_container("A", ptr, true);
    builder.add_container("B", ptr, true);

    auto& root = builder.subject().root();
    auto& body_a = fx.build_nest(root, {"i"}, {n});
    fx.write_scalar(body_a, "A", {symbolic::symbol("i")}, ptr, "0");
    auto& body_b = fx.build_nest(root, {"j"}, {n});
    fx.write_scalar(body_b, "B", {symbolic::symbol("j")}, ptr, "0");

    dump_sdfg(builder.subject(), "0.init");
    EXPECT_TRUE(run(builder));

    auto memsets = collect_memsets(root);
    ASSERT_EQ(memsets.size(), 2);
    std::vector<std::string> targets{memset_target(root, *memsets[0]), memset_target(root, *memsets[1])};
    EXPECT_NE(std::find(targets.begin(), targets.end(), "A"), targets.end());
    EXPECT_NE(std::find(targets.begin(), targets.end(), "B"), targets.end());
}

TEST(ZeroFillToMemsetPassTest, FullAccess_TwoIndependent2DPointers) {
    builder::StructuredSDFGBuilder builder("sdfg_two_2d_ptr", FunctionType_CPU);
    NestFixture fx(builder);

    auto n = fx.size("N");
    auto m = fx.size("M");
    types::Array inner(kElement, m);
    types::Pointer ptr(inner);
    builder.add_container("A", ptr, true);
    builder.add_container("B", ptr, true);

    auto& root = builder.subject().root();
    auto& body_a = fx.build_nest(root, {"i", "j"}, {n, m});
    fx.write_scalar(body_a, "A", {symbolic::symbol("i"), symbolic::symbol("j")}, ptr, "0");
    auto& body_b = fx.build_nest(root, {"p", "q"}, {n, m});
    fx.write_scalar(body_b, "B", {symbolic::symbol("p"), symbolic::symbol("q")}, ptr, "0");

    dump_sdfg(builder.subject(), "0.init");
    EXPECT_TRUE(run(builder));

    auto memsets = collect_memsets(root);
    ASSERT_EQ(memsets.size(), 2);
    std::vector<std::string> targets{memset_target(root, *memsets[0]), memset_target(root, *memsets[1])};
    EXPECT_NE(std::find(targets.begin(), targets.end(), "A"), targets.end());
    EXPECT_NE(std::find(targets.begin(), targets.end(), "B"), targets.end());
}

// ---------------------------------------------------------------------------
// Positive cases: linearized accesses. The nest indexes a flat pointer buffer with a
// single delinearizable expression (e.g. `A[i*M + j]`) instead of one index per loop.
// These reach the memory-layout analysis' delinearization path, which the plain
// per-dimension accesses above never exercise.
//
// Note: linearization is only modeled for flat pointer buffers. A native fixed-size
// array carries its own shape and is never delinearized, so there is no
// "linearized native array" case here.
// ---------------------------------------------------------------------------

TEST(ZeroFillToMemsetPassTest, Linearized_FullAccess_2DPointer) {
    builder::StructuredSDFGBuilder builder("sdfg_lin_2d_ptr", FunctionType_CPU);
    NestFixture fx(builder);

    auto n = fx.size("N");
    auto m = fx.size("M");
    types::Pointer ptr(kElement); // flat pointer, layout inferred from A[i*M + j]
    builder.add_container("A", ptr, true);

    auto& body = fx.build_nest(builder.subject().root(), {"i", "j"}, {n, m});
    auto linearized = symbolic::add(symbolic::mul(symbolic::symbol("i"), m), symbolic::symbol("j"));
    fx.write_scalar(body, "A", {linearized}, ptr, "0");

    dump_sdfg(builder.subject(), "0.init");
    EXPECT_TRUE(run(builder));

    auto& root = builder.subject().root();
    auto memsets = collect_memsets(root);
    ASSERT_EQ(memsets.size(), 1);
    EXPECT_EQ(memset_target(root, *memsets[0]), "A");
    EXPECT_TRUE(symbolic::eq(memsets[0]->num(), symbolic::mul(symbolic::mul(symbolic::integer(kElementBytes), n), m)));
}

TEST(ZeroFillToMemsetPassTest, Linearized_FullAccess_3DPointer) {
    builder::StructuredSDFGBuilder builder("sdfg_lin_3d_ptr", FunctionType_CPU);
    NestFixture fx(builder);

    auto n = fx.size("N");
    auto m = fx.size("M");
    auto k = fx.size("K");
    types::Pointer ptr(kElement);
    builder.add_container("A", ptr, true);

    auto& body = fx.build_nest(builder.subject().root(), {"i", "j", "l"}, {n, m, k});
    // A[i*M*K + j*K + l]
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto l = symbolic::symbol("l");
    auto linearized = symbolic::add(symbolic::add(symbolic::mul(symbolic::mul(i, m), k), symbolic::mul(j, k)), l);
    fx.write_scalar(body, "A", {linearized}, ptr, "0");

    dump_sdfg(builder.subject(), "0.init");
    EXPECT_TRUE(run(builder));

    auto& root = builder.subject().root();
    auto memsets = collect_memsets(root);
    ASSERT_EQ(memsets.size(), 1);
    EXPECT_EQ(memset_target(root, *memsets[0]), "A");
    EXPECT_TRUE(symbolic::
                    eq(memsets[0]->num(),
                       symbolic::mul(symbolic::mul(symbolic::mul(symbolic::integer(kElementBytes), n), m), k)));
}

TEST(ZeroFillToMemsetPassTest, Linearized_FullAccess_TwoIndependent2DPointers) {
    builder::StructuredSDFGBuilder builder("sdfg_lin_two_2d_ptr", FunctionType_CPU);
    NestFixture fx(builder);

    auto n = fx.size("N");
    auto m = fx.size("M");
    types::Pointer ptr(kElement);
    builder.add_container("A", ptr, true);
    builder.add_container("B", ptr, true);

    auto& root = builder.subject().root();
    auto& body_a = fx.build_nest(root, {"i", "j"}, {n, m});
    fx.write_scalar(
        body_a, "A", {symbolic::add(symbolic::mul(symbolic::symbol("i"), m), symbolic::symbol("j"))}, ptr, "0"
    );
    auto& body_b = fx.build_nest(root, {"p", "q"}, {n, m});
    fx.write_scalar(
        body_b, "B", {symbolic::add(symbolic::mul(symbolic::symbol("p"), m), symbolic::symbol("q"))}, ptr, "0"
    );

    dump_sdfg(builder.subject(), "0.init");
    EXPECT_TRUE(run(builder));

    auto memsets = collect_memsets(root);
    ASSERT_EQ(memsets.size(), 2);
    std::vector<std::string> targets{memset_target(root, *memsets[0]), memset_target(root, *memsets[1])};
    EXPECT_NE(std::find(targets.begin(), targets.end(), "A"), targets.end());
    EXPECT_NE(std::find(targets.begin(), targets.end(), "B"), targets.end());
}

// ---------------------------------------------------------------------------
// Negative cases: the nest is left untouched.
// ---------------------------------------------------------------------------

// The loop bound is smaller than the declared array extent, so the fill is partial.
TEST(ZeroFillToMemsetPassTest, PartialAccess_1DArray) {
    builder::StructuredSDFGBuilder builder("sdfg_partial_1d_array", FunctionType_CPU);
    NestFixture fx(builder);

    auto n = fx.size("N");
    auto m = fx.size("M");
    types::Array array(kElement, n); // container has N elements ...
    builder.add_container("A", array);

    auto& body = fx.build_nest(builder.subject().root(), {"i"}, {m}); // ... but only M are written
    fx.write_scalar(body, "A", {symbolic::symbol("i")}, array, "0");

    dump_sdfg(builder.subject(), "0.init");
    EXPECT_FALSE(run(builder));
    EXPECT_TRUE(collect_memsets(builder.subject().root()).empty());
}

// The inner dimension is only partially covered.
TEST(ZeroFillToMemsetPassTest, PartialAccess_2DArray) {
    builder::StructuredSDFGBuilder builder("sdfg_partial_2d_array", FunctionType_CPU);
    NestFixture fx(builder);

    auto n = fx.size("N");
    auto m = fx.size("M");
    auto m2 = fx.size("M2");
    types::Array inner(kElement, m); // inner extent is M ...
    types::Array array(inner, n);
    builder.add_container("A", array);

    auto& body = fx.build_nest(builder.subject().root(), {"i", "j"}, {n, m2}); // ... but only M2 written
    fx.write_scalar(body, "A", {symbolic::symbol("i"), symbolic::symbol("j")}, array, "0");

    dump_sdfg(builder.subject(), "0.init");
    EXPECT_FALSE(run(builder));
    EXPECT_TRUE(collect_memsets(builder.subject().root()).empty());
}

// The pointee array is only partially covered by the inner loop.
TEST(ZeroFillToMemsetPassTest, PartialAccess_2DPointer) {
    builder::StructuredSDFGBuilder builder("sdfg_partial_2d_ptr", FunctionType_CPU);
    NestFixture fx(builder);

    auto n = fx.size("N");
    auto m = fx.size("M");
    auto m2 = fx.size("M2");
    types::Array inner(kElement, m);
    types::Pointer ptr(inner);
    builder.add_container("A", ptr, true);

    auto& body = fx.build_nest(builder.subject().root(), {"i", "j"}, {n, m2});
    fx.write_scalar(body, "A", {symbolic::symbol("i"), symbolic::symbol("j")}, ptr, "0");

    dump_sdfg(builder.subject(), "0.init");
    EXPECT_FALSE(run(builder));
    EXPECT_TRUE(collect_memsets(builder.subject().root()).empty());
}

// Linearized negatives: same linearized shapes as the positive cases above, but the inner
// loop stops short of the linearization stride, so the delinearized inner extent (M / K)
// does not match the loop bound (M2 / K2) -> partial fill, left untouched.

// A[i*M + j] on a flat pointer, but the inner loop only covers M2 < M of the M-wide stride.
TEST(ZeroFillToMemsetPassTest, Linearized_PartialAccess_2DPointer) {
    builder::StructuredSDFGBuilder builder("sdfg_lin_partial_2d_ptr", FunctionType_CPU);
    NestFixture fx(builder);

    auto n = fx.size("N");
    auto m = fx.size("M");
    auto m2 = fx.size("M2");
    types::Pointer ptr(kElement);
    builder.add_container("A", ptr, true);

    auto& body = fx.build_nest(builder.subject().root(), {"i", "j"}, {n, m2});
    auto linearized = symbolic::add(symbolic::mul(symbolic::symbol("i"), m), symbolic::symbol("j"));
    fx.write_scalar(body, "A", {linearized}, ptr, "0");

    dump_sdfg(builder.subject(), "0.init");
    EXPECT_FALSE(run(builder));
    EXPECT_TRUE(collect_memsets(builder.subject().root()).empty());
}

// A[i*M*K + j*K + l] on a flat pointer, but the innermost loop only covers K2 < K.
TEST(ZeroFillToMemsetPassTest, Linearized_PartialAccess_3DPointer) {
    builder::StructuredSDFGBuilder builder("sdfg_lin_partial_3d_ptr", FunctionType_CPU);
    NestFixture fx(builder);

    auto n = fx.size("N");
    auto m = fx.size("M");
    auto k = fx.size("K");
    auto k2 = fx.size("K2");
    types::Pointer ptr(kElement);
    builder.add_container("A", ptr, true);

    auto& body = fx.build_nest(builder.subject().root(), {"i", "j", "l"}, {n, m, k2});
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto l = symbolic::symbol("l");
    auto linearized = symbolic::add(symbolic::add(symbolic::mul(symbolic::mul(i, m), k), symbolic::mul(j, k)), l);
    fx.write_scalar(body, "A", {linearized}, ptr, "0");

    dump_sdfg(builder.subject(), "0.init");
    EXPECT_FALSE(run(builder));
    EXPECT_TRUE(collect_memsets(builder.subject().root()).empty());
}

// A full linearized fill, but the stored constant is non-zero.
TEST(ZeroFillToMemsetPassTest, Linearized_NonZeroConstant) {
    builder::StructuredSDFGBuilder builder("sdfg_lin_nonzero_const", FunctionType_CPU);
    NestFixture fx(builder);

    auto n = fx.size("N");
    auto m = fx.size("M");
    types::Pointer ptr(kElement);
    builder.add_container("A", ptr, true);

    auto& body = fx.build_nest(builder.subject().root(), {"i", "j"}, {n, m});
    auto linearized = symbolic::add(symbolic::mul(symbolic::symbol("i"), m), symbolic::symbol("j"));
    fx.write_scalar(body, "A", {linearized}, ptr, "1.0");

    dump_sdfg(builder.subject(), "0.init");
    EXPECT_FALSE(run(builder));
    EXPECT_TRUE(collect_memsets(builder.subject().root()).empty());
}

// A full linearized nest, but the stored value is a variable load, not a constant.
TEST(ZeroFillToMemsetPassTest, Linearized_VariableInput) {
    builder::StructuredSDFGBuilder builder("sdfg_lin_variable_input", FunctionType_CPU);
    NestFixture fx(builder);

    auto n = fx.size("N");
    auto m = fx.size("M");
    types::Pointer ptr(kElement);
    builder.add_container("A", ptr, true);
    builder.add_container("B", ptr, true);

    auto& body = fx.build_nest(builder.subject().root(), {"i", "j"}, {n, m});
    auto linearized = symbolic::add(symbolic::mul(symbolic::symbol("i"), m), symbolic::symbol("j"));
    fx.copy_scalar(body, "B", "A", {linearized}, ptr);

    dump_sdfg(builder.subject(), "0.init");
    EXPECT_FALSE(run(builder));
    EXPECT_TRUE(collect_memsets(builder.subject().root()).empty());
}

// A strided write (A[2*i]) skips half of the elements: not a full fill.
TEST(ZeroFillToMemsetPassTest, PartialAccess_1DPointerStridedSubset) {
    builder::StructuredSDFGBuilder builder("sdfg_partial_1d_ptr", FunctionType_CPU);
    NestFixture fx(builder);

    auto n = fx.size("N");
    types::Pointer ptr(kElement);
    builder.add_container("A", ptr, true);

    auto& body = fx.build_nest(builder.subject().root(), {"i"}, {n});
    fx.write_scalar(body, "A", {symbolic::mul(symbolic::integer(2), symbolic::symbol("i"))}, ptr, "0");

    dump_sdfg(builder.subject(), "0.init");
    EXPECT_FALSE(run(builder));
    EXPECT_TRUE(collect_memsets(builder.subject().root()).empty());
}

// The stored constant is non-zero.
TEST(ZeroFillToMemsetPassTest, NonZeroConstant) {
    builder::StructuredSDFGBuilder builder("sdfg_nonzero_const", FunctionType_CPU);
    NestFixture fx(builder);

    auto n = fx.size("N");
    types::Array array(kElement, n);
    builder.add_container("A", array);

    auto& body = fx.build_nest(builder.subject().root(), {"i"}, {n});
    fx.write_scalar(body, "A", {symbolic::symbol("i")}, array, "1.0");

    dump_sdfg(builder.subject(), "0.init");
    EXPECT_FALSE(run(builder));
    EXPECT_TRUE(collect_memsets(builder.subject().root()).empty());
}

// The stored value is a variable load, not a constant.
TEST(ZeroFillToMemsetPassTest, VariableInput) {
    builder::StructuredSDFGBuilder builder("sdfg_variable_input", FunctionType_CPU);
    NestFixture fx(builder);

    auto n = fx.size("N");
    types::Pointer ptr(kElement);
    builder.add_container("A", ptr, true);
    builder.add_container("B", ptr, true);

    auto& body = fx.build_nest(builder.subject().root(), {"i"}, {n});
    fx.copy_scalar(body, "B", "A", {symbolic::symbol("i")}, ptr);

    dump_sdfg(builder.subject(), "0.init");
    EXPECT_FALSE(run(builder));
    EXPECT_TRUE(collect_memsets(builder.subject().root()).empty());
}
