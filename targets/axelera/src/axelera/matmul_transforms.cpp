#include "docc/target/axelera/matmul_transforms.h"

#include <cstdlib>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>

#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/library_nodes/math/cmath/cmath_node.h>
#include <sdfg/data_flow/library_nodes/math/tensor/copy_node.h>
#include <sdfg/data_flow/tasklet.h>
#include <sdfg/einsum/einsum_detection.h>
#include <sdfg/einsum/einsum_replacer.h>
#include <sdfg/einsum/replacers/einsum2matmul.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/structured_control_flow/reduce.h>
#include <sdfg/types/array.h>
#include <sdfg/types/pointer.h>
#include <sdfg/types/scalar.h>
#include <sdfg/visitor/for_each.h>

#include "docc/target/axelera/matmul_node.h"
#include "docc/target/axelera/schedule.h"
#include "docc/target/axelera/tile_target.h"

namespace sdfg {
namespace axelera {

namespace {

/// Largest reduction the MVM accumulates in one pass on Europa (IMC rows).
constexpr int64_t MAX_MVM_REDUCTION = 4 * 9 * 8 * PWORD_LANES;

std::optional<int64_t> constant(const symbolic::Expression& expr) {
    if (expr.is_null() || !SymEngine::is_a<SymEngine::Integer>(*expr)) {
        return std::nullopt;
    }
    return SymEngine::down_cast<const SymEngine::Integer&>(*expr).as_int();
}

bool counts_from_zero(const structured_control_flow::StructuredLoop& loop) {
    return symbolic::eq(loop.init(), symbolic::zero()) && symbolic::eq(loop.stride(), symbolic::one());
}

bool is_zero_constant(const data_flow::DataFlowNode& node) {
    auto constant = dynamic_cast<const data_flow::ConstantNode*>(&node);
    if (!constant) {
        return false;
    }
    char* end = nullptr;
    double value = std::strtod(constant->data().c_str(), &end);
    return end != constant->data().c_str() && value == 0.0;
}

/// A dense row-major @p rows x @p columns matrix starting at element 0.
bool is_dense(const math::tensor::TensorLayout& layout, int64_t rows, int64_t columns) {
    return layout.dims() == 2 && constant(layout.get_dim(0)) == rows && constant(layout.get_dim(1)) == columns &&
           constant(layout.get_stride(0)) == columns && constant(layout.get_stride(1)) == 1 &&
           symbolic::eq(layout.offset(), symbolic::zero());
}

/// A @p rows x @p columns matrix stored transposed, as a dense row-major @p columns x @p rows matrix.
bool is_dense_transposed(const math::tensor::TensorLayout& layout, int64_t rows, int64_t columns) {
    return layout.dims() == 2 && constant(layout.get_dim(0)) == rows && constant(layout.get_dim(1)) == columns &&
           constant(layout.get_stride(0)) == 1 && constant(layout.get_stride(1)) == rows &&
           symbolic::eq(layout.offset(), symbolic::zero());
}

/// @p node as a loop counting from zero @p count times around a single child, if it is one.
structured_control_flow::StructuredLoop* zero_based_loop(structured_control_flow::ControlFlowNode& node, int64_t count) {
    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node);
    return loop && counts_from_zero(*loop) && constant(loop->num_iterations()) == count && loop->root().size() == 1
               ? loop
               : nullptr;
}

/// True if @p node is the loop nest `for i < rows: for j < columns: C[i * columns + j] = 0`.
bool zero_fills(
    structured_control_flow::ControlFlowNode& node, const std::string& container, int64_t rows, int64_t columns
) {
    auto outer = zero_based_loop(node, rows);
    auto inner = outer ? zero_based_loop(outer->root().at(0), columns) : nullptr;
    auto block = inner ? dynamic_cast<structured_control_flow::Block*>(&inner->root().at(0)) : nullptr;
    if (!block) {
        return false;
    }
    auto& dataflow = block->dataflow();
    bool zero = false;
    size_t assignments = 0;
    for (auto& node : dataflow.nodes()) {
        zero |= is_zero_constant(node);
        if (auto tasklet = dynamic_cast<const data_flow::Tasklet*>(&node)) {
            if (tasklet->code() != data_flow::TaskletCode::assign) {
                return false;
            }
            ++assignments;
        }
    }
    const data_flow::Memlet* store = nullptr;
    for (auto* access : dataflow.data_nodes()) {
        for (auto& memlet : dataflow.in_edges(*access)) {
            if (store) {
                return false;
            }
            store = &memlet;
        }
    }
    if (!zero || assignments != 1 || !store ||
        static_cast<const data_flow::AccessNode&>(store->dst()).data() != container || store->subset().size() != 1) {
        return false;
    }
    auto expected = symbolic::add(symbolic::mul(outer->indvar(), symbolic::integer(columns)), inner->indvar());
    return symbolic::eq(symbolic::expand(symbolic::sub(store->subset().at(0), expected)), symbolic::zero());
}

/// All of @p k if the IMC holds it, otherwise the most whole PWORDs dividing @p k that it does.
int64_t largest_panel(int64_t k) {
    if (k <= MAX_MVM_REDUCTION) {
        return k;
    }
    for (auto panel = MAX_MVM_REDUCTION; panel > 0; panel -= static_cast<int64_t>(PWORD_LANES)) {
        if (k % panel == 0) {
            return panel;
        }
    }
    return k;
}

/// A float matmul as einsum detection finds it in a loop nest, with the nest zero-initializing its result.
struct MvmMatch {
    std::unique_ptr<einsum::EinsumDetection> detection;
    const einsum::EinsumCluster* cluster = nullptr;
    einsum::Einsum2MatMul::MatMulAnalysis analysis;
    int64_t m = 0, n = 0, k = 0, k_panel = 0, cores = 1;
    bool b_transposed = false;
    structured_control_flow::ControlFlowNode* zero_init = nullptr;
};

std::optional<MvmMatch> match_mvm(structured_control_flow::StructuredLoop& loop, int64_t k_panel, int64_t cores) {
    auto parent = dynamic_cast<structured_control_flow::Sequence*>(loop.get_parent());
    if (!parent || parent->index(loop) < 1 || cores < 1) {
        return std::nullopt;
    }
    MvmMatch match;
    match.detection = std::make_unique<einsum::EinsumDetection>();
    match.detection->run(loop);
    for (auto& cluster : match.detection->einsums()) {
        if (cluster->consumed_loops.empty() || cluster->consumed_loops.front() != &loop || !cluster->output_node) {
            continue;
        }
        einsum::Einsum2MatMul::MatMulAnalysis analysis;
        if (!einsum::Einsum2MatMul().analyze(*cluster, analysis) ||
            analysis.input_type != types::PrimitiveType::Float || analysis.output_type != types::PrimitiveType::Float) {
            continue;
        }
        auto m = constant(analysis.m);
        auto n = constant(analysis.n);
        auto k = constant(analysis.k);
        if (!m || !n || !k) {
            continue;
        }
        auto panel = k_panel == 0 ? largest_panel(*k) : k_panel;
        bool b_transposed = is_dense_transposed(*analysis.layout_b, *k, *n);
        // The MVM works on whole PWORDs of the reduction and the result rows.
        if (*n % PWORD_LANES != 0 || panel % PWORD_LANES != 0 || panel > MAX_MVM_REDUCTION || *k % panel != 0 ||
            *m % cores != 0 || !is_dense(*analysis.layout_a, *m, *k) ||
            !(b_transposed || is_dense(*analysis.layout_b, *k, *n)) || !is_dense(*analysis.layout_y, *m, *n)) {
            continue;
        }
        auto& before = parent->at(parent->index(loop) - 1);
        if (!zero_fills(before, cluster->output_node->data(), *m, *n)) {
            continue;
        }
        match.cluster = cluster.get();
        match.analysis = analysis;
        match.m = *m;
        match.n = *n;
        match.k = *k;
        match.k_panel = panel;
        match.cores = cores;
        match.b_transposed = b_transposed;
        match.zero_init = &before;
        return std::optional<MvmMatch>(std::move(match));
    }
    return std::nullopt;
}

std::string literal(double value) {
    std::ostringstream text;
    text << std::setprecision(17) << value;
    return text.str();
}

/**
 * Builds the quantized MVM computation of a match into a sequence: per operand an absmax reduction and a quantizing
 * map, the MVM over the rows of A in one AI core, and the map scaling its products into C.
 */
class QuantizedMvmBuilder {
    builder::StructuredSDFGBuilder& builder_;
    passes::LibNodeExpander::AccessNodeExpand& access_;
    structured_control_flow::Sequence& sequence_;
    const MvmMatch& match_;
    DebugInfo debug_info_;
    types::Scalar f32_{types::PrimitiveType::Float};
    types::Scalar i8_{types::PrimitiveType::Int8};
    types::Scalar i64_{types::PrimitiveType::Int64};

    std::string container(const std::string& prefix, const types::IType& type) {
        auto name = builder_.find_new_name(prefix);
        builder_.add_container(name, type);
        return name;
    }

    symbolic::Symbol loop_variable(const std::string& prefix) {
        return symbolic::symbol(container(prefix, i64_));
    }

    /// Cluster operand @p index; the output C comes after the inputs.
    std::string operand_name(size_t index) const {
        auto& cluster = *match_.cluster;
        if (index == cluster.inputs.size()) {
            return cluster.output_node->data();
        }
        return static_cast<const data_flow::AccessNode*>(cluster.input_nodes.at(cluster.inputs.at(index)))->data();
    }

    const types::IType& type_of(const std::string& name) const {
        return builder_.subject().type(name);
    }

    data_flow::AccessNode& scalar(structured_control_flow::Block& block, const std::string& name) {
        return builder_.add_access(block, name, debug_info_);
    }

    data_flow::AccessNode& literal_node(structured_control_flow::Block& block, double value) {
        return builder_.add_constant(block, literal(value), f32_, debug_info_);
    }

    void read(
        structured_control_flow::Block& block,
        data_flow::AccessNode& source,
        data_flow::CodeNode& node,
        const std::string& connector,
        const data_flow::Subset& subset = {}
    ) {
        auto& type = subset.empty() ? static_cast<const types::IType&>(f32_) : type_of(source.data());
        builder_.add_computational_memlet(block, source, node, connector, subset, type, debug_info_);
    }

    void write(
        structured_control_flow::Block& block,
        data_flow::CodeNode& node,
        data_flow::AccessNode& target,
        const data_flow::Subset& subset = {}
    ) {
        auto& type = subset.empty() ? static_cast<const types::IType&>(f32_) : type_of(target.data());
        builder_.add_computational_memlet(block, node, "_out", target, subset, type, debug_info_);
    }

    data_flow::Tasklet& tasklet(structured_control_flow::Block& block, data_flow::TaskletCode code, size_t inputs) {
        std::vector<std::string> connectors;
        for (size_t k = 0; k < inputs; ++k) {
            connectors.push_back(inputs == 1 ? "_in" : "_in" + std::to_string(k + 1));
        }
        return builder_.add_tasklet(block, code, "_out", connectors, debug_info_);
    }

    math::cmath::CMathNode& cmath(structured_control_flow::Block& block, math::cmath::CMathFunction function) {
        return static_cast<
            math::cmath::CMathNode&>(builder_.add_library_node<
                                     math::cmath::CMathNode>(block, debug_info_, function, types::PrimitiveType::Float));
    }

    void assign(structured_control_flow::Block& block, data_flow::AccessNode& source, const std::string& target) {
        auto& copy = tasklet(block, data_flow::TaskletCode::assign, 1);
        read(block, source, copy, "_in");
        write(block, copy, scalar(block, target));
    }

    /// `result = max |X[e]|` over the @p count elements of operand @p index, as a reduction.
    std::string absmax(size_t index, int64_t count) {
        auto source = operand_name(index);
        auto accumulator = container(source + "_absmax_acc", f32_);
        auto magnitude = container(source + "_abs", f32_);
        auto result = container(source + "_absmax", f32_);
        auto e = loop_variable(source + "_e");

        auto& init = builder_.add_block(sequence_, {}, debug_info_);
        assign(init, builder_.add_constant(init, "-inf", f32_, debug_info_), accumulator);

        auto& loop = builder_.add_reduce(
            sequence_,
            e,
            symbolic::Lt(e, symbolic::integer(count)),
            symbolic::zero(),
            symbolic::add(e, symbolic::one()),
            {{structured_control_flow::ReductionOperation::Max, accumulator}},
            structured_control_flow::ScheduleType_Sequential::create(),
            debug_info_
        );
        auto& body = builder_.add_block(loop.root(), {}, debug_info_);
        auto& abs = cmath(body, math::cmath::CMathFunction::fabs);
        read(body, access_.add_scalar_input_access(body, index), abs, "_in1", {e});
        auto& magnitude_access = scalar(body, magnitude);
        write(body, abs, magnitude_access);
        auto& max = cmath(body, math::cmath::CMathFunction::fmax);
        read(body, magnitude_access, max, "_in1");
        read(body, scalar(body, accumulator), max, "_in2");
        write(body, max, scalar(body, accumulator));

        auto& store = builder_.add_block(sequence_, {}, debug_info_);
        assign(store, scalar(store, accumulator), result);
        return result;
    }

    /// `Q[e] = rint(X[e] / absmax * 127)` as int8 over the @p count elements of operand @p index.
    std::string quantize(size_t index, const std::string& absmax, int64_t count) {
        auto source = operand_name(index);
        auto quantized = container(source + "_int8", types::Pointer(i8_));
        auto divisor = container(source + "_divisor", f32_);
        auto ratio = container(source + "_ratio", f32_);
        auto scaled = container(source + "_scaled", f32_);
        auto rounded = container(source + "_rounded", f32_);
        auto e = loop_variable(source + "_q");

        auto& map = builder_.add_map(
            sequence_,
            e,
            symbolic::Lt(e, symbolic::integer(count)),
            symbolic::zero(),
            symbolic::add(e, symbolic::one()),
            structured_control_flow::ScheduleType_Sequential::create(),
            debug_info_
        );
        auto& body = builder_.add_block(map.root(), {}, debug_info_);
        // An all-zero tensor quantizes to zeros instead of dividing by zero.
        auto& guard = cmath(body, math::cmath::CMathFunction::fmax);
        read(body, scalar(body, absmax), guard, "_in1");
        read(body, literal_node(body, 1e-30), guard, "_in2");
        auto& divisor_access = scalar(body, divisor);
        write(body, guard, divisor_access);

        auto& divide = tasklet(body, data_flow::TaskletCode::fp_div, 2);
        read(body, access_.add_scalar_input_access(body, index), divide, "_in1", {e});
        read(body, divisor_access, divide, "_in2");
        auto& ratio_access = scalar(body, ratio);
        write(body, divide, ratio_access);

        auto& scale = tasklet(body, data_flow::TaskletCode::fp_mul, 2);
        read(body, ratio_access, scale, "_in1");
        read(body, literal_node(body, 127.0), scale, "_in2");
        auto& scaled_access = scalar(body, scaled);
        write(body, scale, scaled_access);

        auto& round = cmath(body, math::cmath::CMathFunction::rint);
        read(body, scaled_access, round, "_in1");
        auto& rounded_access = scalar(body, rounded);
        write(body, round, rounded_access);

        auto& narrow = tasklet(body, data_flow::TaskletCode::assign, 1);
        read(body, rounded_access, narrow, "_in");
        write(body, narrow, scalar(body, quantized), {e});
        return quantized;
    }

    /// Copies the int8 @p rows x @p columns matrix that @p source stores transposed into row-major order.
    std::string transpose(const std::string& source, int64_t rows, int64_t columns) {
        auto target = container(source + "_t", types::Pointer(i8_));
        auto r = symbolic::integer(rows);
        auto c = symbolic::integer(columns);
        math::tensor::TensorLayout layout_x({r, c}, {symbolic::one(), r}, symbolic::zero());
        math::tensor::TensorLayout layout_y({r, c}, {c, symbolic::one()}, symbolic::zero());
        auto& block = builder_.add_block(sequence_, {}, debug_info_);
        auto& copy = builder_.add_library_node<math::tensor::TensorCopyNode>(block, debug_info_, layout_x, layout_y);
        types::Pointer i8_pointer(i8_);
        builder_.add_computational_memlet(block, scalar(block, source), copy, "X", {}, i8_pointer, debug_info_);
        builder_.add_computational_memlet(block, scalar(block, target), copy, "Y", {}, i8_pointer, debug_info_);
        return target;
    }

    /// `T[i, :] = A[i, :] @ B` on the MVM, each AI core multiplying a block of rows, in panels of the reduction.
    structured_control_flow::StructuredLoop&
    multiply(const std::string& a, const std::string& b, const std::string& products) {
        auto core = loop_variable("mvm_core");
        auto& cores = builder_.add_map(
            sequence_,
            core,
            symbolic::Lt(core, symbolic::integer(match_.cores)),
            symbolic::zero(),
            symbolic::add(core, symbolic::one()),
            ScheduleType_Axelera::create(AxeleraLevel::Core, match_.cores),
            debug_info_
        );
        auto* scope = &cores.root();
        symbolic::Expression first = symbolic::zero();
        symbolic::Condition accumulate = symbolic::__false__();
        auto panels = match_.k / match_.k_panel;
        if (panels > 1) {
            auto panel = loop_variable("mvm_panel");
            auto& loop = builder_.add_for(
                *scope,
                panel,
                symbolic::Lt(panel, symbolic::integer(panels)),
                symbolic::zero(),
                symbolic::add(panel, symbolic::one()),
                debug_info_
            );
            scope = &loop.root();
            first = symbolic::mul(panel, symbolic::integer(match_.k_panel));
            accumulate = symbolic::Lt(symbolic::zero(), panel);
        }
        auto rows_per_core = match_.m / match_.cores;
        auto row = loop_variable("mvm_row");
        auto& rows = builder_.add_for(
            *scope,
            row,
            symbolic::Lt(row, symbolic::integer(rows_per_core)),
            symbolic::zero(),
            symbolic::add(row, symbolic::one()),
            debug_info_
        );
        auto& block = builder_.add_block(rows.root(), {}, debug_info_);
        auto one = symbolic::one();
        auto n = symbolic::integer(match_.n);
        auto k = symbolic::integer(match_.k);
        auto k_panel = symbolic::integer(match_.k_panel);
        auto i = symbolic::add(symbolic::mul(core, symbolic::integer(rows_per_core)), row);
        math::tensor::TensorLayout layout_a({one, k_panel}, {k, one}, symbolic::add(symbolic::mul(i, k), first));
        math::tensor::TensorLayout layout_b({k_panel, n}, {n, one}, symbolic::mul(first, n));
        math::tensor::TensorLayout layout_y({one, n}, {n, one}, symbolic::mul(i, n));
        auto& node =
            builder_.add_library_node<AxeleraMatMulNode>(block, debug_info_, layout_a, layout_b, layout_y, accumulate);
        types::Pointer i8_pointer(i8_);
        types::Pointer f32_pointer(f32_);
        builder_.add_computational_memlet(block, scalar(block, products), node, "Y", {}, f32_pointer, debug_info_);
        builder_.add_computational_memlet(block, scalar(block, a), node, "A", {}, i8_pointer, debug_info_);
        builder_.add_computational_memlet(block, scalar(block, b), node, "B", {}, i8_pointer, debug_info_);
        return rows;
    }

    /// `C[e] = T[e] * absmax_a * absmax_b / 127^2` over all elements of C.
    void dequantize(const std::string& products, const std::string& absmax_a, const std::string& absmax_b, size_t c) {
        auto target = operand_name(c);
        auto both = container(target + "_scale_ab", f32_);
        auto scale = container(target + "_scale", f32_);
        auto e = loop_variable(target + "_e");
        auto& map = builder_.add_map(
            sequence_,
            e,
            symbolic::Lt(e, symbolic::integer(match_.m * match_.n)),
            symbolic::zero(),
            symbolic::add(e, symbolic::one()),
            structured_control_flow::ScheduleType_Sequential::create(),
            debug_info_
        );
        auto& body = builder_.add_block(map.root(), {}, debug_info_);
        auto& product = tasklet(body, data_flow::TaskletCode::fp_mul, 2);
        read(body, scalar(body, absmax_a), product, "_in1");
        read(body, scalar(body, absmax_b), product, "_in2");
        auto& both_access = scalar(body, both);
        write(body, product, both_access);

        auto& normalize = tasklet(body, data_flow::TaskletCode::fp_mul, 2);
        read(body, both_access, normalize, "_in1");
        read(body, literal_node(body, 1.0 / (127.0 * 127.0)), normalize, "_in2");
        auto& scale_access = scalar(body, scale);
        write(body, normalize, scale_access);

        auto& apply = tasklet(body, data_flow::TaskletCode::fp_mul, 2);
        read(body, scalar(body, products), apply, "_in1", {e});
        read(body, scale_access, apply, "_in2");
        write(body, apply, access_.add_scalar_input_access(body, c), {e});
    }

public:
    QuantizedMvmBuilder(
        builder::StructuredSDFGBuilder& builder,
        passes::LibNodeExpander::AccessNodeExpand& access,
        structured_control_flow::Sequence& sequence,
        const MvmMatch& match,
        const DebugInfo& debug_info
    )
        : builder_(builder), access_(access), sequence_(sequence), match_(match), debug_info_(debug_info) {
    }

    structured_control_flow::StructuredLoop& build() {
        auto a = static_cast<size_t>(match_.analysis.a_idx);
        auto b = static_cast<size_t>(match_.analysis.b_idx);
        auto c = match_.cluster->inputs.size();
        auto absmax_a = absmax(a, match_.m * match_.k);
        auto a_int8 = quantize(a, absmax_a, match_.m * match_.k);
        auto absmax_b = absmax(b, match_.k * match_.n);
        auto b_int8 = quantize(b, absmax_b, match_.k * match_.n);
        if (match_.b_transposed) {
            b_int8 = transpose(b_int8, match_.k, match_.n);
        }
        auto products = container(operand_name(c) + "_products", types::Pointer(f32_));
        auto& rows = multiply(a_int8, b_int8, products);
        dequantize(products, absmax_a, absmax_b, c);
        return rows;
    }
};

class QuantizedMvmReplacer : public einsum::EinsumReplacer {
    const MvmMatch& match_;
    mutable structured_control_flow::Sequence* sequence_ = nullptr;
    mutable structured_control_flow::StructuredLoop* rows_ = nullptr;

public:
    explicit QuantizedMvmReplacer(const MvmMatch& match) : match_(match) {
    }

    std::string name() const override {
        return "AxeleraQuantizedMvm";
    }

    einsum::ReplaceOutcome
    replace(einsum::EinsumReplacementContext& context, const einsum::EinsumCluster& cluster) const override {
        std::vector<passes::LibNodeExpander::InputUse>
            uses(cluster.inputs.size() + 1, passes::LibNodeExpander::InputUse::Scalar);
        auto standalone = context.replacement_requires_access_nodes(uses, true);
        if (!standalone) {
            return context.unable();
        }
        sequence_ = &standalone->replace_with_sequence();
        auto debug_info = cluster.consumed_loops.front()->debug_info();
        rows_ = &QuantizedMvmBuilder(standalone->builder(), *standalone, *sequence_, match_, debug_info).build();
        return standalone->successfully_expanded();
    }

    structured_control_flow::Sequence* sequence() const {
        return sequence_;
    }

    structured_control_flow::StructuredLoop* rows() const {
        return rows_;
    }
};

} // namespace

bool MvmEinsumTransform::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    return match_mvm(loop_, k_panel_, cores_).has_value();
}

void MvmEinsumTransform::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto match = match_mvm(loop_, k_panel_, cores_);
    if (!match) {
        throw transformations::InvalidTransformationException(
            "AxeleraMvmEinsumTransform: no dense float matmul into a zero-initialized result"
        );
    }
    auto& parent = static_cast<structured_control_flow::Sequence&>(*loop_.get_parent());
    builder.remove_child(parent, parent.index(*match->zero_init));

    QuantizedMvmReplacer replacer(*match);
    if (!einsum::replace_einsum_cluster(builder, *match->cluster, replacer).applied || !replacer.sequence()) {
        throw transformations::InvalidTransformationException("AxeleraMvmEinsumTransform: replacing the matmul failed");
    }
    // Datapaths and AI-core programs run from the top level, so the replacement joins its parent sequence.
    auto& sequence = *replacer.sequence();
    builder.move_children(sequence, parent, parent.index(sequence));
    builder.remove_child(parent, parent.index(sequence));
    rows_ = replacer.rows();
    analysis_manager.invalidate_all();
}

void MvmEinsumTransform::to_json(nlohmann::json& j) const {
    j["transformation_type"] = name();
    j["loop_element_id"] = loop_element_id_;
    j["k_panel"] = k_panel_;
    j["cores"] = cores_;
}

namespace {

/// The single AxeleraMatMulNode under @p loop, with its block.
std::pair<AxeleraMatMulNode*, structured_control_flow::Block*> single_matmul(structured_control_flow::StructuredLoop& loop) {
    std::pair<AxeleraMatMulNode*, structured_control_flow::Block*> found{nullptr, nullptr};
    int count = 0;
    visitor::for_each_block(loop.root(), [&](structured_control_flow::Block& block) {
        for (auto* node : block.dataflow().library_nodes()) {
            if (auto* matmul = dynamic_cast<AxeleraMatMulNode*>(node)) {
                found = {matmul, &block};
                ++count;
            }
        }
    });
    return count == 1 ? found : std::pair<AxeleraMatMulNode*, structured_control_flow::Block*>{nullptr, nullptr};
}

/// Indvars of @p loop and every loop nested in it.
symbolic::SymbolSet loop_variables(structured_control_flow::StructuredLoop& loop) {
    symbolic::SymbolSet variables{loop.indvar()};
    std::function<void(structured_control_flow::ControlFlowNode&)> visit =
        [&](structured_control_flow::ControlFlowNode& node) {
            if (auto* inner = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
                variables.insert(inner->indvar());
                visit(inner->root());
            } else if (auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
                for (size_t i = 0; i < sequence->size(); ++i) {
                    visit(sequence->at(i));
                }
            }
        };
    visit(loop.root());
    return variables;
}

data_flow::Memlet*
operand_edge(structured_control_flow::Block& block, data_flow::LibraryNode& node, const std::string& connector) {
    for (auto& memlet : block.dataflow().in_edges(node)) {
        if (memlet.dst_conn() == connector) {
            return &memlet;
        }
    }
    return nullptr;
}

} // namespace

bool ImcResidency::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto [matmul, block] = single_matmul(loop_);
    if (!matmul || !dynamic_cast<structured_control_flow::Sequence*>(loop_.get_parent())) {
        return false;
    }
    auto& layout = matmul->layout_b();
    if (!SymEngine::is_a<SymEngine::Integer>(*layout.get_dim(0)) ||
        !SymEngine::is_a<SymEngine::Integer>(*layout.get_dim(1))) {
        return false;
    }
    symbolic::SymbolSet used;
    layout.collect_symbols(used);
    for (auto& variable : loop_variables(loop_)) {
        if (used.count(variable)) {
            return false;
        }
    }
    auto* edge = operand_edge(*block, *matmul, "B");
    if (!edge) {
        return false;
    }
    // The weights must not change while they are resident.
    auto& source = static_cast<const data_flow::AccessNode&>(edge->src()).data();
    bool written = false;
    visitor::for_each_block(loop_.root(), [&](structured_control_flow::Block& b) {
        for (auto* access : b.dataflow().data_nodes()) {
            written |= access->data() == source && b.dataflow().in_degree(*access) > 0;
        }
    });
    return !written;
}

void ImcResidency::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (!can_be_applied(builder, analysis_manager)) {
        throw transformations::InvalidTransformationException("AxeleraImcResidency: no loop-invariant matmul weights");
    }
    auto [matmul, block] = single_matmul(loop_);
    auto* edge = operand_edge(*block, *matmul, "B");
    auto source = static_cast<const data_flow::AccessNode&>(edge->src()).data();
    auto layout = matmul->layout_b();
    auto rows = layout.get_dim(0);
    auto columns = layout.get_dim(1);

    types::Scalar int8(types::PrimitiveType::Int8);
    types::Pointer int8_pointer(int8);
    imc_container_ = builder.find_new_name("__axelera_imc_" + source);
    builder.add_container(imc_container_, types::Array(StorageType_IMC(), 0, "", types::Array(int8, columns), rows));

    auto& parent = static_cast<structured_control_flow::Sequence&>(*loop_.get_parent());
    auto debug_info = loop_.debug_info();
    auto& load_block = builder.add_block_before(parent, loop_, debug_info);
    auto& load = builder.add_library_node<AxeleraImcLoadNode>(load_block, debug_info, layout);
    builder.add_computational_memlet(
        load_block, builder.add_access(load_block, imc_container_), load, "_imc", {}, int8_pointer, debug_info
    );
    builder.add_computational_memlet(
        load_block, builder.add_access(load_block, source), load, "_src", {}, int8_pointer, debug_info
    );

    // The matmul now reads the dense resident copy.
    builder.remove_memlet(*block, *edge);
    builder.add_computational_memlet(
        *block, builder.add_access(*block, imc_container_), *matmul, "B", {}, int8_pointer, debug_info
    );
    matmul->relocalize_operand(
        AxeleraMatMulNode::B_INPUT_IDX, math::tensor::TensorLayout({rows, columns}, {columns, symbolic::one()})
    );
    analysis_manager.invalidate_all();
}

void ImcResidency::to_json(nlohmann::json& j) const {
    j["transformation_type"] = name();
    j["loop_element_id"] = loop_.element_id();
}

} // namespace axelera
} // namespace sdfg
