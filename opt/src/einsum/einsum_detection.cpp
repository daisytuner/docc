#include "sdfg/einsum/einsum_detection.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/einsum/einsum_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/conjunctive_normal_form.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/type.h"
#include "symengine/logic.h"
#include "symengine/symengine_rcp.h"

namespace sdfg::einsum {

namespace {

bool is_constant(const data_flow::DataFlowNode* node) {
    return dynamic_cast<const data_flow::ConstantNode*>(node) != nullptr;
}

// Determine how `indvar` contributes to a single operand's access subset. It contributes
// iff it appears in exactly one index position as an integer-scaled linear term.
struct Contribution {
    enum class Kind { Absent, Invalid, Valid };
    Kind kind = Kind::Absent;
    size_t index = 0;
    int64_t factor = 1;
};

Contribution contribution_of(const data_flow::Subset& subset, const symbolic::Symbol& indvar) {
    for (size_t p = 0; p < subset.size(); p++) {
        if (!symbolic::uses(subset.at(p), indvar)) {
            continue;
        }
        auto decomp = symbolic::affine_decomposition(subset.at(p), indvar);
        if (!decomp.success || !SymEngine::is_a<SymEngine::Integer>(*decomp.coeff)) {
            return {Contribution::Kind::Invalid, 0, 1};
        }
        int64_t factor = SymEngine::rcp_static_cast<const SymEngine::Integer>(decomp.coeff)->as_int();
        return {Contribution::Kind::Valid, p, factor};
    }
    return {};
}

// Count how many index positions, from the innermost outward, are fully reconstructed by the
// recorded contributions to them.
size_t compute_covered(const data_flow::Subset& subset, const std::vector<EinsumIndexContribution>& contributions) {
    size_t covered = 0;
    for (size_t back = 0; back < subset.size(); back++) {
        size_t p = subset.size() - 1 - back;
        symbolic::Expression acc = symbolic::zero();
        bool any = false;
        for (const auto& c : contributions) {
            if (c.index == p) {
                any = true;
                acc = symbolic::add(acc, symbolic::mul(symbolic::integer(c.factor), c.indvar));
            }
        }
        if (any && symbolic::eq(acc, subset.at(p))) {
            covered++;
        } else {
            break;
        }
    }
    return covered;
}

// Add a contribution to an operand's indexing, keeping contributions ordered by subset index
// and refreshing how many innermost positions are fully covered.
void add_contribution(
    EinsumIndexing& indexing, const data_flow::Subset& subset, const EinsumIndexContribution& contribution
) {
    indexing.contributions.push_back(contribution);
    std::sort(
        indexing.contributions.begin(),
        indexing.contributions.end(),
        [](const EinsumIndexContribution& a, const EinsumIndexContribution& b) { return a.index < b.index; }
    );
    indexing.covered = compute_covered(subset, indexing.contributions);
}

// Build the inner-index mapping of a single operand's subset from the cluster's dimensions.
EinsumIndexing compute_indexing(const data_flow::Subset& subset, const std::vector<EinsumDimension>& dims) {
    EinsumIndexing indexing;
    for (const auto& dim : dims) {
        auto contrib = contribution_of(subset, dim.indvar);
        if (contrib.kind == Contribution::Kind::Valid) {
            indexing.contributions.push_back({dim.indvar, contrib.index, contrib.factor});
        }
    }
    std::sort(
        indexing.contributions.begin(),
        indexing.contributions.end(),
        [](const EinsumIndexContribution& a, const EinsumIndexContribution& b) { return a.index < b.index; }
    );
    indexing.covered = compute_covered(subset, indexing.contributions);
    return indexing;
}

} // namespace

symbolic::Expression EinsumIndexContribution::factor_expr() const { return symbolic::integer(this->factor); }

symbolic::Expression EinsumIndexing::get_stride_including_subsets(int idx, const std::vector<symbolic::Expression>& shape)
    const {
    auto& contrib = this->contributions.at(idx);
    auto dims = shape.size();
    symbolic::Expression stride = contrib.factor_expr();
    for (auto i = contrib.index; i < dims - 1; ++i) {
        stride = symbolic::mul(stride, shape.at(i));
    }
    return stride;
}

std::pair<data_flow::AccessNode*, data_flow::Memlet*> EinsumCluster::get_input_for(size_t input_idx) {
    auto* input_src = dyn_cast<data_flow::AccessNode*>(input_nodes.at(inputs.at(input_idx)));
    return {input_src, in_edges.at(input_idx)};
}

const data_flow::Subset& EinsumCluster::get_input_subset(size_t input_idx) const {
    return in_edges.at(input_idx)->subset();
}

const EinsumIndexing& EinsumCluster::get_output_indexing() const { return this->einsum_out_indices; }

// Infer an element-wise TensorLayout for a type, if one is well-defined. Supports actual tensors
// and nested c-style arrays of scalars; returns nullopt otherwise.
std::optional<math::tensor::TensorLayout> try_infer_tensor_layout(const types::IType& type) {
    if (type.type_id() == types::TypeID::Tensor) {
        return static_cast<const types::Tensor&>(type).layout();
    }
    if (type.type_id() == types::TypeID::Array) {
        // Collect the shape of the nested c-style arrays (outermost first).
        symbolic::MultiExpression shape;
        const types::IType* current = &type;
        while (current->type_id() == types::TypeID::Array) {
            const auto& array = static_cast<const types::Array&>(*current);
            shape.push_back(array.num_elements());
            current = &array.element_type();
        }

        // Only scalar element types have a well-defined linear element offset.
        if (current->primitive_type() == types::PrimitiveType::Void) {
            return std::nullopt;
        }
        return math::tensor::TensorLayout(shape);
    }
    return std::nullopt;
}

symbolic::Expression EinsumCluster::get_linearized_outer_offset(int input_idx) const {
    auto& indexing = input_idx >= 0 ? einsum_in_indices.at(input_idx) : einsum_out_indices;
    auto& edge = input_idx >= 0 ? *in_edges.at(input_idx) : *output_edge;
    auto rem_dims = edge.subset().size() - indexing.covered;
    data_flow::Subset rem_subset(edge.subset().begin(), edge.subset().begin() + rem_dims);

    auto layout = try_infer_tensor_layout(edge.base_type());
    if (!layout) {
        throw std::invalid_argument("EinsumCluster::get_linearized_outer_offset: invalid layout");
    }
    return layout->resolve_element(rem_subset, false);
}

const data_flow::Subset& EinsumCluster::out_indices() const { return output_edge->subset(); }


const std::vector<EinsumIndexing>& EinsumCluster::get_input_indexings() const { return this->einsum_in_indices; }

std::string EinsumCluster::toStr() const {
    std::stringstream stream;

    // The innermost `covered` positions are "inside" the einsum ([...]); the outer positions
    // are irrelevant to the operation and shown wrapped in {...}.
    auto print_subset = [&stream](const data_flow::Subset& subset, size_t covered) {
        size_t inner_start = subset.size() - std::min(covered, subset.size());
        for (size_t p = 0; p < subset.size(); p++) {
            bool inside = p >= inner_start;
            stream << (inside ? "[" : "{") << subset.at(p)->__str__() << (inside ? "]" : "}");
        }
    };

    print_subset(this->out_indices(), this->einsum_out_indices.covered);
    stream << " = ";
    size_t num_inputs = this->inputs.size();
    if (num_inputs > 1) {
        for (size_t i = 0; i < num_inputs - 1; i++) {
            if (i > 0) {
                stream << " * ";
            }
            stream << this->inputs.at(i);
            if (auto* edge = this->in_edges.at(i)) {
                print_subset(
                    edge->subset(), i < this->einsum_in_indices.size() ? this->einsum_in_indices.at(i).covered : 0
                );
            }
        }
        stream << " + ";
    }
    stream << this->inputs.at(num_inputs - 1);
    if (auto* edge = this->in_edges.at(num_inputs - 1)) {
        size_t last = num_inputs - 1;
        print_subset(edge->subset(), last < this->einsum_in_indices.size() ? this->einsum_in_indices.at(last).covered : 0);
    }

    for (auto& dim : this->dims) {
        stream << " for " << dim.indvar->__str__() << " = " << dim.init->__str__() << " : " << dim.bound->__str__();
    }

    return stream.str();
}

bool EinsumDetection::subsets_eq(const data_flow::Subset& subset1, const data_flow::Subset& subset2) {
    if (subset1.size() != subset2.size()) {
        return false;
    }
    for (size_t i = 0; i < subset1.size(); i++) {
        if (!symbolic::eq(subset1.at(i), subset2.at(i))) {
            return false;
        }
    }
    return true;
}

EinsumCluster& EinsumDetection::new_cluster(structured_control_flow::Block& block) {
    this->all_einsums_.push_back(std::make_unique<EinsumCluster>());
    auto& cluster = *this->all_einsums_.back();
    cluster.block = &block;
    return cluster;
}

EinsumCluster* EinsumDetection::try_lift(structured_control_flow::Block& block, data_flow::Tasklet& tasklet) {
    auto& dfg = block.dataflow();

    std::vector<std::string> inputs;
    std::vector<data_flow::Memlet*> in_edges;
    std::unordered_map<std::string, data_flow::DataFlowNode*> input_nodes;
    data_flow::AccessNode* output_node = nullptr;
    data_flow::Memlet* out_edge = nullptr;
    bool subtraction = false;

    if (tasklet.code() == data_flow::TaskletCode::fp_add) {
        auto& oedge = *dfg.out_edges(tasklet).begin();
        auto& out = static_cast<data_flow::AccessNode&>(oedge.dst());
        output_node = &out;
        out_edge = &oedge;

        bool found_reduction = false;
        for (auto& iedge : dfg.in_edges(tasklet)) {
            if (!found_reduction && !is_constant(&iedge.src())) {
                auto& in = static_cast<data_flow::AccessNode&>(iedge.src());
                if (in.data() == out.data() && this->subsets_eq(oedge.subset(), iedge.subset())) {
                    found_reduction = true;
                    continue;
                }
            }
            inputs.push_back(iedge.dst_conn());
            in_edges.push_back(&iedge);
            input_nodes.insert({iedge.dst_conn(), &iedge.src()});
        }
        if (!found_reduction) {
            return nullptr;
        }
    } else if (tasklet.code() == data_flow::TaskletCode::fp_fma) {
        auto& oedge = *dfg.out_edges(tasklet).begin();
        auto& out = static_cast<data_flow::AccessNode&>(oedge.dst());
        output_node = &out;
        out_edge = &oedge;

        const std::string& reduction_conn = tasklet.inputs().back();
        bool found_reduction = false;
        for (auto& iedge : dfg.in_edges(tasklet)) {
            if (iedge.dst_conn() == reduction_conn && this->subsets_eq(oedge.subset(), iedge.subset())) {
                if (!is_constant(&iedge.src()) &&
                    static_cast<data_flow::AccessNode&>(iedge.src()).data() == out.data()) {
                    found_reduction = true;
                    continue;
                }
            }
            inputs.push_back(iedge.dst_conn());
            in_edges.push_back(&iedge);
            input_nodes.insert({iedge.dst_conn(), &iedge.src()});
        }
        if (!found_reduction) {
            return nullptr;
        }
    } else if (tasklet.code() == data_flow::TaskletCode::fp_sub) {
        auto& oedge = *dfg.out_edges(tasklet).begin();
        auto& out = static_cast<data_flow::AccessNode&>(oedge.dst());
        output_node = &out;
        out_edge = &oedge;

        const std::string& reduction_conn = tasklet.inputs().front();
        bool found_reduction = false;
        for (auto& iedge : dfg.in_edges(tasklet)) {
            if (iedge.dst_conn() == reduction_conn && this->subsets_eq(oedge.subset(), iedge.subset())) {
                if (!is_constant(&iedge.src()) &&
                    static_cast<data_flow::AccessNode&>(iedge.src()).data() == out.data()) {
                    found_reduction = true;
                    continue;
                }
            }
            inputs.push_back(iedge.dst_conn());
            in_edges.push_back(&iedge);
            input_nodes.insert({iedge.dst_conn(), &iedge.src()});
        }
        if (!found_reduction) {
            return nullptr;
        }

        // Implicit multiplication with -1
        inputs.push_back("__einsum_const");
        in_edges.push_back(nullptr);
        input_nodes.insert({"__einsum_const", nullptr});
        subtraction = true;
    } else {
        return nullptr;
    }

    auto& cluster = this->new_cluster(block);
    cluster.in_edges = in_edges;
    cluster.inputs = inputs;
    cluster.input_nodes = input_nodes;
    cluster.output_node = output_node;
    cluster.output_edge = out_edge;
    cluster.subtraction = subtraction;
    cluster.consumed_nodes.push_back(&tasklet);
    return &cluster;
}

EinsumCluster* EinsumDetection::from_einsum_node(structured_control_flow::Block& block, EinsumNode& einsum_node) {
    // Not representable as cluster, because we use the memlets to hold the subsets. also potentially wrong
    // auto& dfg = block.dataflow();
    //
    // auto& cluster = this->new_cluster(block);
    // cluster.out_indices = einsum_node.out_indices();
    //
    // // Drop the trailing implicit reduction input ("__einsum_out")
    // const auto& node_inputs = einsum_node.inputs();
    // const auto& node_in_indices = einsum_node.in_indices();
    // for (size_t i = 0; i + 1 < node_inputs.size(); i++) {
    //     cluster.inputs.push_back(node_inputs.at(i));
    //     cluster.in_edges.push_back(node_in_indices.at(i));
    // }
    //
    // for (const auto& dim : einsum_node.dims()) {
    //     cluster.dims.push_back(dim);
    // }
    //
    // for (auto& iedge : dfg.in_edges(einsum_node)) {
    //     cluster.input_nodes.insert({iedge.dst_conn(), &iedge.src()});
    // }
    // for (auto& oedge : dfg.out_edges(einsum_node)) {
    //     cluster.output_node = &static_cast<data_flow::AccessNode&>(oedge.dst());
    //     break;
    // }
    //
    // this->recompute_einsum_indices(cluster);
    // cluster.consumed_nodes.push_back(&einsum_node);
    // return &cluster;
    return nullptr;
}

void EinsumDetection::find_einsum_core_ops(BlockState& state, structured_control_flow::Block& block) {
    auto& dfg = block.dataflow();

    // Lift reduction tasklets into fresh einsum cluster cores
    for (auto* tasklet : dfg.tasklets()) {
        if (auto* cluster = this->try_lift(block, *tasklet)) {
            state.clusters.push_back(cluster);
            state.consumed_nodes.insert(tasklet);
        }
    }

    // Mirror already existing einsum nodes
    for (auto* libnode : dfg.library_nodes()) {
        if (auto* einsum_node = dynamic_cast<EinsumNode*>(libnode)) {
            if (auto* cluster = this->from_einsum_node(block, *einsum_node)) {
                state.clusters.push_back(cluster);
                state.consumed_nodes.insert(einsum_node);
            }
        }
    }
}

bool EinsumDetection::try_extend(BlockState& state, EinsumCluster& cluster) {
    // Extension only applies to the plain reduction core, before any loop was consumed
    if (!cluster.dims.empty()) {
        return false;
    }

    auto& dfg = cluster.block->dataflow();

    std::vector<std::string> new_inputs;
    std::vector<data_flow::Memlet*> new_in_edges;
    std::unordered_map<std::string, data_flow::DataFlowNode*> new_input_nodes;
    bool applied = false;

    for (size_t i = 0; i < cluster.inputs.size(); i++) {
        const std::string& conn = cluster.inputs.at(i);
        auto* node = cluster.input_nodes.at(conn);
        auto* access = dynamic_cast<data_flow::AccessNode*>(node);

        data_flow::Tasklet* mul = nullptr;
        if (access && !is_constant(access) && dfg.in_degree(*access) > 0) {
            for (auto& aedge : dfg.in_edges(*access)) {
                auto* tasklet = dynamic_cast<data_flow::Tasklet*>(&aedge.src());
                if (tasklet && tasklet->code() == data_flow::TaskletCode::fp_mul) {
                    mul = tasklet;
                    break;
                }
            }
        }

        if (mul) {
            applied = true;
            for (auto& medge : dfg.in_edges(*mul)) {
                std::string new_conn = conn + medge.dst_conn();
                new_inputs.push_back(new_conn);
                new_in_edges.push_back(&medge);
                new_input_nodes.insert({new_conn, &medge.src()});
            }
            cluster.consumed_nodes.push_back(mul);
            state.consumed_nodes.insert(mul);
            cluster.consumed_nodes.push_back(access);
            state.consumed_nodes.insert(access);
        } else {
            new_inputs.push_back(conn);
            new_in_edges.push_back(cluster.in_edges.at(i));
            new_input_nodes.insert({conn, node});
        }
    }

    if (!applied) {
        return false;
    }

    cluster.inputs = std::move(new_inputs);
    cluster.in_edges = std::move(new_in_edges);
    cluster.input_nodes = std::move(new_input_nodes);
    return true;
}

void EinsumDetection::consume_input_operations(BlockState& state) {
    for (auto* cluster : state.clusters) {
        while (this->try_extend(state, *cluster)) {
        }
        this->recompute_einsum_indices(*cluster);
    }
}

symbolic::Expression EinsumDetection::cnf_to_upper_bound(const symbolic::CNF& cnf, const symbolic::Symbol indvar) {
    std::vector<symbolic::Expression> candidates;

    for (const auto& clause : cnf) {
        for (const auto& literal : clause) {
            if (SymEngine::is_a<SymEngine::StrictLessThan>(*literal)) {
                auto lt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(literal);
                if (symbolic::eq(lt->get_arg1(), indvar) && !symbolic::uses(lt->get_arg2(), indvar)) {
                    candidates.push_back(lt->get_arg2());
                }
            } else if (SymEngine::is_a<SymEngine::LessThan>(*literal)) {
                auto le = SymEngine::rcp_static_cast<const SymEngine::LessThan>(literal);
                if (symbolic::eq(le->get_arg1(), indvar) && !symbolic::uses(le->get_arg2(), indvar)) {
                    candidates.push_back(symbolic::add(le->get_arg2(), symbolic::one()));
                }
            } else if (SymEngine::is_a<SymEngine::Equality>(*literal)) {
                auto eq = SymEngine::rcp_static_cast<const SymEngine::Equality>(literal);
                if (symbolic::eq(eq->get_arg1(), indvar) && !symbolic::uses(eq->get_arg2(), indvar)) {
                    candidates.push_back(symbolic::add(eq->get_arg2(), symbolic::one()));
                }
            }
        }
    }

    if (candidates.empty()) {
        return SymEngine::null;
    }

    symbolic::Expression result = candidates[0];
    for (size_t i = 1; i < candidates.size(); ++i) {
        result = symbolic::min(result, candidates[i]);
    }
    return result;
}

bool EinsumDetection::subset_contains_symbol(const data_flow::Subset& subset, const symbolic::Symbol& symbol) {
    for (auto& expr : subset) {
        if (symbolic::uses(expr, symbol)) {
            return true;
        }
    }
    return false;
}

PromotionCheck EinsumDetection::can_promote(EinsumCluster& cluster, structured_control_flow::StructuredLoop& loop) {
    symbolic::Symbol indvar = loop.indvar();

    // Loop must be of a sufficiently simple form
    try {
        auto cnf = symbolic::conjunctive_normal_form(loop.condition());
        auto ub = this->cnf_to_upper_bound(cnf, indvar);
        if (ub.is_null()) {
            return {};
        }
    } catch (const symbolic::CNFException&) {
        return {};
    }
    if (!symbolic::eq(loop.update(), symbolic::add(indvar, symbolic::one()))) {
        return {};
    }

    // Prevent one of the input containers to be the index variable
    for (auto& [conn, node] : cluster.input_nodes) {
        auto* access = dynamic_cast<data_flow::AccessNode*>(node);
        if (access && access->data() == indvar->get_name()) {
            return {};
        }
    }

    // Check that the index variable does not collide with an existing dimension index
    for (auto& dim : cluster.dims) {
        if (symbolic::eq(indvar, dim.indvar)) {
            return {};
        }
    }

    const std::string& out_container = cluster.output_node->data();

    PromotionCheck result;
    bool any_contribution = false;

    // Output contribution: must match the integer-affine index pattern if the indvar appears.
    {
        auto contrib = contribution_of(cluster.out_indices(), indvar);
        if (contrib.kind == Contribution::Kind::Invalid) {
            return {};
        }
        bool contributes = contrib.kind == Contribution::Kind::Valid;
        result.output = {contributes, contrib.index, contrib.factor};
        any_contribution |= contributes;
    }

    // Input contributions, verifying the pattern and guarding against self-references without
    // the index variable, e.g. x[i] += ... * x[j].
    result.inputs.resize(cluster.inputs.size());
    data_flow::Subset empty_subset;
    for (size_t i = 0; i < cluster.inputs.size(); i++) {
        auto [in_src, in_edge] = cluster.get_input_for(i);
        const data_flow::Subset& subset = in_edge ? in_edge->subset() : empty_subset;

        auto contrib = contribution_of(subset, indvar);
        if (contrib.kind == Contribution::Kind::Invalid) {
            return {};
        }
        bool contributes = contrib.kind == Contribution::Kind::Valid;

        if (in_src && !is_constant(in_src) && in_src->data() == out_container && !contributes) {
            return {};
        }

        result.inputs.at(i) = {contributes, contrib.index, contrib.factor};
        any_contribution |= contributes;
    }

    if (!any_contribution) {
        return {};
    }
    result.ok = true;
    return result;
}

void EinsumDetection::
    apply_promote(EinsumCluster& cluster, structured_control_flow::StructuredLoop& loop, const PromotionCheck& check) {
    symbolic::Symbol indvar = loop.indvar();
    symbolic::Expression init = loop.init();
    auto cnf = symbolic::conjunctive_normal_form(loop.condition());
    symbolic::Expression bound = this->cnf_to_upper_bound(cnf, indvar);

    cluster.dims.emplace(cluster.dims.begin(), indvar, init, bound);
    cluster.consumed_loops.insert(cluster.consumed_loops.begin(), &loop);

    // Merge the verified contributions into each operand's indexing (kept ordered by index).
    if (cluster.einsum_in_indices.size() != cluster.inputs.size()) {
        cluster.einsum_in_indices.assign(cluster.inputs.size(), EinsumIndexing{});
    }
    data_flow::Subset empty_subset;
    for (size_t i = 0; i < cluster.inputs.size(); i++) {
        const auto& contrib = check.inputs.at(i);
        if (!contrib.contributes) {
            continue;
        }
        const data_flow::Subset& subset = cluster.get_input_subset(i);
        add_contribution(cluster.einsum_in_indices.at(i), subset, {indvar, contrib.index, contrib.factor});
    }
    if (check.output.contributes) {
        add_contribution(
            cluster.einsum_out_indices, cluster.out_indices(), {indvar, check.output.index, check.output.factor}
        );
    }
}

void EinsumDetection::recompute_einsum_indices(EinsumCluster& cluster) {
    cluster.einsum_in_indices.clear();
    cluster.einsum_in_indices.reserve(cluster.in_edges.size());
    data_flow::Subset empty_subset;
    for (auto* edge : cluster.in_edges) {
        const data_flow::Subset& subset = edge ? edge->subset() : empty_subset;
        cluster.einsum_in_indices.push_back(compute_indexing(subset, cluster.dims));
    }
    cluster.einsum_out_indices = compute_indexing(cluster.out_indices(), cluster.dims);
}

bool EinsumDetection::try_to_consume_loop(LoopScopeState& state, structured_control_flow::StructuredLoop& loop) {
    EinsumCluster* cluster = state.representing_last_block_;
    if (!cluster) {
        return false;
    }
    auto check = this->can_promote(*cluster, loop);
    if (!check.ok) {
        return false;
    }
    this->apply_promote(*cluster, loop, check);
    return true;
}

void EinsumDetection::filter_for_coverage() {
    std::vector<std::unique_ptr<EinsumCluster>> filtered;
    for (auto& cluster : this->all_einsums_) {
        bool keep = true;
        int out_coverage = cluster->einsum_out_indices.covered;
        if (!cluster->einsum_out_indices.contributions.empty() && out_coverage == 0) {
            keep = false;
        }
        for (size_t i = 0; i < cluster->inputs.size(); i++) {
            const auto& indexing = cluster->einsum_in_indices.at(i);
            if (!indexing.contributions.empty() && indexing.covered == 0) {
                keep = false;
                break;
            }
        }
        if (keep) {
            filtered.push_back(std::move(cluster));
        } else {
            DEBUG_PRINTLN("Einsum: " << cluster->toStr() << " filtered out due to no coverage");
        }
    }
    this->all_einsums_ = std::move(filtered);
}

bool EinsumDetection::block_fully_consumed(BlockState& state, structured_control_flow::Block& block) {
    auto& dfg = block.dataflow();
    for (auto* tasklet : dfg.tasklets()) {
        if (!state.consumed_nodes.count(tasklet)) {
            return false;
        }
    }
    for (auto* libnode : dfg.library_nodes()) {
        if (!state.consumed_nodes.count(libnode)) {
            return false;
        }
    }
    return true;
}

void EinsumDetection::register_cluster_in_scope(EinsumCluster* cluster) {
    if (this->loop_stack_.empty()) {
        return;
    }
    auto& scope = this->loop_stack_.back();
    if (scope.representing_last_block_ != nullptr) {
        scope.non_reducible = true;
    } else {
        scope.representing_last_block_ = cluster;
    }
}

bool EinsumDetection::visit(structured_control_flow::Block& node) {
    BlockState state;
    this->find_einsum_core_ops(state, node);
    this->consume_input_operations(state);

    bool has_ops = !node.dataflow().tasklets().empty() || !node.dataflow().library_nodes().empty();
    if (!this->loop_stack_.empty() && has_ops) {
        auto& scope = this->loop_stack_.back();
        bool clean = state.clusters.size() == 1 && this->block_fully_consumed(state, node);
        if (clean) {
            this->register_cluster_in_scope(state.clusters.front());
        } else {
            scope.non_reducible = true;
        }
    }

    return false;
}

bool EinsumDetection::visit(structured_control_flow::AssignmentBlock& node) {
    if (!this->loop_stack_.empty() && !node.assignments().empty()) {
        this->loop_stack_.back().non_reducible = true;
    }
    return false;
}

bool EinsumDetection::visit(structured_control_flow::IfElse& node) {
    if (!this->loop_stack_.empty()) {
        this->loop_stack_.back().non_reducible = true;
    }
    // Still descend to detect any clusters nested inside the branches
    return visitor::ActualStructuredSDFGVisitor::visit(node);
}

bool EinsumDetection::visit(structured_control_flow::While& node) {
    if (!this->loop_stack_.empty()) {
        this->loop_stack_.back().non_reducible = true;
    }
    return visitor::ActualStructuredSDFGVisitor::visit(node);
}

bool EinsumDetection::visit(structured_control_flow::Return& node) {
    if (!this->loop_stack_.empty()) {
        this->loop_stack_.back().non_reducible = true;
    }
    return false;
}

bool EinsumDetection::handleStructuredLoop(structured_control_flow::StructuredLoop& loop) {
    this->loop_stack_.push_back(LoopScopeState{});
    visitor::ActualStructuredSDFGVisitor::visit(loop.root());
    LoopScopeState scope = this->loop_stack_.back();
    this->loop_stack_.pop_back();

    EinsumCluster* result = nullptr;
    if (!scope.non_reducible && scope.representing_last_block_ != nullptr && loop.root().size() == 1) {
        if (this->try_to_consume_loop(scope, loop)) {
            result = scope.representing_last_block_;
        }
    }

    if (result) {
        this->register_cluster_in_scope(result);
    } else if (!this->loop_stack_.empty()) {
        this->loop_stack_.back().non_reducible = true;
    }

    return true;
}

void EinsumDetection::run(structured_control_flow::ControlFlowNode& start) {
    this->all_einsums_.clear();
    this->loop_stack_.clear();
    this->visit(start);
    this->filter_for_coverage();
}

} // namespace sdfg::einsum
