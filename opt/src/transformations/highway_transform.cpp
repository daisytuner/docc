#include "sdfg/transformations/highway_transform.h"

#include <list>
#include <stdexcept>

#include "sdfg/optimization_report/pass_report_consumer.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/targets/highway/codegen/highway_map_dispatcher.h"

namespace sdfg {
namespace transformations {

HighwayTransform::HighwayTransform(structured_control_flow::Map& map) : map_(map) {}

std::string HighwayTransform::name() const { return "HighwayTransform"; }

bool HighwayTransform::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (map_.schedule_type().value() != structured_control_flow::ScheduleType_Sequential::value()) {
        if (report_) {
            report_->transform_impossible(this, "not sequential");
        }
        return false;
    }

    // Check for contiguous loop stride
    if (!map_.is_contiguous()) {
        if (report_) {
            report_->transform_impossible(this, "not contiguous");
        }
        return false;
    }

    // Check all outputs are pointers
    auto& arguments_analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();
    std::vector<std::string> arguments;
    for (auto& entry : arguments_analysis.arguments(analysis_manager, map_)) {
        if (entry.second.is_output) {
            if (!entry.second.is_ptr) {
                if (report_) {
                    report_->transform_impossible(this, "contains non-pointer output argument " + entry.first);
                }
                return false;
            }
        }
        if (entry.first == map_.indvar()->get_name()) {
            if (report_) {
                report_->transform_impossible(this, "contains induction variable as argument " + entry.first);
            }
            return false;
        }
    }

    // Check: all locals are scalar (implicitly converted into vectors)
    symbolic::SymbolSet local_symbols;
    for (auto& local : arguments_analysis.locals(analysis_manager, map_)) {
        auto& type = builder.subject().type(local);
        if (type.type_id() != types::TypeID::Scalar) {
            if (report_) {
                report_->transform_impossible(this, "contains non-scalar local " + local);
            }
            return false;
        }
        if (local == map_.indvar()->get_name()) {
            continue;
        }
        if (types::is_integer(type.primitive_type())) {
            local_symbols.insert(symbolic::symbol(local));
        }
    }

    size_t bitwidth = 0;
    std::list<structured_control_flow::ControlFlowNode*> queue = {&map_.root()};
    while (!queue.empty()) {
        auto node = queue.front();
        queue.pop_front();

        if (auto block = dynamic_cast<structured_control_flow::Block*>(node)) {
            auto& graph = block->dataflow();
            for (auto& edge : graph.edges()) {
                if (edge.type() != data_flow::MemletType::Computational) {
                    if (report_) {
                        report_->transform_impossible(
                            this,
                            "contains unsupported memlet type on edge with element ID " +
                                std::to_string(edge.element_id())
                        );
                    }
                    return false;
                }

                size_t edge_bitwidth = types::bit_width(edge.base_type().primitive_type());
                if (bitwidth == 0) {
                    bitwidth = edge_bitwidth;
                } else if (edge_bitwidth != bitwidth) {
                    if (report_) {
                        report_->transform_impossible(
                            this,
                            "contains mismatched memlet bitwidth on edge with element ID " +
                                std::to_string(edge.element_id())
                        );
                    }
                    return false;
                }

                // Classify memlet access pattern
                auto access_type = classify_memlet_access_type(edge.subset(), map_.indvar(), local_symbols);
                if (access_type == HighwayTransform::CONSTANT) {
                    continue;
                } else if (access_type == HighwayTransform::CONTIGUOUS) {
                    continue;
                } else {
                    if (report_) {
                        report_->transform_impossible(
                            this,
                            "contains unsupported memlet access pattern on edge with element ID " +
                                std::to_string(edge.element_id())
                        );
                    }
                    return false;
                }
            }

            for (auto& dnode : graph.topological_sort()) {
                if (auto const_node = dynamic_cast<data_flow::ConstantNode*>(dnode)) {
                    if (const_node->type().type_id() != types::TypeID::Scalar) {
                        if (report_) {
                            report_->transform_impossible(
                                this,
                                "contains unsupported constant type on node with element ID " +
                                    std::to_string(const_node->element_id())
                            );
                        }
                        return false;
                    }
                    continue;
                }
                if (auto access_node = dynamic_cast<data_flow::AccessNode*>(dnode)) {
                    continue;
                }

                if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(dnode)) {
                    std::string code = highway::HighwayMapDispatcher::tasklet(*tasklet);
                    if (code.empty()) {
                        if (report_) {
                            report_->transform_impossible(
                                this,
                                "contains unsupported tasklet node with element ID " +
                                    std::to_string(tasklet->element_id())
                            );
                        }
                        return false;
                    }
                    continue;
                } else if (auto cmath_node = dynamic_cast<math::cmath::CMathNode*>(dnode)) {
                    std::string code = highway::HighwayMapDispatcher::cmath_node(*cmath_node);
                    if (code.empty()) {
                        if (report_) {
                            report_->transform_impossible(
                                this,
                                "contains unsupported CMath node with element ID " +
                                    std::to_string(cmath_node->element_id())
                            );
                        }
                        return false;
                    }
                    continue;
                } else {
                    if (report_) {
                        report_->transform_impossible(
                            this, "contains unsupported dataflow node of type " + std::string(typeid(*dnode).name())
                        );
                    }
                    return false;
                }
            }
        } else if (auto sequence = dynamic_cast<structured_control_flow::Sequence*>(node)) {
            for (size_t i = 0; i < sequence->size(); i++) {
                if (sequence->at(i).second.assignments().size() > 0) {
                    if (report_) {
                        report_->transform_impossible(
                            this,
                            "contains unsupported transition with assignments at element ID " +
                                std::to_string(sequence->at(i).second.element_id())
                        );
                    }
                    return false;
                }
                queue.push_back(&sequence->at(i).first);
            }
        } else {
            if (report_) {
                report_->transform_impossible(
                    this, "contains unsupported control flow node of type " + std::string(typeid(*node).name())
                );
            }
            return false;
        }
    }

    return true;
}

void HighwayTransform::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    builder.update_schedule_type(this->map_, highway::ScheduleType_Highway::create());
    if (report_) report_->transform_applied(this);
}

void HighwayTransform::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", this->map_.element_id()}, {"type", "map"}}}};
    j["transformation_type"] = this->name();
}

HighwayTransform HighwayTransform::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto map_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto element = builder.find_element_by_id(map_id);
    if (element == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(map_id) + " not found.");
    }

    auto loop = dynamic_cast<structured_control_flow::Map*>(element);

    if (loop == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(map_id) + " is not a Map.");
    }

    return HighwayTransform(*loop);
}

HighwayTransform::MemletAccessType HighwayTransform::classify_memlet_access_type(
    const data_flow::Subset& subset, const symbolic::Symbol& indvar, const symbolic::SymbolSet& local_symbols
) {
    // Scalar access is constant
    if (subset.empty()) {
        return HighwayTransform::CONSTANT;
    }

    // Criterion: dim0, ..., dimN-1 are constant
    for (size_t i = 0; i < subset.size() - 1; ++i) {
        auto expr = subset.at(i);
        bool is_constant = true;
        for (auto& sym : symbolic::atoms(expr)) {
            if (symbolic::eq(sym, indvar)) {
                is_constant = false;
                break;
            }
            if (local_symbols.find(sym) != local_symbols.end()) {
                is_constant = false;
                break;
            }
        }
        if (!is_constant) {
            return HighwayTransform::UNKNOWN;
        }
    }

    // Classify dimN
    auto dimN = subset.back();
    for (auto& sym : symbolic::atoms(dimN)) {
        // Gather / Scatter - extend here
        if (local_symbols.find(sym) != local_symbols.end()) {
            return HighwayTransform::UNKNOWN;
        }
    }
    if (symbolic::uses(dimN, indvar)) {
        if (symbolic::eq(dimN, indvar)) {
            return HighwayTransform::CONTIGUOUS;
        }

        symbolic::SymbolVec poly_gens = {indvar};
        auto poly = symbolic::polynomial(dimN, poly_gens);
        if (poly.is_null()) {
            return HighwayTransform::UNKNOWN;
        }
        auto affine_coeffs = symbolic::affine_coefficients(poly, poly_gens);
        if (affine_coeffs.size() != 2) {
            return HighwayTransform::UNKNOWN;
        }
        auto mul_coeff = affine_coeffs.at(indvar);
        if (symbolic::eq(mul_coeff, symbolic::zero())) {
            return HighwayTransform::CONSTANT;
        }
        if (symbolic::eq(mul_coeff, symbolic::one())) {
            return HighwayTransform::CONTIGUOUS;
        }

        return HighwayTransform::UNKNOWN;
    }

    // dimN does not use indvar or moving symbols -> constant
    return HighwayTransform::CONSTANT;
}

} // namespace transformations
} // namespace sdfg
