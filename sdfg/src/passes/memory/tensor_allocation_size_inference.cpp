#include "sdfg/passes/memory/tensor_allocation_size_inference.h"

#include <map>
#include <unordered_set>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/utils.h"
#include "sdfg/visitor/immutable_structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

namespace {

class TensorSizeCollector : public visitor::ImmutableStructuredSDFGVisitor {
public:
    std::map<std::string, symbolic::Expression> sizes;
    std::unordered_set<std::string> rejected;

    using visitor::ImmutableStructuredSDFGVisitor::ImmutableStructuredSDFGVisitor;

    bool accept(structured_control_flow::Block& block) override {
        for (auto& memlet : block.dataflow().edges()) {
            const data_flow::AccessNode* access = nullptr;
            if (dynamic_cast<const data_flow::LibraryNode*>(&memlet.dst())) {
                access = dynamic_cast<const data_flow::AccessNode*>(&memlet.src());
            } else if (dynamic_cast<const data_flow::LibraryNode*>(&memlet.src())) {
                access = dynamic_cast<const data_flow::AccessNode*>(&memlet.dst());
            }

            if (access == nullptr || dynamic_cast<const data_flow::ConstantNode*>(access)) {
                continue;
            }

            auto* tensor = dynamic_cast<const types::Tensor*>(&memlet.base_type());
            if (tensor == nullptr) {
                continue;
            }

            const std::string& container = access->data();
            if (rejected.contains(container)) {
                continue;
            }
            if (sdfg_.type(container).type_id() != types::TypeID::Pointer) {
                continue;
            }

            auto element_size = types::get_type_size(tensor->element_type(), false);
            if (element_size.is_null()) {
                reject(container);
                continue;
            }
            auto size = symbolic::expand(symbolic::mul(tensor->layout().memory_span(), element_size));

            // Size must stay valid until offloading, so only parameters may appear in it.
            bool only_arguments = true;
            for (auto& sym : symbolic::atoms(size)) {
                if (!sdfg_.is_argument(sym->get_name())) {
                    only_arguments = false;
                    break;
                }
            }
            if (!only_arguments) {
                reject(container);
                continue;
            }

            auto it = sizes.find(container);
            if (it == sizes.end()) {
                sizes.insert({container, size});
            } else {
                it->second = symbolic::max(it->second, size);
            }
        }
        return false;
    }

private:
    void reject(const std::string& container) {
        rejected.insert(container);
        sizes.erase(container);
    }
};

} // namespace

bool TensorAllocationSizeInference::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    TensorSizeCollector collector(sdfg, analysis_manager);
    collector.visit();

    bool applied = false;
    for (auto& [container, size] : collector.sizes) {
        auto& type = sdfg.type(container);
        auto storage = type.storage_type();
        if (storage.allocation() == types::StorageType::Managed) {
            continue;
        }

        auto new_size = size;
        if (!storage.allocation_size().is_null()) {
            new_size = symbolic::max(storage.allocation_size(), size);
        }
        new_size = symbolic::simplify(new_size);
        if (symbolic::null_safe_eq(storage.allocation_size(), new_size)) {
            continue;
        }

        auto new_type = type.clone();
        new_type->storage_type().allocation_size(new_size);
        builder.change_type(container, *new_type);
        applied = true;
    }

    return applied;
}

} // namespace passes
} // namespace sdfg
