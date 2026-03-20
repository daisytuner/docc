#include "sdfg/transformations/utils.h"

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "symengine/logic.h"

using namespace sdfg;

symbolic::Integer sdfg::get_iteration_count(sdfg::structured_control_flow::StructuredLoop& loop) {
    auto condition = loop.condition();
    // TODO: extend me to support more complex loop conditions and update expressions
    if (SymEngine::is_a<SymEngine::And>(*condition)) {
        auto and_condition = SymEngine::rcp_static_cast<const SymEngine::And>(condition);
        auto container = and_condition->get_container();
        for (auto it = container.begin(); it != container.end(); ++it) {
            if (!SymEngine::is_a<SymEngine::StrictLessThan>(**it)) {
                return SymEngine::null;
            }
            auto stl = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(*it);
            auto lhs = stl->get_args()[0];
            auto rhs = stl->get_args()[1];
            if (!symbolic::eq(lhs, loop.indvar())) {
                std::swap(lhs, rhs);
            }
            if (!symbolic::eq(lhs, loop.indvar())) {
                continue;
            }
            auto bound = rhs;
            auto range = symbolic::sub(bound, loop.init());
            if (SymEngine::is_a<SymEngine::Integer>(*range)) {
                return SymEngine::rcp_static_cast<const SymEngine::Integer>(range);
            }
        }
    } else if (SymEngine::is_a<SymEngine::StrictLessThan>(*condition)) {
        auto stl = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(condition);
        auto lhs = stl->get_args()[0];
        auto rhs = stl->get_args()[1];
        if (!symbolic::eq(lhs, loop.indvar())) {
            std::swap(lhs, rhs);
        }
        if (!symbolic::eq(lhs, loop.indvar())) {
            return SymEngine::null;
        }
        auto bound = rhs;
        auto range = symbolic::sub(bound, loop.init());
        if (SymEngine::is_a<SymEngine::Integer>(*range)) {
            return SymEngine::rcp_static_cast<const SymEngine::Integer>(range);
        }
    }
    return SymEngine::null;
};
