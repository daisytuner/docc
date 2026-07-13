#include "sdfg/structured_control_flow/for.h"

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace structured_control_flow {

For::
    For(size_t element_id,
        const DebugInfo& debug_info,
        ControlFlowNode* parent,
        symbolic::Symbol indvar,
        symbolic::Expression init,
        symbolic::Expression update,
        symbolic::Condition condition)
    : StructuredLoop(element_id, debug_info, parent, indvar, init, update, condition) {};

bool For::accept(visitor::ActualStructuredSDFGVisitor& visitor) { return visitor.visit(*this); }

void For::validate(const Function& function) const { StructuredLoop::validate(function); };

} // namespace structured_control_flow
} // namespace sdfg
