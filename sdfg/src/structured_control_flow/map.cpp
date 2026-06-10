#include "sdfg/structured_control_flow/map.h"
#include <string>

#include "sdfg/serializer/json_serializer.h"
#include "sdfg/symbolic/symbolic.h"
#include "symengine/expression.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace structured_control_flow {

Map::
    Map(size_t element_id,
        const DebugInfo& debug_info,
        ControlFlowNode* parent,
        symbolic::Symbol indvar,
        symbolic::Expression init,
        symbolic::Expression update,
        symbolic::Condition condition,
        const ScheduleType& schedule_type)
    : StructuredLoop(element_id, debug_info, parent, indvar, init, update, condition), schedule_type_(schedule_type) {};

void Map::validate(const Function& function) const { StructuredLoop::validate(function); };

const ScheduleType& Map::schedule_type() const { return this->schedule_type_; };

} // namespace structured_control_flow
} // namespace sdfg
