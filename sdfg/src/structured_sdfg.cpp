#include "sdfg/structured_sdfg.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/element.h"

#include "sdfg/serializer/json_serializer.h"

namespace sdfg {

StructuredSDFG::StructuredSDFG(const std::string& name, FunctionType type, const types::IType& return_type)
    : Function(name, type, return_type) {
    this->root_ = std::unique_ptr<structured_control_flow::Sequence>(
        new structured_control_flow::Sequence(this->element_counter_, DebugInfo(), nullptr)
    );
};

StructuredSDFG::StructuredSDFG(const std::string& name, FunctionType type)
    : StructuredSDFG(name, type, types::Scalar(types::PrimitiveType::Void)) {}

const structured_control_flow::Sequence& StructuredSDFG::root() const { return *this->root_; };

structured_control_flow::Sequence& StructuredSDFG::root() { return *this->root_; };

std::unique_ptr<StructuredSDFG> StructuredSDFG::clone() const {
    serializer::JSONSerializer serializer;
    nlohmann::json j = serializer.serialize(*this);
    return serializer.deserialize(j);
};

void StructuredSDFG::validate() const {
    // Call parent validate
    Function::validate();

    // Validate root
    this->root().validate(*this);
};

} // namespace sdfg
