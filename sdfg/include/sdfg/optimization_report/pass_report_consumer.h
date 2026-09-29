#pragma once
#include <nlohmann/json.hpp>

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {

class PassReportConsumer {
public:
    virtual void transform_impossible(const std::string& transform, const std::string& reason) = 0;
    virtual void transform_impossible(const transformations::Transformation* transform, const std::string& reason) {
        transform_impossible(transform->name(), reason);
    }

    virtual void transform_possible(const std::string& transform) = 0;
    virtual void transform_possible(const transformations::Transformation* transform) {
        transform_possible(transform->name());
    }

    virtual void transform_applied(const std::string& transform, nlohmann::json transform_info = {}) = 0;
    virtual void transform_applied(const sdfg::transformations::Transformation* transform) {
        transform_applied(transform->name());
    }

    virtual void in_scope(StructuredSDFG* scope) = 0;
    void no_scope() {
        in_scope(nullptr);
    }

    virtual void in_outermost_loop(int idx) = 0;

    void no_loop() {
        in_outermost_loop(-1);
    }
    virtual ~PassReportConsumer() = default;

    virtual void target_transform_possible(const std::string basicString, bool b) = 0;
};

} // namespace sdfg
