#pragma once

#include <string>

#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/structure.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace codegen {

class LanguageExtension {
protected:
    sdfg::Function& function_;
    std::string external_prefix_;

public:
    LanguageExtension(sdfg::Function& function, const std::string& external_prefix)
        : function_(function), external_prefix_(external_prefix) {
    }

    virtual ~LanguageExtension() = default;

    const std::string& external_prefix() const {
        return this->external_prefix_;
    }

    virtual const std::string language() const = 0;

    virtual std::string primitive_type(const types::PrimitiveType prim_type) = 0;

    virtual std::string declaration(
        const std::string& name, const types::IType& type, bool use_initializer = false, bool use_alignment = false
    ) = 0;

    virtual std::string type_cast(const std::string& name, const types::IType& type) = 0;

    virtual std::string subset(const types::IType& type, const data_flow::Subset& subset) = 0;

    virtual std::string expression(const symbolic::Expression expr) = 0;

    virtual std::string access_node(const data_flow::AccessNode& node) = 0;

    virtual std::string tasklet(const data_flow::Tasklet& tasklet) = 0;

    virtual std::string zero(const types::PrimitiveType prim_type) = 0;
};

} // namespace codegen
} // namespace sdfg
