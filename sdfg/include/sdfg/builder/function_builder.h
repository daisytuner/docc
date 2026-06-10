#pragma once

#include <utility>

#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace builder {

class FunctionBuilder {
protected:
    virtual Function& function() const = 0;

    size_t new_element_id() const;

    size_t new_element_id_batch(size_t batch) const;

public:
    virtual ~FunctionBuilder() = default;

    /***** Section: Containers *****/

    void set_return_type(const types::IType& type) const;

    const types::IType& add_container(
        const std::string& name, const types::IType& type, bool is_argument = false, bool is_external = false
    ) const;

    const types::IType& add_external(const std::string& name, const types::IType& type, LinkageType linkage_type) const;

    void remove_container(const std::string& name) const;

    void change_type(const std::string& name, const types::IType& type) const;

    virtual void rename_container(const std::string& old_name, const std::string& new_name) const;

    types::StructureDefinition& add_structure(const std::string& name, bool is_packed) const;

    std::unique_ptr<types::Structure> create_vector_type(const types::Scalar& element_type, size_t vector_size);

    std::string find_new_name(std::string prefix = "tmp_") const;

    void set_element_counter(size_t element_counter);

    /** Common Dataflow Operations **/

    void update_tasklet(data_flow::Tasklet& tasklet, const data_flow::TaskletCode code);
};

} // namespace builder
} // namespace sdfg
