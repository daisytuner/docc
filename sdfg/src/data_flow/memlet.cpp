#include <sdfg/data_flow/memlet.h>

#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace data_flow {

Memlet::Memlet(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Edge& edge,
    DataFlowGraph& parent,
    DataFlowNode& src,
    const std::string& src_conn,
    DataFlowNode& dst,
    const std::string& dst_conn,
    const Subset& subset,
    const types::IType& base_type
)
    : Element(element_id, debug_info), edge_(edge), parent_(&parent), src_(src), dst_(dst), src_conn_(src_conn),
      dst_conn_(dst_conn), subset_(subset), base_type_(base_type.clone()) {

      };

void Memlet::validate(const Function& function) const {
    // Validate subset
    for (const auto& dim : this->subset_) {
        // Null ptr check
        if (dim.is_null()) {
            throw InvalidSDFGException("Memlet: Subset dimensions cannot be null");
        }
    }

    // Validate connections
    switch (this->type()) {
        case MemletType::Computational: {
            // Criterion: Must connect a code node and an access node with void connector at access node
            const AccessNode* data_node = nullptr;
            const CodeNode* code_node = nullptr;
            if (this->src_conn_ == "void") {
                data_node = dynamic_cast<const AccessNode*>(&this->src_);
                code_node = dynamic_cast<const CodeNode*>(&this->dst_);
                if (!data_node || !code_node) {
                    throw InvalidSDFGException("Memlet: Computation memlets must connect a code node and an access node"
                    );
                }

                // Criterion: Non-void connector must be an input of the code node
                if (std::find(code_node->inputs().begin(), code_node->inputs().end(), this->dst_conn_) ==
                    code_node->inputs().end()) {
                    throw InvalidSDFGException("Memlet: Computation memlets must have an input in the code node");
                }
            } else if (this->dst_conn_ == "void") {
                data_node = dynamic_cast<const AccessNode*>(&this->dst_);
                code_node = dynamic_cast<const CodeNode*>(&this->src_);
                if (!data_node || !code_node) {
                    throw InvalidSDFGException("Memlet: Computation memlets must connect a code node and an access node"
                    );
                }

                // Criterion: Non-void connector must be an output of the code node
                if (std::find(code_node->outputs().begin(), code_node->outputs().end(), this->src_conn_) ==
                    code_node->outputs().end()) {
                    throw InvalidSDFGException("Memlet: Computation memlets must have an output in the code node");
                }
            } else {
                throw InvalidSDFGException(
                    "Memlet: Computation memlets must have void connector at source or destination"
                );
            }

            // If tensor, check that the type is consistenly defined
            if (this->base_type_->type_id() == types::TypeID::Tensor) {
                auto& tensor_type = dynamic_cast<const types::Tensor&>(*this->base_type_);
                if (tensor_type.is_scalar()) {
                    if (auto const_node = dynamic_cast<const data_flow::ConstantNode*>(data_node)) {
                        if (const_node->type().type_id() != types::TypeID::Scalar) {
                            throw InvalidSDFGException(
                                "Memlet: Scalar tensors must reference scalar buffers. Base type: " +
                                this->base_type_->print() + " Buffer type: " + const_node->type().print()
                            );
                        }
                    } else {
                        auto& buffer_type = function.type(data_node->data());
                        if (buffer_type.type_id() != types::TypeID::Scalar) {
                            throw InvalidSDFGException(
                                "Memlet: Scalar tensors must reference scalar buffers. Base type: " +
                                this->base_type_->print() + " Buffer type: " + buffer_type.print()
                            );
                        }
                    }
                } else {
                    auto& buffer_type = function.type(data_node->data());
                    if (buffer_type.type_id() != types::TypeID::Pointer) {
                        throw InvalidSDFGException(
                            "Memlet: Non-scalar tensors must reference pointer buffers. Base type: " +
                            this->base_type_->print() + " Buffer type: " + buffer_type.print()
                        );
                    }
                    if (this->subset_.size() > tensor_type.shape().size()) {
                        throw InvalidSDFGException(
                            "Memlet: Subset dimensions must match base type dimensions. Base type: " +
                            this->base_type_->print() + " Subset Dim: " + std::to_string(this->subset_.size())
                        );
                    }
                    if (tensor_type.shape().size() != tensor_type.strides().size()) {
                        throw InvalidSDFGException(
                            "Memlet: Tensor types must have the same number of shape and stride dimensions. Base "
                            "type: " +
                            this->base_type_->print()
                        );
                    }
                }
            }
            break;
        }
        case MemletType::Reference: {
            // Criterion: Destination must be an access node with a pointer type
            auto dst_node = dynamic_cast<const AccessNode*>(&this->dst_);
            if (!dst_node) {
                throw InvalidSDFGException("Memlet: Reference memlets must have an access node destination");
            }
            auto dst_data = dst_node->data();
            // Criterion: Destination must be non-constant
            if (helpers::is_number(dst_data) || symbolic::is_nullptr(symbolic::symbol(dst_data))) {
                throw InvalidSDFGException("Memlet: Reference memlets must have a non-constant destination");
            }

            // Criterion: Destination must be a pointer
            auto& dst_type = function.type(dst_data);
            if (dst_type.type_id() != types::TypeID::Pointer) {
                throw InvalidSDFGException("Memlet: Reference memlets must have a pointer destination");
            }

            // Criterion: Source must be an access node
            if (this->src_conn_ != "void") {
                throw InvalidSDFGException("Memlet: Reference memlets must have a void source");
            }
            auto src_node = dynamic_cast<const AccessNode*>(&this->src_);
            if (!src_node) {
                throw InvalidSDFGException("Memlet: Reference memlets must have an access node source");
            }

            // Case: Constant
            if (helpers::is_number(src_node->data()) || symbolic::is_nullptr(symbolic::symbol(src_node->data()))) {
                if (!this->subset_.empty()) {
                    throw InvalidSDFGException("Memlet: Reference memlets for raw addresses must not have a subset");
                }
                return;
            }

            // Case: Container
            // Criterion: Must be contiguous memory reference
            // Throws exception if not contiguous
            types::infer_type(function, *this->base_type_, this->subset_);
            break;
        }
        case MemletType::Dereference_Src: {
            if (this->src_conn_ != "void") {
                throw InvalidSDFGException("Memlet: Dereference memlets must have a void destination");
            }

            auto src_node = dynamic_cast<const AccessNode*>(&this->src_);
            if (!src_node) {
                throw InvalidSDFGException("Memlet: Dereference memlets must have an access node source");
            }
            auto dst_node = dynamic_cast<const AccessNode*>(&this->dst_);
            if (!dst_node) {
                throw InvalidSDFGException("Memlet: Dereference memlets must have an access node destination");
            }

            // Criterion: Dereference memlets must have '0' as the only dimension
            if (this->subset_.size() != 1) {
                throw InvalidSDFGException("Memlet: Dereference memlets must have '0' as the only dimension");
            }
            if (!symbolic::eq(this->subset_[0], symbolic::zero())) {
                throw InvalidSDFGException("Memlet: Dereference memlets must have '0' as the only dimension");
            }

            // Criterion: Source must be a pointer
            if (auto const_node = dynamic_cast<const ConstantNode*>(src_node)) {
                if (const_node->type().type_id() != types::TypeID::Pointer &&
                    const_node->type().type_id() != types::TypeID::Scalar) {
                    throw InvalidSDFGException("Memlet: Dereference memlets must have a pointer source");
                }
            } else {
                auto src_data = src_node->data();
                auto& src_type = function.type(src_data);
                if (src_type.type_id() != types::TypeID::Pointer) {
                    throw InvalidSDFGException("Memlet: Dereference memlets must have a pointer source");
                }
            }

            // Criterion: Must be typed pointer
            auto base_pointer_type = dynamic_cast<const types::Pointer*>(this->base_type_.get());
            if (!base_pointer_type) {
                throw InvalidSDFGException("Memlet: Dereference memlets must have a typed pointer base type");
            }
            if (!base_pointer_type->has_pointee_type()) {
                throw InvalidSDFGException("Memlet: Dereference memlets must have a pointee type");
            }

            break;
        }
        case MemletType::Dereference_Dst: {
            if (this->dst_conn_ != "void") {
                throw InvalidSDFGException("Memlet: Dereference memlets must have a void source");
            }

            auto src_node = dynamic_cast<const AccessNode*>(&this->src_);
            if (!src_node) {
                throw InvalidSDFGException("Memlet: Dereference memlets must have an access node source");
            }
            auto dst_node = dynamic_cast<const AccessNode*>(&this->dst_);
            if (!dst_node) {
                throw InvalidSDFGException("Memlet: Dereference memlets must have an access node destination");
            }

            // Criterion: Dereference memlets must have '0' as the only dimension
            if (this->subset_.size() != 1) {
                throw InvalidSDFGException("Memlet: Dereference memlets must have '0' as the only dimension");
            }
            if (!symbolic::eq(this->subset_[0], symbolic::zero())) {
                throw InvalidSDFGException("Memlet: Dereference memlets must have '0' as the only dimension");
            }

            // Criterion: src type cannot be a function
            const sdfg::types::IType* src_type;
            if (auto const_node = dynamic_cast<const data_flow::ConstantNode*>(src_node)) {
                src_type = &const_node->type();
            } else {
                src_type = &function.type(src_node->data());
            }
            if (src_type->type_id() == types::TypeID::Function) {
                throw InvalidSDFGException("Memlet: Dereference memlets cannot have source of type Function");
            }

            // Criterion: Destination must be a pointer
            if (auto const_node = dynamic_cast<const ConstantNode*>(dst_node)) {
                throw InvalidSDFGException("Memlet: Dereference memlets must have a non-constant destination");
            }
            auto dst_data = dst_node->data();
            auto& dst_type = function.type(dst_data);
            if (dst_type.type_id() != types::TypeID::Pointer) {
                throw InvalidSDFGException("Memlet: Dereference memlets must have a pointer destination");
            }

            // Criterion: Must be typed pointer
            auto base_pointer_type = dynamic_cast<const types::Pointer*>(this->base_type_.get());
            if (!base_pointer_type) {
                throw InvalidSDFGException("Memlet: Dereference memlets must have a typed pointer base type");
            }
            if (!base_pointer_type->has_pointee_type()) {
                throw InvalidSDFGException("Memlet: Dereference memlets must have a pointee type");
            }

            break;
        }
        default:
            throw InvalidSDFGException("Memlet: Invalid memlet type");
    }
};

const graph::Edge Memlet::edge() const { return this->edge_; };

const DataFlowGraph& Memlet::get_parent() const { return *this->parent_; };

DataFlowGraph& Memlet::get_parent() { return *this->parent_; };

MemletType Memlet::type() const {
    if (this->dst_conn_ == "ref") {
        return Reference;
    } else if (this->dst_conn_ == "deref") {
        return Dereference_Src;
    } else if (this->src_conn_ == "deref") {
        return Dereference_Dst;
    } else {
        return Computational;
    }
}

const DataFlowNode& Memlet::src() const { return this->src_; };

DataFlowNode& Memlet::src() { return this->src_; };

const DataFlowNode& Memlet::dst() const { return this->dst_; };

DataFlowNode& Memlet::dst() { return this->dst_; };

const std::string& Memlet::src_conn() const { return this->src_conn_; };

const std::string& Memlet::dst_conn() const { return this->dst_conn_; };

const Subset& Memlet::subset() const { return this->subset_; };

void Memlet::set_subset(const Subset& subset) { this->subset_ = subset; };

const types::IType& Memlet::base_type() const { return *this->base_type_; };

void Memlet::set_base_type(const types::IType& base_type) { this->base_type_ = base_type.clone(); };

std::unique_ptr<types::IType> Memlet::result_type(const Function& function) const {
    return types::infer_type(function, *this->base_type_, this->subset_);
};

std::unique_ptr<Memlet> Memlet::clone(
    size_t element_id, const graph::Edge& edge, DataFlowGraph& parent, DataFlowNode& src, DataFlowNode& dst
) const {
    return std::unique_ptr<Memlet>(new Memlet(
        element_id,
        this->debug_info_,
        edge,
        parent,
        src,
        this->src_conn_,
        dst,
        this->dst_conn_,
        this->subset_,
        *this->base_type_
    ));
};

void Memlet::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    Subset new_subset;
    for (auto& dim : this->subset_) {
        new_subset.push_back(symbolic::subs(dim, old_expression, new_expression));
    }
    this->subset_ = new_subset;
};

} // namespace data_flow
} // namespace sdfg
