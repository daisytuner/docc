#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace types {

class Scalar;

class Tensor : public IType {
private:
    std::unique_ptr<Scalar> element_type_;
    math::tensor::TensorLayout layout_;

public:
    /**
     * @deprecated use TensorLayout
     */
    Tensor(const Scalar& element_type, const symbolic::MultiExpression& shape);

    Tensor(const Scalar& element_type, const math::tensor::TensorLayout& layout);

    /**
     * @deprecated use TensorLayout
     */
    Tensor(
        const Scalar& element_type,
        const symbolic::MultiExpression& shape,
        const symbolic::MultiExpression& strides,
        const symbolic::Expression& offset = symbolic::zero()
    );

    Tensor(
        StorageType storage_type,
        size_t alignment,
        const std::string& initializer,
        const Scalar& element_type,
        const math::tensor::TensorLayout& layout
    );

    /**
     * @deprecated use TensorLayout
     */
    Tensor(
        StorageType storage_type,
        size_t alignment,
        const std::string& initializer,
        const Scalar& element_type,
        const symbolic::MultiExpression& shape,
        const symbolic::MultiExpression& strides,
        const symbolic::Expression& offset = symbolic::zero()
    );

    virtual PrimitiveType primitive_type() const override;

    virtual TypeID type_id() const override;

    virtual bool is_symbol() const override;

    bool is_pointer_like() const override { return true; }

    const Scalar& element_type() const;

    const math::tensor::TensorLayout& layout() const;

    /**
     * @deprecated use TensorLayout
     */
    const symbolic::MultiExpression& shape() const;

    /**
     * @deprecated use TensorLayout
     */
    const symbolic::MultiExpression& strides() const;

    /**
     * @deprecated use TensorLayout
     */
    const symbolic::Expression& offset() const;

    symbolic::Expression total_elements() const;

    bool is_scalar() const;

    virtual bool operator==(const IType& other) const override;

    virtual std::unique_ptr<IType> clone() const override;

    virtual std::string print() const override;

    static symbolic::MultiExpression strides_from_shape(const symbolic::MultiExpression& shape);

    std::unique_ptr<Tensor> newaxis(size_t axis) const;

    std::unique_ptr<Tensor> flip(size_t axis) const;

    std::unique_ptr<Tensor> unsqueeze(size_t axis) const;

    std::unique_ptr<Tensor> squeeze(size_t axis) const;

    std::unique_ptr<Tensor> squeeze() const;

    std::unique_ptr<Tensor> reshape(const symbolic::MultiExpression& new_shape) const;
};

} // namespace types
} // namespace sdfg
