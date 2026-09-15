#pragma once
#include <memory>
#include <nlohmann/json_fwd.hpp>

#include "sdfg/data_flow/memlet.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg::math::tensor {

/**
 * The metadata associated with a addressing a tensor in elements
 * Meant to be used with TensorNodes to describe their input's and outputs layouts and ease handling of such
 *descriptions If the TensorType will keep existing it should also be switched to use this.
 *
 * Datatype is not part of it, as that can change independent of the layout. As long as the layout is strictly in
 *elements, not bytes, there is no conflict
 **/
class TensorLayout {
private:
    /**
     * Shape of input tensor [..., M, K]
     */
    symbolic::MultiExpression shape_;
    /**
     * Strides for tensor (defaults to row-major contiguous)
     */
    symbolic::MultiExpression strides_;
    /**
     * Offset into tensor in elements (defaults to 0)
     */
    symbolic::Expression offset_;

public:
    TensorLayout(
        const symbolic::MultiExpression& shape,
        const symbolic::MultiExpression& strides = {},
        const symbolic::Expression offset = symbolic::integer(0)
    );

    const symbolic::MultiExpression& shape() const { return shape_; }
    const symbolic::MultiExpression& strides() const { return strides_; }

    const symbolic::Expression& offset() const { return offset_; }

    void serialize_to_json(nlohmann::json& j) const;

    std::string toStr() const;

    symbolic::Expression total_elements() const;

    symbolic::MultiExpression linear_strides() const;

    symbolic::Expression resolve_element(const symbolic::MultiExpression& indices, bool require_to_element = true) const;

    static symbolic::MultiExpression linear_strides(const symbolic::MultiExpression& shape);

    void collect_symbols(symbolic::SymbolSet& set) const;

    void replace_symbols(const symbolic::Expression& old, const symbolic::Expression& new_expr);

    void replace_symbols(const symbolic::ExpressionMapping& replacements);

    /**
     *
     * @param i the dimension / entry in shape. 0 is outermost
     */
    const symbolic::Expression& get_dim(int i) const { return shape_.at(i); }
    /**
     *
     * @param i 0 is innermost dim, 1 next level out etc.
     */
    symbolic::Expression get_dim_innermost(int i) const { return shape_.at(shape_.size() - 1 - i); }

    /**
     *
     * @param i the dimension / entry in strides. 0 is outermost
     */
    symbolic::Expression get_stride(int i) const { return strides_.at(i); }
    /**
     *
     * @param i 0 is innermost dim, 1 next level out etc.
     */
    symbolic::Expression get_stride_innermost(int i) const { return strides_.at(strides_.size() - 1 - i); }

    int dims() const { return shape_.size(); }

    bool is_scalar() const;

    static TensorLayout deserialize_from_json(const nlohmann::json& j);

    static bool has_linear_accesses(symbolic::MultiExpression shape, symbolic::MultiExpression strides);

    static bool has_linear_accesses_no_padding(
        symbolic::MultiExpression shape, symbolic::MultiExpression strides, symbolic::Expression offset
    );

    bool has_linear_accesses() const;

    bool has_linear_accesses_no_padding() const;

    bool has_transposed_strides_no_padding() const;

    bool operator==(const TensorLayout& other) const;

    std::unique_ptr<TensorLayout> newaxis(size_t axis) const;

    std::unique_ptr<TensorLayout> flip(size_t axis) const;

    std::unique_ptr<TensorLayout> unsqueeze(size_t axis) const;

    std::unique_ptr<TensorLayout> squeeze(size_t axis) const;

    std::unique_ptr<TensorLayout> squeeze() const;

    std::unique_ptr<TensorLayout> reshape(const symbolic::MultiExpression& new_shape) const;

    /**
     * Broadcast this layout to a bigger shape with strides correctly set.
     */
    std::unique_ptr<TensorLayout> broadcast(const symbolic::MultiExpression& ref_shape) const;

    static std::ostream& emit_symbolic_list(std::ostream& stream, const symbolic::MultiExpression& list);

    static types::PrimitiveType get_tensor_indvar_type_for_shape(const std::vector<symbolic::Expression>& shape);

    enum TensorLayoutType { LAYOUT_OTHER = 0, LAYOUT_ROW_MAJOR = 1, LAYOUT_COL_MAJOR = 2, LAYOUT_ROW_OR_COL_MAJOR = 3 };
    /**
     * Checks if the layout is 2D and either column-major or row-major. A layout is considered column-major if the
     * stride of the first dimension is 1, and row-major if the stride of the second dimension is 1. If neither
     * condition is met, then it is not mappable to code that expects col/row major and the outer line size, like blas
     * GEMM or MMA cores
     * @return LAYOUT_OTHER(0), LAYOUT_ROW_MAJOR(1), LAYOUT_COL_MAJOR(-1)
     */
    TensorLayoutType is_2d_col_or_row_major() const;

    /**
     * Whatever the dimesionality of the entire Tensor, if it has at least 2 dimensions, and the innermost has a stride
     * of "1" Useful to check of certain (hardware) APIs that only support 1 stride parameter and implicitly expect the
     * other stride to be 1 can be used with this Tensor
     */
    bool is_last_dims_expressible_as_row_major() const;

    /**
     * Checks if the given strides can be looked at as col-major or row-major. A layout is considered column-major if
     * the stride of the first dimension is 1, and row-major if the stride of the second dimension is 1. If neither
     * condition is met, then it is not mappable to code that expects col/row major and the outer line size, like blas
     * GEMM or MMA cores Only considers the last 2 dimensions
     * @param strides at least the innermost 2 strides
     * @return
     */
    static TensorLayoutType is_last_dims_col_or_row_major(const symbolic::MultiExpression& strides);
};

std::ostream& operator<<(std::ostream& stream, const TensorLayout& layout);


} // namespace sdfg::math::tensor
