#pragma once

#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/data_flow/library_nodes/math/math.h>
#include <sdfg/data_flow/tasklet.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/structured_control_flow/control_flow_node.h>
#include <sdfg/structured_control_flow/for.h>
#include <sdfg/structured_control_flow/if_else.h>
#include <sdfg/structured_control_flow/sequence.h>
#include <sdfg/structured_control_flow/while.h>
#include <stack>
#include <vector>

#include "py_structured_sdfg.h"
#include "types/py_types.h"

struct Scope {
    sdfg::structured_control_flow::Sequence* sequence;
    sdfg::structured_control_flow::ControlFlowNode* node;
    int branch_index;
};

class PyStructuredSDFGBuilder {
private:
    sdfg::plugins::Context& docc_context_;
    sdfg::builder::StructuredSDFGBuilder builder_;
    std::vector<Scope> scope_stack;

    sdfg::structured_control_flow::Sequence& current_sequence();

public:
    PyStructuredSDFGBuilder(sdfg::plugins::Context& ctx, const std::string& name);
    PyStructuredSDFGBuilder(sdfg::plugins::Context& ctx, const std::string& name, const sdfg::types::IType& return_type);
    PyStructuredSDFGBuilder(PyStructuredSDFG& sdfg);

    sdfg::builder::StructuredSDFGBuilder& builder() { return builder_; }

    sdfg::plugins::Context& docc_context() const;

    PyStructuredSDFG move();

    /***** Containers *****/

    void add_container(const std::string& name, const sdfg::types::IType& type, bool is_argument);

    void add_structure(const std::string& name, const std::vector<const sdfg::types::IType*>& member_types);

    bool exists(const std::string& name);

    void set_return_type(const sdfg::types::IType& type);

    std::string get_sizeof(const sdfg::types::IType& type);

    std::string find_new_name(const std::string& prefix = "tmp_");

    void add_assumption_lb(const std::string& symbol, const std::string& bound);

    void add_assumption_ub(const std::string& symbol, const std::string& bound);

    void add_assumption_const(const std::string& symbol, bool constant);

    /***** Control Flow *****/

    void add_return(const std::string& data, const sdfg::DebugInfo& debug_info = sdfg::DebugInfo());

    void add_constant_return(
        const std::string& value, const sdfg::types::IType& type, const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    sdfg::structured_control_flow::IfElse&
    begin_if(const std::string& condition, const sdfg::DebugInfo& debug_info = sdfg::DebugInfo());

    void begin_else(const sdfg::DebugInfo& debug_info = sdfg::DebugInfo());

    void end_if();

    sdfg::structured_control_flow::While& begin_while(const sdfg::DebugInfo& debug_info = sdfg::DebugInfo());

    void add_break(const sdfg::DebugInfo& debug_info = sdfg::DebugInfo());

    void add_continue(const sdfg::DebugInfo& debug_info = sdfg::DebugInfo());

    void end_while();

    sdfg::structured_control_flow::For& begin_for(
        const std::string& var,
        const std::string& start,
        const std::string& end,
        const std::string& step,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    void end_for();

    void add_transition(
        const std::string& lhs, const std::string& rhs, const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    void add_assignment(
        const std::string& target, const std::string& value, const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    /***** Dataflow *****/

    sdfg::structured_control_flow::Block& add_block(const sdfg::DebugInfo& debug_info = sdfg::DebugInfo());

    sdfg::data_flow::AccessNode& add_access(
        sdfg::structured_control_flow::Block& block,
        const std::string& name,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    sdfg::data_flow::ConstantNode& add_constant(
        sdfg::structured_control_flow::Block& block,
        const std::string& value,
        const sdfg::types::IType& type,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    sdfg::data_flow::Tasklet& add_tasklet(
        sdfg::structured_control_flow::Block& block,
        sdfg::data_flow::TaskletCode code,
        const std::vector<std::string>& inputs,
        const std::vector<std::string>& outputs,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    void add_memlet(
        sdfg::structured_control_flow::Block& block,
        sdfg::data_flow::DataFlowNode& src,
        const std::string& src_conn,
        sdfg::data_flow::DataFlowNode& dst,
        const std::string& dst_conn,
        const std::string& subset = "",
        const sdfg::types::IType* type = nullptr,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    void add_reference_memlet(
        sdfg::structured_control_flow::Block& block,
        sdfg::data_flow::AccessNode& src,
        sdfg::data_flow::AccessNode& dst,
        const std::string& subset = "",
        const sdfg::types::IType* type = nullptr,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    /***** Library Nodes *****/

    sdfg::data_flow::LibraryNode& add_cmath(
        sdfg::structured_control_flow::Block& block,
        sdfg::math::cmath::CMathFunction func,
        sdfg::types::PrimitiveType primitive_type,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    sdfg::data_flow::LibraryNode& add_malloc(
        sdfg::structured_control_flow::Block& block,
        const std::string& size,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    sdfg::data_flow::LibraryNode& add_memset(
        sdfg::structured_control_flow::Block& block,
        const std::string& value,
        const std::string& num,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    sdfg::data_flow::LibraryNode& add_memcpy(
        sdfg::structured_control_flow::Block& block,
        const std::string& count,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    sdfg::data_flow::LibraryNode&
    add_free(sdfg::structured_control_flow::Block& block, const sdfg::DebugInfo& debug_info = sdfg::DebugInfo());

    /**
     * @brief Check if a size expression only depends on function arguments (hoistable to function entry)
     * @param size_expr Size expression string to check
     * @return true if all symbols in the expression are function arguments
     */
    bool is_hoistable_size(const std::string& size_expr);

    /**
     * @brief Insert a block at the very beginning of the root sequence
     * @param debug_info Optional debug info
     * @return Reference to the newly created block
     */
    sdfg::structured_control_flow::Block& insert_block_at_root_start(const sdfg::DebugInfo& debug_info = sdfg::DebugInfo());

    void add_gemm(
        const std::string& A,
        const std::string& B,
        const std::string& C,
        const std::string& alpha,
        const std::string& beta,
        const std::string& m,
        const std::string& n,
        const std::string& k,
        bool trans_a,
        bool trans_b,
        const std::vector<std::string>& a_subset,
        const std::vector<std::string>& b_subset,
        const std::vector<std::string>& c_subset,
        const std::string& lda = "",
        const std::string& ldb = "",
        const std::string& ldc = "",
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    void add_dot(
        const std::string& X,
        const std::string& Y,
        const std::string& result,
        const std::string& n,
        const std::string& incx,
        const std::string& incy,
        const std::vector<std::string>& x_subset,
        const std::vector<std::string>& y_subset,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    void add_elementwise_op(
        const std::string& op_type,
        const std::string& A,
        const sdfg::types::Tensor& A_type,
        const std::string& B,
        const sdfg::types::Tensor& B_type,
        const std::string& C,
        const sdfg::types::Tensor& C_type,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    void add_elementwise_unary_op(
        const std::string& op_type,
        const std::string& A,
        const sdfg::types::Tensor& A_type,
        const std::string& C,
        const sdfg::types::Tensor& C_type,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    void add_cast_op(
        const std::string& A,
        const sdfg::types::Tensor& A_type,
        const std::string& C,
        const sdfg::types::Tensor& C_type,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    void add_reduce_op(
        const std::string& op_type,
        const std::string& input,
        const sdfg::types::Tensor& input_type,
        const std::string& output,
        const sdfg::types::Tensor& output_type,
        const std::vector<int64_t>& axes,
        bool keepdims,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    void add_einsum(
        const std::vector<std::string>& inputs,
        const std::string& output,
        const std::vector<std::tuple<std::string, std::string, std::string>>& dims,
        const std::vector<std::string>& out_indices,
        const std::vector<std::vector<std::string>>& in_indices,
        const std::vector<const sdfg::types::Tensor*>& input_types,
        const sdfg::types::Tensor& output_type,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    void add_conv(
        const std::string& X,
        const std::string& W,
        const std::string& Y,
        const std::vector<std::string>& shape,
        const std::vector<std::string>& kernel_shape,
        const std::vector<std::string>& strides,
        const std::vector<std::string>& pads,
        const std::vector<std::string>& dilations,
        const std::string& output_channels,
        const std::string& group,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );
};
