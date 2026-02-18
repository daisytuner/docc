#pragma once

#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/library_nodes/math/math.h>
#include <sdfg/data_flow/tasklet.h>
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
    sdfg::builder::StructuredSDFGBuilder builder_;
    std::vector<Scope> scope_stack;

    sdfg::structured_control_flow::Sequence& current_sequence();

public:
    PyStructuredSDFGBuilder(const std::string& name);
    PyStructuredSDFGBuilder(const std::string& name, const sdfg::types::IType& return_type);

    sdfg::builder::StructuredSDFGBuilder& builder() { return builder_; }

    PyStructuredSDFG move();

    /***** Containers *****/

    void add_container(const std::string& name, const sdfg::types::IType& type, bool is_argument);

    void add_structure(const std::string& name, const std::vector<const sdfg::types::IType*>& member_types);

    bool exists(const std::string& name);

    void set_return_type(const sdfg::types::IType& type);

    std::string get_sizeof(const sdfg::types::IType& type);

    std::string find_new_name(const std::string& prefix = "tmp_");

    /***** Control Flow *****/

    void add_return(const std::string& data, const sdfg::DebugInfo& debug_info = sdfg::DebugInfo());

    void add_constant_return(
        const std::string& value, const sdfg::types::IType& type, const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    void begin_if(const std::string& condition, const sdfg::DebugInfo& debug_info = sdfg::DebugInfo());

    void begin_else(const sdfg::DebugInfo& debug_info = sdfg::DebugInfo());

    void end_if();

    void begin_while(const sdfg::DebugInfo& debug_info = sdfg::DebugInfo());

    void add_break(const sdfg::DebugInfo& debug_info = sdfg::DebugInfo());

    void add_continue(const sdfg::DebugInfo& debug_info = sdfg::DebugInfo());

    void end_while();

    void begin_for(
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

    size_t add_block(const sdfg::DebugInfo& debug_info = sdfg::DebugInfo());

    size_t add_access(size_t block_ptr, const std::string& name, const sdfg::DebugInfo& debug_info = sdfg::DebugInfo());

    size_t add_constant(
        size_t block_ptr,
        const std::string& value,
        const sdfg::types::IType& type,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    size_t add_tasklet(
        size_t block_ptr,
        sdfg::data_flow::TaskletCode code,
        const std::vector<std::string>& inputs,
        const std::vector<std::string>& outputs,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    void add_memlet(
        size_t block_ptr,
        size_t src_ptr,
        const std::string& src_conn,
        size_t dst_ptr,
        const std::string& dst_conn,
        const std::string& subset = "",
        const sdfg::types::IType* type = nullptr,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    void add_reference_memlet(
        size_t block_ptr,
        size_t src_ptr,
        size_t dst_ptr,
        const std::string& subset = "",
        const sdfg::types::IType* type = nullptr,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    /***** Library Nodes *****/

    size_t add_cmath(
        size_t block_ptr, sdfg::math::cmath::CMathFunction func, const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    size_t add_malloc(size_t block_ptr, const std::string& size, const sdfg::DebugInfo& debug_info = sdfg::DebugInfo());

    size_t add_memset(
        size_t block_ptr,
        const std::string& value,
        const std::string& num,
        const sdfg::DebugInfo& debug_info = sdfg::DebugInfo()
    );

    size_t add_memcpy(size_t block_ptr, const std::string& count, const sdfg::DebugInfo& debug_info = sdfg::DebugInfo());

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
