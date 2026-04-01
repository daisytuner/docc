#pragma once

#include <sdfg/builder/structured_sdfg_builder.h>

#include <sdfg/data_flow/library_nodes/math/cmath/cmath_node.h>

using namespace sdfg;

inline std::unique_ptr<StructuredSDFG> correlation() {
    /***
    for (i = 0; i < _PB_M-1; i++)
    {
        C[i][i] = SCALAR_VAL(1.0);
        for (j = i+1; j < _PB_M; j++)
        {
            C[i][j] = SCALAR_VAL(0.0);
            for (k = 0; k < _PB_N; k++)
                C[i][j] += (D[k][i] * D[k][j]);
            C[j][i] = C[i][j];
        }
    }
    ***/

    /***

    ------>

    for (i = 0; i < _PB_M-1; i++)
    {
        C[i][i] = SCALAR_VAL(1.0);
    }

    for (i=0; i < _PB_M-1; i++)
        for (j = i+1; j < _PB_M; j++)
        {
            C[i][j] = SCALAR_VAL(0.0);
        }
    }

        for (i=0; i < _PB_M-1; i++)
            for (j = i+1; j < _PB_M; j++)
                for (k = 0; k < _PB_N; k++)
                {
                    C[i][j] += (D[k][i] * D[k][j]);
                }
            }
        }

    for (i=0; i < _PB_M-1; i++)
        for (j = i+1; j < _PB_M; j++)
        {
            C[j][i] = C[i][j];
        }
    }
    ***/

    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_symbols(types::PrimitiveType::UInt64);
    builder.add_container("M", desc_symbols, true);
    builder.add_container("N", desc_symbols, true);
    builder.add_container("i", desc_symbols);
    builder.add_container("j", desc_symbols);
    builder.add_container("k", desc_symbols);

    types::Scalar desc_element(types::PrimitiveType::Double);
    builder.add_container("tmp", desc_element);

    types::Array desc_1d(desc_element, symbolic::symbol("M"));
    types::Pointer desc_2d(desc_1d);
    builder.add_container("C", desc_2d, true);
    builder.add_container("D", desc_2d, true);

    auto& root = builder.subject().root();

    auto& loop_i = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::sub(symbolic::symbol("M"), symbolic::integer(1))),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body_i = loop_i.root();
    {
        auto& block = builder.add_block(body_i);
        auto& output_node = builder.add_access(block, "C");
        auto& one_node = builder.add_constant(block, "1.0", desc_element);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, one_node, tasklet, "_in", {});
        builder
            .add_computational_memlet(block, tasklet, "_out", output_node, {symbolic::symbol("i"), symbolic::symbol("i")});
    }

    auto& loop_j = builder.add_for(
        body_i,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );
    auto& body_j = loop_j.root();
    {
        auto& block = builder.add_block(body_j);
        auto& output_node = builder.add_access(block, "C");
        auto& one_node = builder.add_constant(block, "1.0", desc_element);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, one_node, tasklet, "_in", {});
        builder
            .add_computational_memlet(block, tasklet, "_out", output_node, {symbolic::symbol("i"), symbolic::symbol("j")});
    }

    auto& loop_k = builder.add_for(
        body_j,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
    );
    auto& body_k = loop_k.root();
    {
        auto& block = builder.add_block(body_k);
        auto& tmp_node = builder.add_access(block, "tmp");
        auto& D_node = builder.add_access(block, "D");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, D_node, tasklet, "_in1", {symbolic::symbol("k"), symbolic::symbol("i")});
        builder.add_computational_memlet(block, D_node, tasklet, "_in2", {symbolic::symbol("k"), symbolic::symbol("j")});
        builder.add_computational_memlet(block, tasklet, "_out", tmp_node, {});

        auto& C_in_node = builder.add_access(block, "C");
        auto& C_out_node = builder.add_access(block, "C");
        auto& tasklet_2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder
            .add_computational_memlet(block, C_in_node, tasklet_2, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")});
        builder.add_computational_memlet(block, tmp_node, tasklet_2, "_in2", {});
        builder.add_computational_memlet(
            block, tasklet_2, "_out", C_out_node, {symbolic::symbol("i"), symbolic::symbol("j")}
        );
    }
    {
        auto& block = builder.add_block(body_j);
        auto& C_in_node = builder.add_access(block, "C");
        auto& C_out_node = builder.add_access(block, "C");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder
            .add_computational_memlet(block, C_in_node, tasklet, "_in", {symbolic::symbol("i"), symbolic::symbol("j")});
        builder
            .add_computational_memlet(block, tasklet, "_out", C_out_node, {symbolic::symbol("j"), symbolic::symbol("i")});
    }

    return builder.move();
};

inline std::unique_ptr<StructuredSDFG> covariance() {
    /***
    for (i = 0; i < _PB_M; i++)
    {
        for (j = i; j < _PB_M; j++)
        {
            C[i][j] = SCALAR_VAL(0.0);
            for (k = 0; k < _PB_N; k++)
                C[i][j] += (D[k][i] * D[k][j]);
            C[j][i] = C[i][j];
        }
    }
    ***/

    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_symbols(types::PrimitiveType::UInt64);
    builder.add_container("M", desc_symbols, true);
    builder.add_container("N", desc_symbols, true);
    builder.add_container("i", desc_symbols);
    builder.add_container("j", desc_symbols);
    builder.add_container("k", desc_symbols);

    types::Scalar desc_element(types::PrimitiveType::Double);
    builder.add_container("tmp", desc_element);

    types::Array desc_1d(desc_element, symbolic::symbol("M"));
    types::Pointer desc_2d(desc_1d);
    builder.add_container("C", desc_2d, true);
    builder.add_container("D", desc_2d, true);

    auto& root = builder.subject().root();

    auto& loop_i = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body_i = loop_i.root();

    auto& loop_j = builder.add_for(
        body_i,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::symbol("i"),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );
    auto& body_j = loop_j.root();
    {
        auto& block = builder.add_block(body_j);
        auto& output_node = builder.add_access(block, "C");
        auto& one_node = builder.add_constant(block, "1.0", desc_element);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, one_node, tasklet, "_in", {});
        builder
            .add_computational_memlet(block, tasklet, "_out", output_node, {symbolic::symbol("i"), symbolic::symbol("j")});
    }

    auto& loop_k = builder.add_for(
        body_j,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
    );
    auto& body_k = loop_k.root();
    {
        auto& block = builder.add_block(body_k);
        auto& tmp_node = builder.add_access(block, "tmp");
        auto& D_node = builder.add_access(block, "D");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, D_node, tasklet, "_in1", {symbolic::symbol("k"), symbolic::symbol("i")});
        builder.add_computational_memlet(block, D_node, tasklet, "_in2", {symbolic::symbol("k"), symbolic::symbol("j")});
        builder.add_computational_memlet(block, tasklet, "_out", tmp_node, {});

        auto& C_in_node = builder.add_access(block, "C");
        auto& C_out_node = builder.add_access(block, "C");
        auto& tasklet_2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder
            .add_computational_memlet(block, C_in_node, tasklet_2, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")});
        builder.add_computational_memlet(block, tmp_node, tasklet_2, "_in2", {});
        builder.add_computational_memlet(
            block, tasklet_2, "_out", C_out_node, {symbolic::symbol("i"), symbolic::symbol("j")}
        );
    }

    {
        auto& block = builder.add_block(body_j);
        auto& C_in_node = builder.add_access(block, "C");
        auto& C_out_node = builder.add_access(block, "C");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder
            .add_computational_memlet(block, C_in_node, tasklet, "_in", {symbolic::symbol("i"), symbolic::symbol("j")});
        builder
            .add_computational_memlet(block, tasklet, "_out", C_out_node, {symbolic::symbol("j"), symbolic::symbol("i")});
    }

    return builder.move();
};

inline std::unique_ptr<StructuredSDFG> gemm() {
    /***
     for (i = 0; i < _PB_N; i++) {
        for (j = 0; j < _PB_M; j++)
            C[i][j] *= beta;
        for (k = 0; k < _PB_K; k++) {
            for (j = 0; j < _PB_M; j++)
                C[i][j] += alpha * A[i][k] * B[k][j];
        }
    }
    ***/
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_element(types::PrimitiveType::Double);
    builder.add_container("alpha", desc_element, true);
    builder.add_container("beta", desc_element, true);
    builder.add_container("tmp", desc_element);

    types::Array desc_1d(desc_element, symbolic::symbol("M"));
    types::Pointer desc_2d(desc_1d);
    builder.add_container("A", desc_2d, true);
    builder.add_container("B", desc_2d, true);
    builder.add_container("C", desc_2d, true);

    types::Scalar desc_symbols(types::PrimitiveType::UInt64);
    builder.add_container("N", desc_symbols, true);
    builder.add_container("M", desc_symbols, true);
    builder.add_container("K", desc_symbols, true);
    builder.add_container("i", desc_symbols);
    builder.add_container("j_1", desc_symbols);
    builder.add_container("j_2", desc_symbols);
    builder.add_container("k", desc_symbols);

    auto& root = builder.subject().root();

    auto& loop_i = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body_i = loop_i.root();
    auto& loop_j_1 = builder.add_for(
        body_i,
        symbolic::symbol("j_1"),
        symbolic::Lt(symbolic::symbol("j_1"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j_1"), symbolic::integer(1))
    );
    auto& body_j = loop_j_1.root();
    {
        auto& block = builder.add_block(body_j);
        auto& C_in_node = builder.add_access(block, "C");
        auto& C_out_node = builder.add_access(block, "C");
        auto& beta_node = builder.add_access(block, "beta");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder
            .add_computational_memlet(block, C_in_node, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j_1")});
        builder.add_computational_memlet(block, beta_node, tasklet, "_in2", {});
        builder.add_computational_memlet(
            block, tasklet, "_out", C_out_node, {symbolic::symbol("i"), symbolic::symbol("j_1")}
        );
    }

    auto& loop_k = builder.add_for(
        body_i,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("K")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
    );
    auto& body_k = loop_k.root();
    auto& loop_j_2 = builder.add_for(
        body_k,
        symbolic::symbol("j_2"),
        symbolic::Lt(symbolic::symbol("j_2"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j_2"), symbolic::integer(1))
    );
    auto& body_j_2 = loop_j_2.root();
    {
        auto& block = builder.add_block(body_j_2);
        auto& A_node = builder.add_access(block, "A");
        auto& B_node = builder.add_access(block, "B");
        auto& tmp_node = builder.add_access(block, "tmp");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, A_node, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("k")});
        builder
            .add_computational_memlet(block, B_node, tasklet, "_in2", {symbolic::symbol("k"), symbolic::symbol("j_2")});
        builder.add_computational_memlet(block, tasklet, "_out", tmp_node, {});

        auto& tmp_node_2 = builder.add_access(block, "tmp");
        auto& alpha_node = builder.add_access(block, "alpha");
        auto& tasklet_2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, tmp_node, tasklet_2, "_in1", {});
        builder.add_computational_memlet(block, alpha_node, tasklet_2, "_in2", {});
        builder.add_computational_memlet(block, tasklet_2, "_out", tmp_node_2, {});

        auto& C_in_node = builder.add_access(block, "C");
        auto& C_out_node = builder.add_access(block, "C");
        auto& tasklet_3 = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(
            block, C_in_node, tasklet_3, "_in1", {symbolic::symbol("i"), symbolic::symbol("j_2")}
        );
        builder.add_computational_memlet(block, tmp_node_2, tasklet_3, "_in2", {});
        builder.add_computational_memlet(
            block, tasklet_3, "_out", C_out_node, {symbolic::symbol("i"), symbolic::symbol("j_2")}
        );
    }

    return builder.move();
};

inline std::unique_ptr<StructuredSDFG> symm() {
    /***
    for (i = 0; i < _PB_M; i++)
      for (j = 0; j < _PB_N; j++ ) {
        temp2 = 0;
        for (k = 0; k < i; k++) {
           C[k][j] += B[i][j] * A[i][k];
           temp2 += B[k][j] * A[i][k];
        }
        C[i][j] = C[i][j] + temp2;
     }
    ***/

    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_symbols(types::PrimitiveType::UInt64);
    builder.add_container("i", desc_symbols);
    builder.add_container("j", desc_symbols);
    builder.add_container("k", desc_symbols);

    types::Scalar desc_element(types::PrimitiveType::Double);
    builder.add_container("tmp", desc_element);
    builder.add_container("tmp2", desc_element);
    builder.add_container("tmp3", desc_element);

    types::Array desc_1d(desc_element, symbolic::symbol("M"));
    types::Pointer desc_2d(desc_1d);
    builder.add_container("A", desc_2d, true);
    builder.add_container("B", desc_2d, true);
    builder.add_container("C", desc_2d, true);

    auto& root = builder.subject().root();

    auto& loop_i = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(16)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body_i = loop_i.root();
    auto& loop_j = builder.add_for(
        body_i,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::integer(12)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );
    auto& body_j = loop_j.root();
    {
        auto& block = builder.add_block(body_j);
        auto& temp2_node = builder.add_access(block, "tmp");
        auto& const_node = builder.add_constant(block, "0.0", desc_element);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, const_node, tasklet, "_in", {});
        builder.add_computational_memlet(block, tasklet, "_out", temp2_node, {});
    }
    auto& loop_k = builder.add_for(
        body_j,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("i")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
    );
    {
        auto& block = builder.add_block(loop_k.root());
        auto& a_node = builder.add_access(block, "A");
        auto& b_node = builder.add_access(block, "B");
        auto& tmp_node = builder.add_access(block, "tmp2");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, a_node, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("k")});
        builder.add_computational_memlet(block, b_node, tasklet, "_in2", {symbolic::symbol("i"), symbolic::symbol("j")});
        builder.add_computational_memlet(block, tasklet, "_out", tmp_node, {});

        auto& c_in_node = builder.add_access(block, "C");
        auto& c_node = builder.add_access(block, "C");
        auto& tasklet_2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, tmp_node, tasklet_2, "_in1", {});
        builder
            .add_computational_memlet(block, c_in_node, tasklet_2, "_in2", {symbolic::symbol("k"), symbolic::symbol("j")});
        builder
            .add_computational_memlet(block, tasklet_2, "_out", c_node, {symbolic::symbol("k"), symbolic::symbol("j")});

        auto& block2 = builder.add_block(loop_k.root());
        auto& a2_node = builder.add_access(block2, "A");
        auto& b2_node = builder.add_access(block2, "B");
        auto& tmp3_node = builder.add_access(block2, "tmp3");
        auto& tasklet_3 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder
            .add_computational_memlet(block2, b2_node, tasklet_3, "_in1", {symbolic::symbol("k"), symbolic::symbol("j")});
        builder
            .add_computational_memlet(block2, a2_node, tasklet_3, "_in2", {symbolic::symbol("i"), symbolic::symbol("k")});
        builder.add_computational_memlet(block2, tasklet_3, "_out", tmp3_node, {});

        auto& temp2_node = builder.add_access(block2, "tmp");
        auto& temp2_node_2 = builder.add_access(block2, "tmp");
        auto& tasklet_4 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block2, temp2_node, tasklet_4, "_in1", {});
        builder.add_computational_memlet(block2, tmp3_node, tasklet_4, "_in2", {});
        builder.add_computational_memlet(block2, tasklet_4, "_out", temp2_node_2, {});
    }
    {
        auto& block = builder.add_block(body_j);
        auto& c_in_node = builder.add_access(block, "C");
        auto& c_out_node = builder.add_access(block, "C");
        auto& tmp_node = builder.add_access(block, "tmp");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder
            .add_computational_memlet(block, c_in_node, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")});
        builder.add_computational_memlet(block, tmp_node, tasklet, "_in2", {});
        builder
            .add_computational_memlet(block, tasklet, "_out", c_out_node, {symbolic::symbol("i"), symbolic::symbol("j")});
    }

    return builder.move();
}

inline std::unique_ptr<StructuredSDFG> gemver() {
    /***
    for (i = 0; i < _PB_N; i++)
        for (j = 0; j < _PB_N; j++)
            A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

    for (i = 0; i < _PB_N; i++)
        for (j = 0; j < _PB_N; j++)
        x[i] = x[i] + A[j][i] * y[j];

    for (i = 0; i < _PB_N; i++)
        x[i] = x[i] + z[i];

    for (i = 0; i < _PB_N; i++)
        for (j = 0; j < _PB_N; j++)
            w[i] = w[i] + A[i][j] * x[j];
    */

    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_symbols(types::PrimitiveType::UInt64);
    builder.add_container("i_1", desc_symbols);
    builder.add_container("i_2", desc_symbols);
    builder.add_container("i_3", desc_symbols);
    builder.add_container("i_4", desc_symbols);
    builder.add_container("j_1", desc_symbols);
    builder.add_container("j_2", desc_symbols);
    builder.add_container("j_3", desc_symbols);
    builder.add_container("j_4", desc_symbols);
    builder.add_container("N", desc_symbols, true);

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Array desc_1d(desc_element, symbolic::symbol("M"));
    types::Pointer desc_2d(desc_1d);
    builder.add_container("A", desc_2d, true);
    builder.add_container("u1", desc_1d, true);
    builder.add_container("u2", desc_1d, true);
    builder.add_container("v1", desc_1d, true);
    builder.add_container("v2", desc_1d, true);
    builder.add_container("x", desc_1d, true);
    builder.add_container("y", desc_1d, true);
    builder.add_container("z", desc_1d, true);
    builder.add_container("w", desc_1d, true);

    auto& root = builder.subject().root();

    {
        auto& loop_i_1 = builder.add_for(
            root,
            symbolic::symbol("i_1"),
            symbolic::Lt(symbolic::symbol("i_1"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i_1"), symbolic::integer(1))
        );
        auto& body_i_1 = loop_i_1.root();
        auto& loop_j_1 = builder.add_for(
            body_i_1,
            symbolic::symbol("j_1"),
            symbolic::Lt(symbolic::symbol("j_1"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j_1"), symbolic::integer(1))
        );
        auto& body_j_1 = loop_j_1.root();

        builder.add_container("tmp_1", desc_element);

        auto& block = builder.add_block(body_j_1);
        auto& u1_node = builder.add_access(block, "u1");
        auto& v1_node = builder.add_access(block, "v1");
        auto& tmp_node = builder.add_access(block, "tmp_1");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, u1_node, tasklet, "_in1", {symbolic::symbol("i_1")});
        builder.add_computational_memlet(block, v1_node, tasklet, "_in2", {symbolic::symbol("j_1")});
        builder.add_computational_memlet(block, tasklet, "_out", tmp_node, {});

        builder.add_container("tmp_2", desc_element);

        auto& block2 = builder.add_block(body_j_1);
        auto& u2_node = builder.add_access(block2, "u2");
        auto& v2_node = builder.add_access(block2, "v2");
        auto& tmp2_node = builder.add_access(block2, "tmp_2");
        auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block2, u2_node, tasklet2, "_in1", {symbolic::symbol("i_1")});
        builder.add_computational_memlet(block2, v2_node, tasklet2, "_in2", {symbolic::symbol("j_1")});
        builder.add_computational_memlet(block2, tasklet2, "_out", tmp2_node, {});

        builder.add_container("tmp_3", desc_element);

        auto& block3 = builder.add_block(body_j_1);
        auto& tmp_node_1 = builder.add_access(block3, "tmp_1");
        auto& tmp2_node_1 = builder.add_access(block3, "tmp_2");
        auto& tmp3_node = builder.add_access(block3, "tmp_3");
        auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block3, tmp_node_1, tasklet3, "_in1", {});
        builder.add_computational_memlet(block3, tmp2_node_1, tasklet3, "_in2", {});
        builder.add_computational_memlet(block3, tasklet3, "_out", tmp3_node, {});

        auto& A_node = builder.add_access(block3, "A");
        auto& A_node_out = builder.add_access(block3, "A");
        auto& tasklet4 = builder.add_tasklet(block3, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(
            block3, A_node, tasklet4, "_in1", {symbolic::symbol("i_1"), symbolic::symbol("j_1")}
        );
        builder.add_computational_memlet(block3, tmp3_node, tasklet4, "_in2", {});
        builder.add_computational_memlet(
            block3, tasklet4, "_out", A_node_out, {symbolic::symbol("i_1"), symbolic::symbol("j_1")}
        );
    }

    {
        auto& loop_i_2 = builder.add_for(
            root,
            symbolic::symbol("i_2"),
            symbolic::Lt(symbolic::symbol("i_2"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i_2"), symbolic::integer(1))
        );
        auto& body_i_2 = loop_i_2.root();
        auto& loop_j_2 = builder.add_for(
            body_i_2,
            symbolic::symbol("j_2"),
            symbolic::Lt(symbolic::symbol("j_2"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j_2"), symbolic::integer(1))
        );
        auto& body_j_2 = loop_j_2.root();

        auto& block = builder.add_block(body_j_2);
        auto& x_node_in = builder.add_access(block, "x");
        auto& x_node_out = builder.add_access(block, "x");
        auto& y_node = builder.add_access(block, "y");
        auto& A_node = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
        builder.add_computational_memlet(block, x_node_in, tasklet, "_in3", {symbolic::symbol("i_2")});
        builder
            .add_computational_memlet(block, A_node, tasklet, "_in1", {symbolic::symbol("j_2"), symbolic::symbol("i_2")});
        builder.add_computational_memlet(block, y_node, tasklet, "_in2", {symbolic::symbol("j_2")});
        builder.add_computational_memlet(block, tasklet, "_out", x_node_out, {symbolic::symbol("i_2")});
    }

    {
        auto& loop_i_3 = builder.add_for(
            root,
            symbolic::symbol("i_3"),
            symbolic::Lt(symbolic::symbol("i_3"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i_3"), symbolic::integer(1))
        );
        auto& body_i_3 = loop_i_3.root();

        auto& block = builder.add_block(body_i_3);
        auto& x_node_in = builder.add_access(block, "x");
        auto& x_node_out = builder.add_access(block, "x");
        auto& z_node = builder.add_access(block, "z");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, x_node_in, tasklet, "_in1", {symbolic::symbol("i_3")});
        builder.add_computational_memlet(block, z_node, tasklet, "_in2", {symbolic::symbol("i_3")});
        builder.add_computational_memlet(block, tasklet, "_out", x_node_out, {symbolic::symbol("i_3")});
    }

    {
        auto& loop_i_4 = builder.add_for(
            root,
            symbolic::symbol("i_4"),
            symbolic::Lt(symbolic::symbol("i_4"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i_4"), symbolic::integer(1))
        );
        auto& body_i_4 = loop_i_4.root();
        auto& loop_j_4 = builder.add_for(
            body_i_4,
            symbolic::symbol("j_4"),
            symbolic::Lt(symbolic::symbol("j_4"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j_4"), symbolic::integer(1))
        );
        auto& body_j_4 = loop_j_4.root();

        auto& block = builder.add_block(body_j_4);

        auto& w_node_in = builder.add_access(block, "w");
        auto& w_node_out = builder.add_access(block, "w");
        auto& A_node = builder.add_access(block, "A");
        auto& x_node = builder.add_access(block, "x");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
        builder.add_computational_memlet(block, w_node_in, tasklet, "_in3", {symbolic::symbol("i_4")});
        builder
            .add_computational_memlet(block, A_node, tasklet, "_in1", {symbolic::symbol("i_4"), symbolic::symbol("j_4")});
        builder.add_computational_memlet(block, x_node, tasklet, "_in2", {symbolic::symbol("j_4")});
        builder.add_computational_memlet(block, tasklet, "_out", w_node_out, {symbolic::symbol("i_4")});
    }

    return builder.move();
};

inline std::unique_ptr<StructuredSDFG> gesummv() {
    /***
    for (i = 0; i < _PB_N; i++) {
        tmp[i] = SCALAR_VAL(0.0);
        y[i] = SCALAR_VAL(0.0);
        for (j = 0; j < _PB_N; j++) {
            tmp[i] = A[i][j] * x[j] + tmp[i];
            y[i] = B[i][j] * x[j] + y[i];
            }
        y[i] = tmp[i] + y[i];
    }
    */

    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_symbols(types::PrimitiveType::UInt64);
    builder.add_container("i", desc_symbols);
    builder.add_container("j", desc_symbols);
    builder.add_container("N", desc_symbols, true);

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Array desc_1d(desc_element, symbolic::symbol("M"));
    types::Pointer desc_2d(desc_1d);
    builder.add_container("tmp", desc_1d, true);
    builder.add_container("x", desc_1d, true);
    builder.add_container("y", desc_1d, true);
    builder.add_container("A", desc_2d, true);
    builder.add_container("B", desc_2d, true);

    auto& root = builder.subject().root();

    auto& loop_i = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body_i = loop_i.root();

    {
        auto& block = builder.add_block(body_i);
        auto& tmp_node = builder.add_access(block, "tmp");
        auto& const_node = builder.add_constant(block, "0.0", desc_element);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, const_node, tasklet, "_in", {});
        builder.add_computational_memlet(block, tasklet, "_out", tmp_node, {symbolic::symbol("i")});
    }

    {
        auto& block = builder.add_block(body_i);
        auto& y_node = builder.add_access(block, "y");
        auto& const_node = builder.add_constant(block, "0.0", desc_element);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, const_node, tasklet, "_in", {});
        builder.add_computational_memlet(block, tasklet, "_out", y_node, {symbolic::symbol("i")});
    }

    {
        auto& loop_j = builder.add_for(
            body_i,
            symbolic::symbol("j"),
            symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
        );
        auto& body_j = loop_j.root();

        {
            auto& block = builder.add_block(body_j);
            auto& A_node = builder.add_access(block, "A");
            auto& x_node = builder.add_access(block, "x");
            auto& tmp_node = builder.add_access(block, "tmp");
            auto& tmp_node_out = builder.add_access(block, "tmp");
            auto& tasklet =
                builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
            builder
                .add_computational_memlet(block, A_node, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")});
            builder.add_computational_memlet(block, x_node, tasklet, "_in2", {symbolic::symbol("j")});
            builder.add_computational_memlet(block, tmp_node, tasklet, "_in3", {symbolic::symbol("i")});
            builder.add_computational_memlet(block, tasklet, "_out", tmp_node_out, {symbolic::symbol("i")});
        }

        {
            auto& block = builder.add_block(body_j);
            auto& B_node = builder.add_access(block, "B");
            auto& x_node = builder.add_access(block, "x");
            auto& y_node = builder.add_access(block, "y");
            auto& y_node_out = builder.add_access(block, "y");
            auto& tasklet =
                builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
            builder
                .add_computational_memlet(block, B_node, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")});
            builder.add_computational_memlet(block, x_node, tasklet, "_in2", {symbolic::symbol("j")});
            builder.add_computational_memlet(block, y_node, tasklet, "_in3", {symbolic::symbol("i")});
            builder.add_computational_memlet(block, tasklet, "_out", y_node_out, {symbolic::symbol("i")});
        }
    }

    {
        auto& block = builder.add_block(body_i);
        auto& tmp_node = builder.add_access(block, "tmp");
        auto& y_node = builder.add_access(block, "y");
        auto& y_node_out = builder.add_access(block, "y");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, tmp_node, tasklet, "_in1", {symbolic::symbol("i")});
        builder.add_computational_memlet(block, y_node, tasklet, "_in2", {symbolic::symbol("i")});
        builder.add_computational_memlet(block, tasklet, "_out", y_node_out, {symbolic::symbol("i")});
    }

    return builder.move();
};

inline std::unique_ptr<StructuredSDFG> syr2k() {
    /***
    for (i = 0; i < _PB_N; i++) {
        for (j = 0; j <= i; j++) {
            C[i][j] *= beta;
        }

        for (k = 0; k < _PB_M; k++) {
            for (j = 0; j <= i; j++) {
                C[i][j] += A[j][k]*B[i][k] + B[j][k]*A[i][k];
            }
        }
    }
    */

    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_symbols(types::PrimitiveType::UInt64);
    builder.add_container("M", desc_symbols, true);
    builder.add_container("N", desc_symbols, true);
    builder.add_container("i", desc_symbols);
    builder.add_container("j_1", desc_symbols);
    builder.add_container("j_2", desc_symbols);
    builder.add_container("k", desc_symbols);

    types::Scalar desc_element(types::PrimitiveType::Double);
    builder.add_container("tmp", desc_element);
    builder.add_container("tmp2", desc_element);
    builder.add_container("tmp3", desc_element);
    builder.add_container("beta", desc_element, true);

    types::Array desc_1d(desc_element, symbolic::symbol("M"));
    types::Pointer desc_2d(desc_1d);
    builder.add_container("A", desc_2d, true);
    builder.add_container("B", desc_2d, true);
    builder.add_container("C", desc_2d, true);

    auto& root = builder.subject().root();

    auto& loop_i = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body_i = loop_i.root();

    auto& loop_j_1 = builder.add_for(
        body_i,
        symbolic::symbol("j_1"),
        symbolic::Le(symbolic::symbol("j_1"), symbolic::symbol("i")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j_1"), symbolic::integer(1))
    );
    {
        auto& block = builder.add_block(loop_j_1.root());
        auto& C_in_node = builder.add_access(block, "C");
        auto& C_out_node = builder.add_access(block, "C");
        auto& beta_node = builder.add_access(block, "beta");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder
            .add_computational_memlet(block, C_in_node, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j_1")});
        builder.add_computational_memlet(block, beta_node, tasklet, "_in2", {});
        builder.add_computational_memlet(
            block, tasklet, "_out", C_out_node, {symbolic::symbol("i"), symbolic::symbol("j_1")}
        );
    }

    auto& loop_k = builder.add_for(
        body_i,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
    );
    {
        auto& loop_j_2 = builder.add_for(
            loop_k.root(),
            symbolic::symbol("j_2"),
            symbolic::Le(symbolic::symbol("j_2"), symbolic::symbol("i")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j_2"), symbolic::integer(1))
        );

        auto& block = builder.add_block(loop_j_2.root());
        auto& A_node = builder.add_access(block, "A");
        auto& B_node = builder.add_access(block, "B");
        auto& tmp_node = builder.add_access(block, "tmp");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder
            .add_computational_memlet(block, A_node, tasklet, "_in1", {symbolic::symbol("j_2"), symbolic::symbol("k")});
        builder.add_computational_memlet(block, B_node, tasklet, "_in2", {symbolic::symbol("i"), symbolic::symbol("k")});
        builder.add_computational_memlet(block, tasklet, "_out", tmp_node, {});

        auto& block2 = builder.add_block(loop_j_2.root());
        auto& tmp_node_2 = builder.add_access(block2, "tmp");
        auto& B_node_2 = builder.add_access(block2, "B");
        auto& A_node_2 = builder.add_access(block2, "A");
        auto& tmp2_node = builder.add_access(block2, "tmp2");
        auto& tasklet_2 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
        builder.add_computational_memlet(
            block2, B_node_2, tasklet_2, "_in1", {symbolic::symbol("j_2"), symbolic::symbol("k")}
        );
        builder
            .add_computational_memlet(block2, A_node_2, tasklet_2, "_in2", {symbolic::symbol("i"), symbolic::symbol("k")});
        builder.add_computational_memlet(block2, tmp_node_2, tasklet_2, "_in3", {});
        builder.add_computational_memlet(block2, tasklet_2, "_out", tmp2_node, {});

        auto& block3 = builder.add_block(loop_j_2.root());
        auto& C_in_node = builder.add_access(block3, "C");
        auto& C_out_node = builder.add_access(block3, "C");
        auto& tmp2_node_2 = builder.add_access(block3, "tmp2");
        auto& tasklet_3 = builder.add_tasklet(block3, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(
            block3, C_in_node, tasklet_3, "_in1", {symbolic::symbol("i"), symbolic::symbol("j_2")}
        );
        builder.add_computational_memlet(block3, tmp2_node_2, tasklet_3, "_in2", {});
        builder.add_computational_memlet(
            block3, tasklet_3, "_out", C_out_node, {symbolic::symbol("i"), symbolic::symbol("j_2")}
        );
    }

    return builder.move();
};

inline std::unique_ptr<StructuredSDFG> syrk() {
    /***
     for (i = 0; i < _PB_N; i++) {
        for (j = 0; j <= i; j++)
            C[i][j] *= beta;

        for (k = 0; k < _PB_M; k++) {
            for (j = 0; j <= i; j++)
                C[i][j] += A[i][k] * A[j][k];
        }
    }
    */

    builder::StructuredSDFGBuilder sdfg("sdfg_1", FunctionType_CPU);

    types::Scalar desc_symbols(types::PrimitiveType::UInt64);
    sdfg.add_container("M", desc_symbols, true);
    sdfg.add_container("N", desc_symbols, true);
    sdfg.add_container("i", desc_symbols);
    sdfg.add_container("j_1", desc_symbols);
    sdfg.add_container("j_2", desc_symbols);
    sdfg.add_container("k", desc_symbols);

    types::Scalar desc_element(types::PrimitiveType::Double);
    sdfg.add_container("beta", desc_element, true);
    sdfg.add_container("tmp", desc_element);

    types::Array desc_1d(desc_element, symbolic::symbol("M"));
    types::Pointer desc_2d(desc_1d);
    sdfg.add_container("A", desc_2d, true);
    sdfg.add_container("C", desc_2d, true);

    auto& root = sdfg.subject().root();

    auto& loop_i = sdfg.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body_i = loop_i.root();

    auto& loop_j_1 = sdfg.add_for(
        body_i,
        symbolic::symbol("j_1"),
        symbolic::Le(symbolic::symbol("j_1"), symbolic::symbol("i")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j_1"), symbolic::integer(1))
    );
    {
        auto& block = sdfg.add_block(loop_j_1.root());
        auto& C_in_node = sdfg.add_access(block, "C");
        auto& C_out_node = sdfg.add_access(block, "C");
        auto& beta_node = sdfg.add_access(block, "beta");
        auto& tasklet = sdfg.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        sdfg.add_computational_memlet(block, C_in_node, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j_1")});
        sdfg.add_computational_memlet(block, beta_node, tasklet, "_in2", {});
        sdfg.add_computational_memlet(block, tasklet, "_out", C_out_node, {symbolic::symbol("i"), symbolic::symbol("j_1")});
    }

    auto& loop_k = sdfg.add_for(
        body_i,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
    );
    {
        auto& loop_j_2 = sdfg.add_for(
            loop_k.root(),
            symbolic::symbol("j_2"),
            symbolic::Le(symbolic::symbol("j_2"), symbolic::symbol("i")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j_2"), symbolic::integer(1))
        );

        auto& block = sdfg.add_block(loop_j_2.root());
        auto& A_node = sdfg.add_access(block, "A");
        auto& tmp_node = sdfg.add_access(block, "tmp");
        auto& tasklet = sdfg.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        sdfg.add_computational_memlet(block, A_node, tasklet, "_in1", {symbolic::symbol("j_2"), symbolic::symbol("k")});
        sdfg.add_computational_memlet(block, A_node, tasklet, "_in2", {symbolic::symbol("i"), symbolic::symbol("k")});
        sdfg.add_computational_memlet(block, tasklet, "_out", tmp_node, {});

        auto& block2 = sdfg.add_block(loop_j_2.root());
        auto& C_in_node = sdfg.add_access(block2, "C");
        auto& C_out_node = sdfg.add_access(block2, "C");
        auto& tmp_node_2 = sdfg.add_access(block2, "tmp");
        auto& tasklet_2 = sdfg.add_tasklet(block2, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        sdfg.add_computational_memlet(
            block2, C_in_node, tasklet_2, "_in1", {symbolic::symbol("i"), symbolic::symbol("j_2")}
        );
        sdfg.add_computational_memlet(block2, tmp_node_2, tasklet_2, "_in2", {});
        sdfg.add_computational_memlet(
            block2, tasklet_2, "_out", C_out_node, {symbolic::symbol("i"), symbolic::symbol("j_2")}
        );
    }

    return sdfg.move();
};

inline std::unique_ptr<StructuredSDFG> trmm() {
    /***
     for (i = 0; i < _PB_M; i++) {
        for (j = 0; j < _PB_N; j++) {
            for (k = i+1; k < _PB_M; k++) {
                B[i][j] += A[k][i] * B[k][j];
            }

            B[i][j] = alpha * B[i][j];
        }
    }
    */

    builder::StructuredSDFGBuilder sdfg("sdfg_1", FunctionType_CPU);

    types::Scalar desc_symbols(types::PrimitiveType::UInt64);
    sdfg.add_container("M", desc_symbols, true);
    sdfg.add_container("N", desc_symbols, true);
    sdfg.add_container("i", desc_symbols);
    sdfg.add_container("j", desc_symbols);
    sdfg.add_container("k", desc_symbols);

    types::Scalar desc_element(types::PrimitiveType::Double);
    sdfg.add_container("alpha", desc_element, true);
    sdfg.add_container("tmp", desc_element);

    types::Array desc_1d(desc_element, symbolic::symbol("M"));
    types::Pointer desc_2d(desc_1d);
    sdfg.add_container("A", desc_2d, true);
    sdfg.add_container("B", desc_2d, true);

    auto& root = sdfg.subject().root();

    auto& loop_i = sdfg.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body_i = loop_i.root();

    auto& loop_j = sdfg.add_for(
        body_i,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );
    auto& body_j = loop_j.root();

    auto& loop_k = sdfg.add_for(
        body_j,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("M")),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
    );
    {
        auto& block = sdfg.add_block(loop_k.root());
        auto& A_node = sdfg.add_access(block, "A");
        auto& B_node = sdfg.add_access(block, "B");
        auto& tmp_node = sdfg.add_access(block, "tmp");
        auto& tasklet = sdfg.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        sdfg.add_computational_memlet(block, A_node, tasklet, "_in1", {symbolic::symbol("k"), symbolic::symbol("i")});
        sdfg.add_computational_memlet(block, B_node, tasklet, "_in2", {symbolic::symbol("k"), symbolic::symbol("j")});
        sdfg.add_computational_memlet(block, tasklet, "_out", tmp_node, {});

        auto& block2 = sdfg.add_block(loop_k.root());
        auto& B_in_node = sdfg.add_access(block2, "B");
        auto& B_out_node = sdfg.add_access(block2, "B");
        auto& tmp_node_2 = sdfg.add_access(block2, "tmp");
        auto& tasklet_2 = sdfg.add_tasklet(block2, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        sdfg.add_computational_memlet(block2, B_in_node, tasklet_2, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")});
        sdfg.add_computational_memlet(block2, tmp_node_2, tasklet_2, "_in2", {});
        sdfg.add_computational_memlet(
            block2, tasklet_2, "_out", B_out_node, {symbolic::symbol("i"), symbolic::symbol("j")}
        );
    }

    auto& block3 = sdfg.add_block(body_j);
    auto& B_in_node = sdfg.add_access(block3, "B");
    auto& B_out_node = sdfg.add_access(block3, "B");
    auto& alpha_node = sdfg.add_access(block3, "alpha");
    auto& tasklet_3 = sdfg.add_tasklet(block3, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
    sdfg.add_computational_memlet(block3, B_in_node, tasklet_3, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")});
    sdfg.add_computational_memlet(block3, alpha_node, tasklet_3, "_in2", {});
    sdfg.add_computational_memlet(block3, tasklet_3, "_out", B_out_node, {symbolic::symbol("i"), symbolic::symbol("j")});

    return sdfg.move();
};

inline std::unique_ptr<StructuredSDFG> atax() {
    /***
    for (i = 0; i < _PB_N; i++)
        y[i] = 0;

    for (i = 0; i < _PB_M; i++) {
        tmp[i] = SCALAR_VAL(0.0);

        for (j = 0; j < _PB_N; j++)
            tmp[i] = tmp[i] + A[i][j] * x[j];

        for (j = 0; j < _PB_N; j++)
            y[j] = y[j] + A[i][j] * tmp[i];
    }
    */

    builder::StructuredSDFGBuilder sdfg("sdfg_1", FunctionType_CPU);

    types::Scalar desc_symbols(types::PrimitiveType::UInt64);
    sdfg.add_container("M", desc_symbols, true);
    sdfg.add_container("N", desc_symbols, true);
    sdfg.add_container("i_1", desc_symbols);
    sdfg.add_container("i_2", desc_symbols);
    sdfg.add_container("j_1", desc_symbols);
    sdfg.add_container("j_2", desc_symbols);

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Array desc_1d(desc_element, symbolic::symbol("M"));
    types::Pointer desc_2d(desc_1d);
    sdfg.add_container("tmp", desc_1d, true);
    sdfg.add_container("x", desc_1d, true);
    sdfg.add_container("y", desc_1d, true);
    sdfg.add_container("A", desc_2d, true);

    auto& root = sdfg.subject().root();

    {
        auto& loop_i_1 = sdfg.add_for(
            root,
            symbolic::symbol("i_1"),
            symbolic::Lt(symbolic::symbol("i_1"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i_1"), symbolic::integer(1))
        );
        auto& body_i_1 = loop_i_1.root();

        auto& block = sdfg.add_block(body_i_1);
        auto& y_node = sdfg.add_access(block, "y");
        auto& const_node = sdfg.add_constant(block, "0.0", desc_element);
        auto& tasklet = sdfg.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        sdfg.add_computational_memlet(block, const_node, tasklet, "_in", {});
        sdfg.add_computational_memlet(block, tasklet, "_out", y_node, {symbolic::symbol("i_1")});
    }

    {
        auto& loop_i_2 = sdfg.add_for(
            root,
            symbolic::symbol("i_2"),
            symbolic::Lt(symbolic::symbol("i_2"), symbolic::symbol("M")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i_2"), symbolic::integer(1))
        );
        auto& body_i_2 = loop_i_2.root();

        auto& block = sdfg.add_block(body_i_2);
        auto& tmp_node = sdfg.add_access(block, "tmp");
        auto& const_node = sdfg.add_constant(block, "0.0", desc_element);
        auto& tasklet = sdfg.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        sdfg.add_computational_memlet(block, const_node, tasklet, "_in", {});
        sdfg.add_computational_memlet(block, tasklet, "_out", tmp_node, {symbolic::symbol("i_2")});

        auto& loop_j_1 = sdfg.add_for(
            body_i_2,
            symbolic::symbol("j_1"),
            symbolic::Lt(symbolic::symbol("j_1"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j_1"), symbolic::integer(1))
        );
        auto& body_j_1 = loop_j_1.root();

        {
            auto& block = sdfg.add_block(body_j_1);
            auto& A_node = sdfg.add_access(block, "A");
            auto& x_node = sdfg.add_access(block, "x");
            auto& tmp_node = sdfg.add_access(block, "tmp");
            auto& tmp_node_out = sdfg.add_access(block, "tmp");
            auto& tasklet = sdfg.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
            sdfg.add_computational_memlet(
                block, A_node, tasklet, "_in1", {symbolic::symbol("i_2"), symbolic::symbol("j_1")}
            );
            sdfg.add_computational_memlet(block, x_node, tasklet, "_in2", {symbolic::symbol("j_1")});
            sdfg.add_computational_memlet(block, tmp_node, tasklet, "_in3", {symbolic::symbol("i_2")});
            sdfg.add_computational_memlet(block, tasklet, "_out", tmp_node_out, {symbolic::symbol("i_2")});
        }

        {
            auto& loop_j_2 = sdfg.add_for(
                body_i_2,
                symbolic::symbol("j_2"),
                symbolic::Lt(symbolic::symbol("j_2"), symbolic::symbol("N")),
                symbolic::integer(0),
                symbolic::add(symbolic::symbol("j_2"), symbolic::integer(1))
            );
            auto& body_j_2 = loop_j_2.root();

            auto& block = sdfg.add_block(body_j_2);
            auto& A_node = sdfg.add_access(block, "A");
            auto& y_node = sdfg.add_access(block, "y");
            auto& y_node_out = sdfg.add_access(block, "y");
            auto& tmp_node = sdfg.add_access(block, "tmp");
            auto& tasklet = sdfg.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
            sdfg.add_computational_memlet(
                block, A_node, tasklet, "_in1", {symbolic::symbol("i_2"), symbolic::symbol("j_2")}
            );
            sdfg.add_computational_memlet(block, tmp_node, tasklet, "_in2", {symbolic::symbol("i_2")});
            sdfg.add_computational_memlet(block, y_node, tasklet, "_in3", {symbolic::symbol("j_2")});
            sdfg.add_computational_memlet(block, tasklet, "_out", y_node_out, {symbolic::symbol("j_2")});
        }
    }

    return sdfg.move();
}

inline std::unique_ptr<StructuredSDFG> bicg() {
    /***
        for (i = 0; i < _PB_M; i++)
            s[i] = 0;
        for (i = 0; i < _PB_N; i++) {
            q[i] = SCALAR_VAL(0.0);
            for (j = 0; j < _PB_M; j++) {
                s[j] = s[j] + r[i] * A[i][j];
                q[i] = q[i] + A[i][j] * p[j];
            }
        }
    */

    builder::StructuredSDFGBuilder sdfg("sdfg_1", FunctionType_CPU);

    types::Scalar desc_symbols(types::PrimitiveType::UInt64);
    sdfg.add_container("M", desc_symbols, true);
    sdfg.add_container("N", desc_symbols, true);
    sdfg.add_container("i_1", desc_symbols);
    sdfg.add_container("i_2", desc_symbols);
    sdfg.add_container("j", desc_symbols);

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Array desc_1d(desc_element, symbolic::symbol("M"));
    types::Pointer desc_2d(desc_1d);
    sdfg.add_container("s", desc_1d, true);
    sdfg.add_container("q", desc_1d, true);
    sdfg.add_container("r", desc_1d, true);
    sdfg.add_container("p", desc_1d, true);
    sdfg.add_container("A", desc_2d, true);

    auto& root = sdfg.subject().root();

    {
        auto& loop_i_1 = sdfg.add_for(
            root,
            symbolic::symbol("i_1"),
            symbolic::Lt(symbolic::symbol("i_1"), symbolic::symbol("M")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i_1"), symbolic::integer(1))
        );
        auto& body_i_1 = loop_i_1.root();

        auto& block = sdfg.add_block(body_i_1);

        auto& s_node = sdfg.add_access(block, "s");
        auto& const_node = sdfg.add_constant(block, "0.0", desc_element);
        auto& tasklet = sdfg.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        sdfg.add_computational_memlet(block, const_node, tasklet, "_in", {});
        sdfg.add_computational_memlet(block, tasklet, "_out", s_node, {symbolic::symbol("i_1")});
    }

    {
        auto& loop_i_2 = sdfg.add_for(
            root,
            symbolic::symbol("i_2"),
            symbolic::Lt(symbolic::symbol("i_2"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i_2"), symbolic::integer(1))
        );
        auto& body_i_2 = loop_i_2.root();

        auto& block = sdfg.add_block(body_i_2);

        auto& q_node = sdfg.add_access(block, "q");
        auto& zero_node = sdfg.add_constant(block, "0.0", desc_element);
        auto& tasklet = sdfg.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        sdfg.add_computational_memlet(block, zero_node, tasklet, "_in", {});
        sdfg.add_computational_memlet(block, tasklet, "_out", q_node, {symbolic::symbol("i_2")});

        auto& loop_j = sdfg.add_for(
            body_i_2,
            symbolic::symbol("j"),
            symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
        );
        auto& body_j = loop_j.root();

        {
            auto& block = sdfg.add_block(body_j);

            auto& s_node = sdfg.add_access(block, "s");
            auto& r_node = sdfg.add_access(block, "r");
            auto& A_node = sdfg.add_access(block, "A");
            auto& s_node_out = sdfg.add_access(block, "s");
            auto& tasklet = sdfg.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
            sdfg.add_computational_memlet(block, r_node, tasklet, "_in1", {symbolic::symbol("i_2")});
            sdfg.add_computational_memlet(block, A_node, tasklet, "_in2", {symbolic::symbol("i_2"), symbolic::symbol("j")});
            sdfg.add_computational_memlet(block, s_node, tasklet, "_in3", {symbolic::symbol("j")});
            sdfg.add_computational_memlet(block, tasklet, "_out", s_node_out, {symbolic::symbol("j")});
        }

        {
            auto& block = sdfg.add_block(body_j);

            auto& q_node = sdfg.add_access(block, "q");
            auto& A_node = sdfg.add_access(block, "A");
            auto& p_node = sdfg.add_access(block, "p");
            auto& q_node_out = sdfg.add_access(block, "q");
            auto& tasklet = sdfg.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
            sdfg.add_computational_memlet(block, A_node, tasklet, "_in1", {symbolic::symbol("i_2"), symbolic::symbol("j")});
            sdfg.add_computational_memlet(block, p_node, tasklet, "_in2", {symbolic::symbol("j")});
            sdfg.add_computational_memlet(block, q_node, tasklet, "_in3", {symbolic::symbol("i_2")});
            sdfg.add_computational_memlet(block, tasklet, "_out", q_node_out, {symbolic::symbol("i_2")});
        }
    }

    return sdfg.move();
};

inline std::unique_ptr<StructuredSDFG> doitgen() {
    /***
    for (r = 0; r < _PB_NR; r++) {}
        for (q = 0; q < _PB_NQ; q++)  {
            for (p = 0; p < _PB_NP; p++)  {
                    sum[p] = SCALAR_VAL(0.0);
                    for (s = 0; s < _PB_NP; s++)
                        sum[p] += A[r][q][s] * C4[s][p];
            }
            for (p = 0; p < _PB_NP; p++)
                    A[r][q][p] = sum[p];
        }
    }
    */

    builder::StructuredSDFGBuilder sdfg("sdfg_1", FunctionType_CPU);

    types::Scalar desc_symbols(types::PrimitiveType::UInt64);
    sdfg.add_container("NP", desc_symbols, true);
    sdfg.add_container("NQ", desc_symbols, true);
    sdfg.add_container("NR", desc_symbols, true);
    sdfg.add_container("p_1", desc_symbols);
    sdfg.add_container("p_2", desc_symbols);
    sdfg.add_container("q", desc_symbols);
    sdfg.add_container("r", desc_symbols);
    sdfg.add_container("s", desc_symbols);

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Array desc_1d(desc_element, symbolic::symbol("NP"));
    types::Array desc_2d(desc_1d, symbolic::symbol("NQ"));
    types::Pointer desc_3d(desc_2d);
    sdfg.add_container("sum", desc_1d, true);
    sdfg.add_container("A", desc_3d, true);
    sdfg.add_container("C4", desc_2d, true);

    auto& root = sdfg.subject().root();

    auto& loop_r = sdfg.add_for(
        root,
        symbolic::symbol("r"),
        symbolic::Lt(symbolic::symbol("r"), symbolic::symbol("NR")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("r"), symbolic::integer(1))
    );
    auto& body_r = loop_r.root();

    auto& loop_q = sdfg.add_for(
        body_r,
        symbolic::symbol("q"),
        symbolic::Lt(symbolic::symbol("q"), symbolic::symbol("NQ")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("q"), symbolic::integer(1))
    );

    {
        auto& loop_p_1 = sdfg.add_for(
            loop_q.root(),
            symbolic::symbol("p_1"),
            symbolic::Lt(symbolic::symbol("p_1"), symbolic::symbol("NP")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("p_1"), symbolic::integer(1))
        );
        auto& body_p_1 = loop_p_1.root();

        auto& block = sdfg.add_block(body_p_1);
        auto& sum_node = sdfg.add_access(block, "sum");
        auto& const_node = sdfg.add_constant(block, "0.0", desc_element);
        auto& tasklet = sdfg.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        sdfg.add_computational_memlet(block, const_node, tasklet, "_in", {});
        sdfg.add_computational_memlet(block, tasklet, "_out", sum_node, {symbolic::symbol("p_1")});

        {
            auto& loop_s = sdfg.add_for(
                body_p_1,
                symbolic::symbol("s"),
                symbolic::Lt(symbolic::symbol("s"), symbolic::symbol("NP")),
                symbolic::integer(0),
                symbolic::add(symbolic::symbol("s"), symbolic::integer(1))
            );
            auto& body_s = loop_s.root();

            auto& block = sdfg.add_block(body_s);
            auto& A_node = sdfg.add_access(block, "A");
            auto& C4_node = sdfg.add_access(block, "C4");
            auto& sum_node = sdfg.add_access(block, "sum");
            auto& sum_node_out = sdfg.add_access(block, "sum");
            auto& tasklet = sdfg.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
            sdfg.add_computational_memlet(
                block, A_node, tasklet, "_in1", {symbolic::symbol("r"), symbolic::symbol("q"), symbolic::symbol("s")}
            );
            sdfg.add_computational_memlet(
                block, C4_node, tasklet, "_in2", {symbolic::symbol("s"), symbolic::symbol("p_1")}
            );
            sdfg.add_computational_memlet(block, sum_node, tasklet, "_in3", {symbolic::symbol("p_1")});
            sdfg.add_computational_memlet(block, tasklet, "_out", sum_node_out, {symbolic::symbol("p_1")});
        }
    }

    {
        auto& loop_p_2 = sdfg.add_for(
            loop_q.root(),
            symbolic::symbol("p_2"),
            symbolic::Lt(symbolic::symbol("p_2"), symbolic::symbol("NP")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("p_2"), symbolic::integer(1))
        );
        auto& body_p_2 = loop_p_2.root();

        auto& block = sdfg.add_block(body_p_2);
        auto& A_node = sdfg.add_access(block, "A");
        auto& sum_node = sdfg.add_access(block, "sum");
        auto& A_node_out = sdfg.add_access(block, "A");
        auto& tasklet = sdfg.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
        sdfg.add_computational_memlet(block, sum_node, tasklet, "_in1", {symbolic::symbol("p_2")});
        sdfg.add_computational_memlet(
            block,
            tasklet,
            "_out",
            A_node_out,

            {symbolic::symbol("r"), symbolic::symbol("q"), symbolic::symbol("p_2")}
        );
    }

    return sdfg.move();
};

inline std::unique_ptr<StructuredSDFG> mvt() {
    /*
    for (i = 0; i < _PB_N; i++)
        for (j = 0; j < _PB_N; j++)
            x1[i] = x1[i] + A[i][j] * y_1[j];
    for (i = 0; i < _PB_N; i++)
        for (j = 0; j < _PB_N; j++)
            x2[i] = x2[i] + A[j][i] * y_2[j];
    */

    builder::StructuredSDFGBuilder sdfg("sdfg_1", FunctionType_CPU);

    types::Scalar desc_symbols(types::PrimitiveType::UInt64);
    sdfg.add_container("N", desc_symbols, true);
    sdfg.add_container("i_1", desc_symbols);
    sdfg.add_container("i_2", desc_symbols);
    sdfg.add_container("j_1", desc_symbols);
    sdfg.add_container("j_2", desc_symbols);

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Array desc_1d(desc_element, symbolic::symbol("M"));
    types::Pointer desc_2d(desc_1d);
    sdfg.add_container("x1", desc_1d, true);
    sdfg.add_container("x2", desc_1d, true);
    sdfg.add_container("y_1", desc_1d, true);
    sdfg.add_container("y_2", desc_1d, true);
    sdfg.add_container("A", desc_2d, true);

    auto& root = sdfg.subject().root();

    {
        auto& loop_i_1 = sdfg.add_for(
            root,
            symbolic::symbol("i_1"),
            symbolic::Lt(symbolic::symbol("i_1"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i_1"), symbolic::integer(1))
        );
        auto& body_i_1 = loop_i_1.root();

        auto& loop_j_1 = sdfg.add_for(
            body_i_1,
            symbolic::symbol("j_1"),
            symbolic::Lt(symbolic::symbol("j_1"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j_1"), symbolic::integer(1))
        );
        auto& body_j_1 = loop_j_1.root();

        auto& block = sdfg.add_block(body_j_1);
        auto& x1_node = sdfg.add_access(block, "x1");
        auto& A_node = sdfg.add_access(block, "A");
        auto& y_1_node = sdfg.add_access(block, "y_1");
        auto& x1_node_out = sdfg.add_access(block, "x1");
        auto& tasklet = sdfg.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
        sdfg.add_computational_memlet(block, x1_node, tasklet, "_in3", {symbolic::symbol("i_1")});
        sdfg.add_computational_memlet(block, A_node, tasklet, "_in2", {symbolic::symbol("i_1"), symbolic::symbol("j_1")});
        sdfg.add_computational_memlet(block, y_1_node, tasklet, "_in1", {symbolic::symbol("j_1")});
        sdfg.add_computational_memlet(block, tasklet, "_out", x1_node_out, {symbolic::symbol("i_1")});
    }

    {
        auto& loop_i_2 = sdfg.add_for(
            root,
            symbolic::symbol("i_2"),
            symbolic::Lt(symbolic::symbol("i_2"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i_2"), symbolic::integer(1))
        );
        auto& body_i_2 = loop_i_2.root();

        auto& loop_j_2 = sdfg.add_for(
            body_i_2,
            symbolic::symbol("j_2"),
            symbolic::Lt(symbolic::symbol("j_2"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j_2"), symbolic::integer(1))
        );
        auto& body_j_2 = loop_j_2.root();

        auto& block = sdfg.add_block(body_j_2);
        auto& x2_node = sdfg.add_access(block, "x2");
        auto& A_node = sdfg.add_access(block, "A");
        auto& y_2_node = sdfg.add_access(block, "y_2");
        auto& x2_node_out = sdfg.add_access(block, "x2");
        auto& tasklet = sdfg.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
        sdfg.add_computational_memlet(block, x2_node, tasklet, "_in3", {symbolic::symbol("i_2")});
        sdfg.add_computational_memlet(block, A_node, tasklet, "_in2", {symbolic::symbol("j_2"), symbolic::symbol("i_2")});
        sdfg.add_computational_memlet(block, y_2_node, tasklet, "_in1", {symbolic::symbol("j_2")});
        sdfg.add_computational_memlet(block, tasklet, "_out", x2_node_out, {symbolic::symbol("i_2")});
    }

    return sdfg.move();
}

inline std::unique_ptr<StructuredSDFG> cholesky() {
    /***
     for (i = 0; i < _PB_N; i++) {
        //j<i
        for (j = 0; j < i; j++) {
            for (k = 0; k < j; k++) {
                A[i][j] -= A[i][k] * A[j][k];
            }

            A[i][j] /= A[j][j];
        }
        // i==j case
        for (k = 0; k < i; k++) {
            A[i][i] -= A[i][k] * A[i][k];
        }
        A[i][i] = sqrt(A[i][i]);
    }
    */

    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_symbols(types::PrimitiveType::UInt64);
    builder.add_container("N", desc_symbols, true);
    builder.add_container("i", desc_symbols);
    builder.add_container("j", desc_symbols);
    builder.add_container("k_1", desc_symbols);
    builder.add_container("k_2", desc_symbols);

    types::Scalar desc_element(types::PrimitiveType::Double);
    builder.add_container("tmp", desc_element);

    types::Array desc_1d(desc_element, symbolic::symbol("M"));
    types::Pointer desc_2d(desc_1d);
    builder.add_container("A", desc_2d, true);

    auto& root = builder.subject().root();

    auto& loop_i = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body_i = loop_i.root();
    {
        auto& loop_j = builder.add_for(
            body_i,
            symbolic::symbol("j"),
            symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("i")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
        );
        auto& body_j = loop_j.root();
        {
            auto& loop_k_1 = builder.add_for(
                body_j,
                symbolic::symbol("k_1"),
                symbolic::Lt(symbolic::symbol("k_1"), symbolic::symbol("j")),
                symbolic::integer(0),
                symbolic::add(symbolic::symbol("k_1"), symbolic::integer(1))
            );
            auto& body_k_1 = loop_k_1.root();
            {
                auto& block = builder.add_block(body_k_1);
                auto& A_node = builder.add_access(block, "A");
                auto& tmp_node = builder.add_access(block, "tmp");
                auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
                builder.add_computational_memlet(
                    block, A_node, tasklet, "_in1", {symbolic::symbol("j"), symbolic::symbol("k_1")}
                );
                builder.add_computational_memlet(
                    block, A_node, tasklet, "_in2", {symbolic::symbol("i"), symbolic::symbol("k_1")}
                );
                builder.add_computational_memlet(block, tasklet, "_out", tmp_node, {});

                auto& block2 = builder.add_block(body_k_1);
                auto& A_in_node = builder.add_access(block2, "A");
                auto& A_out_node = builder.add_access(block2, "A");
                auto& tmp_node_2 = builder.add_access(block2, "tmp");
                auto& tasklet_2 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
                builder.add_computational_memlet(
                    block2, A_in_node, tasklet_2, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}
                );
                builder.add_computational_memlet(block2, tmp_node_2, tasklet_2, "_in2", {});
                builder.add_computational_memlet(
                    block2, tasklet_2, "_out", A_out_node, {symbolic::symbol("i"), symbolic::symbol("j")}
                );
            }

            auto& block3 = builder.add_block(body_j);
            auto& A_in_node = builder.add_access(block3, "A");
            auto& A_out_node = builder.add_access(block3, "A");
            auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::fp_div, "_out", {"_in1", "_in2"});
            builder.add_computational_memlet(
                block3, A_in_node, tasklet3, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}
            );
            builder.add_computational_memlet(
                block3, A_in_node, tasklet3, "_in2", {symbolic::symbol("j"), symbolic::symbol("j")}
            );
            builder.add_computational_memlet(
                block3, tasklet3, "_out", A_out_node, {symbolic::symbol("i"), symbolic::symbol("j")}
            );
        }

        auto& loop_k_2 = builder.add_for(
            body_i,
            symbolic::symbol("k_2"),
            symbolic::Lt(symbolic::symbol("k_2"), symbolic::symbol("i")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("k_2"), symbolic::integer(1))
        );
        auto& body_k_2 = loop_k_2.root();
        {
            auto& block = builder.add_block(body_k_2);
            auto& A_node = builder.add_access(block, "A");
            auto& tmp_node = builder.add_access(block, "tmp");
            auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
            builder.add_computational_memlet(
                block, A_node, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("k_2")}
            );
            builder.add_computational_memlet(
                block, A_node, tasklet, "_in2", {symbolic::symbol("i"), symbolic::symbol("k_2")}
            );
            builder.add_computational_memlet(block, tasklet, "_out", tmp_node, {});

            auto& block2 = builder.add_block(body_k_2);
            auto& A_in_node = builder.add_access(block2, "A");
            auto& A_out_node = builder.add_access(block2, "A");
            auto& tmp_node_2 = builder.add_access(block2, "tmp");
            auto& tasklet_2 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
            builder.add_computational_memlet(
                block2, A_in_node, tasklet_2, "_in1", {symbolic::symbol("i"), symbolic::symbol("i")}
            );
            builder.add_computational_memlet(block2, tmp_node_2, tasklet_2, "_in2", {});
            builder.add_computational_memlet(
                block2, tasklet_2, "_out", A_out_node, {symbolic::symbol("i"), symbolic::symbol("i")}
            );
        }

        auto& block3 = builder.add_block(body_i);
        auto& A_in_node = builder.add_access(block3, "A");
        auto& A_out_node = builder.add_access(block3, "A");
        auto& tasklet = builder.add_library_node<math::cmath::CMathNode>(
            block3, body_i.debug_info(), math::cmath::CMathFunction::sqrt, types::PrimitiveType::Double
        );
        builder.add_computational_memlet(
            block3, A_in_node, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("i")}, desc_2d
        );
        builder.add_computational_memlet(
            block3, tasklet, "_out", A_out_node, {symbolic::symbol("i"), symbolic::symbol("i")}, desc_2d
        );
    }

    return builder.move();
};

inline std::unique_ptr<StructuredSDFG> fdtd_2d() {
    /***
    for(t = 0; t < _PB_TMAX; t++)
    {
      for (j = 0; j < _PB_NY; j++)
            ey[0][j] = _fict_[t];

      for (i = 1; i < _PB_NX; i++)
            for (j = 0; j < _PB_NY; j++)
                ey[i][j] -= (hz[i][j]-hz[i-1][j]);

      for (i = 0; i < _PB_NX; i++)
            for (j = 1; j < _PB_NY; j++)
                ex[i][j] -= (hz[i][j]-hz[i][j-1]);

      for (i = 0; i < _PB_NX - 1; i++)
            for (j = 0; j < _PB_NY - 1; j++)
                hz[i][j] += (ex[i][j+1] - ex[i][j]) * (ey[i+1][j] - ey[i][j]);
    }
    */

    builder::StructuredSDFGBuilder builder("fdtd_2d", FunctionType_CPU);

    types::Scalar desc_symbols(types::PrimitiveType::UInt64);
    builder.add_container("TMAX", desc_symbols, true);
    builder.add_container("NX", desc_symbols, true);
    builder.add_container("NY", desc_symbols, true);
    builder.add_container("t", desc_symbols);
    builder.add_container("i_1", desc_symbols);
    builder.add_container("i_2", desc_symbols);
    builder.add_container("i_3", desc_symbols);
    builder.add_container("j_1", desc_symbols);
    builder.add_container("j_2", desc_symbols);
    builder.add_container("j_3", desc_symbols);
    builder.add_container("j_4", desc_symbols);

    types::Scalar desc_element(types::PrimitiveType::Double);
    builder.add_container("tmp", desc_element);
    builder.add_container("tmp2", desc_element);
    builder.add_container("tmp3", desc_element);
    builder.add_container("tmp4", desc_element);

    types::Array desc_1d(desc_element, symbolic::symbol("M"));
    types::Pointer desc_2d(desc_1d);
    builder.add_container("_fict_", desc_1d, true);
    builder.add_container("ey", desc_2d, true);
    builder.add_container("hz", desc_2d, true);
    builder.add_container("ex", desc_2d, true);

    auto& root = builder.subject().root();

    auto& loop_t = builder.add_for(
        root,
        symbolic::symbol("t"),
        symbolic::Lt(symbolic::symbol("t"), symbolic::symbol("TMAX")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("t"), symbolic::integer(1))
    );
    auto& body_t = loop_t.root();

    auto& loop_j_1 = builder.add_for(
        body_t,
        symbolic::symbol("j_1"),
        symbolic::Lt(symbolic::symbol("j_1"), symbolic::symbol("NY")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j_1"), symbolic::integer(1))
    );
    {
        auto& block = builder.add_block(loop_j_1.root());
        auto& ey_node = builder.add_access(block, "ey");
        auto& fict_node = builder.add_access(block, "_fict_");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
        builder.add_computational_memlet(block, fict_node, tasklet, "_in1", {symbolic::symbol("t")});
        builder
            .add_computational_memlet(block, tasklet, "_out", ey_node, {symbolic::integer(0), symbolic::symbol("j_1")});
    }

    /**
     * for (i = 1; i < _PB_NX; i++)
            for (j = 0; j < _PB_NY; j++)
                ey[i][j] -= (hz[i][j]-hz[i-1][j]);
     */
    auto& loop_i_1 = builder.add_for(
        body_t,
        symbolic::symbol("i_1"),
        symbolic::Lt(symbolic::symbol("i_1"), symbolic::symbol("NX")),
        symbolic::integer(1),
        symbolic::add(symbolic::symbol("i_1"), symbolic::integer(1))
    );
    {
        auto& loop_j_2 = builder.add_for(
            loop_i_1.root(),
            symbolic::symbol("j_2"),
            symbolic::Lt(symbolic::symbol("j_2"), symbolic::symbol("NY")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j_2"), symbolic::integer(1))
        );
        {
            auto& block = builder.add_block(loop_j_2.root());
            auto& hz_node = builder.add_access(block, "hz");
            auto& tmp_node = builder.add_access(block, "tmp");
            auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
            builder.add_computational_memlet(
                block, hz_node, tasklet, "_in1", {symbolic::symbol("i_1"), symbolic::symbol("j_2")}
            );
            builder.add_computational_memlet(
                block,
                hz_node,

                tasklet,
                "_in2",
                {symbolic::sub(symbolic::symbol("i_1"), symbolic::integer(1)), symbolic::symbol("j_2")}
            );
            builder.add_computational_memlet(block, tasklet, "_out", tmp_node, {});

            auto& block2 = builder.add_block(loop_j_2.root());
            auto& ey_node = builder.add_access(block2, "ey");
            auto& ey_node_2 = builder.add_access(block2, "ey");
            auto& tmp_node_2 = builder.add_access(block2, "tmp");
            auto& tasklet_2 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
            builder.add_computational_memlet(
                block2, ey_node, tasklet_2, "_in1", {symbolic::symbol("i_1"), symbolic::symbol("j_2")}
            );
            builder.add_computational_memlet(block2, tmp_node_2, tasklet_2, "_in2", {});
            builder.add_computational_memlet(
                block2, tasklet_2, "_out", ey_node_2, {symbolic::symbol("i_1"), symbolic::symbol("j_2")}
            );
        }
    }

    /*
        for (i = 0; i < _PB_NX; i++)
            for (j = 1; j < _PB_NY; j++)
                ex[i][j] -= (hz[i][j]-hz[i][j-1]);
    */
    auto& loop_i_2 = builder.add_for(
        body_t,
        symbolic::symbol("i_2"),
        symbolic::Lt(symbolic::symbol("i_2"), symbolic::symbol("NX")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i_2"), symbolic::integer(1))
    );
    {
        auto& loop_j_3 = builder.add_for(
            loop_i_2.root(),
            symbolic::symbol("j_3"),
            symbolic::Lt(symbolic::symbol("j_3"), symbolic::symbol("NY")),
            symbolic::integer(1),
            symbolic::add(symbolic::symbol("j_3"), symbolic::integer(1))
        );
        {
            auto& block = builder.add_block(loop_j_3.root());
            auto& hz_node = builder.add_access(block, "hz");
            auto& tmp_node = builder.add_access(block, "tmp2");
            auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
            builder.add_computational_memlet(
                block, hz_node, tasklet, "_in1", {symbolic::symbol("i_2"), symbolic::symbol("j_3")}
            );
            builder.add_computational_memlet(
                block,
                hz_node,

                tasklet,
                "_in2",
                {symbolic::symbol("i_2"), symbolic::sub(symbolic::symbol("j_3"), symbolic::integer(1))}
            );
            builder.add_computational_memlet(block, tasklet, "_out", tmp_node, {});

            auto& block2 = builder.add_block(loop_j_3.root());
            auto& ex_node = builder.add_access(block2, "ex");
            auto& ex_node_2 = builder.add_access(block2, "ex");
            auto& tmp_node_2 = builder.add_access(block2, "tmp2");
            auto& tasklet_2 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
            builder.add_computational_memlet(
                block2, ex_node, tasklet_2, "_in1", {symbolic::symbol("i_2"), symbolic::symbol("j_3")}
            );
            builder.add_computational_memlet(block2, tmp_node_2, tasklet_2, "_in2", {});
            builder.add_computational_memlet(
                block2, tasklet_2, "_out", ex_node_2, {symbolic::symbol("i_2"), symbolic::symbol("j_3")}
            );
        }
    }

    /*
    for (i = 0; i < _PB_NX - 1; i++)
            for (j = 0; j < _PB_NY - 1; j++)
                hz[i][j] += (ex[i][j+1] - ex[i][j]) * (ey[i+1][j] - ey[i][j]);
    */
    auto& loop_i_3 = builder.add_for(
        body_t,
        symbolic::symbol("i_3"),
        symbolic::Lt(symbolic::symbol("i_3"), symbolic::sub(symbolic::symbol("NX"), symbolic::integer(1))),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i_3"), symbolic::integer(1))
    );
    {
        auto& loop_j_4 = builder.add_for(
            loop_i_3.root(),
            symbolic::symbol("j_4"),
            symbolic::Lt(symbolic::symbol("j_4"), symbolic::sub(symbolic::symbol("NY"), symbolic::integer(1))),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j_4"), symbolic::integer(1))
        );
        {
            auto& block = builder.add_block(loop_j_4.root());
            auto& ex_node = builder.add_access(block, "ex");
            auto& tmp_node = builder.add_access(block, "tmp3");
            auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
            builder.add_computational_memlet(
                block,
                ex_node,

                tasklet,
                "_in1",
                {symbolic::symbol("i_3"), symbolic::add(symbolic::symbol("j_4"), symbolic::integer(1))}
            );
            builder.add_computational_memlet(
                block, ex_node, tasklet, "_in2", {symbolic::symbol("i_3"), symbolic::symbol("j_4")}
            );
            builder.add_computational_memlet(block, tasklet, "_out", tmp_node, {});

            auto& block2 = builder.add_block(loop_j_4.root());
            auto& ey_node = builder.add_access(block2, "ey");
            auto& tmp_node_2 = builder.add_access(block2, "tmp4");
            auto& tasklet_2 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
            builder.add_computational_memlet(
                block2,
                ey_node,

                tasklet_2,
                "_in1",
                {symbolic::add(symbolic::symbol("i_3"), symbolic::integer(1)), symbolic::symbol("j_4")}
            );
            builder.add_computational_memlet(
                block2, ey_node, tasklet_2, "_in2", {symbolic::symbol("i_3"), symbolic::symbol("j_4")}
            );
            builder.add_computational_memlet(block2, tasklet_2, "_out", tmp_node_2, {});

            auto& block3 = builder.add_block(loop_j_4.root());
            auto& hz_node = builder.add_access(block3, "hz");
            auto& hz_node_2 = builder.add_access(block3, "hz");
            auto& tmp_node_3 = builder.add_access(block3, "tmp3");
            auto& tmp_node_4 = builder.add_access(block3, "tmp4");
            auto& tasklet_3 =
                builder.add_tasklet(block3, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
            builder.add_computational_memlet(block3, tmp_node_3, tasklet_3, "_in1", {});
            builder.add_computational_memlet(block3, tmp_node_4, tasklet_3, "_in2", {});
            builder.add_computational_memlet(
                block3, hz_node, tasklet_3, "_in3", {symbolic::symbol("i_3"), symbolic::symbol("j_4")}
            );
            builder.add_computational_memlet(
                block3, tasklet_3, "_out", hz_node_2, {symbolic::symbol("i_3"), symbolic::symbol("j_4")}
            );
        }
    }

    return builder.move();
}
