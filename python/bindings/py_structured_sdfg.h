#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sdfg/passes/rpc/rpc_context.h>
#include <sdfg/structured_control_flow/sequence.h>
#include <sdfg/structured_sdfg.h>

#include "sdfg/plugins/plugins.h"

class PyStructuredSDFGBuilder;

class PyStructuredSDFG {
    friend class PyStructuredSDFGBuilder;

private:
    sdfg::plugins::Context& docc_context_;
    std::unique_ptr<sdfg::StructuredSDFG> sdfg_;

    PyStructuredSDFG(sdfg::plugins::Context& ctx, std::unique_ptr<sdfg::StructuredSDFG>& sdfg);

public:
    static PyStructuredSDFG parse(sdfg::plugins::Context& ctx, const std::string& sdfg_text);

    static PyStructuredSDFG from_file(sdfg::plugins::Context& ctx, const std::string& file_path);

    /**
     * @brief Create a PyStructuredSDFG from a unique_ptr
     *
     * This factory method is used internally for operations like cutout
     * that create new SDFGs.
     */
    static PyStructuredSDFG from_sdfg(sdfg::plugins::Context& ctx, std::unique_ptr<sdfg::StructuredSDFG> sdfg);

    std::string name() const;

    sdfg::plugins::Context& docc_context() const;

    sdfg::StructuredSDFG& sdfg() { return *sdfg_; }

    sdfg::structured_control_flow::Sequence& root() { return sdfg_->root(); }

    const sdfg::types::IType& return_type() const;

    const sdfg::types::IType& type(const std::string& name) const;

    bool exists(const std::string& name) const;

    bool is_argument(const std::string& name) const;

    bool is_transient(const std::string& name) const;

    std::vector<std::string> arguments() const;

    pybind11::dict containers() const;

    void validate();

    void expand();

    void simplify();

    void dump(
        const std::string& path,
        const std::string& type = "",
        bool dump_dot = false,
        bool dump_json = true,
        bool record_for_instrumentation = false
    );

    void normalize();

    void schedule(const std::string& target, const std::string& category, bool remote_tuning = false);

    /**
     * Build the shared library containing the SDFG
     * @param output_folder will contain src and binary files
     * @param target
     * @param instrumentation_mode
     * @param capture_args
     * @param debug_build build with debug info
     * @param threads number of threads to use for the compile. 0 or negative means auto (max. hardware threads)
     * @return
     */
    std::string compile(
        const std::string& output_folder,
        const std::string& target,
        const std::string& instrumentation_mode = "",
        bool capture_args = false,
        bool debug_build = false,
        int threads = 0
    ) const;

    std::string metadata(const std::string& key) const;

    pybind11::dict loop_report() const;

    std::string to_json() const;

    std::string to_dot() const;

    std::string to_cpp() const;
};
