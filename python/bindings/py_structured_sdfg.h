#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sdfg/passes/rpc/rpc_context.h>
#include <sdfg/structured_sdfg.h>

class PyStructuredSDFGBuilder;

class PyStructuredSDFG {
    friend class PyStructuredSDFGBuilder;

private:
    std::unique_ptr<sdfg::StructuredSDFG> sdfg_;

    PyStructuredSDFG(std::unique_ptr<sdfg::StructuredSDFG>& sdfg);

public:
    static PyStructuredSDFG parse(const std::string& sdfg_text);

    static PyStructuredSDFG from_file(const std::string& file_path);

    std::string name() const;

    sdfg::StructuredSDFG& sdfg() { return *sdfg_; }

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

    void dump(const std::string& path);

    void normalize();

    void schedule(const std::string& target, const std::string& category, bool remote_tuning = false);

    std::string compile(
        const std::string& output_folder,
        const std::string& target,
        const std::string& instrumentation_mode = "",
        bool capture_args = false
    ) const;

    std::string metadata(const std::string& key) const;

    pybind11::dict loop_report() const;
};
