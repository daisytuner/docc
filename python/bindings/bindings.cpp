#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <fstream>

#include "analysis/py_analysis.h"
#include "builder/py_structured_sdfg_builder.h"
#include "control_flow/py_control_flow.h"
#include "cutouts/py_cutout.h"
#include "data_flow/py_cmath.h"
#include "data_flow/py_code_node.h"
#include "data_flow/py_data_flow_graph.h"
#include "data_flow/py_data_flow_node.h"
#include "data_flow/py_memlet.h"
#include "data_flow/py_tasklet.h"
#include "py_structured_sdfg.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/passes/rpc/daisytuner_rpc_context.h"
#include "sdfg/passes/rpc/rpc_context.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/targets/cuda/plugin.h"
#include "transformations/py_replayer.h"
#include "transformations/py_transformations.h"
#include "types/py_types.h"

#include <sdfg/data_flow/tasklet.h>
#include <sdfg/element.h>
#include <sdfg/passes/rpc/rpc_context.h>
#include <sdfg/types/array.h>
#include <sdfg/types/pointer.h>
#include <sdfg/types/scalar.h>
#include <sdfg/types/structure.h>
#include <sdfg/types/type.h>

#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/plugins/plugins.h>
#include <sdfg/serializer/json_serializer.h>
#include <sdfg/targets/cuda/plugin.h>
#include <sdfg/targets/highway/plugin.h>
#include <sdfg/targets/omp/plugin.h>
#include <sdfg/targets/rocm/plugin.h>

#include <sdfg/passes/statistics.h>
#include "sdfg/passes/rpc/rpc_scheduler.h"
#include "sdfg/passes/scheduler/cuda_scheduler.h"

#ifdef DOCC_HAS_TARGET_ET
#include <docc/target/et/target.h>
#endif

namespace py = pybind11;
using namespace sdfg::types;

PYBIND11_MODULE(_sdfg, m) {
    m.doc() = "A JIT compiler for Numpy-based Python programs targeting various hardware backends.";

    static sdfg::plugins::Context docc_context = sdfg::plugins::Context::global_context();
    sdfg::codegen::register_default_dispatchers();
    sdfg::serializer::register_default_serializers();
    sdfg::omp::register_omp_plugin();
    sdfg::highway::register_highway_plugin();
    sdfg::cuda::register_cuda_plugin();
    sdfg::rocm::register_rocm_plugin();
#ifdef DOCC_HAS_TARGET_ET
    docc::target::et::register_plugin(docc_context);
#endif

    register_types(m);
    register_data_flow_node(m);
    register_code_node(m);
    register_tasklet(m);
    register_memlet(m);
    register_data_flow_graph(m);
    register_control_flow(m);
    register_cmath(m);
    register_analysis(m);
    register_replayer(m);
    register_transformations(m);
    register_cutout(m);

    py::class_<sdfg::passes::rpc::RpcContext>(m, "RpcContext");

    py::class_<sdfg::passes::rpc::SimpleRpcContext, sdfg::passes::rpc::RpcContext>(m, "SimpleRpcContext")
        .def(
            py::init<std::string, std::string, std::unordered_map<std::string, std::string>>(),
            py::arg("host"),
            py::arg("endpoint"),
            py::arg("headers")
        )
        .def_static(
            "build_from_file",
            &sdfg::passes::rpc::build_rpc_context_from_file,
            py::arg("path"),
            "Read Server Context from JSON file"
        )
        .def_static(
            "build_from_env",
            []() {
                sdfg::passes::rpc::SimpleRpcContextBuilder b;
                return b.from_env().build();
            },
            "Read from the file pointed to by $SDFG_RPC_CONFIG"
        )
        .def_static(
            "build_auto",
            &sdfg::passes::rpc::build_rpc_context_auto,
            "Use whatever config you can find to build a context. Default to local server"
        )
        .def_static(
            "build_local", &sdfg::passes::rpc::build_rpc_context_local, "Use localhost:8080/docc as in example server"
        );


    py::class_<sdfg::passes::rpc::DaisytunerRpcContext, sdfg::passes::rpc::SimpleRpcContext>(m, "DaisytunerRpcContext")
        .def(py::init<std::string, bool>(), py::arg("license_token"), py::arg("is_job_token") = false)
        .def_static(
            "from_docc_config",
            sdfg::passes::rpc::DaisytunerRpcContext::from_docc_config,
            "Read license config from an already setup DOCC"
        );

    py::class_<sdfg::DebugInfo>(m, "DebugInfo")
        .def(py::init<>())
        .def(
            py::init<std::string, size_t, size_t, size_t, size_t>(),
            py::arg("filename"),
            py::arg("start_line"),
            py::arg("start_column"),
            py::arg("end_line"),
            py::arg("end_column")
        )
        .def(
            py::init<std::string, std::string, size_t, size_t, size_t, size_t>(),
            py::arg("filename"),
            py::arg("function"),
            py::arg("start_line"),
            py::arg("start_column"),
            py::arg("end_line"),
            py::arg("end_column")
        )
        .def_property_readonly("filename", &sdfg::DebugInfo::filename)
        .def_property_readonly("function", &sdfg::DebugInfo::function)
        .def_property_readonly("start_line", &sdfg::DebugInfo::start_line)
        .def_property_readonly("start_column", &sdfg::DebugInfo::start_column)
        .def_property_readonly("end_line", &sdfg::DebugInfo::end_line)
        .def_property_readonly("end_column", &sdfg::DebugInfo::end_column);

    // Register SDFG class
    py::class_<PyStructuredSDFG>(m, "StructuredSDFG")
        .def_static("from_file", &PyStructuredSDFG::from_file, py::arg("file_path"), "Load a StructuredSDFG from file")
        .def_static("parse", &PyStructuredSDFG::parse, py::arg("sdfg_text"), "Parse a StructuredSDFG from text")
        .def_property_readonly("name", &PyStructuredSDFG::name)
        .def_property_readonly(
            "_ptr",
            [](PyStructuredSDFG& self) { return reinterpret_cast<uintptr_t>(&self.sdfg()); },
            "Get native pointer to StructuredSDFG for external plugin use"
        )
        .def_property_readonly(
            "root",
            [](PyStructuredSDFG& self) -> sdfg::structured_control_flow::Sequence& { return self.root(); },
            py::return_value_policy::reference,
            "Get the root sequence of the SDFG"
        )
        .def_property_readonly("return_type", &PyStructuredSDFG::return_type, py::return_value_policy::reference)
        .def("type", &PyStructuredSDFG::type, py::arg("name"), py::return_value_policy::reference)
        .def("exists", &PyStructuredSDFG::exists, py::arg("name"))
        .def("is_argument", &PyStructuredSDFG::is_argument, py::arg("name"))
        .def("is_transient", &PyStructuredSDFG::is_transient, py::arg("name"))
        .def_property_readonly("arguments", &PyStructuredSDFG::arguments)
        .def_property_readonly("containers", &PyStructuredSDFG::containers)
        .def("validate", &PyStructuredSDFG::validate, "Validates the SDFG")
        .def("expand", &PyStructuredSDFG::expand, "Expands all library nodes")
        .def("simplify", &PyStructuredSDFG::simplify, "Simplify the SDFG")
        .def(
            "dump",
            &PyStructuredSDFG::dump,
            py::arg("path"),
            py::arg("type") = "",
            py::arg("dump_dot") = false,
            py::arg("dump_json") = true,
            py::arg("record_for_instrumentation") = false
        )
        .def("normalize", &PyStructuredSDFG::normalize, "Normalize the SDFG")
        .def(
            "schedule",
            &PyStructuredSDFG::schedule,
            py::arg("target"),
            py::arg("category"),
            py::arg("remote_tuning") = false,
            "Schedule the SDFG"
        )
        .def(
            "_compile",
            &PyStructuredSDFG::compile,
            py::arg("output_folder"),
            py::arg("target"),
            py::arg("instrumentation_mode") = "",
            py::arg("capture_args") = false
        )
        .def("metadata", &PyStructuredSDFG::metadata, py::arg("key"), "Get metadata value")
        .def("loop_report", &PyStructuredSDFG::loop_report, "Get loop statistics from the SDFG")
        .def("to_json", &PyStructuredSDFG::to_json, "Serialize the SDFG to a JSON string")
        .def("to_dot", &PyStructuredSDFG::to_dot, "Serialize the SDFG to a DOT graph string")
        .def("to_cpp", &PyStructuredSDFG::to_cpp, "Generate C++ code from the SDFG");

    // Register StructuredSDFGBuilder class
    py::class_<PyStructuredSDFGBuilder>(m, "StructuredSDFGBuilder")
        .def(py::init<const std::string&>(), py::arg("name"), "Create a StructuredSDFGBuilder with the given name")
        .def(
            py::init<const std::string&, const IType&>(),
            py::arg("name"),
            py::arg("return_type"),
            "Create a StructuredSDFGBuilder with the given name and return type"
        )
        .def(py::init<PyStructuredSDFG&>(), py::arg("sdfg"), "Create a StructuredSDFGBuilder to modify an existing SDFG")
        .def("move", &PyStructuredSDFGBuilder::move, "Move the built StructuredSDFG and return it")
        .def(
            "add_container",
            &PyStructuredSDFGBuilder::add_container,
            py::arg("name"),
            py::arg("type"),
            py::arg("is_argument") = false,
            "Add a container to the SDFG"
        )
        .def("exists", &PyStructuredSDFGBuilder::exists, py::arg("name"), "Check if a container exists in the SDFG")
        .def(
            "set_return_type",
            &PyStructuredSDFGBuilder::set_return_type,
            py::arg("type"),
            "Set the return type of the SDFG"
        )
        .def(
            "find_new_name",
            &PyStructuredSDFGBuilder::find_new_name,
            py::arg("prefix") = "tmp_",
            "Find a new unique name in the SDFG with the given prefix"
        )
        .def(
            "add_return",
            &PyStructuredSDFGBuilder::add_return,
            py::arg("data"),
            py::arg("debug_info") = sdfg::DebugInfo(),
            "Add a return statement to the SDFG"
        )
        .def(
            "add_constant_return",
            &PyStructuredSDFGBuilder::add_constant_return,
            py::arg("value"),
            py::arg("type"),
            py::arg("debug_info") = sdfg::DebugInfo(),
            "Add a constant return statement to the SDFG"
        )
        .def(
            "add_assignment",
            &PyStructuredSDFGBuilder::add_assignment,
            py::arg("target"),
            py::arg("value"),
            py::arg("debug_info") = sdfg::DebugInfo(),
            "Add an assignment to the SDFG"
        )
        .def(
            "begin_if",
            &PyStructuredSDFGBuilder::begin_if,
            py::arg("condition"),
            py::arg("debug_info") = sdfg::DebugInfo(),
            py::return_value_policy::reference
        )
        .def("begin_else", &PyStructuredSDFGBuilder::begin_else, py::arg("debug_info") = sdfg::DebugInfo())
        .def("end_if", &PyStructuredSDFGBuilder::end_if)
        .def(
            "begin_while",
            &PyStructuredSDFGBuilder::begin_while,
            py::arg("debug_info") = sdfg::DebugInfo(),
            py::return_value_policy::reference
        )
        .def("add_break", &PyStructuredSDFGBuilder::add_break, py::arg("debug_info") = sdfg::DebugInfo())
        .def("add_continue", &PyStructuredSDFGBuilder::add_continue, py::arg("debug_info") = sdfg::DebugInfo())
        .def("end_while", &PyStructuredSDFGBuilder::end_while)
        .def(
            "begin_for",
            &PyStructuredSDFGBuilder::begin_for,
            py::arg("var"),
            py::arg("start"),
            py::arg("end"),
            py::arg("step"),
            py::arg("debug_info") = sdfg::DebugInfo(),
            py::return_value_policy::reference
        )
        .def("end_for", &PyStructuredSDFGBuilder::end_for)
        .def(
            "add_transition",
            &PyStructuredSDFGBuilder::add_transition,
            py::arg("lhs"),
            py::arg("rhs"),
            py::arg("debug_info") = sdfg::DebugInfo()
        )
        .def(
            "add_gemm",
            &PyStructuredSDFGBuilder::add_gemm,
            py::arg("A"),
            py::arg("B"),
            py::arg("C"),
            py::arg("alpha"),
            py::arg("beta"),
            py::arg("m"),
            py::arg("n"),
            py::arg("k"),
            py::arg("trans_a") = false,
            py::arg("trans_b") = false,
            py::arg("a_subset") = std::vector<std::string>(),
            py::arg("b_subset") = std::vector<std::string>(),
            py::arg("c_subset") = std::vector<std::string>(),
            py::arg("lda") = "",
            py::arg("ldb") = "",
            py::arg("ldc") = "",
            py::arg("debug_info") = sdfg::DebugInfo()
        )
        .def(
            "add_dot",
            &PyStructuredSDFGBuilder::add_dot,
            py::arg("X"),
            py::arg("Y"),
            py::arg("result"),
            py::arg("n"),
            py::arg("incx"),
            py::arg("incy"),
            py::arg("x_subset") = std::vector<std::string>(),
            py::arg("y_subset") = std::vector<std::string>(),
            py::arg("debug_info") = sdfg::DebugInfo()
        )
        .def(
            "add_elementwise_op",
            &PyStructuredSDFGBuilder::add_elementwise_op,
            py::arg("op_type"),
            py::arg("A"),
            py::arg("A_type"),
            py::arg("B"),
            py::arg("B_type"),
            py::arg("C"),
            py::arg("C_type"),
            py::arg("debug_info") = sdfg::DebugInfo()
        )
        .def(
            "add_elementwise_unary_op",
            &PyStructuredSDFGBuilder::add_elementwise_unary_op,
            py::arg("op_type"),
            py::arg("A"),
            py::arg("A_type"),
            py::arg("C"),
            py::arg("C_type"),
            py::arg("debug_info") = sdfg::DebugInfo()
        )
        .def(
            "add_conv",
            &PyStructuredSDFGBuilder::add_conv,
            py::arg("X"),
            py::arg("W"),
            py::arg("Y"),
            py::arg("shape"),
            py::arg("kernel_shape"),
            py::arg("strides"),
            py::arg("pads"),
            py::arg("dilations"),
            py::arg("output_channels"),
            py::arg("group"),
            py::arg("debug_info") = sdfg::DebugInfo()
        )
        .def(
            "add_cast_op",
            &PyStructuredSDFGBuilder::add_cast_op,
            py::arg("A"),
            py::arg("A_type"),
            py::arg("C"),
            py::arg("C_type"),
            py::arg("debug_info") = sdfg::DebugInfo()
        )
        .def(
            "add_reduce_op",
            &PyStructuredSDFGBuilder::add_reduce_op,
            py::arg("op_type"),
            py::arg("input"),
            py::arg("input_type"),
            py::arg("output"),
            py::arg("output_type"),
            py::arg("axes"),
            py::arg("keepdims"),
            py::arg("debug_info") = sdfg::DebugInfo()
        )
        .def(
            "add_einsum",
            [](PyStructuredSDFGBuilder& self,
               const std::vector<std::string>& inputs,
               const std::string& output,
               const std::vector<std::tuple<std::string, std::string, std::string>>& dims,
               const std::vector<std::string>& out_indices,
               const std::vector<std::vector<std::string>>& in_indices,
               py::list input_types,
               const sdfg::types::Tensor& output_type,
               const sdfg::DebugInfo& debug_info) {
                std::vector<const sdfg::types::Tensor*> types;
                for (auto item : input_types) {
                    types.push_back(&item.cast<const sdfg::types::Tensor&>());
                }
                self.add_einsum(inputs, output, dims, out_indices, in_indices, types, output_type, debug_info);
            },
            py::arg("inputs"),
            py::arg("output"),
            py::arg("dims"),
            py::arg("out_indices"),
            py::arg("in_indices"),
            py::arg("input_types"),
            py::arg("output_type"),
            py::arg("debug_info") = sdfg::DebugInfo()
        )
        .def(
            "add_block",
            &PyStructuredSDFGBuilder::add_block,
            py::arg("debug_info") = sdfg::DebugInfo(),
            py::return_value_policy::reference
        )
        .def(
            "add_access",
            &PyStructuredSDFGBuilder::add_access,
            py::arg("block"),
            py::arg("name"),
            py::arg("debug_info") = sdfg::DebugInfo(),
            py::return_value_policy::reference
        )
        .def(
            "add_constant",
            &PyStructuredSDFGBuilder::add_constant,
            py::arg("block"),
            py::arg("value"),
            py::arg("type"),
            py::arg("debug_info") = sdfg::DebugInfo(),
            py::return_value_policy::reference
        )
        .def(
            "add_tasklet",
            &PyStructuredSDFGBuilder::add_tasklet,
            py::arg("block"),
            py::arg("code"),
            py::arg("inputs"),
            py::arg("outputs"),
            py::arg("debug_info") = sdfg::DebugInfo(),
            py::return_value_policy::reference
        )
        .def(
            "add_cmath",
            &PyStructuredSDFGBuilder::add_cmath,
            py::arg("block"),
            py::arg("func"),
            py::arg("primitive_type"),
            py::arg("debug_info") = sdfg::DebugInfo(),
            py::return_value_policy::reference
        )
        .def(
            "add_malloc",
            &PyStructuredSDFGBuilder::add_malloc,
            py::arg("block"),
            py::arg("size"),
            py::arg("debug_info") = sdfg::DebugInfo(),
            py::return_value_policy::reference
        )
        .def(
            "add_memset",
            &PyStructuredSDFGBuilder::add_memset,
            py::arg("block"),
            py::arg("value"),
            py::arg("num"),
            py::arg("debug_info") = sdfg::DebugInfo(),
            py::return_value_policy::reference
        )
        .def(
            "add_memcpy",
            &PyStructuredSDFGBuilder::add_memcpy,
            py::arg("block"),
            py::arg("count"),
            py::arg("debug_info") = sdfg::DebugInfo(),
            py::return_value_policy::reference
        )
        .def(
            "add_free",
            &PyStructuredSDFGBuilder::add_free,
            py::arg("block"),
            py::arg("debug_info") = sdfg::DebugInfo(),
            py::return_value_policy::reference
        )
        .def(
            "is_hoistable_size",
            &PyStructuredSDFGBuilder::is_hoistable_size,
            py::arg("size_expr"),
            "Check if a size expression only depends on function arguments (can be hoisted to function entry)"
        )
        .def(
            "insert_block_at_root_start",
            &PyStructuredSDFGBuilder::insert_block_at_root_start,
            py::arg("debug_info") = sdfg::DebugInfo(),
            py::return_value_policy::reference,
            "Insert a block at the very beginning of the root sequence"
        )
        .def("get_sizeof", &PyStructuredSDFGBuilder::get_sizeof, py::arg("type"))
        .def(
            "add_reference_memlet",
            &PyStructuredSDFGBuilder::add_reference_memlet,
            py::arg("block"),
            py::arg("src"),
            py::arg("dst"),
            py::arg("subset") = "",
            py::arg("type") = nullptr,
            py::arg("debug_info") = sdfg::DebugInfo()
        )
        .def(
            "add_memlet",
            [](PyStructuredSDFGBuilder& self,
               sdfg::structured_control_flow::Block& block,
               sdfg::data_flow::DataFlowNode& src,
               std::string src_conn,
               sdfg::data_flow::DataFlowNode& dst,
               std::string dst_conn,
               std::string subset,
               py::object type_obj,
               const sdfg::DebugInfo& debug_info) {
                const sdfg::types::IType* type = nullptr;
                if (!type_obj.is_none()) {
                    if (py::isinstance<sdfg::types::Pointer>(type_obj)) {
                        type = &type_obj.cast<const sdfg::types::Pointer&>();
                    } else if (py::isinstance<sdfg::types::Scalar>(type_obj)) {
                        type = &type_obj.cast<const sdfg::types::Scalar&>();
                    } else if (py::isinstance<sdfg::types::Array>(type_obj)) {
                        type = &type_obj.cast<const sdfg::types::Array&>();
                    } else {
                        type = &type_obj.cast<const sdfg::types::IType&>();
                    }
                }
                self.add_memlet(block, src, src_conn, dst, dst_conn, subset, type, debug_info);
            },
            py::arg("block"),
            py::arg("src"),
            py::arg("src_conn"),
            py::arg("dst"),
            py::arg("dst_conn"),
            py::arg("subset") = "",
            py::arg("type") = py::none(),
            py::arg("debug_info") = sdfg::DebugInfo()
        )
        .def(
            "add_structure",
            [](PyStructuredSDFGBuilder& self, const std::string& name, py::list member_types) {
                std::vector<const sdfg::types::IType*> types;
                for (auto item : member_types) {
                    types.push_back(&item.cast<const sdfg::types::IType&>());
                }
                self.add_structure(name, types);
            },
            py::arg("name"),
            py::arg("member_types"),
            "Define a structure type with the given name and member types"
        );

    // Plugin infrastructure - global context and registration callback
    m.def(
        "_plugin_context",
        []() { return reinterpret_cast<uintptr_t>(&docc_context); },
        "Get native pointer to the global plugin context"
    );

    // Statistics
    m.def(
        "_enable_statistics",
        []() {
            sdfg::passes::PassStatistics::instance().enable();
            sdfg::passes::PipelineStatistics::instance().enable();
            sdfg::passes::AnalysisStatistics::instance().enable();
        },
        "Enable pass, pipeline, and analysis statistics collection"
    );
    m.def(
        "_statistics_enabled_by_env",
        &sdfg::passes::statistics_enabled_by_env,
        "Check if DOCC_STATISTICS envvar is set to 1"
    );
    m.def(
        "_statistics_summary",
        []() {
            std::string result;
            result += sdfg::passes::PassStatistics::instance().summary();
            result += sdfg::passes::PipelineStatistics::instance().summary();
            result += sdfg::passes::AnalysisStatistics::instance().summary();
            return result;
        },
        "Get pass and pipeline statistics summary"
    );
}
