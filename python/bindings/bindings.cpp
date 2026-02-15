#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <fstream>

#include "analysis/py_analysis.h"
#include "builder/py_structured_sdfg_builder.h"
#include "data_flow/py_cmath.h"
#include "data_flow/py_tasklet.h"
#include "py_structured_sdfg.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/passes/rpc/rpc_context.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/targets/cuda/plugin.h"
#include "transformations/py_replayer.h"
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
#include <sdfg/targets/onnx/plugin.h>

#include "sdfg/passes/rpc/daisytuner_rpc_context.h"
#include "sdfg/passes/rpc/rpc_loop_opt.h"
#include "sdfg/passes/scheduler/cuda_scheduler.h"

namespace py = pybind11;
using namespace sdfg::types;

PYBIND11_MODULE(_sdfg, m) {
    m.doc() = "A JIT compiler for Numpy-based Python programs targeting various hardware backends.";

    sdfg::codegen::register_default_dispatchers();
    sdfg::serializer::register_default_serializers();
    sdfg::omp::register_omp_plugin();
    sdfg::onnx::register_onnx_plugin();
    sdfg::highway::register_highway_plugin();
    sdfg::cuda::register_cuda_plugin();

    register_types(m);
    register_tasklet(m);
    register_cmath(m);
    register_analysis(m);
    register_replayer(m);

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


    py::class_<
        sdfg::passes::rpc::DaisytunerTransfertuningRpcContext,
        sdfg::passes::rpc::SimpleRpcContext>(m, "DaisytunerTransfertuningRpcContext")
        .def(py::init<std::string>(), py::arg("license_token"))
        .def_static(
            "from_docc_config",
            sdfg::passes::rpc::DaisytunerTransfertuningRpcContext::from_docc_config,
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
        .def("dump", &PyStructuredSDFG::dump, py::arg("path"))
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
        .def("loop_report", &PyStructuredSDFG::loop_report, "Get loop statistics from the SDFG");

    // Register StructuredSDFGBuilder class
    py::class_<PyStructuredSDFGBuilder>(m, "StructuredSDFGBuilder")
        .def(py::init<const std::string&>(), py::arg("name"), "Create a StructuredSDFGBuilder with the given name")
        .def(
            py::init<const std::string&, const IType&>(),
            py::arg("name"),
            py::arg("return_type"),
            "Create a StructuredSDFGBuilder with the given name and return type"
        )
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
            py::arg("debug_info") = sdfg::DebugInfo()
        )
        .def("begin_else", &PyStructuredSDFGBuilder::begin_else, py::arg("debug_info") = sdfg::DebugInfo())
        .def("end_if", &PyStructuredSDFGBuilder::end_if)
        .def("begin_while", &PyStructuredSDFGBuilder::begin_while, py::arg("debug_info") = sdfg::DebugInfo())
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
            py::arg("debug_info") = sdfg::DebugInfo()
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
        .def("add_block", &PyStructuredSDFGBuilder::add_block, py::arg("debug_info") = sdfg::DebugInfo())
        .def(
            "add_access",
            &PyStructuredSDFGBuilder::add_access,
            py::arg("block_ptr"),
            py::arg("name"),
            py::arg("debug_info") = sdfg::DebugInfo()
        )
        .def(
            "add_constant",
            &PyStructuredSDFGBuilder::add_constant,
            py::arg("block_ptr"),
            py::arg("value"),
            py::arg("type"),
            py::arg("debug_info") = sdfg::DebugInfo()
        )
        .def(
            "add_tasklet",
            &PyStructuredSDFGBuilder::add_tasklet,
            py::arg("block_ptr"),
            py::arg("code"),
            py::arg("inputs"),
            py::arg("outputs"),
            py::arg("debug_info") = sdfg::DebugInfo()
        )
        .def(
            "add_cmath",
            &PyStructuredSDFGBuilder::add_cmath,
            py::arg("block_ptr"),
            py::arg("func"),
            py::arg("debug_info") = sdfg::DebugInfo()
        )
        .def(
            "add_malloc",
            &PyStructuredSDFGBuilder::add_malloc,
            py::arg("block_ptr"),
            py::arg("size"),
            py::arg("debug_info") = sdfg::DebugInfo()
        )
        .def(
            "add_memset",
            &PyStructuredSDFGBuilder::add_memset,
            py::arg("block_ptr"),
            py::arg("value"),
            py::arg("num"),
            py::arg("debug_info") = sdfg::DebugInfo()
        )
        .def(
            "add_memcpy",
            &PyStructuredSDFGBuilder::add_memcpy,
            py::arg("block_ptr"),
            py::arg("count"),
            py::arg("debug_info") = sdfg::DebugInfo()
        )
        .def("get_sizeof", &PyStructuredSDFGBuilder::get_sizeof, py::arg("type"))
        .def(
            "add_reference_memlet",
            &PyStructuredSDFGBuilder::add_reference_memlet,
            py::arg("block_ptr"),
            py::arg("src_ptr"),
            py::arg("dst_ptr"),
            py::arg("subset") = "",
            py::arg("type") = nullptr,
            py::arg("debug_info") = sdfg::DebugInfo()
        )
        .def(
            "add_memlet",
            [](PyStructuredSDFGBuilder& self,
               size_t block_ptr,
               size_t src_ptr,
               std::string src_conn,
               size_t dst_ptr,
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
                self.add_memlet(block_ptr, src_ptr, src_conn, dst_ptr, dst_conn, subset, type, debug_info);
            },
            py::arg("block_ptr"),
            py::arg("src_ptr"),
            py::arg("src_conn"),
            py::arg("dst_ptr"),
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
}
