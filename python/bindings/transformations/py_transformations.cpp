#include "py_transformations.h"

#include <nlohmann/json.hpp>
#include <sstream>

#include <sdfg/data_flow/access_node.h>
#include <sdfg/transformations/in_local_storage.h>
#include <sdfg/transformations/loop_distribute.h>
#include <sdfg/transformations/loop_interchange.h>
#include <sdfg/transformations/loop_peeling.h>
#include <sdfg/transformations/loop_skewing.h>
#include <sdfg/transformations/loop_tiling.h>
#include <sdfg/transformations/map_collapse.h>
#include <sdfg/transformations/map_fusion.h>
#include <sdfg/transformations/offloading/cuda_parallelize_nested_map.h>
#include <sdfg/transformations/offloading/cuda_transform.h>
#include <sdfg/transformations/out_local_storage.h>
#include <sdfg/transformations/recorder.h>
#include <sdfg/transformations/tile_fusion.h>
#include <sdfg/transformations/transformation.h>
#include <sdfg/transformations/vectorize_transform.h>
#include <sdfg/types/type.h>

#include "analysis/py_analysis.h"
#include "builder/py_structured_sdfg_builder.h"

using namespace sdfg::transformations;
using namespace sdfg::structured_control_flow;

void register_transformations(py::module& m) {
    // Base Transformation class (abstract)
    py::class_<Transformation>(m, "Transformation")
        .def_property_readonly("name", &Transformation::name, "Get the transformation name")
        .def(
            "can_be_applied",
            [](Transformation& self, PyStructuredSDFGBuilder& builder, PyAnalysisManager& analysis_manager) {
                return self.can_be_applied(builder.builder(), analysis_manager.manager());
            },
            py::arg("builder"),
            py::arg("analysis_manager"),
            "Check if this transformation can be applied"
        )
        .def(
            "apply",
            [](Transformation& self, PyStructuredSDFGBuilder& builder, PyAnalysisManager& analysis_manager) {
                self.apply(builder.builder(), analysis_manager.manager());
            },
            py::arg("builder"),
            py::arg("analysis_manager"),
            "Apply the transformation"
        )
        .def(
            "try_apply",
            [](Transformation& self, PyStructuredSDFGBuilder& builder, PyAnalysisManager& analysis_manager) {
                return self.try_apply(builder.builder(), analysis_manager.manager());
            },
            py::arg("builder"),
            py::arg("analysis_manager"),
            "Try to apply the transformation, returning True if successful"
        )
        .def(
            "to_json",
            [](const Transformation& self) {
                nlohmann::json j;
                self.to_json(j);
                return j.dump();
            },
            "Serialize the transformation to a JSON string"
        );

    // LoopTiling transformation
    py::class_<LoopTiling, Transformation>(m, "LoopTiling")
        .def(
            py::init<StructuredLoop&, size_t>(),
            py::arg("loop"),
            py::arg("tile_size"),
            "Create a loop tiling transformation.\n\n"
            "Args:\n"
            "    loop: The loop to tile\n"
            "    tile_size: The tile size (must be > 1)"
        )
        .def_property_readonly(
            "inner_loop",
            &LoopTiling::inner_loop,
            py::return_value_policy::reference,
            "Get the inner (tiled) loop after apply"
        )
        .def_property_readonly(
            "outer_loop",
            &LoopTiling::outer_loop,
            py::return_value_policy::reference,
            "Get the outer (tile) loop after apply"
        )
        .def("__repr__", [](const LoopTiling& t) {
            std::ostringstream oss;
            oss << "<LoopTiling name='" << t.name() << "'>";
            return oss.str();
        });

    // LoopInterchange transformation
    py::class_<LoopInterchange, Transformation>(m, "LoopInterchange")
        .def(
            py::init<StructuredLoop&, StructuredLoop&>(),
            py::arg("outer_loop"),
            py::arg("inner_loop"),
            "Create a loop interchange transformation.\n\n"
            "Args:\n"
            "    outer_loop: The outer loop to interchange\n"
            "    inner_loop: The inner loop to interchange"
        )
        .def("__repr__", [](const LoopInterchange& t) {
            std::ostringstream oss;
            oss << "<LoopInterchange name='" << t.name() << "'>";
            return oss.str();
        });

    // LoopDistribute transformation
    py::class_<LoopDistribute, Transformation>(m, "LoopDistribute")
        .def(
            py::init<StructuredLoop&>(),
            py::arg("loop"),
            "Create a loop distribution transformation.\n\n"
            "Args:\n"
            "    loop: The loop to distribute"
        )
        .def("__repr__", [](const LoopDistribute& t) {
            std::ostringstream oss;
            oss << "<LoopDistribute name='" << t.name() << "'>";
            return oss.str();
        });

    // LoopSkewing transformation
    py::class_<LoopSkewing, Transformation>(m, "LoopSkewing")
        .def(
            py::init<StructuredLoop&, StructuredLoop&, int>(),
            py::arg("outer_loop"),
            py::arg("inner_loop"),
            py::arg("skew_factor") = 1,
            "Create a loop skewing transformation.\n\n"
            "Args:\n"
            "    outer_loop: The outer loop\n"
            "    inner_loop: The inner loop\n"
            "    skew_factor: The skewing factor (default: 1)"
        )
        .def("__repr__", [](const LoopSkewing& t) {
            std::ostringstream oss;
            oss << "<LoopSkewing name='" << t.name() << "'>";
            return oss.str();
        });

    // OutLocalStorage transformation
    py::class_<OutLocalStorage, Transformation>(m, "OutLocalStorage")
        .def(
            py::init<StructuredLoop&, const sdfg::data_flow::AccessNode&>(),
            py::arg("loop"),
            py::arg("access_node"),
            "Create an out-of-loop local storage transformation.\n\n"
            "Args:\n"
            "    loop: The loop to optimize\n"
            "    access_node: The access node to extract to local storage"
        )
        .def("__repr__", [](const OutLocalStorage& t) {
            std::ostringstream oss;
            oss << "<OutLocalStorage name='" << t.name() << "'>";
            return oss.str();
        });

    // MapCollapse transformation
    py::class_<MapCollapse, Transformation>(m, "MapCollapse")
        .def(
            py::init<Map&, size_t>(),
            py::arg("loop"),
            py::arg("count"),
            "Create a map collapse transformation.\n\n"
            "Args:\n"
            "    loop: The outermost map of the nest to collapse\n"
            "    count: The number of maps to collapse (must be >= 2)"
        )
        .def_property_readonly(
            "collapsed_loop",
            &MapCollapse::collapsed_loop,
            py::return_value_policy::reference,
            "Get the collapsed map after apply"
        )
        .def("__repr__", [](const MapCollapse& t) {
            std::ostringstream oss;
            oss << "<MapCollapse name='" << t.name() << "'>";
            return oss.str();
        });

    // MapFusion transformation
    py::class_<MapFusion, Transformation>(m, "MapFusion")
        .def(
            py::init<Map&, StructuredLoop&>(),
            py::arg("first_map"),
            py::arg("second_loop"),
            "Create a map fusion transformation.\n\n"
            "Args:\n"
            "    first_map: The first (producer) map\n"
            "    second_loop: The second (consumer) loop"
        )
        .def("__repr__", [](const MapFusion& t) {
            std::ostringstream oss;
            oss << "<MapFusion name='" << t.name() << "'>";
            return oss.str();
        });

    // TileFusion transformation
    py::class_<TileFusion, Transformation>(m, "TileFusion")
        .def(
            py::init<Map&, Map&>(),
            py::arg("first_map"),
            py::arg("second_map"),
            "Create a tile fusion transformation.\n\n"
            "Args:\n"
            "    first_map: The first (producer) tiled map\n"
            "    second_map: The second (consumer) tiled map"
        )
        .def_property_readonly(
            "fused_loop", &TileFusion::fused_loop, py::return_value_policy::reference, "Get the fused loop after apply"
        )
        .def_property_readonly("radius", &TileFusion::radius, "Get the computed radius")
        .def("__repr__", [](const TileFusion& t) {
            std::ostringstream oss;
            oss << "<TileFusion name='" << t.name() << "'>";
            return oss.str();
        });

    // CUDATransform transformation (offload a top-level map to a CUDA kernel, X grid dim)
    py::class_<sdfg::cuda::CUDATransform, Transformation>(m, "CUDATransform")
        .def(
            py::init<Map&, int, bool>(),
            py::arg("map"),
            py::arg("block_size") = 32,
            py::arg("allow_dynamic_sizes") = false,
            "Create a CUDA offload transformation.\n\n"
            "Args:\n"
            "    map: The top-level map to offload to a CUDA kernel (X dimension)\n"
            "    block_size: Threads per block along X (default: 32)\n"
            "    allow_dynamic_sizes: Permit non-constant iteration counts (default: False)"
        )
        .def("__repr__", [](const sdfg::cuda::CUDATransform& t) {
            std::ostringstream oss;
            oss << "<CUDATransform name='" << t.name() << "'>";
            return oss.str();
        });

    // CUDAParallelizeNestedMap transformation (add a nested map as the next grid dim)
    py::class_<CUDAParallelizeNestedMap, Transformation>(m, "CUDAParallelizeNestedMap")
        .def(
            py::init<Map&, size_t>(),
            py::arg("loop"),
            py::arg("block_size"),
            "Parallelize a nested map as the next CUDA grid dimension (parent X->Y, Y->Z).\n\n"
            "Args:\n"
            "    loop: The nested (sequential) map to parallelize\n"
            "    block_size: Threads per block along this dimension"
        )
        .def("__repr__", [](const CUDAParallelizeNestedMap& t) {
            std::ostringstream oss;
            oss << "<CUDAParallelizeNestedMap name='" << t.name() << "'>";
            return oss.str();
        });

    // LoopPeeling transformation
    py::class_<LoopPeeling, Transformation>(m, "LoopPeeling")
        .def(
            py::init<StructuredLoop&>(),
            py::arg("loop"),
            "Create a loop peeling transformation.\n\n"
            "Args:\n"
            "    loop: The loop with compound conditions to peel"
        )
        .def("__repr__", [](const LoopPeeling& t) {
            std::ostringstream oss;
            oss << "<LoopPeeling name='" << t.name() << "'>";
            return oss.str();
        });

    // VectorizeTransform transformation
    py::class_<VectorizeTransform, Transformation>(m, "VectorizeTransform")
        .def(
            py::init<StructuredLoop&>(),
            py::arg("loop"),
            "Create a vectorize transformation.\n\n"
            "Args:\n"
            "    loop: The sequential loop to vectorize"
        )
        .def("__repr__", [](const VectorizeTransform& t) {
            std::ostringstream oss;
            oss << "<VectorizeTransform name='" << t.name() << "'>";
            return oss.str();
        });

    // InLocalStorage transformation (stage a read tile into local/shared storage)
    py::class_<InLocalStorage, Transformation>(m, "InLocalStorage")
        .def(
            py::init([](StructuredLoop& loop,
                        const sdfg::data_flow::AccessNode& access_node,
                        const std::string& storage_type) {
                sdfg::types::StorageType st = sdfg::types::StorageType::CPU_Stack();
                if (storage_type == "NV_Shared") {
                    st = sdfg::types::StorageType::NV_Shared();
                } else if (storage_type == "CPU_Stack") {
                    st = sdfg::types::StorageType::CPU_Stack();
                } else {
                    throw std::invalid_argument("Unsupported storage_type: " + storage_type);
                }
                return std::make_unique<InLocalStorage>(loop, access_node, st);
            }),
            py::arg("loop"),
            py::arg("access_node"),
            py::arg("storage_type") = "CPU_Stack",
            "Create an in-local-storage transformation (stage a read tile).\n\n"
            "Args:\n"
            "    loop: The loop defining the localization scope\n"
            "    access_node: The access node for the container to localize\n"
            "    storage_type: 'CPU_Stack' (registers) or 'NV_Shared' (shared memory)"
        )
        .def("__repr__", [](const InLocalStorage& t) {
            std::ostringstream oss;
            oss << "<InLocalStorage name='" << t.name() << "'>";
            return oss.str();
        });

    // Recorder class for recording transformation history
    py::class_<Recorder>(m, "Recorder")
        .def(py::init<>(), "Create an empty transformation recorder")
        .def(
            "apply",
            [](Recorder& self,
               Transformation& transformation,
               PyStructuredSDFGBuilder& builder,
               PyAnalysisManager& analysis_manager,
               bool skip_if_not_applicable) {
                // Delegate to the C++ ``record`` so the virtual ``enrich`` hook
                // runs: the base Recorder attaches ``loop_info`` and subclasses
                // such as EmbeddingRecorder additionally attach node embeddings.
                return self
                    .record(transformation, builder.builder(), analysis_manager.manager(), skip_if_not_applicable);
            },
            py::arg("transformation"),
            py::arg("builder"),
            py::arg("analysis_manager"),
            py::arg("skip_if_not_applicable") = false,
            "Apply a transformation and record it.\n\n"
            "Args:\n"
            "    transformation: The transformation to apply\n"
            "    builder: The SDFG builder\n"
            "    analysis_manager: The analysis manager\n"
            "    skip_if_not_applicable: If True, skip if transformation cannot be applied\n\n"
            "Returns:\n"
            "    True if the transformation was applied, False if skipped"
        )
        .def("save", &Recorder::save, py::arg("path"), "Save the recorded transformation history to a file")
        .def(
            "get_history",
            [](const Recorder& self) { return self.get_history().dump(); },
            "Get the transformation history as a JSON string"
        )
        .def_property_readonly(
            "history",
            [](const Recorder& self) { return self.get_history().dump(); },
            "Get the transformation history as a JSON string"
        )
        .def("__repr__", [](const Recorder& self) {
            std::ostringstream oss;
            oss << "<Recorder transformations=" << self.get_history().size() << ">";
            return oss.str();
        });

    // InvalidTransformationException
    py::register_exception<InvalidTransformationException>(m, "InvalidTransformationException");

    // InvalidTransformationDescriptionException
    py::register_exception<InvalidTransformationDescriptionException>(m, "InvalidTransformationDescriptionException");
}
