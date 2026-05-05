#include "py_transformations.h"

#include <nlohmann/json.hpp>
#include <sstream>

#include <sdfg/data_flow/access_node.h>
#include <sdfg/transformations/loop_distribute.h>
#include <sdfg/transformations/loop_interchange.h>
#include <sdfg/transformations/loop_skewing.h>
#include <sdfg/transformations/loop_tiling.h>
#include <sdfg/transformations/map_fusion.h>
#include <sdfg/transformations/out_local_storage.h>
#include <sdfg/transformations/recorder.h>
#include <sdfg/transformations/tile_fusion.h>
#include <sdfg/transformations/transformation.h>

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
                if (!transformation.can_be_applied(builder.builder(), analysis_manager.manager())) {
                    if (!skip_if_not_applicable) {
                        throw InvalidTransformationException(
                            "Transformation " + transformation.name() + " cannot be applied."
                        );
                    }
                    return false;
                }

                // Record the transformation
                nlohmann::json desc;
                transformation.to_json(desc);
                self.history().push_back(desc);

                // Apply the transformation
                transformation.apply(builder.builder(), analysis_manager.manager());
                return true;
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
