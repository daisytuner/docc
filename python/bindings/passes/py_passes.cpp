#include "py_passes.h"

#include <sstream>

#include <sdfg/passes/offloading/lanes_map_predication.h>
#include <sdfg/passes/offloading/sync_condition_propagation.h>
#include <sdfg/passes/pass.h>
#include <sdfg/passes/symbolic/symbol_promotion.h>
#include <sdfg/passes/symbolic/symbol_propagation.h>
#include <sdfg/scheduling/loops/loop_scheduling_pass.h>
#include <sdfg/tiles/transformations/tile_guard_normalization.h>
#include <sdfg/transformations/recorder.h>

#include "analysis/py_analysis.h"
#include "builder/py_structured_sdfg_builder.h"

using namespace sdfg::passes;

namespace {

struct OwnedLoopSchedulers {
    std::vector<std::shared_ptr<scheduler::LoopScheduler>> loop_schedulers;
};

/// A LoopSchedulingPass that owns the loop schedulers created by the targets.
class TargetLoopSchedulingPass : private OwnedLoopSchedulers, public sdfg::scheduling::loops::LoopSchedulingPass {
    static std::vector<scheduler::LoopScheduler*>
    pointers(const std::vector<std::shared_ptr<scheduler::LoopScheduler>>& loop_schedulers) {
        std::vector<scheduler::LoopScheduler*> result;
        for (auto& loop_scheduler : loop_schedulers) {
            result.push_back(loop_scheduler.get());
        }
        return result;
    }

public:
    TargetLoopSchedulingPass(
        std::vector<std::shared_ptr<scheduler::LoopScheduler>> loop_schedulers, bool offload_unknown_sizes
    )
        : OwnedLoopSchedulers{std::move(loop_schedulers)},
          LoopSchedulingPass(pointers(this->loop_schedulers), nullptr, offload_unknown_sizes) {
    }
};

} // namespace

void register_passes(py::module& m, sdfg::plugins::Context& context) {
    // Base Pass class (abstract)
    py::class_<Pass>(m, "Pass")
        .def_property_readonly("name", &Pass::name, "Get the pass name")
        .def(
            "run",
            [](Pass& self, PyStructuredSDFGBuilder& builder, PyAnalysisManager& analysis_manager) {
                return self.run(builder.builder(), analysis_manager.manager());
            },
            py::arg("builder"),
            py::arg("analysis_manager"),
            "Run the pass, returning True if the SDFG was modified"
        );

    // SymbolPropagation pass
    py::class_<SymbolPropagation, Pass>(m, "SymbolPropagation")
        .def(
            py::init<>(),
            "Create a symbol propagation pass.\n\n"
            "Propagates symbolic assignments through the SDFG, replacing symbol\n"
            "uses with their assigned values where appropriate."
        )
        .def("__repr__", [](SymbolPropagation& self) {
            std::ostringstream oss;
            oss << "<SymbolPropagation name='" << self.name() << "'>";
            return oss.str();
        });

    // SymbolPromotion pass
    py::class_<SymbolPromotion, Pass>(m, "SymbolPromotion")
        .def(
            py::init<>(),
            "Create a symbol promotion pass.\n\n"
            "Promotes symbols from dataflow to symbolic expressions."
        )
        .def("__repr__", [](SymbolPromotion& self) {
            std::ostringstream oss;
            oss << "<SymbolPromotion name='" << self.name() << "'>";
            return oss.str();
        });

    // SyncConditionPropagation pass
    py::class_<SyncConditionPropagation, Pass>(m, "SyncConditionPropagation")
        .def(
            py::init<>(),
            "Create a synchronization condition propagation pass.\n\n"
            "Propagates synchronization conditions into GPU-scheduled maps."
        )
        .def("__repr__", [](SyncConditionPropagation& self) {
            std::ostringstream oss;
            oss << "<SyncConditionPropagation name='" << self.name() << "'>";
            return oss.str();
        });

    // LanesMapPredication pass
    py::class_<LanesMapPredication, Pass>(m, "LanesMapPredication")
        .def(
            py::init<>(),
            "Create a lanes-map predication pass.\n\n"
            "Every lane of an X_BLOCK map with lanes > 1 executes the body of its\n"
            "iteration. Wraps the blocks writing lane-shared (device) memory in a\n"
            "leader-lane guard (threadIdx.x % lanes == 0) and inserts a wave barrier\n"
            "where other lanes read the result. Code generation runs it after\n"
            "materializing reduction buffers; it is idempotent."
        )
        .def("__repr__", [](LanesMapPredication& self) {
            std::ostringstream oss;
            oss << "<LanesMapPredication name='" << self.name() << "'>";
            return oss.str();
        });

    // TileGuardNormalization pass
    py::class_<sdfg::tiles::TileGuardNormalization, Pass>(m, "TileGuardNormalization")
        .def(
            py::init<>(),
            "Create a tile-copy guard normalization pass.\n\n"
            "Re-discharges every TileCopyNode's ragged boundary guard against the\n"
            "current assumptions, dropping guard dims a later pass (loop peeling,\n"
            "condition propagation) has proven fully covering."
        )
        .def("__repr__", [](sdfg::tiles::TileGuardNormalization& self) {
            std::ostringstream oss;
            oss << "<TileGuardNormalization name='" << self.name() << "'>";
            return oss.str();
        });

    // LoopSchedulingPass: run the target loop schedulers over the SDFG.
    py::class_<sdfg::scheduling::loops::LoopSchedulingPass, Pass>(m, "LoopSchedulingPass")
        .def(
            py::init([&context](const std::vector<std::string>& targets, bool offload_unknown_sizes) {
                std::vector<std::shared_ptr<scheduler::LoopScheduler>> loop_schedulers;
                for (const auto& target : targets) {
                    auto* handler = context.get_target_handler(target);
                    if (!handler) {
                        throw std::runtime_error("Unknown target '" + target + "'");
                    }
                    auto target_loop_schedulers = handler->scheduling.create_loop_schedulers({.target = target});
                    loop_schedulers
                        .insert(loop_schedulers.end(), target_loop_schedulers.begin(), target_loop_schedulers.end());
                }
                return std::unique_ptr<sdfg::scheduling::loops::LoopSchedulingPass>(
                    new TargetLoopSchedulingPass(std::move(loop_schedulers), offload_unknown_sizes)
                );
            }),
            py::arg("targets"),
            py::arg("offload_unknown_sizes") = false,
            "Create a loop-scheduling pass for the given targets.\n\n"
            "Applies the loop schedulers of each target (e.g. 'openmp',\n"
            "'cuda', 'rocm') to the SDFG's loop nests -- the same\n"
            "schedulers the compiler uses, i.e. the default schedule / baseline."
        )
        .def(
            "set_recorder",
            [](sdfg::scheduling::loops::LoopSchedulingPass& self, sdfg::transformations::Recorder& recorder) {
                self.set_recorder(&recorder);
            },
            py::arg("recorder"),
            "Attach a Recorder to capture the scheduling transformations."
        )
        .def("__repr__", [](sdfg::scheduling::loops::LoopSchedulingPass&) {
            return "<LoopSchedulingPass>";
        });
}
