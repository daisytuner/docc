#include <iostream>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include <sdfg/analysis/analysis.h>
#include <sdfg/codegen/code_generators/c_code_generator.h>
#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/codegen/instrumentation/arg_capture_plan.h>
#include <sdfg/codegen/instrumentation/instrumentation_plan.h>
#include <sdfg/serializer/json_serializer.h>

int main(int argc, char* argv[]) {
    sdfg::codegen::register_default_dispatchers();
    sdfg::serializer::register_default_serializers();

    std::stringstream jsonstream;
    for (std::string line; std::getline(std::cin, line);) {
        jsonstream << line;
    }

    nlohmann::json json = nlohmann::json::parse(jsonstream);

    sdfg::serializer::JSONSerializer serializer;
    auto sdfg = serializer.deserialize(json);

    sdfg::analysis::AnalysisManager analysis_manager(*sdfg);
    auto instrumentation_plan = sdfg::codegen::InstrumentationPlan::none(*sdfg);
    auto arg_capture_plan = sdfg::codegen::ArgCapturePlan::none(*sdfg);
    sdfg::codegen::CCodeGenerator gen(*sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan);
    if (!gen.generate()) {
        std::cerr << "Could not generate code for SDFG" << std::endl;
        return 1;
    }

    std::cout << gen.includes().str() << gen.globals().str() << gen.function_definition() << " {" << std::endl
              << gen.main().str() << "}" << std::endl;

    return 0;
}
