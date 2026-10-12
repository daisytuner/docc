#include "docc/target/axelera/datapath_dispatcher.h"

#include "docc/target/axelera/axscript_language_extension.h"
#include "docc/target/axelera/datapath.h"
#include "docc/target/axelera/schedule.h"

namespace sdfg {
namespace axelera {

namespace {

[[noreturn]] void unsupported(const std::string& what) {
    throw std::runtime_error("AxScript codegen: " + what + " is not supported");
}

bool is_integer(const symbolic::Expression& expr, int64_t value) {
    return SymEngine::is_a<SymEngine::Integer>(*expr) &&
           SymEngine::down_cast<const SymEngine::Integer&>(*expr).as_int() == value;
}

bool is_integer(const symbolic::Expression& expr) {
    return SymEngine::is_a<SymEngine::Integer>(*expr);
}

} // namespace

DatapathDispatcher::DatapathDispatcher(
    codegen::LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Map& node,
    codegen::InstrumentationPlan& instrumentation_plan,
    codegen::ArgCapturePlan& arg_capture_plan
)
    : NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan),
      map_(node) {
}

void DatapathDispatcher::dispatch_node(
    codegen::PrettyPrinter& main_stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto lang = dynamic_cast<AxScriptLanguageExtension*>(&language_extension_);
    if (!lang) {
        throw std::runtime_error("Axelera datapaths require the axscript codegen, not " + language_extension_.language());
    }

    auto block = map_.indvar();
    auto count = map_.num_iterations();
    if (!is_integer(map_.init(), 0) || !is_integer(map_.stride(), PWORD_LANES) || !is_integer(count)) {
        unsupported("a datapath loop that does not step over whole PWORDs from zero");
    }
    auto lane_loop = map_.root().size() == 1 ? dynamic_cast<structured_control_flow::Map*>(&map_.root().at(0))
                                             : nullptr;
    if (!lane_loop || lane_loop->schedule_type().value() != ScheduleType_Axelera_Lane::value() ||
        !symbolic::eq(lane_loop->init(), block) || !is_integer(lane_loop->stride(), 1) ||
        !is_integer(lane_loop->num_iterations(), PWORD_LANES)) {
        unsupported("a datapath loop whose body is not one loop over the lanes of a PWORD");
    }
    auto lane = lane_loop->indvar();
    auto plan = plan_datapath(sdfg_, lane_loop->root(), lane);

    // A.grid counts PWORDs where the loop counted elements: element offset b is PWORD index b / PWORD_LANES.
    auto pwords = symbolic::integer(PWORD_LANES);
    auto pword_access = [&](const DatapathStream& stream) {
        auto first = symbolic::subs(stream.index, lane, block);
        auto index =
            symbolic::expand(SymEngine::div(symbolic::subs(first, block, symbolic::mul(pwords, block)), pwords));
        auto offset = symbolic::expand(symbolic::subs(index, block, symbolic::zero()));
        auto step = symbolic::expand(symbolic::sub(symbolic::subs(index, block, symbolic::one()), offset));
        if (!is_integer(offset) || !is_integer(step) ||
            !symbolic::eq(index, symbolic::expand(symbolic::add(symbolic::mul(step, block), offset)))) {
            unsupported("access " + stream.container + "[" + stream.index->__str__() + "], which is not PWORD-aligned,");
        }
        return stream.container + "[" + lang->expression(index) + "]";
    };

    auto id = std::to_string(lang->next_datapath());
    auto input_stream = [&](size_t k) {
        return "s" + id + "_" + std::to_string(k);
    };
    auto output_stream = "s" + id + "_out";
    auto registers = "r" + id;
    auto reg = [&](size_t r) {
        return registers + "[" + std::to_string(r) + "]";
    };
    auto dtype = "\"" + lang->vector_type(types::PrimitiveType::Float) + "\"";
    auto dpu_dtype = "\"" + AxScriptLanguageExtension::dpu_type() + "\"";

    main_stream << "for " << block->get_name() << " in A.grid(" << lang->expression(count) << "):" << std::endl;
    main_stream.changeIndent(+4);
    main_stream << "with A.Datapath():" << std::endl;
    main_stream.changeIndent(+4);
    for (size_t k = 0; k < plan.inputs.size(); ++k) {
        main_stream << input_stream(k) << " = A.decl_stream(" << dtype << ")" << std::endl;
    }
    main_stream << output_stream << " = A.decl_stream(" << dtype << ")" << std::endl;

    for (size_t k = 0; k < plan.inputs.size(); ++k) {
        main_stream << "with A.IFD(num=" << k << "):" << std::endl;
        main_stream.changeIndent(+4);
        main_stream << input_stream(k) << ".push(" << pword_access(plan.inputs[k]) << ")" << std::endl;
        main_stream.changeIndent(-4);
    }

    main_stream << "with A.DPU():" << std::endl;
    main_stream.changeIndent(+4);
    main_stream << registers << " = A.array(" << dpu_dtype << ")" << std::endl;
    for (auto& instruction : plan.program) {
        std::string value;
        if (instruction.input_stream) {
            value = "A.cast(" + input_stream(*instruction.input_stream) + ".pop(), " + dpu_dtype + ")";
        } else {
            std::vector<std::string> operands;
            for (auto operand : instruction.operands) {
                operands.push_back(reg(operand));
            }
            value = lang->dpu_operation(instruction.code, operands);
        }
        main_stream << reg(instruction.destination) << " = " << value << std::endl;
    }
    main_stream << output_stream << ".push(" << reg(plan.result) << ")" << std::endl;
    main_stream.changeIndent(-4);

    main_stream << "with A.ODR():" << std::endl;
    main_stream.changeIndent(+4);
    main_stream << pword_access(plan.output) << " = " << output_stream << ".pop()" << std::endl;
    main_stream.changeIndent(-12);
}

} // namespace axelera
} // namespace sdfg
