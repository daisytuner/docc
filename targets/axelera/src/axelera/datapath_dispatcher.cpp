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

bool is_integer(const symbolic::Expression& expr) {
    return SymEngine::is_a<SymEngine::Integer>(*expr);
}

bool is_integer(const symbolic::Expression& expr, int64_t value) {
    return is_integer(expr) && SymEngine::down_cast<const SymEngine::Integer&>(*expr).as_int() == value;
}

int64_t as_int(const symbolic::Expression& expr) {
    return SymEngine::down_cast<const SymEngine::Integer&>(*expr).as_int();
}

structured_control_flow::Map* only_child_loop(structured_control_flow::Map& loop, AxeleraLevel level) {
    if (loop.root().size() != 1) {
        return nullptr;
    }
    auto child = dynamic_cast<structured_control_flow::Map*>(&loop.root().at(0));
    return child && ScheduleType_Axelera::is(child->schedule_type(), level) ? child : nullptr;
}

} // namespace

AxeleraMapDispatcher::AxeleraMapDispatcher(
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

void AxeleraMapDispatcher::dispatch_node(
    codegen::PrettyPrinter& main_stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto lang = dynamic_cast<AxScriptLanguageExtension*>(&language_extension_);
    if (!lang) {
        throw std::runtime_error("AXELERA loops require the axscript codegen, not " + language_extension_.language());
    }
    if (ScheduleType_Axelera::is(map_.schedule_type(), AxeleraLevel::Core)) {
        dispatch_cores(*lang, main_stream);
    } else if (ScheduleType_Axelera::is(map_.schedule_type(), AxeleraLevel::Stream)) {
        dispatch_stream(*lang, main_stream, map_, map_.init());
    } else {
        unsupported("a lane loop outside a datapath loop");
    }
}

void AxeleraMapDispatcher::dispatch_cores(AxScriptLanguageExtension& lang, codegen::PrettyPrinter& stream) {
    auto cores = ScheduleType_Axelera::num_cores(map_.schedule_type());
    auto stream_loop = only_child_loop(map_, AxeleraLevel::Stream);
    if (!is_integer(map_.init(), 0) || !is_integer(map_.stride()) || !is_integer(map_.num_iterations(), cores) ||
        !stream_loop) {
        unsupported("a core loop that is not one contiguous chunk per core around one datapath loop");
    }

    std::string all_cores;
    for (int64_t c = 0; c < cores; ++c) {
        all_cores += (c > 0 ? ", " : "") + std::to_string(c);
    }
    auto chunk = as_int(map_.stride());
    for (int64_t c = 0; c < cores; ++c) {
        stream << "with A.AICore(core=" << c << "):" << std::endl;
        stream.changeIndent(+4);
        auto first_element = symbolic::subs(stream_loop->init(), map_.indvar(), symbolic::integer(c * chunk));
        dispatch_stream(lang, stream, *stream_loop, first_element);
        // alto only accepts barriers inside core regions; each core waits for all others.
        stream << "A.barrier(" << all_cores << ")" << std::endl;
        stream.changeIndent(-4);
    }
}

void AxeleraMapDispatcher::dispatch_stream(
    AxScriptLanguageExtension& lang,
    codegen::PrettyPrinter& stream,
    structured_control_flow::Map& stream_loop,
    const symbolic::Expression& first_element
) {
    auto block = stream_loop.indvar();
    auto count = stream_loop.num_iterations();
    auto lane_loop = only_child_loop(stream_loop, AxeleraLevel::Lane);
    if (!is_integer(stream_loop.stride(), PWORD_LANES) || !is_integer(count) || !lane_loop ||
        !symbolic::eq(lane_loop->init(), block) || !is_integer(lane_loop->stride(), 1) ||
        !is_integer(lane_loop->num_iterations(), PWORD_LANES)) {
        unsupported("a datapath loop that is not a whole number of PWORDs around one lane loop");
    }
    auto lane = lane_loop->indvar();
    auto plan = plan_datapath(sdfg_, lane_loop->root(), lane);

    // A.grid counts PWORDs from zero where the loop counted elements from first_element.
    auto pwords = symbolic::integer(PWORD_LANES);
    auto element_offset = symbolic::add(first_element, symbolic::mul(pwords, block));
    auto pword_access = [&](const DatapathStream& access) {
        auto first = symbolic::subs(symbolic::subs(access.index, lane, block), block, element_offset);
        auto index = symbolic::expand(SymEngine::div(first, pwords));
        auto offset = symbolic::expand(symbolic::subs(index, block, symbolic::zero()));
        auto step = symbolic::expand(symbolic::sub(symbolic::subs(index, block, symbolic::one()), offset));
        if (!is_integer(offset) || !is_integer(step) ||
            !symbolic::eq(index, symbolic::expand(symbolic::add(symbolic::mul(step, block), offset)))) {
            unsupported("access " + access.container + "[" + access.index->__str__() + "], which is not PWORD-aligned,");
        }
        return access.container + "[" + lang.expression(index) + "]";
    };

    auto id = std::to_string(lang.next_datapath());
    auto input_stream = [&](size_t k) {
        return "s" + id + "_" + std::to_string(k);
    };
    auto output_stream = "s" + id + "_out";
    auto registers = "r" + id;
    auto reg = [&](size_t r) {
        return registers + "[" + std::to_string(r) + "]";
    };
    auto dtype = "\"" + lang.vector_type(types::PrimitiveType::Float) + "\"";
    auto dpu_dtype = "\"" + AxScriptLanguageExtension::dpu_type() + "\"";

    stream << "for " << block->get_name() << " in A.grid(" << lang.expression(count) << "):" << std::endl;
    stream.changeIndent(+4);
    stream << "with A.Datapath():" << std::endl;
    stream.changeIndent(+4);
    for (size_t k = 0; k < plan.inputs.size(); ++k) {
        stream << input_stream(k) << " = A.decl_stream(" << dtype << ")" << std::endl;
    }
    stream << output_stream << " = A.decl_stream(" << dtype << ")" << std::endl;

    for (size_t k = 0; k < plan.inputs.size(); ++k) {
        stream << "with A.IFD(num=" << k << "):" << std::endl;
        stream.changeIndent(+4);
        stream << input_stream(k) << ".push(" << pword_access(plan.inputs[k]) << ")" << std::endl;
        stream.changeIndent(-4);
    }

    stream << "with A.DPU():" << std::endl;
    stream.changeIndent(+4);
    stream << registers << " = A.array(" << dpu_dtype << ")" << std::endl;
    for (auto& instruction : plan.program) {
        std::string value;
        if (instruction.input_stream) {
            value = "A.cast(" + input_stream(*instruction.input_stream) + ".pop(), " + dpu_dtype + ")";
        } else {
            std::vector<std::string> operands;
            for (auto operand : instruction.operands) {
                operands.push_back(reg(operand));
            }
            value = lang.dpu_operation(instruction.code, operands);
        }
        stream << reg(instruction.destination) << " = " << value << std::endl;
    }
    stream << output_stream << ".push(" << reg(plan.result) << ")" << std::endl;
    stream.changeIndent(-4);

    stream << "with A.ODR():" << std::endl;
    stream.changeIndent(+4);
    stream << pword_access(plan.output) << " = " << output_stream << ".pop()" << std::endl;
    stream.changeIndent(-12);
}

} // namespace axelera
} // namespace sdfg
