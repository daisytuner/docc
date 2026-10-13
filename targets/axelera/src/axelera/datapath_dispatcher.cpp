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

AxScriptLanguageExtension& axscript(codegen::LanguageExtension& language_extension) {
    auto lang = dynamic_cast<AxScriptLanguageExtension*>(&language_extension);
    if (!lang) {
        throw std::runtime_error("AXELERA loops require the axscript codegen, not " + language_extension.language());
    }
    return *lang;
}

structured_control_flow::StructuredLoop* only_child_loop(structured_control_flow::StructuredLoop& loop, AxeleraLevel level) {
    if (loop.root().size() != 1) {
        return nullptr;
    }
    auto child = dynamic_cast<structured_control_flow::StructuredLoop*>(&loop.root().at(0));
    return child && ScheduleType_Axelera::is(child->schedule_type(), level) ? child : nullptr;
}

std::string join_cores(int64_t cores) {
    std::string result;
    for (int64_t c = 0; c < cores; ++c) {
        result += (c > 0 ? ", " : "") + std::to_string(c);
    }
    return result;
}

/// A Stream loop over PWORDs whose body is one Lane loop, with the element offset of its first PWORD resolved.
struct PwordLoop {
    symbolic::Symbol block;
    int64_t count;
    symbolic::Expression first_element;
    structured_control_flow::StructuredLoop* lanes;
};

PwordLoop pword_loop(structured_control_flow::StructuredLoop& stream_loop, const symbolic::Expression& first_element) {
    auto block = stream_loop.indvar();
    auto count = stream_loop.num_iterations();
    auto lanes = only_child_loop(stream_loop, AxeleraLevel::Lane);
    if (!is_integer(stream_loop.stride(), PWORD_LANES) || !is_integer(count) || !lanes ||
        !symbolic::eq(lanes->init(), block) || !is_integer(lanes->stride(), 1) ||
        !is_integer(lanes->num_iterations(), PWORD_LANES)) {
        unsupported("a datapath loop that is not a whole number of PWORDs around one lane loop");
    }
    return {block, as_int(count), first_element, lanes};
}

/// `container[pword]` for the PWORD a stream accesses in the iteration of the loop's block variable.
std::string pword_access(AxScriptLanguageExtension& lang, const PwordLoop& loop, const DatapathStream& access) {
    auto pwords = symbolic::integer(PWORD_LANES);
    auto element_offset = symbolic::add(loop.first_element, symbolic::mul(pwords, loop.block));
    auto first =
        symbolic::subs(symbolic::subs(access.index, loop.lanes->indvar(), loop.block), loop.block, element_offset);
    auto index = symbolic::expand(SymEngine::div(first, pwords));
    auto offset = symbolic::expand(symbolic::subs(index, loop.block, symbolic::zero()));
    auto step = symbolic::expand(symbolic::sub(symbolic::subs(index, loop.block, symbolic::one()), offset));
    if (!is_integer(offset) || !is_integer(step) ||
        !symbolic::eq(index, symbolic::expand(symbolic::add(symbolic::mul(step, loop.block), offset)))) {
        unsupported("access " + access.container + "[" + access.index->__str__() + "], which is not PWORD-aligned,");
    }
    return access.container + "[" + lang.expression(index) + "]";
}

/// Names of one datapath's streams and register file.
struct DatapathNames {
    std::string id;

    std::string input(size_t k) const {
        return "s" + id + "_" + std::to_string(k);
    }
    std::string output() const {
        return "s" + id + "_out";
    }
    std::string registers() const {
        return "r" + id;
    }
    std::string reg(size_t r) const {
        return registers() + "[" + std::to_string(r) + "]";
    }
};

std::string quoted(const std::string& text) {
    return "\"" + text + "\"";
}

void declare_streams(
    AxScriptLanguageExtension& lang, codegen::PrettyPrinter& out, const DatapathNames& names, size_t inputs
) {
    auto dtype = quoted(lang.vector_type(types::PrimitiveType::Float));
    for (size_t k = 0; k < inputs; ++k) {
        out << names.input(k) << " = A.decl_stream(" << dtype << ")" << std::endl;
    }
    out << names.output() << " = A.decl_stream(" << dtype << ")" << std::endl;
}

void emit_program(
    AxScriptLanguageExtension& lang, codegen::PrettyPrinter& out, const DatapathNames& names, const DatapathPlan& plan
) {
    auto dpu_dtype = quoted(AxScriptLanguageExtension::dpu_type());
    for (auto& instruction : plan.program) {
        std::string value;
        if (instruction.input_stream) {
            value = "A.cast(" + names.input(*instruction.input_stream) + ".pop(), " + dpu_dtype + ")";
        } else {
            std::vector<std::string> operands;
            for (auto operand : instruction.operands) {
                operands.push_back(names.reg(operand));
            }
            value = lang.dpu_operation(instruction.op, operands);
        }
        out << names.reg(instruction.destination) << " = " << value << std::endl;
    }
}

void emit_map_datapath(
    AxScriptLanguageExtension& lang,
    codegen::PrettyPrinter& out,
    const StructuredSDFG& sdfg,
    structured_control_flow::StructuredLoop& stream_loop,
    const symbolic::Expression& first_element
) {
    auto loop = pword_loop(stream_loop, first_element);
    auto plan = plan_datapath(sdfg, loop.lanes->root(), loop.lanes->indvar());
    DatapathNames names{std::to_string(lang.next_datapath())};

    out << "for " << loop.block->get_name() << " in A.grid(" << loop.count << "):" << std::endl;
    out.changeIndent(+4);
    out << "with A.Datapath():" << std::endl;
    out.changeIndent(+4);
    declare_streams(lang, out, names, plan.inputs.size());
    for (size_t k = 0; k < plan.inputs.size(); ++k) {
        out << "with A.IFD(num=" << k << "):" << std::endl;
        out.changeIndent(+4);
        out << names.input(k) << ".push(" << pword_access(lang, loop, plan.inputs[k]) << ")" << std::endl;
        out.changeIndent(-4);
    }
    out << "with A.DPU():" << std::endl;
    out.changeIndent(+4);
    out << names.registers() << " = A.array(" << quoted(AxScriptLanguageExtension::dpu_type())
        << ", size=" << plan.num_registers << ")" << std::endl;
    emit_program(lang, out, names, plan);
    out << names.output() << ".push(" << names.reg(plan.result) << ")" << std::endl;
    out.changeIndent(-4);
    out << "with A.ODR():" << std::endl;
    out.changeIndent(+4);
    out << pword_access(lang, loop, *plan.output) << " = " << names.output() << ".pop()" << std::endl;
    out.changeIndent(-12);
}

/// One datapath folding all PWORDs of the loop into a DPU register, optionally reducing its lanes, stored to @p store.
void emit_reduction_datapath(
    AxScriptLanguageExtension& lang,
    codegen::PrettyPrinter& out,
    const StructuredSDFG& sdfg,
    structured_control_flow::StructuredLoop& stream_loop,
    const symbolic::Expression& first_element,
    const ReductionKernel& kernel,
    bool reduce_lanes,
    const std::string& store
) {
    auto loop = pword_loop(stream_loop, first_element);
    auto plan = plan_datapath(sdfg, loop.lanes->root(), loop.lanes->indvar(), kernel.accumulator);
    DatapathNames names{std::to_string(lang.next_datapath())};
    auto accumulator = plan.num_registers;
    auto block = loop.block->get_name();

    out << "with A.Datapath():" << std::endl;
    out.changeIndent(+4);
    declare_streams(lang, out, names, plan.inputs.size());
    for (size_t k = 0; k < plan.inputs.size(); ++k) {
        out << "with A.IFD(num=" << k << "):" << std::endl;
        out.changeIndent(+4);
        out << "for " << block << " in A.serial(" << loop.count << "):" << std::endl;
        out << "    " << names.input(k) << ".push(" << pword_access(lang, loop, plan.inputs[k]) << ")" << std::endl;
        out.changeIndent(-4);
    }

    out << "with A.DPU():" << std::endl;
    out.changeIndent(+4);
    out << names.registers() << " = A.array(" << quoted(AxScriptLanguageExtension::dpu_type())
        << ", size=" << accumulator + 1 << ")" << std::endl;
    // There are no float18 literals to start from the identity, so the first PWORD seeds the accumulator.
    emit_program(lang, out, names, plan);
    out << names.reg(accumulator) << " = " << names.reg(plan.result) << std::endl;
    if (loop.count > 1) {
        out << "for " << block << " in A.serial(" << loop.count - 1 << "):" << std::endl;
        out.changeIndent(+4);
        emit_program(lang, out, names, plan);
        out << names.reg(accumulator) << " = "
            << lang.dpu_operation(kernel.combine, {names.reg(accumulator), names.reg(plan.result)}) << std::endl;
        out.changeIndent(-4);
    }
    if (reduce_lanes) {
        out << names.reg(accumulator) << " = " << lang.lane_reduction(kernel.combine, names.reg(accumulator))
            << std::endl;
    }
    out << names.output() << ".push(" << names.reg(accumulator) << ")" << std::endl;
    out.changeIndent(-4);

    out << "with A.ODR():" << std::endl;
    out.changeIndent(+4);
    out << store << " = " << names.output() << ".pop()" << std::endl;
    out.changeIndent(-8);
}

/// One datapath folding the @p cores partial PWORDs and reducing their lanes, stored to @p store.
void emit_combine_datapath(
    AxScriptLanguageExtension& lang,
    codegen::PrettyPrinter& out,
    const std::string& partials,
    int64_t cores,
    const ReductionKernel& kernel,
    const std::string& store
) {
    DatapathNames names{std::to_string(lang.next_datapath())};
    auto dpu_dtype = quoted(AxScriptLanguageExtension::dpu_type());
    auto pop = "A.cast(" + names.input(0) + ".pop(), " + dpu_dtype + ")";
    auto core = "c" + names.id;

    out << "with A.Datapath():" << std::endl;
    out.changeIndent(+4);
    declare_streams(lang, out, names, 1);
    out << "with A.IFD(num=0):" << std::endl;
    out << "    for " << core << " in A.serial(" << cores << "):" << std::endl;
    out << "        " << names.input(0) << ".push(" << partials << "[" << core << "])" << std::endl;
    out << "with A.DPU():" << std::endl;
    out.changeIndent(+4);
    out << names.registers() << " = A.array(" << dpu_dtype << ", size=2)" << std::endl;
    out << names.reg(0) << " = " << pop << std::endl;
    out << "for " << core << " in A.serial(" << cores - 1 << "):" << std::endl;
    out << "    " << names.reg(1) << " = " << pop << std::endl;
    out << "    " << names.reg(0) << " = " << lang.dpu_operation(kernel.combine, {names.reg(0), names.reg(1)})
        << std::endl;
    out << names.reg(0) << " = " << lang.lane_reduction(kernel.combine, names.reg(0)) << std::endl;
    out << names.output() << ".push(" << names.reg(0) << ")" << std::endl;
    out.changeIndent(-4);
    out << "with A.ODR():" << std::endl;
    out << "    " << store << " = " << names.output() << ".pop()" << std::endl;
    out.changeIndent(-4);
}

/// The single Stream loop of a Core loop of @p cores contiguous chunks.
structured_control_flow::StructuredLoop& core_chunks(structured_control_flow::StructuredLoop& loop, int64_t cores) {
    auto stream_loop = only_child_loop(loop, AxeleraLevel::Stream);
    if (!is_integer(loop.init(), 0) || !is_integer(loop.stride()) || !is_integer(loop.num_iterations(), cores) ||
        !stream_loop) {
        unsupported("a core loop that is not one contiguous chunk per core around one datapath loop");
    }
    return *stream_loop;
}

symbolic::Expression chunk_start(
    structured_control_flow::StructuredLoop& core_loop, structured_control_flow::StructuredLoop& stream_loop, int64_t c
) {
    auto chunk = as_int(core_loop.stride());
    return symbolic::subs(stream_loop.init(), core_loop.indvar(), symbolic::integer(c * chunk));
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
    auto& lang = axscript(language_extension_);
    auto& schedule = map_.schedule_type();
    if (ScheduleType_Axelera::is(schedule, AxeleraLevel::Stream)) {
        emit_map_datapath(lang, main_stream, sdfg_, map_, map_.init());
        return;
    }
    if (!ScheduleType_Axelera::is(schedule, AxeleraLevel::Core)) {
        unsupported("a lane loop outside a datapath loop");
    }
    auto cores = ScheduleType_Axelera::num_cores(schedule);
    auto& stream_loop = core_chunks(map_, cores);
    for (int64_t c = 0; c < cores; ++c) {
        main_stream << "with A.AICore(core=" << c << "):" << std::endl;
        main_stream.changeIndent(+4);
        emit_map_datapath(lang, main_stream, sdfg_, stream_loop, chunk_start(map_, stream_loop, c));
        // alto only accepts barriers inside core regions; each core waits for all others.
        main_stream << "A.barrier(" << join_cores(cores) << ")" << std::endl;
        main_stream.changeIndent(-4);
    }
}

AxeleraReduceDispatcher::AxeleraReduceDispatcher(
    codegen::LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Reduce& node,
    codegen::InstrumentationPlan& instrumentation_plan,
    codegen::ArgCapturePlan& arg_capture_plan
)
    : NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan),
      reduce_(node) {
}

void AxeleraReduceDispatcher::dispatch_node(
    codegen::PrettyPrinter& main_stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& lang = axscript(language_extension_);
    auto kernel = match_reduction(sdfg_, analysis_manager_, reduce_);
    auto store = kernel.output.container + "[0]";
    auto& schedule = reduce_.schedule_type();
    if (ScheduleType_Axelera::is(schedule, AxeleraLevel::Stream)) {
        emit_reduction_datapath(lang, main_stream, sdfg_, reduce_, reduce_.init(), kernel, true, store);
        return;
    }
    if (!ScheduleType_Axelera::is(schedule, AxeleraLevel::Core)) {
        unsupported("a lane loop outside a datapath loop");
    }

    auto cores = ScheduleType_Axelera::num_cores(schedule);
    auto& stream_loop = core_chunks(reduce_, cores);
    auto partials = kernel.accumulator + "_partials";
    main_stream << partials << " = A.alloc_buffer((" << cores << ",), "
                << quoted(lang.vector_type(types::PrimitiveType::Float)) << ", scope=A.L2)" << std::endl;
    for (int64_t c = 0; c < cores; ++c) {
        main_stream << "with A.AICore(core=" << c << "):" << std::endl;
        main_stream.changeIndent(+4);
        auto partial = partials + "[" + std::to_string(c) + "]";
        emit_reduction_datapath(
            lang, main_stream, sdfg_, stream_loop, chunk_start(reduce_, stream_loop, c), kernel, false, partial
        );
        main_stream << "A.barrier(" << join_cores(cores) << ")" << std::endl;
        // Any core could combine after the barrier; taking the last one keeps the partials written before they
        // are read even when the core regions run one after another, as in the CPU emulator.
        if (c + 1 == cores) {
            emit_combine_datapath(lang, main_stream, partials, cores, kernel, store);
        }
        main_stream.changeIndent(-4);
    }
}

} // namespace axelera
} // namespace sdfg
