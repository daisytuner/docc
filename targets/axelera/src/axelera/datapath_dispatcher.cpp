#include "docc/target/axelera/datapath_dispatcher.h"

#include <algorithm>

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

/// Sum of integer multiples of symbols plus an integer, as PWORD indices must be.
bool is_integer_affine(const symbolic::Expression& expr) {
    auto terms = SymEngine::is_a<SymEngine::Add>(*expr) ? expr->get_args() : SymEngine::vec_basic{expr};
    for (auto& term : terms) {
        if (is_integer(term) || SymEngine::is_a<SymEngine::Symbol>(*term)) {
            continue;
        }
        if (!SymEngine::is_a<SymEngine::Mul>(*term)) {
            return false;
        }
        auto& mul = SymEngine::down_cast<const SymEngine::Mul&>(*term);
        if (!is_integer(mul.get_coef()) || mul.get_dict().size() != 1 ||
            !SymEngine::is_a<SymEngine::Symbol>(*mul.get_dict().begin()->first) ||
            !is_integer(mul.get_dict().begin()->second, 1)) {
            return false;
        }
    }
    return true;
}

/// The PWORD a stream accesses as `step * block + offset` in the iteration of the loop's block variable.
std::pair<int64_t, symbolic::Expression> pword_index(const PwordLoop& loop, const DatapathStream& access) {
    auto pwords = symbolic::integer(PWORD_LANES);
    auto element_offset = symbolic::add(loop.first_element, symbolic::mul(pwords, loop.block));
    auto first =
        symbolic::subs(symbolic::subs(access.index, loop.lanes->indvar(), loop.block), loop.block, element_offset);
    auto index = symbolic::expand(SymEngine::div(first, pwords));
    auto offset = symbolic::expand(symbolic::subs(index, loop.block, symbolic::zero()));
    auto step = symbolic::expand(symbolic::sub(symbolic::subs(index, loop.block, symbolic::one()), offset));
    if (!is_integer_affine(offset) || !is_integer(step) ||
        !symbolic::eq(index, symbolic::expand(symbolic::add(symbolic::mul(step, loop.block), offset)))) {
        unsupported("access " + access.container + "[" + access.index->__str__() + "], which is not PWORD-aligned,");
    }
    return {as_int(step), offset};
}

/// The PWORDs of a datapath loop as AxScript iterates them: one flat grid, or rows of a matrix buffer's PWORDs when
/// the datapath touches one, since matrix buffers take a (row, PWORD) index.
struct PwordGrid {
    const PwordLoop& loop;
    int64_t row_pwords = 0;

    std::string row() const {
        return loop.block->get_name() + "_row";
    }

    std::string column() const {
        return loop.block->get_name() + "_col";
    }

    std::string header(const std::string& kind) const {
        if (row_pwords == 0) {
            return "for " + loop.block->get_name() + " in A." + kind + "(" + std::to_string(loop.count) + "):";
        }
        return "for " + row() + ", " + column() + " in A." + kind + "(" + std::to_string(loop.count / row_pwords) +
               ", " + std::to_string(row_pwords) + "):";
    }

    std::string access(AxScriptLanguageExtension& lang, const DatapathStream& stream) const {
        auto [step, offset] = pword_index(loop, stream);
        auto matrix = lang.matrix_pwords(stream.container);
        if (row_pwords == 0) {
            return stream.container + "[" +
                   lang.expression(symbolic::add(symbolic::mul(symbolic::integer(step), loop.block), offset)) + "]";
        }
        auto row_symbol = symbolic::symbol(row());
        auto column_symbol = symbolic::symbol(column());
        if (matrix == 0) {
            auto block = symbolic::add(symbolic::mul(symbolic::integer(row_pwords), row_symbol), column_symbol);
            return stream.container + "[" +
                   lang.expression(symbolic::expand(symbolic::add(symbolic::mul(symbolic::integer(step), block), offset))) +
                   "]";
        }
        auto first_row = symbolic::expand(SymEngine::div(offset, symbolic::integer(matrix)));
        if (step != 1 || !is_integer_affine(first_row)) {
            unsupported("access to " + stream.container + " other than along whole rows");
        }
        return stream.container + "[" + lang.expression(symbolic::add(first_row, row_symbol)) + ", " + column() + "]";
    }
};

PwordGrid pword_grid(AxScriptLanguageExtension& lang, const PwordLoop& loop, const DatapathPlan& plan) {
    PwordGrid grid{loop};
    auto streams = plan.inputs;
    if (plan.output) {
        streams.push_back(*plan.output);
    }
    for (auto& stream : streams) {
        auto matrix = lang.matrix_pwords(stream.container);
        if (matrix == 0) {
            continue;
        }
        if ((grid.row_pwords != 0 && grid.row_pwords != matrix) || loop.count % matrix != 0) {
            unsupported("a datapath over matrices of different row lengths or partial rows");
        }
        grid.row_pwords = matrix;
    }
    return grid;
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
    AxScriptLanguageExtension& lang,
    codegen::PrettyPrinter& out,
    const DatapathNames& names,
    size_t inputs,
    types::PrimitiveType output = types::PrimitiveType::Float
) {
    auto dtype = quoted(lang.vector_type(types::PrimitiveType::Float));
    for (size_t k = 0; k < inputs; ++k) {
        out << names.input(k) << " = A.decl_stream(" << dtype << ")" << std::endl;
    }
    out << names.output() << " = A.decl_stream(" << quoted(lang.vector_type(output)) << ")" << std::endl;
}

void emit_program(
    AxScriptLanguageExtension& lang, codegen::PrettyPrinter& out, const DatapathNames& names, const DatapathPlan& plan
) {
    auto dpu_dtype = quoted(AxScriptLanguageExtension::dpu_type());
    for (auto& instruction : plan.program) {
        std::string value;
        if (instruction.input_stream) {
            value = "A.cast(" + names.input(*instruction.input_stream) + ".pop(), " + dpu_dtype + ")";
        } else if (instruction.constant) {
            value = "A.cast(" + lang.constant_buffer(*instruction.constant) + "[0], " + dpu_dtype + ")";
        } else if (instruction.uniform) {
            if (!lang.is_uniform(*instruction.uniform)) {
                unsupported("reading " + *instruction.uniform + ", which no reduction stores,");
            }
            value = "A.cast(" + *instruction.uniform + "[0], " + dpu_dtype + ")";
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
    auto grid = pword_grid(lang, loop, plan);
    DatapathNames names{std::to_string(lang.next_datapath())};

    out << grid.header("grid") << std::endl;
    out.changeIndent(+4);
    out << "with A.Datapath():" << std::endl;
    out.changeIndent(+4);
    declare_streams(lang, out, names, plan.inputs.size(), plan.output->element);
    for (size_t k = 0; k < plan.inputs.size(); ++k) {
        out << "with A.IFD(num=" << k << "):" << std::endl;
        out.changeIndent(+4);
        out << names.input(k) << ".push(" << grid.access(lang, plan.inputs[k]) << ")" << std::endl;
        out.changeIndent(-4);
    }
    out << "with A.DPU():" << std::endl;
    out.changeIndent(+4);
    out << names.registers() << " = A.array(" << quoted(AxScriptLanguageExtension::dpu_type())
        << ", size=" << plan.num_registers << ")" << std::endl;
    emit_program(lang, out, names, plan);
    auto result = names.reg(plan.result);
    if (plan.output->element != types::PrimitiveType::Float) {
        result = "A.cast(" + result + ", " + quoted(lang.vector_type(plan.output->element)) + ")";
    }
    out << names.output() << ".push(" << result << ")" << std::endl;
    out.changeIndent(-4);
    out << "with A.ODR():" << std::endl;
    out.changeIndent(+4);
    out << grid.access(lang, *plan.output) << " = " << names.output() << ".pop()" << std::endl;
    out.changeIndent(-12);
}

/// PWORDs one reduction datapath folds; alto cannot fit the DPU loops of much longer ones.
constexpr int64_t REDUCTION_CHUNK = 64;

/// One datapath folding all PWORDs of the loop into a DPU register, optionally reducing its lanes, stored to @p store.
void emit_reduction_datapath(
    AxScriptLanguageExtension& lang,
    codegen::PrettyPrinter& out,
    const StructuredSDFG& sdfg,
    const PwordLoop& loop,
    const ReductionKernel& kernel,
    bool reduce_lanes,
    const std::string& store
) {
    auto plan = plan_datapath(sdfg, loop.lanes->root(), loop.lanes->indvar(), kernel.accumulator);
    if (pword_grid(lang, loop, plan).row_pwords != 0) {
        unsupported("reducing a matrix buffer");
    }
    PwordGrid grid{loop};
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
        out << "    " << names.input(k) << ".push(" << grid.access(lang, plan.inputs[k]) << ")" << std::endl;
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

/// One datapath folding the @p count partial PWORDs and reducing their lanes, stored to @p store.
void emit_combine_datapath(
    AxScriptLanguageExtension& lang,
    codegen::PrettyPrinter& out,
    const std::string& partials,
    int64_t count,
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
    out << "    for " << core << " in A.serial(" << count << "):" << std::endl;
    out << "        " << names.input(0) << ".push(" << partials << "[" << core << "])" << std::endl;
    out << "with A.DPU():" << std::endl;
    out.changeIndent(+4);
    out << names.registers() << " = A.array(" << dpu_dtype << ", size=2)" << std::endl;
    out << names.reg(0) << " = " << pop << std::endl;
    out << "for " << core << " in A.serial(" << count - 1 << "):" << std::endl;
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

/// Datapaths a reduction over @p count PWORDs is split into.
int64_t reduction_chunks(int64_t count) {
    int64_t chunk = std::min(count, REDUCTION_CHUNK);
    while (count % chunk != 0) {
        --chunk;
    }
    return count / chunk;
}

/// Folds the PWORDs of @p loop into @p chunks partial PWORDs, `partials[first + c]` for chunk c.
void emit_partial_reductions(
    AxScriptLanguageExtension& lang,
    codegen::PrettyPrinter& out,
    const StructuredSDFG& sdfg,
    const PwordLoop& loop,
    int64_t chunks,
    const ReductionKernel& kernel,
    const std::string& partials,
    int64_t first
) {
    if (chunks == 1) {
        emit_reduction_datapath(lang, out, sdfg, loop, kernel, false, partials + "[" + std::to_string(first) + "]");
        return;
    }
    auto chunk = loop.block->get_name() + "_chunk";
    PwordLoop part = loop;
    part.count = loop.count / chunks;
    part.first_element = symbolic::
        add(loop.first_element, symbolic::mul(symbolic::integer(part.count * PWORD_LANES), symbolic::symbol(chunk)));
    out << "for " << chunk << " in A.grid(" << chunks << "):" << std::endl;
    out.changeIndent(+4);
    auto index = first == 0 ? chunk : std::to_string(first) + " + " + chunk;
    emit_reduction_datapath(lang, out, sdfg, part, kernel, false, partials + "[" + index + "]");
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
    auto partials = kernel.accumulator + "_partials";
    auto allocate_partials = [&](int64_t count) {
        main_stream << partials << " = A.alloc_buffer((" << count << ",), "
                    << quoted(lang.vector_type(types::PrimitiveType::Float)) << ", scope=A.L2)" << std::endl;
    };
    auto& schedule = reduce_.schedule_type();
    if (ScheduleType_Axelera::is(schedule, AxeleraLevel::Stream)) {
        auto loop = pword_loop(reduce_, reduce_.init());
        auto chunks = reduction_chunks(loop.count);
        if (chunks == 1) {
            emit_reduction_datapath(lang, main_stream, sdfg_, loop, kernel, true, store);
            return;
        }
        allocate_partials(chunks);
        emit_partial_reductions(lang, main_stream, sdfg_, loop, chunks, kernel, partials, 0);
        emit_combine_datapath(lang, main_stream, partials, chunks, kernel, store);
        return;
    }
    if (!ScheduleType_Axelera::is(schedule, AxeleraLevel::Core)) {
        unsupported("a lane loop outside a datapath loop");
    }

    auto cores = ScheduleType_Axelera::num_cores(schedule);
    auto& stream_loop = core_chunks(reduce_, cores);
    auto chunks = reduction_chunks(pword_loop(stream_loop, stream_loop.init()).count);
    allocate_partials(cores * chunks);
    for (int64_t c = 0; c < cores; ++c) {
        main_stream << "with A.AICore(core=" << c << "):" << std::endl;
        main_stream.changeIndent(+4);
        auto loop = pword_loop(stream_loop, chunk_start(reduce_, stream_loop, c));
        emit_partial_reductions(lang, main_stream, sdfg_, loop, chunks, kernel, partials, c * chunks);
        main_stream << "A.barrier(" << join_cores(cores) << ")" << std::endl;
        // Any core could combine after the barrier; taking the last one keeps the partials written before they
        // are read even when the core regions run one after another, as in the CPU emulator.
        if (c + 1 == cores) {
            emit_combine_datapath(lang, main_stream, partials, cores * chunks, kernel, store);
        }
        main_stream.changeIndent(-4);
    }
}

} // namespace axelera
} // namespace sdfg
