#include "docc/target/axelera/axscript_program.h"

#include <algorithm>
#include <iterator>

#include <sdfg/data_flow/access_node.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/tiles/library_nodes/pipeline_node.h>
#include <sdfg/tiles/library_nodes/tile_copy_node.h>
#include <sdfg/types/array.h>
#include <sdfg/types/pointer.h>
#include <sdfg/visitor/for_each.h>

#include "docc/target/axelera/matmul_node.h"
#include "docc/target/axelera/schedule.h"
#include "docc/target/axelera/tile_target.h"

namespace sdfg {
namespace axelera {

namespace {

[[noreturn]] void unsupported(const std::string& what) {
    throw std::runtime_error("AxScript codegen: " + what + " is not supported");
}

int64_t as_int(const symbolic::Expression& expr, const std::string& what) {
    if (!SymEngine::is_a<SymEngine::Integer>(*expr)) {
        unsupported(what + " " + expr->__str__() + ", which is not a compile-time constant,");
    }
    return SymEngine::down_cast<const SymEngine::Integer&>(*expr).as_int();
}

bool is_program_node(const data_flow::LibraryNode& node) {
    return dynamic_cast<const tiles::TileCopyNode*>(&node) || dynamic_cast<const AxeleraMatMulNode*>(&node) ||
           dynamic_cast<const AxeleraImcLoadNode*>(&node);
}

/// Container feeding @p connector of @p node.
std::string
operand(structured_control_flow::Block& block, const data_flow::LibraryNode& node, const std::string& connector) {
    for (auto& memlet : block.dataflow().in_edges(node)) {
        if (memlet.dst_conn() == connector) {
            return static_cast<const data_flow::AccessNode&>(memlet.src()).data();
        }
    }
    unsupported("a " + node.code().value() + " node without operand " + connector);
}

/// The single-row matmul a loop over its rows consists of, if any.
const AxeleraMatMulNode* row_matmul(structured_control_flow::StructuredLoop& loop) {
    if (loop.root().size() != 1) {
        return nullptr;
    }
    auto block = dynamic_cast<structured_control_flow::Block*>(&loop.root().at(0));
    if (!block || block->dataflow().library_nodes().size() != 1) {
        return nullptr;
    }
    auto matmul = dynamic_cast<const AxeleraMatMulNode*>(*block->dataflow().library_nodes().begin());
    return matmul && symbolic::eq(matmul->layout_a().get_dim(0), symbolic::one()) ? matmul : nullptr;
}

std::string quoted(const std::string& text) {
    return "\"" + text + "\"";
}

std::vector<symbolic::Expression> terms(const symbolic::Expression& expr) {
    if (SymEngine::is_a<SymEngine::Add>(*expr)) {
        return expr->get_args();
    }
    return {expr};
}

/// Integer coefficient of the additive term @p term, e.g. 32 for `32*i`.
int64_t coefficient(const symbolic::Expression& term) {
    if (SymEngine::is_a<SymEngine::Integer>(*term)) {
        return SymEngine::down_cast<const SymEngine::Integer&>(*term).as_int();
    }
    if (SymEngine::is_a<SymEngine::Mul>(*term)) {
        auto factor = SymEngine::down_cast<const SymEngine::Mul&>(*term).get_coef();
        if (SymEngine::is_a<SymEngine::Integer>(*factor)) {
            return SymEngine::down_cast<const SymEngine::Integer&>(*factor).as_int();
        }
    }
    return 1;
}

/// Folds integer divisions and remainders of multiples of the divisor, as tile loops produce them, e.g.
/// `idiv(32*t + 32, 32)` to `t + 1` and `imod(idiv(0, 32), 2)` to 0.
symbolic::Expression fold_divisions(symbolic::Expression expr) {
    for (bool changed = true; changed;) {
        changed = false;
        for (auto& call : SymEngine::atoms<SymEngine::FunctionSymbol>(*expr)) {
            auto& function = SymEngine::down_cast<const SymEngine::FunctionSymbol&>(*call);
            auto args = function.get_args();
            bool division = function.get_name() == "idiv";
            if ((!division && function.get_name() != "imod") || !SymEngine::is_a<SymEngine::Integer>(*args[1])) {
                continue;
            }
            auto divisor = SymEngine::down_cast<const SymEngine::Integer&>(*args[1]).as_int();
            auto dividend = symbolic::expand(args[0]);
            bool exact = divisor > 0;
            for (auto& term : terms(dividend)) {
                exact = exact && coefficient(term) % divisor == 0;
            }
            symbolic::Expression folded;
            if (exact && division) {
                folded = symbolic::expand(SymEngine::div(dividend, args[1]));
            } else if (exact) {
                folded = symbolic::zero();
            } else if (SymEngine::is_a<SymEngine::Integer>(*dividend)) {
                folded = division ? symbolic::div(dividend, args[1]) : symbolic::mod(dividend, args[1]);
            } else {
                continue;
            }
            expr = symbolic::expand(symbolic::subs(expr, call, folded));
            changed = true;
            break;
        }
    }
    return expr;
}

/// Splits the flat offset @p offset into indices of dimensions with element strides @p strides (outermost first,
/// the last 1). A term goes to the outermost dimension whose stride divides its coefficient; constants spread over
/// the dimensions by truncating division.
std::vector<symbolic::Expression> indices(const symbolic::Expression& offset, const std::vector<int64_t>& strides) {
    std::vector<symbolic::Expression> index(strides.size(), symbolic::zero());
    for (auto& term : terms(offset)) {
        auto factor = coefficient(term);
        if (SymEngine::is_a<SymEngine::Integer>(*term)) {
            for (size_t d = 0; d < strides.size(); ++d) {
                index[d] = symbolic::add(index[d], symbolic::integer(factor / strides[d]));
                factor %= strides[d];
            }
            continue;
        }
        size_t d = 0;
        while (factor % strides[d] != 0) {
            ++d;
        }
        index[d] = symbolic::add(index[d], symbolic::expand(SymEngine::div(term, symbolic::integer(strides[d]))));
    }
    return index;
}

} // namespace

bool is_program(structured_control_flow::StructuredLoop& loop) {
    bool found = false;
    visitor::for_each_block(loop.root(), [&](structured_control_flow::Block& block) {
        for (auto* node : block.dataflow().library_nodes()) {
            found |= is_program_node(*node);
        }
    });
    return found;
}

std::map<std::string, MatrixArgument> matrix_arguments(StructuredSDFG& sdfg) {
    std::map<std::string, MatrixArgument> matrices;
    visitor::for_each_block(sdfg.root(), [&](structured_control_flow::Block& block) {
        for (auto* node : block.dataflow().library_nodes()) {
            auto copy = dynamic_cast<const tiles::TileCopyNode*>(node);
            if (!copy) {
                continue;
            }
            auto global = operand(block, *copy, copy->direction() == tiles::CopyDirection::In ? "_src" : "_dst");
            auto& layout = copy->plan().src;
            if (layout.dims() != 2 || !symbolic::eq(layout.get_stride(1), symbolic::one())) {
                continue;
            }
            // The last iteration of every enclosing loop reaches furthest, unless the conditions of the branches
            // in between (e.g. a pipeline's prefetch guard) exclude it; then the last iteration they admit does.
            auto end = layout.memory_span();
            std::vector<symbolic::Condition> conditions;
            structured_control_flow::ControlFlowNode* child = &block;
            bool runs = true;
            for (auto* ancestor : structured_control_flow::ControlFlowNode::parent_chain(block)) {
                if (auto branches = dynamic_cast<structured_control_flow::IfElse*>(ancestor)) {
                    for (size_t k = 0; k < branches->size(); ++k) {
                        if (&branches->at(k).first == child) {
                            conditions.push_back(branches->at(k).second);
                        }
                    }
                } else if (auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(ancestor)) {
                    auto iterations = loop->num_iterations();
                    auto bounded = SymEngine::is_a<SymEngine::Integer>(*iterations);
                    auto attempts = bounded ? as_int(iterations, "the trip count") : 1;
                    runs = false;
                    for (int64_t back = 1; back <= attempts && !runs; ++back) {
                        auto last = symbolic::
                            add(loop->init(),
                                symbolic::mul(loop->stride(), symbolic::sub(iterations, symbolic::integer(back))));
                        runs = std::none_of(conditions.begin(), conditions.end(), [&](auto& condition) {
                            return symbolic::is_false(symbolic::subs(condition, loop->indvar(), last));
                        });
                        if (runs) {
                            end = symbolic::subs(end, loop->indvar(), last);
                            for (auto& condition : conditions) {
                                condition = symbolic::subs(condition, loop->indvar(), last);
                            }
                        }
                    }
                    if (!runs) {
                        break;
                    }
                }
                child = ancestor;
            }
            if (!runs) {
                continue;
            }
            auto& matrix = matrices[global];
            matrix.row_length = as_int(layout.get_stride(0), "the row length");
            matrix.elements = std::max(matrix.elements, as_int(symbolic::expand(end), "the extent"));
        }
    });
    return matrices;
}

ProgramEmitter::ProgramEmitter(
    const StructuredSDFG& sdfg,
    AxScriptLanguageExtension& lang,
    codegen::PrettyPrinter& out,
    std::map<std::string, BufferShape> argument_shapes
)
    : sdfg_(sdfg), lang_(lang), out_(out), shapes_(std::move(argument_shapes)) {
    for (auto& container : sdfg.containers()) {
        auto& type = sdfg.type(container);
        if (type.storage_type() != StorageType_L1() && type.storage_type() != StorageType_IMC()) {
            continue;
        }
        std::vector<int64_t> extents;
        const types::IType* level = &type;
        while (auto array = dynamic_cast<const types::Array*>(level)) {
            extents.push_back(as_int(array->num_elements(), "the buffer extent"));
            level = &array->element_type();
        }
        if (extents.size() < 2 || (type.storage_type() == StorageType_IMC() && extents.size() != 2)) {
            unsupported("local buffer " + container + " of type " + type.print());
        }
        auto rows = extents[extents.size() - 2];
        auto row_length = extents.back();
        extents.resize(extents.size() - 2);
        shapes_[container] = {level->primitive_type(), rows, row_length, std::move(extents)};
    }
}

std::string ProgramEmitter::name(const std::string& container) const {
    auto start = container.find_first_not_of('_');
    return start == std::string::npos || start == 0 ? container : container.substr(start);
}

std::string ProgramEmitter::expression(const symbolic::Expression& expr) const {
    return lang_.expression(fold_divisions(symbolic::expand(symbolic::subs(expr, counters_))));
}

std::string ProgramEmitter::scope(const std::string& container) const {
    auto storage = sdfg_.type(container).storage_type();
    return storage == StorageType_L1() ? "A.L1" : storage == StorageType_IMC() ? "A.IMC" : "A.DDR";
}

std::string ProgramEmitter::predicate() const {
    std::string conjunction;
    for (auto& condition : predicates_) {
        auto term = lang_.expression(fold_divisions(symbolic::subs(condition, counters_)));
        conjunction = conjunction.empty() ? term : "A.And(" + conjunction + ", " + term + ")";
    }
    return conjunction;
}

std::string ProgramEmitter::where() const {
    auto condition = predicate();
    return condition.empty() ? "" : ", where=" + condition;
}

std::string ProgramEmitter::region(
    const std::string& container, const math::tensor::TensorLayout& layout, symbolic::Expression rows, bool clamp
) const {
    auto it = shapes_.find(container);
    if (it == shapes_.end()) {
        unsupported("addressing " + container + " as a matrix");
    }
    auto& shape = it->second;
    if (layout.dims() != 2 || !symbolic::eq(layout.get_stride(1), symbolic::one()) ||
        (!symbolic::eq(layout.get_dim(0), symbolic::one()) &&
         !symbolic::eq(layout.get_stride(0), symbolic::integer(shape.row_length)))) {
        unsupported("the access " + layout.toStr() + " to " + container + ", which is not whole rows,");
    }
    if (rows.is_null()) {
        rows = layout.get_dim(0);
    }
    std::vector<int64_t> strides(shape.slots.size());
    int64_t stride = shape.rows * shape.row_length;
    for (size_t d = shape.slots.size(); d-- > 0;) {
        strides[d] = stride;
        stride *= shape.slots[d];
    }
    strides.push_back(shape.row_length);
    strides.push_back(1);
    auto index = indices(fold_divisions(symbolic::expand(symbolic::subs(layout.offset(), counters_))), strides);
    auto first_row = index[shape.slots.size()];
    if (!symbolic::eq(index.back(), symbolic::zero()) ||
        as_int(layout.get_dim(1), "the row extent") != shape.row_length) {
        unsupported("the access " + layout.toStr() + " to part of the rows of " + container);
    }
    if (shape.slots.empty() && symbolic::eq(first_row, symbolic::zero()) &&
        symbolic::eq(rows, symbolic::integer(shape.rows))) {
        return name(container);
    }
    // A predicated access is still evaluated when its predicate is false, so keep it within the buffer.
    if (clamp && !SymEngine::is_a<SymEngine::Integer>(*first_row)) {
        first_row = symbolic::min(first_row, symbolic::sub(symbolic::integer(shape.rows), rows));
    }
    std::string result = name(container) + "[";
    for (size_t d = 0; d < shape.slots.size(); ++d) {
        result += lang_.expression(index[d]) + ", ";
    }
    auto pwords = shape.row_length / static_cast<int64_t>(PWORD_LANES);
    return result + "A.Ramp(" + lang_.expression(first_row) + ", 1, " + lang_.expression(rows) + "), A.Ramp(0, 1, " +
           std::to_string(pwords) + ")]";
}

void ProgramEmitter::emit_allocations() {
    for (auto& [container, shape] : shapes_) {
        auto storage = sdfg_.type(container).storage_type();
        if (shape.row_length % static_cast<int64_t>(PWORD_LANES) != 0) {
            unsupported("buffer " + container + " with rows of " + std::to_string(shape.row_length) + " elements");
        }
        auto pwords = std::to_string(shape.row_length / static_cast<int64_t>(PWORD_LANES));
        if (storage == StorageType_L1()) {
            out_ << name(container) << " = A.alloc_buffer((";
            for (auto slots : shape.slots) {
                out_ << slots << ", ";
            }
            out_ << shape.rows << ", " << pwords << "), " << quoted(lang_.vector_type(shape.element)) << ", scope=A.L1)"
                 << std::endl;
        } else if (storage == StorageType_IMC()) {
            if (shape.rows % static_cast<int64_t>(PWORD_LANES) != 0) {
                unsupported("IMC buffer " + container + " with " + std::to_string(shape.rows) + " rows");
            }
            out_ << name(container) << " = A.alloc_imc((" << shape.rows / static_cast<int64_t>(PWORD_LANES) << ", "
                 << pwords << "))" << std::endl;
        }
    }
}

void ProgramEmitter::emit_cores(structured_control_flow::Map& cores) {
    auto count = ScheduleType_Axelera::num_cores(cores.schedule_type());
    if (count != 1 || !symbolic::eq(cores.init(), symbolic::zero())) {
        unsupported("explicit memory management on " + std::to_string(count) + " cores");
    }
    out_ << "with A.AICore(core=0):" << std::endl;
    out_.changeIndent(+4);
    counters_[cores.indvar()] = symbolic::zero();
    emit_allocations();
    emit_sequence(cores.root());
    wait_all();
    out_.changeIndent(-4);
}

void ProgramEmitter::emit_sequence(structured_control_flow::Sequence& sequence) {
    for (size_t i = 0; i < sequence.size(); ++i) {
        auto& child = sequence.at(i);
        if (auto block = dynamic_cast<structured_control_flow::Block*>(&child)) {
            emit_block(*block);
        } else if (auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&child)) {
            emit_loop(*loop);
        } else if (auto inner = dynamic_cast<structured_control_flow::Sequence*>(&child)) {
            emit_sequence(*inner);
        } else if (auto branches = dynamic_cast<structured_control_flow::IfElse*>(&child)) {
            emit_branches(*branches);
        } else {
            unsupported("control flow element " + std::to_string(child.element_id()) + " in a memory-managed region");
        }
    }
}

void ProgramEmitter::emit_branches(structured_control_flow::IfElse& branches) {
    if (!open_.empty()) {
        unsupported("a branch after uncommitted asynchronous copies");
    }
    // Each case sees the groups pending before the branch and its own. A wait inside a case keeps the predicates of
    // the copies it waits on, so it waits on them for all cases.
    auto before = committed_;
    std::vector<Group> issued;
    std::vector<Transfer> open;
    for (size_t k = 0; k < branches.size(); ++k) {
        committed_.clear();
        std::copy_if(before.begin(), before.end(), std::back_inserter(committed_), [&](auto& group) {
            return !waited_.contains(group.number);
        });
        auto first = groups_;
        predicates_.push_back(branches.at(k).second);
        emit_sequence(branches.at(k).first);
        predicates_.pop_back();
        std::copy_if(committed_.begin(), committed_.end(), std::back_inserter(issued), [&](auto& group) {
            return group.number >= first;
        });
        open.insert(open.end(), open_.begin(), open_.end());
        open_.clear();
    }
    committed_.clear();
    std::copy_if(before.begin(), before.end(), std::back_inserter(committed_), [&](auto& group) {
        return !waited_.contains(group.number);
    });
    committed_.insert(committed_.end(), issued.begin(), issued.end());
    open_ = std::move(open);
}

void ProgramEmitter::emit_loop(structured_control_flow::StructuredLoop& loop) {
    auto count = loop.num_iterations();
    if (auto matmul = row_matmul(loop)) {
        auto& block = static_cast<structured_control_flow::Block&>(loop.root().at(0));
        counters_[loop.indvar()] = symbolic::subs(loop.init(), counters_);
        auto imc = operand(block, *matmul, "B");
        if (sdfg_.type(imc).storage_type() != StorageType_IMC()) {
            unsupported("a matmul whose weights are not resident in the IMC");
        }
        bool predicated = !predicates_.empty();
        out_ << "atomic.matmul(matrix_ifd="
             << region(operand(block, *matmul, "A"), matmul->layout_a(), count, predicated)
             << ", matrix_imc=" << name(imc)
             << ", output=" << region(operand(block, *matmul, "Y"), matmul->layout_y(), count, predicated) << where()
             << ")" << std::endl;
        counters_.erase(loop.indvar());
        return;
    }
    if (!predicates_.empty()) {
        unsupported("a loop inside a branch");
    }
    // Copies are waited on in the iteration that issued them.
    wait_all();
    auto counter = loop.indvar();
    out_ << "for " << counter->get_name() << " in A.serial(" << expression(count) << "):" << std::endl;
    out_.changeIndent(+4);
    counters_[counter] = symbolic::add(symbolic::subs(loop.init(), counters_), symbolic::mul(loop.stride(), counter));
    emit_sequence(loop.root());
    wait_all();
    counters_.erase(counter);
    out_.changeIndent(-4);
}

void ProgramEmitter::emit_block(structured_control_flow::Block& block) {
    auto& dataflow = block.dataflow();
    for (auto& node : dataflow.nodes()) {
        if (dynamic_cast<const data_flow::AccessNode*>(&node) || dynamic_cast<const data_flow::LibraryNode*>(&node)) {
            continue;
        }
        unsupported("computation outside a datapath in a memory-managed region");
    }
    bool predicated = !predicates_.empty();
    for (auto* node : dataflow.library_nodes()) {
        if (auto copy = dynamic_cast<const tiles::TileCopyNode*>(node)) {
            if (!copy->guard().trivial()) {
                unsupported("a tile copy of a ragged tile");
            }
            bool in = copy->direction() == tiles::CopyDirection::In;
            auto global = operand(block, *copy, in ? "_src" : "_dst");
            auto local = operand(block, *copy, in ? "_dst" : "_src");
            auto global_region = region(global, copy->plan().src, SymEngine::null, predicated);
            auto local_region = region(local, copy->plan().dst, SymEngine::null, predicated);
            auto destination = in ? local : global;
            out_ << "atomic.copy(" << (in ? global_region : local_region) << ", scope=" << scope(destination)
                 << ", output=" << (in ? local_region : global_region);
            if (copy->atom() == tiles::CopyAtom::CpAsync) {
                auto id = name(local) + "_" + std::to_string(transfers_++);
                out_ << ", id=" << quoted(id);
                open_.push_back({id, predicate()});
            }
            out_ << where() << ")" << std::endl;
        } else if (auto load = dynamic_cast<const AxeleraImcLoadNode*>(node)) {
            out_ << "atomic.load_imc(buffer="
                 << region(operand(block, *load, "_src"), load->layout(), SymEngine::null, predicated)
                 << ", imc=" << name(operand(block, *load, "_imc")) << where() << ")" << std::endl;
        } else if (dynamic_cast<const tiles::PipelineCommitNode*>(node)) {
            committed_.push_back({groups_++, std::move(open_)});
            open_.clear();
        } else if (auto wait = dynamic_cast<const tiles::PipelineWaitNode*>(node)) {
            while (committed_.size() > wait->keep_outstanding()) {
                waited_.insert(committed_.front().number);
                emit_waits(committed_.front().transfers);
                committed_.erase(committed_.begin());
            }
        } else {
            unsupported("library node " + node->code().value() + " in a memory-managed region");
        }
    }
}

void ProgramEmitter::emit_waits(const std::vector<Transfer>& transfers) {
    std::vector<std::pair<std::string, std::string>> waits; // predicate, quoted ids
    for (auto& transfer : transfers) {
        auto it = std::find_if(waits.begin(), waits.end(), [&](auto& wait) {
            return wait.first == transfer.predicate;
        });
        if (it == waits.end()) {
            waits.push_back({transfer.predicate, quoted(transfer.id)});
        } else {
            it->second += ", " + quoted(transfer.id);
        }
    }
    for (auto& [condition, ids] : waits) {
        if (!condition.empty()) {
            out_ << "if " << condition << ":" << std::endl;
            out_.changeIndent(+4);
        }
        out_ << "A.waits(" << ids << ")" << std::endl;
        if (!condition.empty()) {
            out_.changeIndent(-4);
        }
    }
}

void ProgramEmitter::wait_all() {
    std::vector<Transfer> transfers;
    for (auto& group : committed_) {
        waited_.insert(group.number);
        transfers.insert(transfers.end(), group.transfers.begin(), group.transfers.end());
    }
    transfers.insert(transfers.end(), open_.begin(), open_.end());
    committed_.clear();
    open_.clear();
    emit_waits(transfers);
}

} // namespace axelera
} // namespace sdfg
