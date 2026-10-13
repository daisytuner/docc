#pragma once

#include <map>
#include <set>
#include <vector>

#include <sdfg/codegen/utils.h>
#include <sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h>
#include <sdfg/structured_control_flow/if_else.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/structured_sdfg.h>

#include "docc/target/axelera/axscript_language_extension.h"

namespace sdfg {
namespace axelera {

/// A buffer as AxScript sees it: rows of PWORD blocks, possibly several slots of them (e.g. pipeline stages).
struct BufferShape {
    types::PrimitiveType element;
    int64_t rows;
    int64_t row_length; ///< elements per row, a multiple of the PWORD lanes
    std::vector<int64_t> slots = {}; ///< extents of the dimensions above the rows, outermost first
};

/// True if @p loop holds explicit memory management (tile copies, IMC loads, matmuls) rather than datapaths.
bool is_program(structured_control_flow::StructuredLoop& loop);

struct MatrixArgument {
    int64_t row_length;
    int64_t elements;
};

/// Pointer arguments that tile copies address as matrices, with the extent the copies reach over all iterations.
std::map<std::string, MatrixArgument> matrix_arguments(StructuredSDFG& sdfg);

/**
 * Emits an AI-core region holding explicit memory management: L1 and IMC allocations, serial tile loops, DMA copies
 * between memory levels (`atomic.copy`), IMC loads (`atomic.load_imc`) and matmuls (`atomic.matmul`). A loop of
 * single-row matmuls becomes one matmul over its rows.
 *
 * Branches are predicated rather than emitted as control flow: their operations carry the branch condition as
 * `where=`. Asynchronous copies (a software pipeline's prefetches) carry an id that `A.waits` waits on. AxScript needs
 * each copy waited on in the loop iteration that issued it, so pending copies are waited on at the latest at the end
 * of the iteration and before entering a loop; a pipeline wait waits on all but its most recent commit groups.
 */
class ProgramEmitter {
    /// An asynchronous copy, with the predicate it was issued under.
    struct Transfer {
        std::string id;
        std::string predicate;
    };

    /// The copies a pipeline commit closed.
    struct Group {
        size_t number;
        std::vector<Transfer> transfers;
    };

    const StructuredSDFG& sdfg_;
    AxScriptLanguageExtension& lang_;
    codegen::PrettyPrinter& out_;
    std::map<std::string, BufferShape> shapes_;
    symbolic::ExpressionMapping counters_;
    std::vector<symbolic::Condition> predicates_;

    std::vector<Group> committed_; ///< oldest first
    std::vector<Transfer> open_; ///< issued since the last commit
    std::set<size_t> waited_;
    size_t groups_ = 0;
    size_t transfers_ = 0;
    /// Appended to local buffer names in the region of one of several cores.
    std::string core_suffix_;

    std::string name(const std::string& container) const;
    std::string expression(const symbolic::Expression& expr) const;
    std::string region(
        const std::string& container,
        const math::tensor::TensorLayout& layout,
        symbolic::Expression rows = SymEngine::null,
        bool clamp = false
    ) const;
    std::string scope(const std::string& container) const;
    /// Conjunction of the enclosing branch conditions and @p extra, if any.
    std::string predicate(const symbolic::Condition& extra = SymEngine::null) const;
    std::string where(const symbolic::Condition& extra = SymEngine::null) const;

    void emit_allocations();
    void emit_sequence(structured_control_flow::Sequence& sequence);
    void emit_branches(structured_control_flow::IfElse& branches);
    void emit_loop(structured_control_flow::StructuredLoop& loop);
    void emit_block(structured_control_flow::Block& block);
    void emit_waits(const std::vector<Transfer>& transfers);
    void wait_all();

public:
    ProgramEmitter(
        const StructuredSDFG& sdfg,
        AxScriptLanguageExtension& lang,
        codegen::PrettyPrinter& out,
        std::map<std::string, BufferShape> argument_shapes
    );

    void emit_cores(structured_control_flow::Map& cores);
};

} // namespace axelera
} // namespace sdfg
