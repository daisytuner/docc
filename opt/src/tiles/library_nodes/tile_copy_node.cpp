#include "sdfg/tiles/library_nodes/tile_copy_node.h"

#include <symengine/add.h>
#include <symengine/mul.h>

#include "sdfg/data_flow/pointer_metadata.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace tiles {

namespace {

const char* atom_to_string(CopyAtom atom) {
    switch (atom) {
        case CopyAtom::ScalarSync:
            return "scalar_sync";
        case CopyAtom::VectorSync:
            return "vector_sync";
        case CopyAtom::CpAsync:
            return "cp_async";
        case CopyAtom::CoalesceSync:
            return "coalesce_sync";
    }
    return "scalar_sync";
}

CopyAtom atom_from_string(const std::string& s) {
    if (s == "vector_sync") {
        return CopyAtom::VectorSync;
    }
    if (s == "cp_async") {
        return CopyAtom::CpAsync;
    }
    // "transpose_sync" is the pre-rename name in serialized SDFGs.
    if (s == "coalesce_sync" || s == "transpose_sync") {
        return CopyAtom::CoalesceSync;
    }
    return CopyAtom::ScalarSync;
}

nlohmann::json layout_to_json(const Layout& layout) {
    nlohmann::json j;
    j["shape"] = nlohmann::json::array();
    for (const auto& e : layout.shape()) {
        j["shape"].push_back(serializer::JSONSerializer::expression(e));
    }
    j["stride"] = nlohmann::json::array();
    for (const auto& e : layout.strides()) {
        j["stride"].push_back(serializer::JSONSerializer::expression(e));
    }
    j["offset"] = serializer::JSONSerializer::expression(layout.offset());
    return j;
}

Layout layout_from_json(const nlohmann::json& j) {
    symbolic::MultiExpression shape;
    for (const auto& e : j.at("shape")) {
        shape.push_back(symbolic::parse(e.get<std::string>()));
    }
    symbolic::MultiExpression stride;
    for (const auto& e : j.at("stride")) {
        stride.push_back(symbolic::parse(e.get<std::string>()));
    }
    auto offset = symbolic::parse(j.at("offset").get<std::string>());
    return Layout(shape, stride, offset);
}

} // namespace

TileCopyNode::TileCopyNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::ImplementationType& implementation_type,
    TiledCopy plan,
    CopyDirection direction,
    size_t bytes,
    TileGuard guard,
    std::vector<int> coop_axes,
    symbolic::Expression coop_threads,
    size_t coop_lanes
)
    : data_flow::LibraryNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_TileCopy, {}, {"_dst", "_src"}, true, implementation_type
      ),
      plan_(std::move(plan)), direction_(direction), bytes_(bytes), guard_(std::move(guard)),
      coop_axes_(std::move(coop_axes)), coop_threads_(std::move(coop_threads)), coop_lanes_(coop_lanes) {
    if (coop_lanes_ > 1 && std::find(coop_axes_.begin(), coop_axes_.end(), 0) != coop_axes_.end()) {
        throw InvalidSDFGException("TileCopyNode: coop_lanes require x to be a slot (non-cooperative) axis");
    }
}

void TileCopyNode::validate(const Function& function) const {
    data_flow::LibraryNode::validate(function);
}

symbolic::SymbolSet TileCopyNode::symbols() const {
    symbolic::SymbolSet set;
    plan_.src.collect_symbols(set);
    plan_.dst.collect_symbols(set);
    guard_.collect_symbols(set);
    return set;
}

std::unique_ptr<data_flow::DataFlowNode> TileCopyNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    auto copy = std::unique_ptr<TileCopyNode>(new TileCopyNode(
        element_id,
        this->debug_info_,
        vertex,
        parent,
        this->implementation_type_,
        plan_,
        direction_,
        bytes_,
        guard_,
        coop_axes_,
        coop_threads_,
        coop_lanes_
    ));
    copy->phase_ = phase_;
    return copy;
}

void TileCopyNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    symbolic::ExpressionMapping replacements;
    replacements[old_expression] = new_expression;
    replace(replacements);
}

void TileCopyNode::replace(const symbolic::ExpressionMapping& replacements) {
    plan_.src.replace_symbols(replacements);
    plan_.dst.replace_symbols(replacements);
    guard_.replace_symbols(replacements);
}

static std::ostream& operator<<(std::ostream& os, const symbolic::Expression& expr) {
    if (expr.is_null()) {
        os << "null";
    } else {
        os << expr->__str__();
    }
    return os;
}

std::string TileCopyNode::toStr() const {
    std::stringstream ss;
    ss << "tile_copy(";
    ss << "dir: " << (direction_ == CopyDirection::In ? "in" : "out") << ", ";
    if (phase_ != CopyPhase::Full) {
        ss << "phase: " << (phase_ == CopyPhase::LoadRegs ? "load_regs" : "store_regs") << ", ";
    }
    ss << "cothr: " << coop_threads_ << ", ";
    ss << "src: " << plan_.src << ", ";
    ss << "dst: " << plan_.dst;
    ss << ")";
    return ss.str();
}

data_flow::PointerAccessType TileCopyNode::pointer_access_type(int input_idx) const {
    auto size = symbolic::integer(static_cast<long long>(bytes_));
    if (input_idx == 0) { // _dst
        return data_flow::PointerAccessMeta::create_full_write_only(size, /*no_capture=*/true);
    }
    if (input_idx == 1) { // _src
        return data_flow::PointerAccessMeta::create_read_only(size, /*no_capture=*/true);
    }
    return data_flow::LibraryNode::pointer_access_type(input_idx);
}

// ---- Serializer ----------------------------------------------------------

nlohmann::json TileCopyNodeSerializer::serialize(const sdfg::data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const TileCopyNode&>(library_node);
    const auto& plan = node.plan();
    nlohmann::json j;
    j["code"] = std::string(node.code().value());
    j["bytes"] = node.bytes();
    j["direction"] = node.direction() == CopyDirection::In ? "in" : "out";
    j["atom"] = atom_to_string(plan.atom);
    j["src"] = layout_to_json(plan.src);
    j["dst"] = layout_to_json(plan.dst);
    j["swizzle"] = {
        {"bits", plan.dst_swizzle.bits}, {"base", plan.dst_swizzle.base}, {"shift", plan.dst_swizzle.shift}
    };
    const auto& guard = node.guard();
    j["guard"]["tile_sizes"] = nlohmann::json::array();
    for (const auto& s : guard.tile_sizes) {
        j["guard"]["tile_sizes"].push_back(serializer::JSONSerializer::expression(s));
    }
    j["guard"]["dims"] = nlohmann::json::array();
    for (const auto& d : guard.dims) {
        j["guard"]["dims"].push_back(
            {{"axis", d.axis},
             {"base", serializer::JSONSerializer::expression(d.base)},
             {"max", serializer::JSONSerializer::expression(d.max)}}
        );
    }
    j["coop_axes"] = node.coop_axes();
    if (!node.coop_threads().is_null()) {
        j["coop_threads"] = serializer::JSONSerializer::expression(node.coop_threads());
    }
    if (node.coop_lanes() > 1) {
        j["coop_lanes"] = node.coop_lanes();
    }
    if (node.phase() != CopyPhase::Full) {
        j["phase"] = node.phase() == CopyPhase::LoadRegs ? "load_regs" : "store_regs";
    }
    return j;
}

data_flow::LibraryNode& TileCopyNodeSerializer::deserialize(
    const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder, sdfg::structured_control_flow::Block& parent
) {
    TiledCopy plan;
    plan.src = layout_from_json(j.at("src"));
    plan.dst = layout_from_json(j.at("dst"));
    plan.atom = atom_from_string(j.at("atom").get<std::string>());
    if (j.contains("swizzle")) {
        const auto& sw = j.at("swizzle");
        plan.dst_swizzle = Swizzle{sw.at("bits").get<int>(), sw.at("base").get<int>(), sw.at("shift").get<int>()};
    }
    auto direction = j.at("direction").get<std::string>() == "out" ? CopyDirection::Out : CopyDirection::In;
    TileGuard guard;
    if (j.contains("guard") && j.at("guard").is_object()) {
        for (const auto& s : j.at("guard").at("tile_sizes")) {
            guard.tile_sizes.push_back(symbolic::parse(s.get<std::string>()));
        }
        for (const auto& d : j.at("guard").at("dims")) {
            guard.dims.push_back(
                {d.at("axis").get<size_t>(),
                 symbolic::parse(d.at("base").get<std::string>()),
                 symbolic::parse(d.at("max").get<std::string>())}
            );
        }
    }
    std::vector<int> coop_axes;
    if (j.contains("coop_axes")) {
        coop_axes = j.at("coop_axes").get<std::vector<int>>();
    }
    symbolic::Expression coop_threads;
    if (j.contains("coop_threads")) {
        coop_threads = symbolic::parse(j.at("coop_threads").get<std::string>());
    }
    size_t coop_lanes = j.contains("coop_lanes") ? j.at("coop_lanes").get<size_t>() : 1;
    auto& node = static_cast<TileCopyNode&>(builder.add_library_node<TileCopyNode>(
        parent,
        DebugInfo(),
        j.at("implementation_type").get<std::string>(),
        plan,
        direction,
        j.at("bytes").get<size_t>(),
        guard,
        coop_axes,
        coop_threads,
        coop_lanes
    ));
    if (j.contains("phase")) {
        node.set_phase(j.at("phase").get<std::string>() == "load_regs" ? CopyPhase::LoadRegs : CopyPhase::StoreRegs);
    }
    return node;
}

// ---- Reference dispatcher ------------------------------------------------

bool provably_multiple_of(symbolic::Expression e, long long m, const TileCopyNode& node) {
    const auto* scope = dynamic_cast<const structured_control_flow::ControlFlowNode*>(node.get_parent().get_parent());
    for (const auto* n = scope; n != nullptr; n = n->get_parent()) {
        const auto* loop = dynamic_cast<const structured_control_flow::StructuredLoop*>(n);
        if (loop == nullptr) {
            continue;
        }
        auto stride = loop->stride();
        if (stride.is_null()) {
            continue;
        }
        auto t = symbolic::symbol("__tc_iter_" + loop->indvar()->get_name());
        e = symbolic::subs(e, loop->indvar(), symbolic::add(loop->init(), symbolic::mul(stride, t)));
    }
    e = symbolic::expand(e);
    std::vector<symbolic::Expression> terms;
    if (SymEngine::is_a<SymEngine::Add>(*e)) {
        for (const auto& term : e->get_args()) {
            terms.push_back(term);
        }
    } else {
        terms.push_back(e);
    }
    for (const auto& term : terms) {
        symbolic::Expression coef = term;
        if (SymEngine::is_a<SymEngine::Mul>(*term)) {
            coef = SymEngine::rcp_static_cast<const SymEngine::Mul>(term)->get_coef();
        }
        auto* c = dynamic_cast<const SymEngine::Integer*>(coef.get());
        if (c == nullptr || c->as_int() % m != 0) {
            return false;
        }
    }
    return true;
}

namespace {

// `e + offset` is a multiple of m for every value of the enclosing loops; a min/max is one if all its
// arguments are (distributing an addend over a single min/max term).
bool bound_multiple_of(const symbolic::Expression& e, long long offset, long long m, const TileCopyNode& node) {
    if (SymEngine::is_a<SymEngine::Min>(*e) || SymEngine::is_a<SymEngine::Max>(*e)) {
        for (const auto& arg : e->get_args()) {
            if (!bound_multiple_of(arg, offset, m, node)) {
                return false;
            }
        }
        return true;
    }
    if (SymEngine::is_a<SymEngine::Add>(*e)) {
        symbolic::Expression minmax = SymEngine::null;
        symbolic::Expression rest = symbolic::zero();
        for (const auto& arg : e->get_args()) {
            if (minmax.is_null() && (SymEngine::is_a<SymEngine::Min>(*arg) || SymEngine::is_a<SymEngine::Max>(*arg))) {
                minmax = arg;
            } else {
                rest = symbolic::add(rest, arg);
            }
        }
        if (!minmax.is_null()) {
            for (const auto& arg : minmax->get_args()) {
                if (!bound_multiple_of(symbolic::add(arg, rest), offset, m, node)) {
                    return false;
                }
            }
            return true;
        }
    }
    return provably_multiple_of(symbolic::add(e, symbolic::integer(offset)), m, node);
}

} // namespace

bool guard_axis_aligned(const TileCopyNode& node, size_t axis, long long factor) {
    for (const auto& d : node.guard().dims) {
        if (d.axis != axis) {
            continue;
        }
        // Coordinates are swept in factor-aligned runs: with base and the exclusive bound (max + 1) both
        // multiples of factor, a run is entirely in or entirely out of bounds.
        if (!provably_multiple_of(d.base, factor, node) || !bound_multiple_of(d.max, 1, factor, node)) {
            return false;
        }
    }
    return true;
}

size_t widest_legal_transfer(const TileCopyNode& node, CopyAtom atom, const std::vector<size_t>& widths) {
    const auto& plan = node.plan();
    const size_t elem_bytes = node.bytes();
    if (plan.src.dims() == 0 || elem_bytes == 0) {
        return 0;
    }
    // A swizzled buffer XORs the inner offset, so consecutive tile coords are not consecutive in it.
    if (!plan.dst_swizzle.is_identity()) {
        return 0;
    }
    if (!symbolic::eq(plan.src.strides().back(), symbolic::one()) ||
        !symbolic::eq(plan.dst.strides().back(), symbolic::one())) {
        return 0;
    }
    // The guard is checked once per transfer at its first element; outer-axis bounds hold for the whole run, an
    // inner-axis bound only when it is aligned to the transfer (checked per width below).
    const size_t inner_axis = static_cast<size_t>(plan.src.dims() - 1);
    bool inner_guarded = false;
    for (const auto& d : node.guard().dims) {
        inner_guarded |= d.axis == inner_axis;
    }
    auto* inner_i = dynamic_cast<const SymEngine::Integer*>(plan.src.shape().back().get());
    auto total = plan.src.total_elements();
    auto* total_i = dynamic_cast<const SymEngine::Integer*>(total.get());
    if (inner_i == nullptr || total_i == nullptr) {
        return 0;
    }
    const long long inner_n = inner_i->as_int();
    const long long total_n = total_i->as_int();
    for (size_t width : widths) {
        if (width <= elem_bytes || width % elem_bytes != 0) {
            continue;
        }
        const long long factor = static_cast<long long>(width / elem_bytes);
        if (inner_n % factor != 0 || total_n % factor != 0) {
            continue;
        }
        if (inner_guarded && !guard_axis_aligned(node, inner_axis, factor)) {
            continue;
        }
        // Synchronous copies split buffer-side accesses to the row alignment at emission;
        // an async copy writes the buffer directly, so its rows and both offsets must hold the full width.
        bool outer_ok = true;
        std::vector<const symbolic::MultiExpression*> sides = {&plan.src.strides()};
        if (atom == CopyAtom::CpAsync) {
            sides.push_back(&plan.dst.strides());
            outer_ok = provably_multiple_of(plan.src.offset(), factor, node) &&
                       provably_multiple_of(plan.dst.offset(), factor, node);
        }
        for (const auto* strides : sides) {
            for (size_t d = 0; outer_ok && d + 1 < strides->size(); ++d) {
                auto* s = dynamic_cast<const SymEngine::Integer*>((*strides)[d].get());
                outer_ok = s != nullptr && s->as_int() % factor == 0;
            }
        }
        if (outer_ok) {
            return width;
        }
    }
    return 0;
}

std::string emit_tile_copy_stmt(
    codegen::LanguageExtension& language_extension,
    const TileCopyNode& node,
    const std::string& dst_expr,
    const std::string& src_expr,
    types::PrimitiveType elem,
    const symbolic::Expression& index
) {
    const auto& plan = node.plan();
    // plan.src is always the global geometry, plan.dst the buffer geometry; the
    // direction fixes which connector each side is wired to.
    const bool copy_in = node.direction() == CopyDirection::In;

    // Index both sides as flat element pointers (the buffer connector is a nested
    // array; reinterpreting to elem* lets the linear layout offset address it).
    types::Pointer elem_ptr{types::Scalar(elem)};
    const std::string dcast = language_extension.type_cast(dst_expr, elem_ptr);
    const std::string scast = language_extension.type_cast(src_expr, elem_ptr);
    // Row-major delinearize the flat index over the tile shape, then apply each side's
    // geometry — both sides share the tile coordinate. The buffer side carries the
    // (possibly non-identity) XOR swizzle on its offset.
    const auto coords = delinearize_rowmajor(index, plan.src.shape());
    auto buffer_off = plan.dst_swizzle.apply(plan.dst.resolve_element(coords, /*require_to_element=*/false));
    auto global_off = plan.src.resolve_element(coords, /*require_to_element=*/false);
    const std::string dst_off = language_extension.expression(copy_in ? buffer_off : global_off);
    const std::string src_off = language_extension.expression(copy_in ? global_off : buffer_off);
    std::string stmt = "(" + dcast + ")[" + dst_off + "] = (" + scast + ")[" + src_off + "];";
    // Ragged tiles: skip the over-approximated elements that fall out of bounds. A copy-in zero-fills them, so
    // consumers of the padded buffer (e.g. MMA fragments) read zeros rather than stale data.
    if (!node.guard().trivial()) {
        stmt = "if (" + language_extension.expression(node.guard().predicate(coords)) + ") { " + stmt + " }";
        if (copy_in) {
            stmt += " else { (" + dcast + ")[" + dst_off + "] = static_cast<" +
                    language_extension.primitive_type(elem) + ">(0); }";
        }
    }
    return stmt;
}

std::string emit_predicated_guarded_load(
    const std::string& type, const std::string& var, const std::string& base, const std::string& index
) {
    return type + " " + var + "{}; if (__tc_ok) { " + var + " = *reinterpret_cast<const " + type + "*>(&(" + base +
           ")[" + index + "]); }";
}

namespace {

// GuardedLoadFn adapter of emit_predicated_guarded_load (the widths do not matter for a plain predicated load).
std::string default_guarded_load(
    const std::string& type, const std::string& var, const std::string& base, const std::string& index, size_t, size_t
) {
    return emit_predicated_guarded_load(type, var, base, index);
}

// The register array of a staged phase, addressed as 32-bit words.
std::string register_words(
    codegen::LanguageExtension& language_extension,
    const TileCopyNode& node,
    const std::string& dst_expr,
    const std::string& src_expr
) {
    types::Pointer word_ptr{types::Scalar(types::PrimitiveType::UInt32)};
    return language_extension.type_cast(node.phase() == CopyPhase::LoadRegs ? dst_expr : src_expr, word_ptr);
}

void require_constant_steps(const TileCopyNode& node) {
    const auto& ct = node.coop_threads();
    if (ct.is_null() || !SymEngine::is_a<SymEngine::Integer>(*ct)) {
        throw InvalidSDFGException("TileCopyNode: a register-staged phase needs a constant thread count");
    }
}

// Widest access (at most the transfer width) that keeps every buffer row aligned: padded
// rows may not be multiples of the global-side transfer.
size_t buffer_access_bytes(const TileCopyNode& node, size_t elem_bytes) {
    const auto& strides = node.plan().dst.strides();
    size_t width = std::max(node.bytes(), elem_bytes);
    for (size_t d = 0; d + 1 < strides.size() && width > elem_bytes; ++d) {
        auto* s = dynamic_cast<const SymEngine::Integer*>(strides[d].get());
        if (s == nullptr) {
            return elem_bytes;
        }
        while (width > elem_bytes && (s->as_int() * static_cast<long long>(elem_bytes)) % width != 0) {
            width /= 2;
        }
    }
    return width;
}

const char* int_type(size_t bytes) {
    switch (bytes) {
        case 16:
            return "int4";
        case 8:
            return "int2";
        case 4:
            return "int";
        case 2:
            return "short";
        default:
            return "char";
    }
}

// CoalesceSync body: the 2-D tile [R][C] (source unit-stride along C, buffer along R) is
// split into 4x4 blocks; each lane loads four source rows (8 bytes for 16-bit, 16 bytes for
// 32-bit elements), regroups them in registers and stores four buffer rows of the same width.
// When the block grid allows, 16-lane groups cover 4x4 blocks, so a wave reads whole source
// rows and writes distinct banks.
void emit_coalesce_copy_blocks(
    codegen::LanguageExtension& language_extension,
    codegen::PrettyPrinter& stream,
    const TileCopyNode& node,
    const std::string& dst_expr,
    const std::string& src_expr,
    types::PrimitiveType elem,
    const std::string& runtime_threads,
    const GuardedLoadFn& guarded_load
) {
    const auto& plan = node.plan();
    const long long rows = SymEngine::rcp_static_cast<const SymEngine::Integer>(plan.src.shape()[0])->as_int();
    const long long cols = SymEngine::rcp_static_cast<const SymEngine::Integer>(plan.src.shape()[1])->as_int();
    const long long rb = rows / 4, cb = cols / 4, blocks = rb * cb;

    types::Pointer elem_ptr{types::Scalar(elem)};
    const std::string dcast = language_extension.type_cast(dst_expr, elem_ptr);
    const std::string scast = language_extension.type_cast(src_expr, elem_ptr);

    const auto& ct = node.coop_threads();
    const bool known = !ct.is_null() && SymEngine::is_a<SymEngine::Integer>(*ct);
    const auto phase = node.phase();
    if (phase != CopyPhase::Full) {
        require_constant_steps(node);
    }
    const std::string regs = phase == CopyPhase::Full ? ""
                                                      : register_words(language_extension, node, dst_expr, src_expr);
    if (known) {
        const long long t = SymEngine::rcp_static_cast<const SymEngine::Integer>(ct)->as_int();
        if (phase != CopyPhase::Full) {
            stream << "#pragma unroll" << std::endl;
        }
        stream << "for (int __tc_i = 0; __tc_i < " << (blocks + t - 1) / t << "; __tc_i++) {" << std::endl;
        stream << "int __tc_b = __tc_i * " << t << " + __tc_tid;" << std::endl;
        if (blocks % t != 0) {
            stream << "if (__tc_b < " << blocks << ") {" << std::endl;
        }
    } else {
        stream << "for (int __tc_b = __tc_tid; __tc_b < " << blocks << "; __tc_b += " << runtime_threads << ") {"
               << std::endl;
    }
    if (rb % 4 == 0 && cb % 4 == 0) {
        stream << "int __tc_r4 = (__tc_b % 16) / 4 + 4 * ((__tc_b / 16) / " << cb / 4 << ");" << std::endl;
        stream << "int __tc_c4 = (__tc_b % 4) + 4 * ((__tc_b / 16) % " << cb / 4 << ");" << std::endl;
    } else {
        stream << "int __tc_r4 = __tc_b / " << cb << ";" << std::endl;
        stream << "int __tc_c4 = __tc_b % " << cb << ";" << std::endl;
    }

    auto r4 = symbolic::mul(symbolic::integer(4), symbolic::symbol("__tc_r4"));
    auto c4 = symbolic::mul(symbolic::integer(4), symbolic::symbol("__tc_c4"));
    // 32-bit words per 4-element row: 2 for 16-bit, 4 for 32-bit elements.
    const int row_words = static_cast<int>(types::bit_width(elem) / 8);
    const std::string row_t = row_words == 4 ? "uint4" : "uint2";
    const char* fields[4] = {"x", "y", "z", "w"};
    // Word w of source row q: a local in a full copy, the register array in a staged phase.
    auto x = [&](int q, int w) {
        if (phase == CopyPhase::Full) {
            return "__tc_x[" + std::to_string(q) + "][" + std::to_string(w) + "]";
        }
        return "(" + regs + ")[" + std::to_string(4 * row_words) + " * __tc_i + " + std::to_string(row_words * q + w) +
               "]";
    };
    if (phase == CopyPhase::Full) {
        stream << "unsigned __tc_x[4][" << row_words << "];" << std::endl;
    }
    // Ragged tile: each 4-element source row is in or out of bounds as a whole (the vectorizer requires the
    // inner-axis bound to be 4-aligned). Out-of-bounds rows read zero through @p guarded_load (no select on the
    // loaded value); stores then write the zero-padded block unguarded, which is exact for a copy-in.
    const bool guarded = !node.guard().trivial();
    for (int q = 0; q < 4 && phase != CopyPhase::StoreRegs; ++q) {
        auto row = symbolic::add(r4, symbolic::integer(q));
        std::string off = language_extension.expression(plan.src.resolve_element({row, c4}, false));
        stream << "{ ";
        if (guarded) {
            stream << "const bool __tc_ok = "
                   << language_extension.expression(node.guard().predicate(symbolic::MultiExpression{row, c4})) << "; ";
            stream << guarded_load(row_t, "__tc_v", scast, off, types::bit_width(elem) / 8, 4 * row_words);
        } else {
            stream << "const " << row_t << " __tc_v = *reinterpret_cast<const " << row_t << "*>(&(" << scast << ")["
                   << off << "]);";
        }
        for (int w = 0; w < row_words; ++w) {
            stream << " " << x(q, w) << " = __tc_v." << fields[w] << ";";
        }
        stream << " }" << std::endl;
    }
    // Buffer row c = source column c of the block. 32-bit: column c is word c of each source
    // row. 16-bit: even columns are the low halves of a source word, odd columns the high halves.
    for (int c = 0; c < 4 && phase != CopyPhase::LoadRegs; ++c) {
        std::string value;
        if (row_words == 4) {
            value = "make_uint4(" + x(0, c) + ", " + x(1, c) + ", " + x(2, c) + ", " + x(3, c) + ")";
        } else {
            const int w = c / 2;
            auto pick = [&](int lo_row, int hi_row) {
                const std::string lo = x(lo_row, w);
                const std::string hi = x(hi_row, w);
                return c % 2 == 0 ? "((" + lo + " & 0xffffu) | (" + hi + " << 16))"
                                  : "((" + lo + " >> 16) | (" + hi + " & 0xffff0000u))";
            };
            value = "make_uint2(" + pick(0, 1) + ", " + pick(2, 3) + ")";
        }
        auto off = plan.dst.resolve_element({r4, symbolic::add(c4, symbolic::integer(c))}, false);
        stream << "*reinterpret_cast<" << row_t << "*>(&(" << dcast << ")[" << language_extension.expression(off)
               << "]) = " << value << ";" << std::endl;
    }
    if (known && blocks % SymEngine::rcp_static_cast<const SymEngine::Integer>(ct)->as_int() != 0) {
        stream << "}" << std::endl;
    }
    stream << "}" << std::endl;
}

} // namespace

void emit_cooperative_copy_loop(
    codegen::LanguageExtension& language_extension,
    codegen::PrettyPrinter& stream,
    const TileCopyNode& node,
    const std::string& dst_expr,
    const std::string& src_expr,
    types::PrimitiveType elem,
    const std::function<std::string(const std::string&, const std::string&, size_t)>& async_stmt,
    const GuardedLoadFn& guarded_load_fn
) {
    const auto& plan = node.plan();
    const bool copy_in = node.direction() == CopyDirection::In;
    const std::string size = language_extension.expression(plan.src.total_elements());
    auto c = symbolic::symbol("__tc_c");
    const GuardedLoadFn guarded_load = guarded_load_fn ? guarded_load_fn : GuardedLoadFn(default_guarded_load);

    // A vector (VectorSync) or async (CpAsync) transfer moves `bytes` per step, so the
    // loop strides by the transfer's element count; a scalar copy strides by one.
    const bool wide = node.atom() == CopyAtom::VectorSync || node.atom() == CopyAtom::CpAsync;
    const size_t elem_bytes = types::bit_width(elem) / 8;
    const size_t factor = (wide && elem_bytes > 0) ? node.bytes() / elem_bytes : 1;

    // Flat thread index + thread count over the cooperating axes. Empty coop_axes =
    // the whole block (a slot-free tile); a subset is the per-thread-slot mode (only
    // those axes split the tile, the slot is fixed in the plan offsets).
    const char* names[3] = {"x", "y", "z"};
    std::vector<int> axes = node.coop_axes();
    if (axes.empty() && node.coop_lanes() == 1) {
        axes = {0, 1, 2};
    }
    std::string tid;
    std::string n = "1";
    std::string blk_prod = "1";
    if (node.coop_lanes() > 1) {
        const std::string lanes = std::to_string(node.coop_lanes());
        tid = "threadIdx.x % " + lanes;
        blk_prod = lanes;
        n = lanes;
    }
    for (int a : axes) {
        std::string t = std::string("threadIdx.") + names[a];
        if (blk_prod != "1") {
            t += " * (" + blk_prod + ")";
        }
        tid = tid.empty() ? t : tid + " + " + t;
        blk_prod = (blk_prod == "1") ? std::string("blockDim.") + names[a] : blk_prod + " * blockDim." + names[a];
        n = (n == "1") ? std::string("blockDim.") + names[a] : n + " * blockDim." + names[a];
    }

    stream << "{" << std::endl;
    stream << "int __tc_tid = " << tid << ";" << std::endl;

    if (node.atom() == CopyAtom::CoalesceSync) {
        emit_coalesce_copy_blocks(language_extension, stream, node, dst_expr, src_expr, elem, n, guarded_load);
        stream << "}" << std::endl;
        return;
    }

    // When the cooperating thread count is known, emit a from-zero loop whose trip
    // count `ceil(size/(threads*factor))` is a symbolic expression that folds to a
    // constant (block dims are compile-time) the backend can unroll — restoring
    // memory-level parallelism (esp. where cp.async degrades to a synchronous copy).
    // Otherwise fall back to a runtime thread-strided loop.
    const bool unrolled = !node.coop_threads().is_null();
    // An exact split (size divisible by threads*factor) needs no bounds guard; dropping
    // it keeps the loads unconditional so the backend can batch them before one wait.
    bool exact = false;
    if (unrolled) {
        auto total = plan.src.total_elements();
        auto threads = node.coop_threads();
        if (SymEngine::is_a<SymEngine::Integer>(*total) && SymEngine::is_a<SymEngine::Integer>(*threads)) {
            const long long t = SymEngine::rcp_static_cast<const SymEngine::Integer>(total)->as_int();
            const long long ct = SymEngine::rcp_static_cast<const SymEngine::Integer>(threads)->as_int();
            exact = ct > 0 && t % (ct * static_cast<long long>(factor)) == 0;
        }
    }
    if (unrolled) {
        const std::string ct = language_extension.expression(node.coop_threads());
        const std::string f = std::to_string(factor);
        if (node.phase() != CopyPhase::Full) {
            stream << "#pragma unroll" << std::endl;
        }
        stream << "for (int __tc_i = 0; __tc_i < ((" << size << ") + (" << ct << ") * " << f << " - 1) / ((" << ct
               << ") * " << f << "); __tc_i++) {" << std::endl;
        stream << "int __tc_c = " << f << " * (__tc_i * (" << ct << ") + __tc_tid);" << std::endl;
        if (!exact) {
            stream << "if (__tc_c < " << size << ") {" << std::endl;
        }
    } else {
        stream << "int __tc_n = " << n << ";" << std::endl;
        const std::string step = factor > 1 ? " * " + std::to_string(factor) : "";
        stream << "for (int __tc_c = __tc_tid" << step << "; __tc_c < " << size << "; __tc_c += __tc_n" << step << ") {"
               << std::endl;
    }

    const auto coords = delinearize_rowmajor(c, plan.src.shape());
    types::Pointer elem_ptr{types::Scalar(elem)};
    const std::string dcast = language_extension.type_cast(dst_expr, elem_ptr);
    const std::string scast = language_extension.type_cast(src_expr, elem_ptr);
    auto buffer_off = plan.dst_swizzle.apply(plan.dst.resolve_element(coords, /*require_to_element=*/false));
    auto global_off = plan.src.resolve_element(coords, /*require_to_element=*/false);
    const std::string doff = language_extension.expression(copy_in ? buffer_off : global_off);
    const std::string soff = language_extension.expression(copy_in ? global_off : buffer_off);

    // A guarded copy-in loads out-of-bounds lanes as zero through `guarded_load`, without a select on the loaded
    // value (which would force a wait right after the load and serialize register-staged prefetches).
    const bool guarded = !node.guard().trivial();
    const bool masked = guarded && copy_in && node.phase() != CopyPhase::StoreRegs && node.atom() != CopyAtom::CpAsync;
    const std::string pred = guarded ? language_extension.expression(node.guard().predicate(coords)) : "";
    auto guarded_decl = [&](const std::string& type, const std::string& var, size_t bytes) {
        return guarded_load(type, var, scast, soff, elem_bytes, bytes);
    };

    std::string stmt;
    if (node.phase() != CopyPhase::Full) {
        // Staged phase: one register word per scalar step, bytes/4 words per vector step.
        require_constant_steps(node);
        const std::string regs = register_words(language_extension, node, dst_expr, src_expr);
        const bool vec = node.atom() == CopyAtom::VectorSync && node.bytes() >= 4;
        const size_t words = vec ? node.bytes() / 4 : 1;
        auto reg = [&](size_t k) {
            return "(" + regs + ")[" + std::to_string(words) + " * __tc_i + " + std::to_string(k) + "]";
        };
        const char* lane = !vec && elem_bytes == 2   ? "unsigned short"
                           : !vec && elem_bytes == 1 ? "unsigned char"
                                                     : "unsigned";
        const size_t lane_bytes = !vec && elem_bytes <= 2 ? elem_bytes : 4;
        const char* wide_t = words == 4 ? "uint4" : "uint2";
        const char* fields[4] = {"x", "y", "z", "w"};
        if (node.phase() == CopyPhase::LoadRegs) {
            const std::string src_addr = "&(" + scast + ")[" + soff + "]";
            if (words > 1) {
                stmt = std::string("{ ") +
                       (masked ? guarded_decl(wide_t, "__tc_v", 4 * words)
                               : std::string("const ") + wide_t + " __tc_v = *reinterpret_cast<const " + wide_t +
                                     "*>(" + src_addr + ");");
                for (size_t k = 0; k < words; ++k) {
                    stmt += " " + reg(k) + " = __tc_v." + fields[k] + ";";
                }
                stmt += " }";
            } else if (masked) {
                stmt = "{ " + guarded_decl(lane, "__tc_s", lane_bytes) + " " + reg(0) +
                       " = static_cast<unsigned>(__tc_s); }";
            } else {
                stmt = reg(0) + " = static_cast<unsigned>(*reinterpret_cast<const " + lane + "*>(" + src_addr + "));";
            }
        } else {
            const std::string dst_addr = "&(" + dcast + ")[" + doff + "]";
            const size_t chunk = buffer_access_bytes(node, elem_bytes);
            if (words > 1 && chunk >= node.bytes()) {
                stmt = std::string("*reinterpret_cast<") + wide_t + "*>(" + dst_addr + ") = make_" + wide_t + "(";
                for (size_t k = 0; k < words; ++k) {
                    stmt += (k ? ", " : "") + reg(k);
                }
                stmt += ");";
            } else if (words > 1) {
                // Padded buffer rows: store the transfer in row-aligned pieces.
                const char* piece = int_type(chunk);
                stmt = std::string("{ const ") + wide_t + " __tc_w = make_" + wide_t + "(";
                for (size_t k = 0; k < words; ++k) {
                    stmt += (k ? ", " : "") + reg(k);
                }
                stmt += "); ";
                for (size_t k = 0; k < node.bytes() / chunk; ++k) {
                    stmt += std::string("reinterpret_cast<") + piece + "*>(" + dst_addr + ")[" + std::to_string(k) +
                            "] = reinterpret_cast<const " + piece + "*>(&__tc_w)[" + std::to_string(k) + "]; ";
                }
                stmt += "}";
            } else {
                stmt = std::string("*reinterpret_cast<") + lane + "*>(" + dst_addr + ") = static_cast<" + lane + ">(" +
                       reg(0) + ");";
            }
        }
    } else if (node.atom() == CopyAtom::CpAsync && async_stmt) {
        const std::string dst_addr = "&(" + dcast + ")[" + doff + "]";
        const std::string src_addr = "&(" + scast + ")[" + soff + "]";
        stmt = async_stmt(dst_addr, src_addr, node.bytes());
    } else if (node.atom() == CopyAtom::VectorSync) {
        const char* vec = int_type(node.bytes());
        const size_t chunk = buffer_access_bytes(node, elem_bytes);
        const std::string dst_addr = "&(" + dcast + ")[" + doff + "]";
        const std::string src_addr = "&(" + scast + ")[" + soff + "]";
        const std::string load = std::string("*reinterpret_cast<const ") + vec + "*>(" + src_addr + ")";
        if (chunk >= node.bytes()) {
            if (masked) {
                stmt = "{ " + guarded_decl(vec, "__tc_w", node.bytes()) + " *reinterpret_cast<" + vec + "*>(" +
                       dst_addr + ") = __tc_w; }";
            } else {
                stmt = std::string("*reinterpret_cast<") + vec + "*>(" + dst_addr +
                       ") = " + "*reinterpret_cast<const " + vec + "*>(" + src_addr + ");";
            }
        } else {
            // Padded buffer rows: the buffer side moves in row-aligned pieces.
            const char* piece = int_type(chunk);
            auto pieces = [&](const std::string& to, const std::string& from) {
                std::string out;
                for (size_t k = 0; k < node.bytes() / chunk; ++k) {
                    const std::string idx = "[" + std::to_string(k) + "]";
                    out += std::string("reinterpret_cast<") + piece + "*>(" + to + ")" + idx +
                           " = reinterpret_cast<const " + piece + "*>(" + from + ")" + idx + "; ";
                }
                return out;
            };
            if (copy_in) {
                stmt = "{ " +
                       (masked ? guarded_decl(vec, "__tc_w", node.bytes())
                               : std::string("const ") + vec + " __tc_w = " + load + ";") +
                       " " + pieces(dst_addr, "&__tc_w") + "}";
            } else {
                stmt = std::string("{ ") + vec + " __tc_w; " + pieces("&__tc_w", src_addr) + "*reinterpret_cast<" +
                       vec + "*>(" + dst_addr + ") = __tc_w; }";
            }
        }
    } else if (masked) {
        const std::string t = language_extension.primitive_type(elem);
        stmt = "{ " + guarded_decl(t, "__tc_s", elem_bytes) + " (" + dcast + ")[" + doff + "] = __tc_s; }";
    } else {
        stmt = "(" + dcast + ")[" + doff + "] = (" + scast + ")[" + soff + "];";
    }
    if (masked) {
        stmt = "{ const bool __tc_ok = " + pred + "; " + stmt + " }";
    } else if (guarded && !copy_in) {
        stmt = "if (" + pred + ") { " + stmt + " }";
    } else if (guarded && node.phase() != CopyPhase::StoreRegs) {
        // An asynchronous copy-in cannot select: zero-fill the out-of-bounds part of the tile directly.
        const size_t chunk = buffer_access_bytes(node, elem_bytes);
        const std::string dst_addr = "&(" + dcast + ")[" + doff + "]";
        std::string zero;
        for (size_t k = 0; k < node.bytes() / chunk; ++k) {
            zero += std::string("reinterpret_cast<") + int_type(chunk) + "*>(" + dst_addr + ")[" + std::to_string(k) +
                    "] = " + int_type(chunk) + "{}; ";
        }
        stmt = "if (" + pred + ") { " + stmt + " } else { " + zero + " }";
    }
    stream << stmt << std::endl;
    if (unrolled && !exact) {
        stream << "}" << std::endl; // close the `__tc_c < size` bounds guard
    }
    stream << "}" << std::endl;
    stream << "}" << std::endl;
}

TileCopyNodeDispatcher::TileCopyNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const TileCopyNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {
}

void TileCopyNodeDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    const auto& node = static_cast<const TileCopyNode&>(node_);
    const auto& plan = node.plan();
    // Connector order {"_dst", "_src"} — both are the bare base pointers.
    const std::string& dst = inputs.at(0).expr;
    const std::string& src = inputs.at(1).expr;
    const auto pt = inputs.at(0).edge.base_type().primitive_type();
    const bool copy_in = node.direction() == CopyDirection::In;
    const auto& shape = plan.src.shape();

    types::Pointer elem_ptr{types::Scalar(pt)};
    const std::string dcast = language_extension_.type_cast(dst, elem_ptr);
    const std::string scast = language_extension_.type_cast(src, elem_ptr);

    // Nested per-mode loops: clean per-axis indices (no idiv/imod) so the host
    // compiler vectorizes the innermost (unit-stride) copy.
    std::vector<symbolic::Expression> coords;
    for (size_t d = 0; d < shape.size(); ++d) {
        const std::string iv = "__tc_i" + std::to_string(d);
        out.stream << "for (long long " << iv << " = 0; " << iv << " < " << language_extension_.expression(shape[d])
                   << "; ++" << iv << ") {" << std::endl;
        coords.push_back(symbolic::symbol(iv));
    }

    auto buffer_off = plan.dst_swizzle.apply(plan.dst.resolve_element(coords, /*require_to_element=*/false));
    auto global_off = plan.src.resolve_element(coords, /*require_to_element=*/false);
    const std::string doff = language_extension_.expression(copy_in ? buffer_off : global_off);
    const std::string soff = language_extension_.expression(copy_in ? global_off : buffer_off);
    std::string stmt = "(" + dcast + ")[" + doff + "] = (" + scast + ")[" + soff + "];";
    if (!node.guard().trivial()) {
        // The guard is per-dim over the tile coordinates, so it applies directly to
        // the nested per-mode indices.
        stmt = "if (" + language_extension_.expression(node.guard().predicate(coords)) + ") { " + stmt + " }";
        if (copy_in) {
            stmt += " else { (" + dcast + ")[" + doff + "] = static_cast<" + language_extension_.primitive_type(pt) +
                    ">(0); }";
        }
    }
    out.stream << stmt << std::endl;
    for (size_t d = 0; d < shape.size(); ++d) {
        out.stream << "}" << std::endl;
    }
}

} // namespace tiles
} // namespace sdfg
