#include "sdfg/tiles/library_nodes/tile_copy_node.h"

#include "sdfg/data_flow/pointer_metadata.h"
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
    }
    return "scalar_sync";
}

CopyAtom atom_from_string(const std::string& s) {
    if (s == "vector_sync") return CopyAtom::VectorSync;
    if (s == "cp_async") return CopyAtom::CpAsync;
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
    symbolic::Expression coop_threads
)
    : data_flow::LibraryNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_TileCopy, {}, {"_dst", "_src"}, true, implementation_type
      ),
      plan_(std::move(plan)), direction_(direction), bytes_(bytes), guard_(std::move(guard)),
      coop_axes_(std::move(coop_axes)), coop_threads_(std::move(coop_threads)) {}

void TileCopyNode::validate(const Function& function) const { data_flow::LibraryNode::validate(function); }

symbolic::SymbolSet TileCopyNode::symbols() const {
    symbolic::SymbolSet set;
    plan_.src.collect_symbols(set);
    plan_.dst.collect_symbols(set);
    guard_.collect_symbols(set);
    return set;
}

std::unique_ptr<data_flow::DataFlowNode> TileCopyNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<TileCopyNode>(new TileCopyNode(
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
        coop_threads_
    ));
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
    return builder.add_library_node<TileCopyNode>(
        parent,
        DebugInfo(),
        j.at("implementation_type").get<std::string>(),
        plan,
        direction,
        j.at("bytes").get<size_t>(),
        guard,
        coop_axes,
        coop_threads
    );
}

// ---- Reference dispatcher ------------------------------------------------

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
    // Ragged tiles: skip the over-approximated elements that fall out of bounds.
    if (!node.guard().trivial()) {
        stmt = "if (" + language_extension.expression(node.guard().predicate(coords)) + ") { " + stmt + " }";
    }
    return stmt;
}

void emit_cooperative_copy_loop(
    codegen::LanguageExtension& language_extension,
    codegen::PrettyPrinter& stream,
    const TileCopyNode& node,
    const std::string& dst_expr,
    const std::string& src_expr,
    types::PrimitiveType elem,
    const std::function<std::string(const std::string&, const std::string&, size_t)>& async_stmt
) {
    const auto& plan = node.plan();
    const bool copy_in = node.direction() == CopyDirection::In;
    const std::string size = language_extension.expression(plan.src.total_elements());
    auto c = symbolic::symbol("__tc_c");

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
    if (axes.empty()) {
        axes = {0, 1, 2};
    }
    std::string tid;
    std::string n = "1";
    std::string blk_prod = "1";
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

    // When the cooperating thread count is known, emit a from-zero loop whose trip
    // count `ceil(size/(threads*factor))` is a symbolic expression that folds to a
    // constant (block dims are compile-time) the backend can unroll — restoring
    // memory-level parallelism (esp. where cp.async degrades to a synchronous copy).
    // Otherwise fall back to a runtime thread-strided loop.
    const bool unrolled = !node.coop_threads().is_null();
    if (unrolled) {
        const std::string ct = language_extension.expression(node.coop_threads());
        const std::string f = std::to_string(factor);
        stream << "for (int __tc_i = 0; __tc_i < ((" << size << ") + (" << ct << ") * " << f << " - 1) / ((" << ct
               << ") * " << f << "); __tc_i++) {" << std::endl;
        stream << "int __tc_c = " << f << " * (__tc_i * (" << ct << ") + __tc_tid);" << std::endl;
        stream << "if (__tc_c < " << size << ") {" << std::endl;
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

    std::string stmt;
    if (node.atom() == CopyAtom::CpAsync && async_stmt) {
        const std::string dst_addr = "&(" + dcast + ")[" + doff + "]";
        const std::string src_addr = "&(" + scast + ")[" + soff + "]";
        stmt = async_stmt(dst_addr, src_addr, node.bytes());
    } else if (node.atom() == CopyAtom::VectorSync) {
        const char* vec = node.bytes() == 16 ? "int4" : node.bytes() == 8 ? "int2" : "int";
        stmt = std::string("*reinterpret_cast<") + vec + "*>(&(" + dcast + ")[" + doff +
               "]) = " + "*reinterpret_cast<const " + vec + "*>(&(" + scast + ")[" + soff + "]);";
    } else {
        stmt = "(" + dcast + ")[" + doff + "] = (" + scast + ")[" + soff + "];";
    }
    if (!node.guard().trivial()) {
        stmt = "if (" + language_extension.expression(node.guard().predicate(coords)) + ") { " + stmt + " }";
    }
    stream << stmt << std::endl;
    if (unrolled) {
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
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

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
    }
    out.stream << stmt << std::endl;
    for (size_t d = 0; d < shape.size(); ++d) {
        out.stream << "}" << std::endl;
    }
}

} // namespace tiles
} // namespace sdfg
