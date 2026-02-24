#pragma once

#include "sdfg/data_flow/library_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace data_flow {

inline LibraryNodeCode LibraryNodeType_LoadConst("LoadConst");

class ConstSource {
    // TODO may need format info (endianness, multi-dim alignment? at least for producing the data)

public:
    using byte_iterator = std::vector<uint8_t>::const_iterator;

    virtual ~ConstSource() = default;

    /**
     * for passes to access the data to potentially fold operations into it etc.
     */
    // virtual bool read_data(void* output) const = 0;

    virtual std::unique_ptr<ConstSource> clone() const = 0;

    virtual bool in_external_file() const = 0;
    virtual const std::filesystem::path& filename() const = 0;

    virtual bool is_inlineable() const = 0;
    virtual byte_iterator inline_begin() const = 0;
    virtual byte_iterator inline_end() const = 0;
    virtual size_t num_bytes() const = 0;

    virtual std::string toStr() const = 0;
};

/**
 * Constant data stored inline in SDFG
 */
class InMemoryConstSource : public ConstSource {
    friend class LoadConstNodeSerializer;

    std::vector<uint8_t> data_;

public:
    InMemoryConstSource(std::vector<uint8_t>&& data) : data_(std::move(data)) {}
    InMemoryConstSource(int size) : data_(size) {}
    template<class InputIt>
    InMemoryConstSource(InputIt begin, InputIt end) : data_(begin, end) {}


    std::unique_ptr<ConstSource> clone() const override {
        std::vector<uint8_t> data(data_.begin(), data_.end());
        return std::make_unique<InMemoryConstSource>(std::move(data));
    }

    bool is_inlineable() const override { return true; }
    byte_iterator inline_begin() const override { return data_.begin(); }
    byte_iterator inline_end() const override { return data_.end(); }

    size_t num_bytes() const override { return data_.size(); }

    bool in_external_file() const override { return false; }
    const std::filesystem::path& filename() const override { throw std::runtime_error("no external file"); }

    std::string toStr() const override { return "InMemory(" + std::to_string(data_.size()) + " B)"; }
};

/**
 * Constant data stored in external files, using arg_capture_io code to read (raw data, no meta data inside the files
 * themselves, meta data stored here)
 */
class RawFileConstSource : public ConstSource {
    std::filesystem::path filename_;
    size_t num_bytes_;

public:
    RawFileConstSource(std::filesystem::path filename, size_t num_bytes)
        : filename_(std::move(filename)), num_bytes_(num_bytes) {}
    // virtual bool read_data(void* output) override;

    std::unique_ptr<ConstSource> clone() const override {
        return std::make_unique<RawFileConstSource>(filename_, num_bytes_);
    }

    bool is_inlineable() const override { return false; }

    bool in_external_file() const override { return true; }
    const std::filesystem::path& filename() const override { return filename_; }

    byte_iterator inline_begin() const override { throw std::runtime_error("inlining not supported"); }
    byte_iterator inline_end() const override { throw std::runtime_error("inlining not supported"); }

    size_t num_bytes() const override { return num_bytes_; }

    std::string toStr() const override {
        return "RawFile(" + filename_.string() + ", " + std::to_string(num_bytes_) + " B)";
    }
};

class LoadConstNode : public LibraryNode {
protected:
    std::unique_ptr<ConstSource> data_source_;
    std::unique_ptr<types::IType> type_;

public:
    LoadConstNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        DataFlowGraph& parent,
        std::unique_ptr<types::IType> type,
        std::unique_ptr<ConstSource> data_source
    );

    symbolic::SymbolSet symbols() const override;

    const types::IType& type() const;

    ConstSource& data_source() const;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
        const override;

    std::string toStr() const override;
};

class LoadConstNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const LibraryNode& library_node) override;
    static void const_source_to_json(nlohmann::json& j, const ConstSource& source);
    static std::unique_ptr<ConstSource> json_to_const_source(const nlohmann::json& j);

    LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

class LoadConstNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    LoadConstNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const DataFlowGraph& data_flow_graph,
        const LoadConstNode& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

    void dispatch_inline_const(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory,
        const ConstSource& source,
        const std::string& output_container
    );

    static void dispatch_runtime_load(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory,
        const RawFileConstSource& source,
        const std::string& output_container
    );

    static void ensure_arg_capture_dep(
        codegen::PrettyPrinter& globals_stream, codegen::CodeSnippetFactory& library_snippet_factory
    );
};

} // namespace data_flow
} // namespace sdfg
