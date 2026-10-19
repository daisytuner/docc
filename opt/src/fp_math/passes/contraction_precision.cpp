#include "sdfg/fp_math/passes/contraction_precision.h"

#include <list>
#include <regex>
#include <stdexcept>
#include <unordered_set>

#include "sdfg/fp_math/transformations/quantize_contraction.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

#include "../contractions/contraction_kernels.h"

namespace sdfg {
namespace fp_math {

ContractionPrecision ContractionPrecision::parse(const std::string& spec) {
    static const std::regex pattern(R"(native|int([2-8])|ozaki|ozaki([12])(?::([1-9][0-9]*))?)");
    std::smatch match;
    if (!std::regex_match(spec, match, pattern)) {
        throw std::invalid_argument(
            "Invalid contraction precision '" + spec +
            "': expected native, int2 ... int8, ozaki, ozaki1[:s] or ozaki2[:L]"
        );
    }
    ContractionPrecision precision;
    if (spec == "native") {
        return precision;
    }
    if (match[1].matched) {
        precision.kind = Kind::Quantize;
        precision.parameter = std::stoi(match[1].str());
        return precision;
    }
    precision.kind = Kind::Ozaki;
    if (match[2].matched) {
        precision.variant = match[2].str() == "1" ? OzakiVariant::Slicing : OzakiVariant::Modular;
    }
    if (match[3].matched) {
        precision.parameter = std::stoi(match[3].str());
    }
    return precision;
}

std::string ContractionPrecision::str() const {
    switch (kind) {
        case Kind::Native:
            return "native";
        case Kind::Quantize:
            return "int" + std::to_string(parameter);
        case Kind::Ozaki: {
            std::string spec = variant == OzakiVariant::Slicing ? "ozaki1" : "ozaki2";
            return parameter == 0 ? spec : spec + ":" + std::to_string(parameter);
        }
    }
    return "native";
}

namespace {

class BlockCollector : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    std::list<structured_control_flow::Block*> blocks;

    BlockCollector(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {
    }

    static std::string name() {
        return "BlockCollector";
    }

    bool accept(structured_control_flow::Block& block) override {
        blocks.push_back(&block);
        return false;
    }
};

} // namespace

ContractionPrecisionPass::ContractionPrecisionPass(const ContractionPrecision& precision) : precision_(precision) {
}

std::string ContractionPrecisionPass::name() {
    return "ContractionPrecision";
}

bool ContractionPrecisionPass::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (precision_.kind == ContractionPrecision::Kind::Native) {
        return false;
    }
    // Each application replaces a block by a sequence, so look for the next contraction afresh every time.
    std::unordered_set<size_t> rejected;
    bool applied = false;
    for (bool found = true; found;) {
        found = false;
        BlockCollector collector(builder, analysis_manager);
        collector.visit();
        for (auto* block : collector.blocks) {
            for (auto* node : block->dataflow().library_nodes()) {
                if (!detail::is_contraction(*node) || rejected.count(node->element_id())) {
                    continue;
                }
                bool done = false;
                try {
                    if (precision_.kind == ContractionPrecision::Kind::Quantize) {
                        QuantizeContraction transformation(*node, precision_.parameter, precision_.rescaling);
                        done = transformation.try_apply(builder, analysis_manager);
                    } else {
                        OzakiContraction
                            transformation(*node, precision_.variant, precision_.parameter, precision_.rescaling);
                        done = transformation.try_apply(builder, analysis_manager);
                    }
                } catch (const transformations::InvalidTransformationException&) {
                    // Operands that could not be isolated leave the contraction native.
                    done = false;
                }
                if (!done) {
                    rejected.insert(node->element_id());
                    continue;
                }
                applied = found = true;
                break;
            }
            if (found) {
                break;
            }
        }
    }
    return applied;
}

} // namespace fp_math
} // namespace sdfg
