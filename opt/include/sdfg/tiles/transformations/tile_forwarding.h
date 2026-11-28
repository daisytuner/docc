#pragma once

#include "sdfg/structured_control_flow/block.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Lets a copy into a local buffer take the tile an earlier copy left in another buffer
 *
 * @p earlier and @p later are blocks of the same sequence, each a staging copy: one TileCopyNode between a local
 * buffer and a global container. After the earlier copy, in or out, its buffer holds the global tile; if nothing
 * changes either before the later copy reads the same tile in, the later copy is redundant, and its buffer can be
 * the earlier one. Legal if
 * - the later copy copies in, both copy whole tiles (no register phase, identity swizzle, trivial guard) of the same
 *   global container, region and buffer layout, and the buffers have the same type,
 * - no step between them writes what the earlier copy accesses: the global tile, its buffer or the symbols of its plan
 *   (see code_motion::may_conflict), and
 * - the later buffer is used in the sequence only, from the later copy on, and nothing from the later copy on writes
 *   either buffer, so that renaming one to the other changes no value read.
 *
 * Applied, the later copy goes and its buffer's uses become uses of the earlier buffer.
 */
class TileForwarding : public Transformation {
    structured_control_flow::Block& earlier_;
    structured_control_flow::Block& later_;

public:
    TileForwarding(structured_control_flow::Block& earlier, structured_control_flow::Block& later);

    std::string name() const override;

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void to_json(nlohmann::json& j) const override;

    static TileForwarding from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
