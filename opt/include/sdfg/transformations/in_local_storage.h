#pragma once

#include <unordered_set>

#include "sdfg/data_flow/memlet.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace transformations {

/**
 * @brief In-loop local storage transformation for read-only data
 *
 * This transformation creates a local buffer for read-only array data accessed
 * within a loop. It copies the accessed tile into a contiguous buffer before
 * the loop, then redirects all reads to the local buffer.
 *
 * This is the read-only counterpart to OutLocalStorage. While OutLocalStorage
 * handles write/read-write containers, InLocalStorage handles pure inputs.
 *
 * Uses MemoryLayoutAnalysis tile API to compute bounding-box extents for the
 * accessed region, supporting:
 * - Constant index access (e.g. A[5]) → tile extent {1}
 * - Loop-dependent access (e.g. A[i]) → tile extent from loop bounds
 * - Delinearized Pointer access (e.g. A[i*K+j]) → multi-dim tile extents
 * - Non-identical subsets across uses (bounding box union)
 *
 * @note The container must be read-only within the loop scope (no writes)
 * @note All tile extents must resolve to integer constants
 */
class InLocalStorage : public Transformation {
public:
    /// Tile information populated by can_be_applied
    struct TileInfo {
        /// Overapproximated integer extents per delinearized dimension
        std::vector<symbolic::Expression> dimensions;
        /// Tile min indices per dimension (bases for index subtraction)
        std::vector<symbolic::Expression> bases;
        /// Layout strides from MemoryLayoutAnalysis (for Pointer re-linearization)
        std::vector<symbolic::Expression> strides;
        /// Layout offset from MemoryLayoutAnalysis
        symbolic::Expression offset = symbolic::integer(0);
    };

private:
    structured_control_flow::StructuredLoop& loop_;
    const data_flow::AccessNode& access_node_;
    std::string container_;
    std::string local_name_; ///< Name of the created local buffer
    TileInfo tile_info_; ///< Populated by can_be_applied
    types::StorageType storage_type_; ///< Storage type for the local buffer
    std::unordered_set<const data_flow::Memlet*> group_memlets_; ///< Memlets in the selected tile group

public:
    /**
     * @brief Construct an in-local storage transformation
     * @param loop The loop defining the scope for localization
     * @param access_node The access node referencing the container to localize
     * @param storage_type Storage type for the local buffer (default: CPU_Stack)
     */
    InLocalStorage(
        structured_control_flow::StructuredLoop& loop,
        const data_flow::AccessNode& access_node,
        const types::StorageType& storage_type = types::StorageType::CPU_Stack()
    );

    /**
     * @brief Get the name of this transformation
     * @return "InLocalStorage"
     */
    virtual std::string name() const override;

    /**
     * @brief Check if this transformation can be applied
     *
     * Criteria:
     * - Container exists and is an array/pointer type
     * - Container is read-only within the loop (no writes)
     * - MemoryLayoutAnalysis provides a tile with integer extents
     *
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     * @return true if the transformation can be applied safely
     */
    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    /**
     * @brief Apply the in-local storage transformation
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     */
    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    /**
     * @brief Serialize this transformation to JSON
     * @param j JSON object to populate
     */
    virtual void to_json(nlohmann::json& j) const override;

    /**
     * @brief Deserialize an in-local storage transformation from JSON
     * @param builder The SDFG builder
     * @param j JSON description of the transformation
     * @return The deserialized transformation
     */
    static InLocalStorage from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    /**
     * @brief Get the name of the local buffer container
     * @return The local buffer container name (valid after apply())
     */
    const std::string& local_container() const { return local_name_; }

    /**
     * @brief Get the tile information
     * @return The tile info (valid after can_be_applied() returns true)
     */
    const TileInfo& tile_info() const { return tile_info_; }
};

} // namespace transformations
} // namespace sdfg
