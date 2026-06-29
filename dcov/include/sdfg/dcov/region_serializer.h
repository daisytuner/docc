#pragma once

#include <ostream>
#include <string>

#include "sdfg/dcov/region.h"

namespace sdfg {
namespace dcov {

/**
 * @brief Abstract serializer for a region @ref Module.
 *
 * Concrete serializers render the same region model into different on-disk
 * encodings (the line-oriented `.dcov` record format vs. canonical JSON).
 */
class RegionSerializer {
public:
    virtual ~RegionSerializer() = default;

    /// Serialize @p module to @p os.
    virtual void serialize(const Module& module, std::ostream& os) = 0;

    /// Convenience: serialize to a string.
    std::string serialize(const Module& module);
};

/**
 * @brief Line-oriented `.dcov` record serializer (canonical, git-friendly).
 *
 * Each region is a small block of `KEY value` records, one per line, so that
 * diffs/blame operate at field granularity:
 * ```
 * MOD <name>  src=<file>  module_id=<hash>
 * CFG <k=v> ...
 * RGN <region_key>  <display_key>  parent=<key|->  instr=<0|1>
 *   TYP <element_type>  op=<op|->  src=<file:line-line>
 *   LOOP depth=<n>  par=<i,j>  red=<k>  trip=<NI,NJ>
 *   ATOM <atom_key>  <op>  <dtype>  in=<a,b>  out=<c>
 * END
 * ```
 */
class DcovRecordSerializer : public RegionSerializer {
public:
    using RegionSerializer::serialize;
    void serialize(const Module& module, std::ostream& os) override;
};

/**
 * @brief Canonical JSON serializer for the region model (interchange format).
 */
class DcovJsonSerializer : public RegionSerializer {
public:
    using RegionSerializer::serialize;
    void serialize(const Module& module, std::ostream& os) override;
};

} // namespace dcov
} // namespace sdfg
