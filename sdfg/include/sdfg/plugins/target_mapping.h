#pragma once
#include <memory>

#include "sdfg/analysis/analysis.h"
#include "sdfg/data_flow/library_node.h"

namespace sdfg::plugins {

struct TargetMappingPlan {};

class TargetMapper {
public:
    virtual ~TargetMapper() = default;
    virtual bool try_map(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        data_flow::LibraryNode& node
    ) const = 0;
    // don't cement this into the API before we have a use-case to test it well
    // virtual std::unique_ptr<TargetMappingPlan> try_create_plan(data_flow::LibraryNode& node) const;
    // virtual void apply_plan(data_flow::LibraryNode& node, std::unique_ptr<TargetMappingPlan> plan) const;
};

class TargetMapperFactory {
public:
    virtual ~TargetMapperFactory() = default;
    virtual std::unique_ptr<TargetMapper> create() const = 0;
    virtual int priority() = 0;
    virtual const std::string& for_target() const = 0;
};


class TargetMapperRegistry {
public:
    const std::vector<std::shared_ptr<TargetMapperFactory>> get_all_mappers();
    void register_target_mapper(std::shared_ptr<TargetMapperFactory> factory);
};

} // namespace sdfg::plugins
