#include "sdfg/passes/pipeline.h"

#include <gtest/gtest.h>

#include "sdfg/passes/symbolic/symbol_promotion.h"

using namespace sdfg;

TEST(PipelineTest, RegisterPass) {
    passes::Pipeline pipeline("pipeline");
    pipeline.register_pass<passes::SymbolPromotion>();

    EXPECT_EQ(pipeline.size(), 1);
}

TEST(PipelineTest, MapFusionPipeline) {
    passes::Pipeline pipeline = passes::Pipeline::map_fusion();

    EXPECT_EQ(pipeline.name(), "MapFusion");
    EXPECT_EQ(pipeline.size(), 4);
}
