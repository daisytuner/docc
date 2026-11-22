# sdfg-opt: Transformations for the Optimization of SDFGs

This module provides transformations for the optimization of SDFGs.

## Transformations: An API to Modify SDFGs

Transformations are a structured, unified way to modify SDFGs under correctness constraints.
A transformation is a class inheriting from the abstract `Transformation`.

```cpp
class MyTransformation : public Transformation {
public:
    MyTransformation(
        /* subgraph */,
        /* parameters */
    );

    bool can_be_applied(builder::StructuredSDFGBuilder& builder,
                        analysis::AnalysisManager& am) override;

    void apply(builder::StructuredSDFGBuilder& builder,
               analysis::AnalysisManager& am) override;

    void to_json(nlohmann::json& j) const override;

    static MyTransformation from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

};
```

```cpp
// Explicit: Create transformation with target and parameters
loops::transformations::LoopTiling tiling(loop, tile_size);
if (tiling.can_be_applied(builder, am)) {
    tiling.apply(builder, am);
}

// Recorder:
recorder.apply<loops::transformations::LoopTiling>(builder, am, false, loop, tile_size);
```
