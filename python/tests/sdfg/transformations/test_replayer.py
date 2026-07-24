import json
from docc.sdfg import (
    StructuredSDFGBuilder,
    Replayer,
    AnalysisManager,
    Scalar,
    PrimitiveType,
)


def test_replayer():
    builder = StructuredSDFGBuilder("sdfg")
    builder.add_container("i", Scalar(PrimitiveType.Int32))
    builder.add_container("j", Scalar(PrimitiveType.Int32))

    # for i in range(0, 32):
    #    for j in range(0, 32):
    #       pass
    builder.begin_for("i", "0", "32", "1")  # Element id = 1
    builder.begin_for("j", "0", "32", "1")  # Element id = 3
    builder.end_for()
    builder.end_for()

    # for i_tile in range(0, 32, 16):
    #    for i in range(i_tile, min(i_tile + 16, 32)):
    #       for j_tile in range(0, 32, 16):
    #            for j in range(j_tile, min(j_tile + 16, 32)):
    #                pass
    desc = [
        {
            "transformation_type": "LoopTiling",
            "parameters": {
                "tile_size": 16,
            },
            "subgraph": {"0": {"element_id": 3, "type": "for"}},
        },
        {
            "transformation_type": "LoopTiling",
            "parameters": {"tile_size": 16},
            "subgraph": {"0": {"element_id": 1, "type": "for"}},
        },
    ]
    print(desc)

    analysis_manager = AnalysisManager(builder)
    replayer = Replayer()
    replayer.apply(builder, analysis_manager, json.dumps(desc))
