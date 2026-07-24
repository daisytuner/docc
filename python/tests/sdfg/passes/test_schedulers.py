"""Tests for the loop-scheduler bindings (``LoopSchedulingPass`` + ``Recorder``).

Each test builds a parallelizable map nest with ``StructuredSDFGBuilder``, runs
a target's standard loop scheduler over it, and inspects the transformations the
scheduler applied via an attached ``Recorder`` -- the "default schedule" that
serves as the tuning baseline.
"""

import json

import pytest

from docc.sdfg import (
    AnalysisManager,
    LoopSchedulingPass,
    Pointer,
    PrimitiveType,
    Recorder,
    Scalar,
    StructuredSDFGBuilder,
    TaskletCode,
)


def _build_map_nest():
    """A 2-D elementwise map nest ``B[i, j] = A[i, j]`` (fully parallelizable)."""
    builder = StructuredSDFGBuilder("scheduler_test")
    f = Scalar(PrimitiveType.Float)
    u = Scalar(PrimitiveType.UInt64)

    builder.add_container("N", u, is_argument=True)
    builder.add_container("M", u, is_argument=True)
    builder.add_container("A", Pointer(f), is_argument=True)
    builder.add_container("i", u)
    builder.add_container("j", u)

    builder.begin_map("i", "0", "N", "1")
    builder.begin_map("j", "0", "M", "1")

    block_ptr = builder.add_block()
    A = builder.add_access(block_ptr, "A")
    zero = builder.add_constant(block_ptr, "0", Scalar(PrimitiveType.Int32))
    tasklet_ptr = builder.add_tasklet(block_ptr, TaskletCode.assign, ["_in"], ["_out"])
    builder.add_memlet(block_ptr, zero, "", tasklet_ptr, "_in")
    builder.add_memlet(block_ptr, tasklet_ptr, "_out", A, "i * M + j")

    builder.end_map()
    builder.end_map()
    return builder


def test_openmp_scheduler_records_transformations():
    """The OpenMP scheduler parallelizes the nest and the recorder captures it."""
    builder = _build_map_nest()
    analysis_manager = AnalysisManager(builder)

    recorder = Recorder()
    scheduler = LoopSchedulingPass(["openmp"])
    scheduler.set_recorder(recorder)

    modified = scheduler.run(builder, analysis_manager)
    assert modified is True

    history = json.loads(recorder.history)
    assert isinstance(history, list) and len(history) == 1
    assert any(entry["transformation_type"] == "OMPTransform" for entry in history)


def test_scheduler_runs_without_recorder():
    """The pass applies the schedule even with no recorder attached."""
    builder = _build_map_nest()
    analysis_manager = AnalysisManager(builder)

    scheduler = LoopSchedulingPass(["openmp"])
    assert scheduler.run(builder, analysis_manager) is True


def test_vectorize_scheduler_runs():
    """The vectorize scheduler is registered and runs on the nest."""
    builder = _build_map_nest()
    analysis_manager = AnalysisManager(builder)

    recorder = Recorder()
    scheduler = LoopSchedulingPass(["vectorize"])
    scheduler.set_recorder(recorder)

    # May or may not vectorize this exact nest; just require it runs and the
    # recording is a well-formed transformation list.
    scheduler.run(builder, analysis_manager)
    history = json.loads(recorder.history)
    assert isinstance(history, list) and len(history) == 1


def test_unknown_target_raises():
    """An unregistered scheduling target raises rather than silently no-oping."""
    builder = _build_map_nest()
    analysis_manager = AnalysisManager(builder)

    with pytest.raises(Exception):
        scheduler = LoopSchedulingPass(["does_not_exist"])
        scheduler.run(builder, analysis_manager)
