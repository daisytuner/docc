from docc.sdfg import StructuredSDFGBuilder, AnalysisManager


def test_analysis_manager():
    builder = StructuredSDFGBuilder("sdfg")
    sdfg = builder.move()

    analysis = AnalysisManager(sdfg)

    analysis1 = analysis.arguments_analysis()
    analysis2 = analysis.arguments_analysis()
    assert analysis1 is analysis2


def test_arguments_analysis():
    builder = StructuredSDFGBuilder("sdfg")
    sdfg = builder.move()

    analysis = AnalysisManager(sdfg)
    args_analysis = analysis.arguments_analysis()

    assert str(args_analysis) == "<ArgumentsAnalysis>"


def test_assumptions_analysis():
    builder = StructuredSDFGBuilder("sdfg")
    sdfg = builder.move()

    analysis = AnalysisManager(sdfg)
    assumptions_analysis = analysis.assumptions_analysis()

    assert str(assumptions_analysis) == "<AssumptionsAnalysis>"


def test_control_flow_analysis():
    builder = StructuredSDFGBuilder("sdfg")
    sdfg = builder.move()

    analysis = AnalysisManager(sdfg)
    control_flow_analysis = analysis.control_flow_analysis()

    assert str(control_flow_analysis) == "<ControlFlowAnalysis>"


def test_dominance_analysis():
    builder = StructuredSDFGBuilder("sdfg")
    sdfg = builder.move()

    analysis = AnalysisManager(sdfg)
    dominance_analysis = analysis.dominance_analysis()

    assert str(dominance_analysis) == "<DominanceAnalysis>"


def test_escape_analysis():
    builder = StructuredSDFGBuilder("sdfg")
    sdfg = builder.move()

    analysis = AnalysisManager(sdfg)
    escape_analysis = analysis.escape_analysis()

    assert str(escape_analysis) == "<EscapeAnalysis>"


def test_flop_analysis():
    builder = StructuredSDFGBuilder("sdfg")
    sdfg = builder.move()

    analysis = AnalysisManager(sdfg)
    flop_analysis = analysis.flop_analysis()

    assert str(flop_analysis) == "<FlopAnalysis>"


def test_loop_analysis():
    builder = StructuredSDFGBuilder("sdfg")
    sdfg = builder.move()

    analysis = AnalysisManager(sdfg)
    loop_analysis = analysis.loop_analysis()

    assert str(loop_analysis) == "<LoopAnalysis>"


def test_type_analysis():
    builder = StructuredSDFGBuilder("sdfg")
    sdfg = builder.move()

    analysis = AnalysisManager(sdfg)
    type_analysis = analysis.type_analysis()

    assert str(type_analysis) == "<TypeAnalysis>"


def test_users():
    builder = StructuredSDFGBuilder("sdfg")
    sdfg = builder.move()

    analysis = AnalysisManager(sdfg)
    users = analysis.users()

    assert str(users) == "<Users>"
