#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sdfg/structured_control_flow/block.h>
#include <sdfg/structured_control_flow/control_flow_node.h>
#include <sdfg/structured_control_flow/for.h>
#include <sdfg/structured_control_flow/if_else.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/structured_control_flow/reduce.h>
#include <sdfg/structured_control_flow/return.h>
#include <sdfg/structured_control_flow/sequence.h>
#include <sdfg/structured_control_flow/while.h>

namespace py = pybind11;

/**
 * @brief Register all control flow node bindings
 *
 * This function registers the Python bindings for the control flow hierarchy:
 * - ControlFlowNode: Abstract base class
 * - Block: Basic block containing a DataFlowGraph
 * - AssignmentBlock: Unordered set of symbol-assignments, executed simultaneously
 * - Sequence: Sequential container of control flow nodes
 * - Transition: Element connecting nodes in a Sequence
 * - IfElse: Conditional branching
 * - For: Traditional for loop
 * - Reduce: Reduction operation
 * - Map: Parallel map loop
 * - While: While loop
 * - Break: Loop break statement
 * - Continue: Loop continue statement
 * - Return: Function return
 */
void register_control_flow(py::module& m);
