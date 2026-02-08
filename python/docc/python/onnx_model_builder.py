"""
ONNX Model Builder

Converts JSON graph descriptions emitted by the ONNX tensor dispatchers
into valid ONNX model files that can be loaded by ONNX Runtime.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple


def build_onnx_model(json_path: str, output_path: str) -> str:
    """
    Build an ONNX model from a JSON graph description.

    Args:
        json_path: Path to the JSON file containing node definitions
        output_path: Path where the .onnx model should be written

    Returns:
        Path to the generated ONNX model file
    """
    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError:
        raise ImportError(
            "The 'onnx' package is required for ONNX model generation. "
            "Install it with: pip install onnx"
        )

    # Read the JSON file
    with open(json_path, "r") as f:
        content = f.read().strip()

    # The JSON file contains comma-separated node definitions
    # Wrap in array brackets to make it valid JSON
    if content.endswith(","):
        content = content[:-1]  # Remove trailing comma
    json_content = f"[{content}]"

    try:
        nodes_data = json.loads(json_content)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse ONNX graph JSON: {e}\nContent: {json_content[:500]}"
        )

    # Collect all inputs and outputs
    all_inputs: Set[str] = set()
    all_outputs: Set[str] = set()
    intermediate: Set[str] = set()
    elem_type = TensorProto.FLOAT  # Default

    # Build ONNX nodes
    onnx_nodes = []
    for node_data in nodes_data:
        op_type = node_data.get("op_type", "Unknown")
        name = node_data.get("name", "unnamed")
        inputs = node_data.get("inputs", [])
        outputs = node_data.get("outputs", [])
        attributes = node_data.get("attributes", {})

        if "elem_type" in node_data:
            elem_type = node_data["elem_type"]

        # Track inputs/outputs
        for inp in inputs:
            if inp not in intermediate:
                all_inputs.add(inp)
        for out in outputs:
            intermediate.add(out)
            all_outputs.add(out)

        # Build attribute list
        onnx_attrs = []
        for attr_name, attr_value in attributes.items():
            if isinstance(attr_value, list):
                if all(isinstance(x, int) for x in attr_value):
                    onnx_attrs.append(helper.make_attribute(attr_name, attr_value))
                elif all(isinstance(x, float) for x in attr_value):
                    onnx_attrs.append(helper.make_attribute(attr_name, attr_value))
            elif isinstance(attr_value, int):
                onnx_attrs.append(helper.make_attribute(attr_name, attr_value))
            elif isinstance(attr_value, float):
                onnx_attrs.append(helper.make_attribute(attr_name, attr_value))
            elif isinstance(attr_value, str):
                onnx_attrs.append(helper.make_attribute(attr_name, attr_value))

        # Create the ONNX node
        node = helper.make_node(
            op_type,
            inputs=inputs,
            outputs=outputs,
            name=name,
        )
        # Add attributes
        node.attribute.extend(onnx_attrs)
        onnx_nodes.append(node)

    # Remove intermediate values from outputs (only keep final outputs)
    # Final outputs are those not consumed by any other node
    consumed = set()
    for node_data in nodes_data:
        for inp in node_data.get("inputs", []):
            consumed.add(inp)

    final_outputs = all_outputs - consumed
    graph_inputs = all_inputs - intermediate

    # If no final outputs, use all outputs
    if not final_outputs:
        final_outputs = all_outputs

    # Create graph inputs (with dynamic shapes using dim_param)
    input_tensors = []
    for inp_name in sorted(graph_inputs):
        # Use dynamic shape with symbolic dimension
        input_tensor = helper.make_tensor_value_info(
            inp_name, elem_type, None  # Dynamic shape
        )
        input_tensors.append(input_tensor)

    # Create graph outputs
    output_tensors = []
    for out_name in sorted(final_outputs):
        output_tensor = helper.make_tensor_value_info(
            out_name, elem_type, None  # Dynamic shape
        )
        output_tensors.append(output_tensor)

    # Create the graph
    graph = helper.make_graph(
        onnx_nodes,
        "docc_onnx_graph",
        input_tensors,
        output_tensors,
    )

    # Create the model
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]  # ONNX opset 13
    )

    # Set IR version
    model.ir_version = 7

    # Validate the model
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        # Log warning but continue - dynamic shapes may cause validation issues
        print(f"ONNX validation warning: {e}")

    # Save the model
    onnx.save(model, output_path)

    return output_path


def convert_json_to_onnx(build_path: str) -> str:
    """
    Find and convert ONNX JSON files in the build directory.

    Args:
        build_path: Path to the build directory

    Returns:
        Path to the generated ONNX model, or None if no JSON found
    """
    json_paths = Path(build_path).glob("*.onnx.json")
    models = []
    for json_path in json_paths:
        onnx_path = json_path.parent / (json_path.name.replace(".onnx.json", ".onnx"))
        build_onnx_model(str(json_path), str(onnx_path))
        models.append(str(onnx_path))

    return models
