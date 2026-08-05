#!/usr/bin/env python3
"""Validate daisy RTL instrumentation traces against the JSON schema.

The schema lives next to this file at ../schema/daisy_trace.schema.json and
describes the Chrome-trace subset emitted by rtl/src/instrumentation.cpp.

Usage:
    python validate_trace.py trace.json [more.json ...]
    python validate_trace.py --schema /path/to/schema.json trace.json

Exit code is 0 when every file validates, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "schema" / "daisy_trace.schema.json"
)


def load_schema(schema_path: Path) -> dict:
    with open(schema_path, "r") as f:
        return json.load(f)


def build_validator(schema_path: Path = DEFAULT_SCHEMA_PATH) -> Draft202012Validator:
    schema = load_schema(schema_path)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def iter_errors(instance: object, validator: Draft202012Validator):
    """Yield human-readable validation error strings, sorted by location."""
    for error in sorted(
        validator.iter_errors(instance), key=lambda e: list(e.absolute_path)
    ):
        location = "/".join(str(p) for p in error.absolute_path) or "<root>"
        yield f"{location}: {error.message}"


def validate_trace(
    instance: object, validator: Draft202012Validator | None = None
) -> list[str]:
    """Validate an already-parsed trace object. Returns a list of error strings."""
    if validator is None:
        validator = build_validator()
    return list(iter_errors(instance, validator))


def validate_file(path: Path, validator: Draft202012Validator) -> list[str]:
    try:
        with open(path, "r") as f:
            instance = json.load(f)
    except json.JSONDecodeError as exc:
        return [f"invalid JSON: {exc}"]
    return validate_trace(instance, validator)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "traces", nargs="+", type=Path, help="Trace JSON files to validate"
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Path to the JSON schema (defaults to the bundled daisy_trace schema)",
    )
    args = parser.parse_args(argv)

    validator = build_validator(args.schema)

    ok = True
    for trace in args.traces:
        errors = validate_file(trace, validator)
        if errors:
            ok = False
            print(f"FAIL {trace}")
            for err in errors:
                print(f"  - {err}")
        else:
            print(f"OK   {trace}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
