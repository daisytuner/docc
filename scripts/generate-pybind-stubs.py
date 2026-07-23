#!/usr/bin/env python3

from pathlib import Path
import glob

UNRESOLVED_NAMES = [
    ("PyStructuredSDFGBuilder", "StructuredSDFGBuilder"),
    ("PyStructuredSDFG", "StructuredSDFG"),
    ("PyArgumentsAnalysis", "ArgumentsAnalysis"),
    ("PyAssumptionsAnalysis", "AssumptionsAnalysis"),
    ("PyControlFlowAnalysis", "ControlFlowAnalysis"),
    ("PyDominanceAnalysis", "DominanceAnalysis"),
    ("PyEscapeAnalysis", "EscapeAnalysis"),
    ("PyFlopAnalysis", "FlopAnalysis"),
    ("PyLoopAnalysis", "LoopAnalysis"),
    ("PyTypeAnalysis", "TypeAnalysis"),
    ("PyUsers", "Users"),
]


def run_pybind11_stubgen(python_path: Path) -> None:
    import pybind11_stubgen.__init__

    unresolved_names_regex = ""
    for old_name, _ in UNRESOLVED_NAMES:
        if unresolved_names_regex == "":
            unresolved_names_regex = old_name
        else:
            unresolved_names_regex += "|" + old_name

    pybind11_stubgen.__init__.main(
        [
            "-o",
            str(python_path),
            "--ignore-unresolved-names",
            unresolved_names_regex,
            "docc.sdfg",
        ]
    )


def find_pyi_files(python_path: Path) -> list[Path]:
    return [
        Path(pyi_file)
        for pyi_file in glob.glob(str(python_path) + "/**/*.pyi", recursive=True)
    ]


def replace_all_names(pyi_file: Path) -> None:
    content: str = pyi_file.read_text(encoding="utf-8")
    for old_name, new_name in UNRESOLVED_NAMES:
        content: str = content.replace(old_name, new_name)
    pyi_file.write_text(content, encoding="utf-8")


def main() -> int:
    docc_root = Path(__file__).parent.parent
    if not docc_root.exists():
        print("Error: Could not find docc repository root: " + str(docc_root))
        return 1

    python_path = docc_root / "python"
    run_pybind11_stubgen(python_path)

    pyi_files = find_pyi_files(python_path)
    for pyi_file in pyi_files:
        replace_all_names(pyi_file)

    return 0


if __name__ == "__main__":
    try:
        import pybind11_stubgen
    except:
        print("Error: Could not find pybind11-stubgen package")
        print("       Install it with: pip install pybind11-stubgen")
        exit(1)
    exit(main())
