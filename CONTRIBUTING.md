# Contributing to docc

Thanks for your interest in contributing to the Daisytuner Optimizing Compiler
Collection (`docc`)! This guide explains how to get set up, propose changes, and
get them merged.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct
Please be respectful and constructive in all interactions. Assume good intent,
keep discussions focused on the technical problem, and help maintain a welcoming
community.

## Ways to Contribute
- Report bugs using the [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md) template.
- Request features using the [Feature Request](.github/ISSUE_TEMPLATE/enhancement_request.md) template.
- Improve documentation.
- Submit bug fixes or new features via pull requests.

## Getting Started
1. **Fork** the repository and clone your fork.
2. Make sure submodules are initialized:
   ```bash
   git submodule update --init --recursive
   ```
3. Install the prerequisites for the component you want to work on. The project
   is organized into modules, each with its own `README.md`:
   - `sdfg/` — core SDFG intermediate representation, interfaces, passes, and analyses.
   - `python/` — Python (JIT) frontend and bindings.
   - `mlir/` — MLIR dialect and PyTorch frontend.
   - `opt/` — optional pass and transformation logic.
   - `rtl/` — runtime libraries for instrumentation.
   - `targets/` — target-specific code generation and definitions.
   - `c-compile/` — C/C++ emission, compilation, and linking logic.
4. `docc` requires `clang-19` (see [LLVM releases](https://apt.llvm.org/)).
5. Install the developer tooling used for formatting and checks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Development Workflow
1. Create a topic branch from `main`:
   ```bash
   git checkout -b my-feature
   ```
2. Make your changes in small, focused commits with clear messages.
3. Keep your branch up to date with `main` and resolve conflicts early.
4. Run formatters, linters, and tests locally before pushing.

## Coding Standards
This repo enforces formatting via [pre-commit](https://pre-commit.com/):
- **C/C++** is formatted with `clang-format` (`-style=file`, version 19).
- **Python** is formatted with [`black`](https://github.com/psf/black).
- Trailing whitespace, end-of-file newlines, and YAML validity are checked
  automatically.

Run all hooks against your changes:
```bash
pre-commit run --all-files
```

## Testing
All changes should be covered by tests. Depending on the component:
- **Python tests** use `pytest`, e.g.:
  ```bash
  python -m pytest python/tests/ -q
  ```
- Add **unit tests** for new logic and edge cases.
- Add or update **integration tests** when behavior crosses component or target
  boundaries (e.g. frontend → SDFG → target codegen).
- Keep **test coverage** from regressing; cover new and critical code paths.

When relevant, test against the supported targets (`sequential`, `openmp`,
`cuda`, `rocm`) for the code you touch.

## Submitting Changes
1. Push your branch and open a pull request against `main`.
2. Fill out the [pull request template](.github/PULL_REQUEST_TEMPLATE.md),
   including the unit test, integration test, documentation, and test coverage
   checklists.
3. Link any related issues (e.g. `Closes #123`).
4. Ensure CI passes and address review feedback.
5. A maintainer will merge once the PR is approved and green.

## Reporting Issues
Before opening a new issue, search existing issues to avoid duplicates. Then use
the appropriate template and provide as much detail as possible (environment,
versions, reproduction steps, logs).

---

By contributing, you agree that your contributions will be licensed under the
project's [BSD license](LICENSE).
