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
- [Versioning and Releases](#versioning-and-releases)
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
   - `opt/` — transformations and built-in targets (OpenMP, CUDA, ROCm).
   - `rpc/` — remote optimization passes (transfer tuning).
   - `targets/` — accelerator backends (Tenstorrent, ETSoC).
   - `c-compile/` — C/C++ emission, compilation, and linking logic.
   - `llvm/` — C/C++ frontend: compiler drivers and LLVM plugin.
   - `python/` — Python (JIT) frontend and bindings.
   - `pytorch/` — PyTorch `torch.compile` backend.
   - `rtl/`, `arg-capture-io/` — runtime libraries for instrumentation and argument capture.
   - `mlir/` — legacy MLIR frontend (no longer used for PyTorch).

   The [architecture overview](https://docs.daisytuner.com/docc/architecture/overview) explains how
   the modules work together.
4. `docc` requires `clang-21` (see [LLVM releases](https://apt.llvm.org/)).
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
- **C/C++** is formatted with `clang-format` (`-style=file`, version 21).
- **Python** is formatted with [`black`](https://github.com/psf/black).
- Trailing whitespace, end-of-file newlines, and YAML validity are checked
  automatically.

Run all hooks against your changes:
```bash
pre-commit run --all-files
```

## Testing
All changes should be covered by tests. Depending on the component:
- **C++ unit tests** use GoogleTest and run with `ctest` from the build directory, e.g.:
  ```bash
  ctest --test-dir build --parallel
  ```
- **Python tests** use `pytest`, e.g.:
  ```bash
  python -m pytest python/tests/ -q -m unmarked
  ```
- **PyTorch tests** use `pytest` and select the target with `--target`, e.g.:
  ```bash
  python -m pytest pytorch/tests/ -q --target=openmp
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

## Versioning and Releases
Versions are `MAJOR.MINOR.PATCH`. There is no version file to bump and no tag to push for a release:
- [`VERSION_BASE`](VERSION_BASE) holds `MAJOR.MINOR` (e.g. `0.8`). Edit it only for a minor or major
  bump; the first release after the change is `MAJOR.MINOR.0`.
- The patch number is allocated by the Daisytuner backend when a release runs.
- Local and CI builds that are not releases report `MAJOR.MINOR.0` with the commit hash, e.g.
  `docc version: 0.8.0+gabc1234` (Python wheels: `0.8.0.dev0+gabc1234`). They never contact the backend.

To release, run the **Release** workflow (`.github/workflows/release.yml`) from `main`
(Actions → Release → Run workflow). Its first job allocates the next patch, e.g. `0.8.4`, and all wheels
and packages of that run are built with it. Tick **dry_run** to preview the version and build everything
without publishing; dry runs also work from other branches. A hotfix for an older series runs from a
`release/**` branch whose `VERSION_BASE` still names that series.

A release from `main` also deploys the docs (docs.daisytuner.com/docc) once its packages are published,
so the site shows the released version and matches the released code. Docs changes merged to `main` go
live with the next release. To redeploy in between, e.g. for an urgent fix, run Actions → Docs → Run
workflow on `main` with the currently released version (PyPI or the last Release run shows it). Hotfix
releases do not deploy the docs, and pull requests only build them.

## Reporting Issues
Before opening a new issue, search existing issues to avoid duplicates. Then use
the appropriate template and provide as much detail as possible (environment,
versions, reproduction steps, logs).

---

By contributing, you agree that your contributions will be licensed under the
project's [BSD license](LICENSE).
