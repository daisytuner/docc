---
name: code-review
description: Review a pull request by combining an architecture review and a dependency / deprecation review. Use when reviewing pull requests, code changes, or diffs, especially changes that affect component structure, APIs, shared libraries, schemas, configuration, or deprecated functionality.
---

# Code Review

## Purpose

Review a pull request by applying two complementary analyses:

1. **Architecture Review / Analysis** — [`../architecture-review/SKILL.md`](../architecture-review/SKILL.md)
2. **Dependency / Deprecation Analysis** — [`../dependency-review/SKILL.md`](../dependency-review/SKILL.md)

Read both skill files before starting and follow their analysis processes and rules. This skill defines when each analysis applies, how the results are combined, and how findings are reported in the review.

## Review process

### 1. Understand the pull request

* Read the pull request title, description, and linked issues.
* Read the full diff.
* Inspect surrounding code and configuration as needed to understand the changed behavior.
* Identify the intent of the change and whether the diff matches that intent.

### 2. Decide which analyses apply

Run the **architecture review** when the change:

* adds, removes, or restructures components, modules, or libraries,
* changes responsibilities or boundaries between components,
* introduces new abstractions, integrations, or dependencies,
* changes control flow or data flow between components.

Run the **dependency review** when the change:

* modifies a public or internal API, function signature, or type,
* changes a shared library, schema, file format, event, or configuration,
* removes, renames, or deprecates functionality,
* changes behavior that other components or repositories may rely on.

Run both when both sets of criteria apply. If neither applies (for example, a purely local bug fix, test, or documentation change), skip both analyses and perform only the general checks in step 3.

### 3. General checks

Independently of the two analyses, check the diff for:

* correctness and logic errors,
* unhandled edge cases and error paths,
* security issues (input validation, injection, secrets, unsafe memory handling),
* concurrency and resource-lifetime issues,
* missing or inadequate tests for changed behavior,
* inconsistencies with existing conventions in the surrounding code.

### 4. Combine findings

* Deduplicate findings reported by more than one analysis.
* Attach each finding to the most specific file and line in the diff.
* Assign each finding a severity:
  * **Blocking** — incorrect behavior, confirmed breaking change without migration, security issue, or architecturally problematic design.
  * **Should fix** — significant risk, missing tests, or an architectural or compatibility concern with a clear mitigation.
  * **Suggestion** — optional improvement.
* Do not report formatting or style issues that are handled by automated tooling.

## Output

### Inline comments

For each finding, leave an inline comment on the relevant line containing:

* the severity,
* what the problem is,
* why it matters,
* a concrete fix or mitigation, with a code suggestion when practical.

### Review summary

Provide a summary comment with the following sections. For each analysis that was skipped, retain its section and state why it was skipped.

#### Change Overview

A short description of what the pull request changes and why.

#### Architecture Assessment

The **Design Assessment** and the most important **Architectural Risks** from the architecture review. Include the **Component Impact** table when more than one component is affected.

#### Dependency Assessment

The **Final Assessment** from the dependency review, the **Cross-Repository Impact** table, any **Breaking Changes** and **Deprecations**, and the **Unknown / Unverified Dependencies**.

#### Findings

| Severity | Location | Finding |
| -------- | -------- | ------- |
| ...      | ...      | ...     |

#### Verdict

One of:

* **Approve** — no blocking findings.
* **Approve with comments** — only should-fix or suggestion findings.
* **Request changes** — at least one blocking finding.

## Rules

* Follow the rules of both referenced skills.
* Base every finding on the actual diff, code, or configuration; do not speculate without stating the uncertainty.
* Prefer few, high-confidence findings over many low-value comments.
* Do not claim a change is non-breaking because no consumer was found.
* Do not modify code or other repositories as part of the review.
