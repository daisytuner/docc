---
name: dependency-review
description: Determine how a change affects dependencies, consumers, APIs, contracts, and deprecated functionality across the organization. Use when a change modifies public or internal APIs, shared libraries, schemas, events, configuration, removes or renames functionality, or deprecates features.
---

# Dependency / Deprecation Analysis

## Purpose

Determine how a proposed or existing change affects dependencies, consumers, APIs, contracts, and deprecated functionality across the GitHub organization.

The primary goal is to identify whether a change:

* breaks existing consumers,
* requires changes in other repositories,
* makes existing functionality obsolete,
* affects shared libraries or APIs,
* or requires a coordinated migration.

This skill complements the Architecture Review / Analysis skill. It focuses on **dependency and compatibility analysis**, not general architectural design.

## When to use

Use this skill when a change:

* modifies a public or internal API,
* changes a shared library,
* changes a schema,
* changes an event or message,
* changes configuration consumed by another component,
* removes or renames functionality,
* deprecates an API or feature,
* changes service behavior relied upon by other repositories,
* or may affect consumers outside the current repository.

## Analysis process

### 1. Identify changed contracts

Identify every interface affected by the change.

Consider:

* REST/HTTP APIs,
* GraphQL APIs,
* gRPC/protobuf interfaces,
* OpenAPI contracts,
* events and messages,
* shared libraries,
* SDKs,
* generated clients,
* database schemas,
* configuration formats,
* authentication and authorization contracts,
* command-line interfaces,
* file formats.

For each contract record:

* current behavior,
* proposed behavior,
* affected symbols/endpoints/types,
* compatibility implications.

### 2. Find consumers

Search the GitHub organization for actual consumers of changed contracts.

Look for:

* API calls,
* imports,
* package dependencies,
* generated clients,
* event subscriptions,
* schema references,
* configuration references,
* shared-library usage,
* tests,
* deployment configuration,
* documentation describing integration.

Prefer source-code evidence over repository naming conventions or assumptions.

For each consumer identify:

* repository,
* component,
* reference/usage,
* dependency,
* affected behavior.

### 3. Classify compatibility

Classify each affected contract as one of:

**Backward compatible**

Existing consumers continue to work without changes.

**Conditionally compatible**

Existing consumers continue to work only under specific conditions, configuration, or migration behavior.

**Breaking**

Existing consumers require changes to continue functioning correctly.

**Unknown**

There is insufficient evidence to determine compatibility.

Never classify a change as non-breaking simply because no consumer was found.

### 4. Analyze breaking changes

Explicitly check for:

* removed endpoints,
* renamed endpoints,
* changed HTTP methods,
* changed required request fields,
* removed response fields,
* changed response field types,
* changed enum values,
* changed error semantics,
* changed authentication requirements,
* changed event names,
* changed event payloads,
* incompatible protobuf changes,
* removed library APIs,
* changed method signatures,
* changed configuration formats,
* changed behavioral contracts.

For each confirmed breaking change explain:

* what changed,
* why it is breaking,
* which consumers are affected,
* what each consumer must change.

### 5. Analyze transitive dependencies

Follow dependency chains where practical.

Example:

```
repo-a
   |
   v
shared-library
   |
   v
repo-b
```

If the shared library changes, determine whether `repo-b` is affected even if the original change originated in `repo-a`.

Consider:

* direct dependencies,
* indirect dependencies,
* generated artifacts,
* shared schemas,
* SDKs,
* services consuming another service's API.

### 6. Analyze deprecations

When functionality is removed, replaced, or marked deprecated, identify:

* what is being deprecated,
* known consumers,
* whether consumers have migration paths,
* replacement functionality,
* remaining dependencies on the deprecated functionality,
* whether the deprecated functionality can safely be removed.

Determine whether the deprecation is:

* unused,
* internally consumed,
* externally consumed within the organization,
* or of unknown usage.

### 7. Determine required changes

For every affected repository, describe the required action.

Examples:

* no action required,
* update API client,
* change request payload,
* migrate to new endpoint,
* update shared-library version,
* regenerate client,
* migrate event consumer,
* update configuration,
* migrate to replacement API,
* coordinate deployment.

Do not modify other repositories unless explicitly requested.

### 8. Recommend migration strategy

When a breaking change exists, recommend a safe migration sequence.

For example:

1. Introduce the new API.
2. Keep the existing API available.
3. Migrate all known consumers.
4. Verify that consumers no longer use the old API.
5. Deprecate the old API.
6. Remove the old API in a later change.

Choose a different sequence when the architecture requires it.

## Output

### Dependency Overview

Summarize the contracts and dependencies affected by the change.

### Cross-Repository Impact

| Repository | Consumer | Contract | Impact          | Breaking? | Required action |
| ---------- | -------- | -------- | --------------- | --------- | --------------- |
| repo-a     | Client X | REST API | Request changed | Yes       | Update request  |
| repo-b     | Worker Y | Event X  | None            | No        | None            |

### Breaking Changes

List every confirmed breaking change.

For each one include:

* changed contract,
* affected consumers,
* reason it breaks,
* required migration.

### Deprecations

| Deprecated item | Consumers | Replacement | Removable? | Required action |
| --------------- | --------- | ----------- | ---------- | --------------- |
| ...             | ...       | ...         | ...        | ...             |

### Unknown / Unverified Dependencies

Explicitly list relationships that could not be verified.

For example:

> No consumers of `POST /orders/v1` were found in the repositories accessible to the analysis. External consumers cannot be ruled out.

### Migration Plan

Provide the recommended order of changes across repositories and services.

### Final Assessment

State one of:

* **No dependency impact**
* **Non-breaking dependency impact**
* **Breaking change — consumer changes required**
* **Deprecation — migration required**
* **Unknown — further dependency information required**

Explain the conclusion.

## Rules

* Search beyond the current repository when the change affects a shared contract or public interface.
* Prefer actual source-code references over assumptions.
* Distinguish confirmed consumers from inferred consumers.
* Consider direct and transitive dependencies.
* Include generated clients and shared libraries.
* Consider both API shape and behavioral compatibility.
* Do not assume that "internal" means "safe to break."
* Do not claim that an API has no consumers merely because none were found.
* Clearly distinguish "no consumer found" from "no consumer exists."
* Report uncertainty explicitly.
* Do not make changes to other repositories unless explicitly requested.
