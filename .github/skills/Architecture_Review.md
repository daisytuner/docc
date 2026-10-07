# Architecture Review / Analysis

## Purpose

Analyze a proposed or existing code change from an architectural perspective.

The goal is to explain:

* what is changing,
* why it is changing,
* which architectural components are involved,
* how those components interact,
* and how the change fits into the existing system architecture.

This skill focuses on understanding the architecture and the design of the change. It does not perform a comprehensive cross-repository dependency or deprecation analysis; that is handled by the Dependency / Deprecation Analysis skill.

## When to use

Use this skill when:

* reviewing a pull request,
* implementing a feature,
* refactoring an existing component,
* changing service boundaries,
* introducing a new component or integration,
* changing APIs or other architectural contracts,
* or evaluating whether a proposed design fits the existing architecture.

## Analysis process

### 1. Understand the change

Identify:

* the components being added, removed, or modified,
* the current behavior,
* the proposed behavior,
* the motivation for the change,
* architectural responsibilities affected by the change.

Do not limit the analysis to files explicitly mentioned in the task. Inspect surrounding code and configuration as necessary to understand the design.

### 2. Identify architectural components

Identify relevant:

* applications,
* services,
* modules,
* libraries,
* APIs,
* databases,
* message brokers,
* events,
* external integrations,
* authentication and authorization boundaries,
* configuration,
* infrastructure components.

Explain the responsibility of each component relevant to the change.

### 3. Trace interactions

Describe how the affected components interact.

Consider:

* synchronous calls,
* asynchronous messaging,
* data flow,
* control flow,
* persistence,
* error handling,
* authentication and authorization,
* configuration,
* retries and failure handling.

Prefer a simple dependency or data-flow diagram when it improves understanding.

Example:

```
Client
  |
  | HTTP
  v
API Service
  |
  +----> Database
  |
  +----> Message Broker
              |
              v
         Worker Service
```

### 4. Analyze the proposed design

Determine:

* whether the change follows the existing architectural patterns,
* whether responsibilities remain appropriately separated,
* whether new coupling is introduced,
* whether existing abstractions are reused appropriately,
* whether a new abstraction is justified,
* whether the change introduces architectural inconsistencies.

Call out architectural risks explicitly.

### 5. Analyze boundaries

Pay particular attention to changes crossing:

* service boundaries,
* repository boundaries,
* API boundaries,
* library boundaries,
* database boundaries,
* messaging/event boundaries.

Identify what information or behavior crosses each boundary.

### 6. Identify architectural consequences

Describe consequences such as:

* increased coupling,
* reduced coupling,
* new dependencies,
* changed ownership,
* scalability implications,
* availability implications,
* consistency implications,
* security implications,
* operational complexity,
* deployment implications.

Only discuss consequences that are relevant and supported by the code or architecture.

## Output

Produce the following sections.

### Architectural Overview

A concise explanation of the relevant architecture and the role of the affected components.

### Change Flow

Explain how the proposed change flows through the system.

Include a diagram when useful.

### Component Impact

| Component | Responsibility | Change | Architectural impact |
| --------- | -------------- | ------ | -------------------- |
| ...       | ...            | ...    | ...                  |

### Interfaces and Boundaries

Describe any affected boundaries and the contracts or data flowing across them.

### Architectural Risks

List significant architectural risks or inconsistencies.

For each risk explain:

* what the risk is,
* why it exists,
* and how it could be mitigated.

### Design Assessment

State whether the proposed design is:

* consistent with the existing architecture,
* acceptable with reservations,
* or architecturally problematic.

Explain the reasoning.

### Recommendations

Provide concrete recommendations where the design could be improved.

## Rules

* Base conclusions on actual code and configuration wherever possible.
* Do not invent architectural components or relationships.
* Distinguish observed architecture from inferred architecture.
* Keep the analysis focused on architecture rather than implementation minutiae.
* Do not perform an exhaustive search of other repositories unless required to understand an architectural boundary.
* Do not claim that a change is backward compatible or non-breaking based solely on this analysis.
  Cross-repository compatibility is handled by the Dependency / Deprecation Analysis skill.
