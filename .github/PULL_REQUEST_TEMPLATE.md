<!--
Thanks for contributing! Please fill out this template to help reviewers
understand your changes. Remove sections that are not relevant.
-->

## Description
A clear and concise description of what this PR does and why.

## Related Issues
<!-- Link issues this PR closes or relates to, e.g. "Closes #123" -->
- Closes #

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Refactor / internal change (no functional change)
- [ ] Documentation update

## Quality Checklist
*Please ensure the following are completed before requesting review:*

* [ ] Runtime-compatible (no externally visible existing method has its parameters changed, was deleted or has its exception behavior changed)
* [ ] Compile-time compatible (new arguments & properties added have default values, but users need to recompile against the new version to work)

IR:
* [ ] Introduces new types into the SDFG IR
* [ ]  Changed / added properties on SDFG IR Elements (LibraryNodes, ControlFlowNodes DataFlowNodes etc.)
  *  [ ] New / modified types have serialization / deserialization support
  *  [ ] Deserialization is backward compatible (new serialization can deserialize old format and represent it in the new format if applicable)
  *  [ ] Serialized format is backward compatible (previous deserializer can read new serialized format without incorrectness (just lacking additional guarantees that are optional or did not exist before)
  *  [ ] Serializer has an option to emit the previous format

### Unit Tests
- [ ] New unit tests have been added for the changes.
- [ ] Existing unit tests pass locally.
- [ ] Edge cases and error paths are covered.

### Integration Tests
- [ ] New integration tests have been added (or existing ones updated).
- [ ] Integration tests pass locally.
- [ ] Interactions with other components/services are verified.

### Documentation
- [ ] Public APIs, options, and behavior changes are documented.
- [ ] README / docs site / inline comments updated where applicable.
- [ ] Changelog / release notes updated (if applicable).

### Test Coverage
- [ ] Test coverage has not decreased as a result of this change.
- [ ] Coverage report reviewed and acceptable.
- [ ] Critical / new code paths are covered.

## How Has This Been Tested?
Describe the tests you ran to verify your changes and provide instructions to reproduce.
Include relevant details for your test configuration (OS, Python/LLVM/Docc version, hardware/target).

## Additional Context
Add any other context about the PR here (design decisions, trade-offs, follow-ups).
