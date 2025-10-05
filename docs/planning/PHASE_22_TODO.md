# Phase 22: Paper2Agent Integration - TODO List

**Status**: In Progress
**Started**: 2025-01-10
**Target Completion**: 2025-01-13

---

## Priority 1: Naming & Documentation (Foundation)

### Naming Clarification
- [ ] **TODO-001**: Add mcp_server naming clarification notice to root README.md
  - Insert warning after badges, before "Core Philosophy" section
  - Use standard warning template from implementation plan

- [ ] **TODO-002**: Add mcp_server naming clarification notice to code/README.md
  - Insert warning after title, before "Project Overview"
  - Use standard warning template

- [ ] **TODO-003**: Create `code/mcp_server/README.md` with detailed naming explanation
  - Full explanation of historic naming
  - Clarify NOT Model Context Protocol
  - Link to migration plan

### Migration Planning
- [ ] **TODO-004**: Create `MIGRATION_PLAN.md` with future renaming strategy
  - Document `mcp_server` → `agent_core` transition roadmap
  - Symlink strategy for backward compatibility
  - URL preservation approach
  - Timeline (6-12 month window)

- [ ] **TODO-005**: Create `code/mcp_server/FUTURE_NAMING.md` for internal transition plan
  - Internal reference for new naming conventions
  - Docstring update strategy
  - Gradual transition milestones

---

## Priority 2: Five Laws Governance Tools (Core Functionality)

### Five Laws Validator (HIGHEST PRIORITY)
- [ ] **TODO-006**: Create `code/tools/lib/` directory for shared libraries

- [ ] **TODO-007**: Create `code/tools/lib/five_laws_evaluator.py` - unified evaluation logic
  - Import all 5 law validators from governance/five_laws_validator/
  - Create unified interface for evaluating any text
  - Aggregate scores from all laws
  - Generate compliance report with recommendations
  - Export `evaluate_text()` function

- [ ] **TODO-008**: Create `code/tools/run_five_laws_validator.py` (HIGHEST PRIORITY)
  - CLI wrapper for Five Laws evaluation
  - Support --text, --file, and stdin input modes
  - JSON output with compliance scores
  - Include MCP_TOOL_ENTRYPOINT marker
  - Comprehensive docstring with usage examples

### Governance Orchestration Tools
- [ ] **TODO-009**: Create `code/tools/run_governed_response.py` - governed response generator
  - Accept user prompt via CLI
  - Execute through protocol stack (REP, ESL, VVP)
  - Apply Five Laws validation automatically
  - Return governed response + compliance report
  - Support --protocols flag to select which protocols to use
  - Support --min-compliance threshold

- [ ] **TODO-010**: Create `code/tools/run_cognitive_workflow.py` - multi-step orchestrator
  - Support predefined workflows: reason_validate_govern, analyze_govern, full_governance
  - Accept --workflow flag to select workflow type
  - Accept --input for text/file input
  - Chain multiple protocols in sequence
  - Generate comprehensive execution trace
  - Output final result + Five Laws compliance

---

## Priority 3: Individual Protocol Wrappers

### Core Protocols (Priority 3A)
- [ ] **TODO-011**: Create `code/tools/run_rep_tool.py` - Reasoning & Explanation Protocol wrapper
  - CLI wrapper for REP protocol
  - Support multiple reasoning types (deductive, inductive, abductive, etc.)
  - Accept --facts, --rules, --reasoning-type flags
  - JSON output with reasoning chain

- [ ] **TODO-012**: Create `code/tools/run_esl_tool.py` - Emotional State Layer Protocol wrapper
  - CLI wrapper for ESL protocol
  - Accept --user-input flag or stdin
  - Output emotional analysis with detected emotions
  - Include valence, intensity, confidence scores

- [ ] **TODO-013**: Create `code/tools/run_vvp_tool.py` - Validation & Verification Protocol wrapper
  - CLI wrapper for VVP protocol
  - Accept --rules flag for validation rules
  - Output validation status and reason
  - Support JSON input/output

- [ ] **TODO-014**: Create `code/tools/run_ccp_tool.py` - Cognitive Control Protocol wrapper
  - CLI wrapper for CCP protocol
  - Accept workflow definition via JSON
  - Output coordination results and metrics
  - Support async execution

- [ ] **TODO-015**: Create `code/tools/run_sep_tool.py` - Semantic Encoding Protocol wrapper
  - CLI wrapper for SEP protocol (if exists, check first)
  - Accept text for semantic encoding
  - Output encoded representation

- [ ] **TODO-016**: Create `code/tools/run_mtp_tool.py` - Memory Tagging Protocol wrapper
  - CLI wrapper for MTP protocol
  - Accept entity extraction input
  - Output tagged entities and patterns

### Additional Protocols (Priority 3B)
- [ ] **TODO-017**: Create `code/tools/run_eep_tool.py` - Energy Efficiency Protocol wrapper
  - CLI wrapper for EEP protocol (check if exists)
  - Measure energy/resource usage
  - Output efficiency metrics

- [ ] **TODO-018**: Create `code/tools/run_hip_tool.py` - Human Interaction Protocol wrapper
  - CLI wrapper for HIP protocol
  - Handle human interaction patterns
  - Output interaction analysis

- [ ] **TODO-019**: Create `code/tools/run_sp_tool.py` - Security Protocol wrapper
  - CLI wrapper for SP protocol (check if exists)
  - Security validation and checks
  - Output security assessment

- [ ] **TODO-020**: Create `code/tools/run_pocp_tool.py` - Protocol Orchestration wrapper
  - CLI wrapper for POCP (check if exists)
  - Orchestrate multiple protocols
  - Output orchestration results

---

## Priority 4: Workflow Wrappers

- [ ] **TODO-021**: Create `code/tools/run_writing_team_workflow.py` - multi-agent writing workflow
  - Execute Ideator → Drafter → Critic → Revisor → Summarizer workflow
  - Accept writing prompt/topic
  - Output final document + workflow trace
  - Apply Five Laws validation to final output

- [ ] **TODO-022**: Create `code/tools/run_reasoning_workflow.py` - reasoning pipeline workflow
  - Execute full reasoning pipeline
  - REP → VVP → Five Laws validation
  - Output reasoning chain + compliance report

- [ ] **TODO-023**: Create `code/tools/run_analysis_workflow.py` - analysis workflow
  - Execute analysis-focused workflow
  - ESL + REP → VVP → Five Laws
  - Output comprehensive analysis + governance report

---

## Priority 5: Discovery & Manifest System

- [ ] **TODO-024**: Create `code/tools/tools_manifest.json` - complete tool discovery manifest
  - JSON structure with all tools categorized
  - Include governance_tools, orchestration_tools, protocol_tools, workflow_tools
  - Specify input/output for each tool
  - Mark composable tools
  - Include version and framework info

- [ ] **TODO-025**: Create `PAPER2AGENT_INTEGRATION.md` - comprehensive integration guide
  - Overview of Paper2Agent + SIM-ONE integration
  - Quick Start section with one-line examples
  - Five Laws Governance section with detailed examples
  - Protocol Composition patterns
  - Example Scenarios (5+ scenarios)
  - Tool Discovery guide
  - API Reference for all tools

- [ ] **TODO-026**: Create `code/tools/README.md` - tool catalog and usage documentation
  - Complete catalog of all available tools
  - Usage examples for each wrapper
  - Protocol composition patterns
  - Input/output specifications for each tool
  - Error handling guide
  - Troubleshooting section

---

## Priority 6: Main Documentation Updates

- [ ] **TODO-027**: Update root `README.md` with Tool Entrypoints section
  - Add section after Case Studies: "🛠️ Tool Entrypoints for AI Agent Integration"
  - Include Five Laws Governance subsection
  - Include Available Tools list
  - Link to PAPER2AGENT_INTEGRATION.md, tools/README.md, tools_manifest.json

- [ ] **TODO-028**: Update `code/README.md` with standalone tools usage section
  - Add section after Quick Start: "Using SIM-ONE Protocols as Standalone Tools"
  - Include Individual Protocol Tools subsection
  - Include Governance Tools subsection
  - Include Example Usage with code blocks
  - Link to tools/README.md

---

## Priority 7: Examples & Use Cases

- [ ] **TODO-029**: Create `examples/paper2agent_usage_examples.sh` - example scripts
  - Example 1: Validate external AI response
  - Example 2: Generate governed response
  - Example 3: Multi-step workflow
  - Example 4: Protocol composition/chaining
  - Make executable (chmod +x)

- [ ] **TODO-030**: Create `examples/USE_CASES.md` - use case scenarios documentation
  - Scenario 1: Paper2Agent Self-Governance
  - Scenario 2: Multi-Agent Validation
  - Scenario 3: Governed Content Generation
  - Scenario 4: Reasoning Chain Validation
  - Scenario 5: Emotional Intelligence Analysis
  - Include code examples for each scenario

---

## Priority 8: Testing & Validation

- [ ] **TODO-031**: Create `code/tests/test_tool_wrappers.py` - wrapper testing suite
  - Test each wrapper executes successfully
  - Test CLI argument parsing
  - Test JSON output validity
  - Test error handling
  - Test stdin/file input modes
  - Test protocol integration correctness

- [ ] **TODO-032**: Create `code/tests/test_five_laws_tools.py` - Five Laws validator tests
  - Test validate compliant response (expect high scores)
  - Test validate non-compliant response (expect low scores)
  - Test detect truth violations
  - Test detect governance failures
  - Test recommendations are actionable
  - Test edge cases (empty input, invalid JSON, etc.)

- [ ] **TODO-033**: Create `code/tests/test_paper2agent_integration.py` - integration tests
  - Test manifest file is valid JSON
  - Test all listed tools exist in filesystem
  - Test all tools are executable
  - Test tool discovery mechanism
  - Test protocol composition
  - Test end-to-end workflow

- [ ] **TODO-034**: Create `code/tests/test_backward_compatibility.py` - compatibility tests
  - Test existing API endpoints still work
  - Test FastAPI server functions unchanged
  - Test all original workflows execute correctly
  - Test no breaking changes in core protocols
  - Regression test suite

---

## Priority 9: Contributing Guidelines

- [ ] **TODO-035**: Update `CONTRIBUTING.md` with tool wrapper creation guidelines
  - Add "Creating Tool Wrappers" section
  - Document wrapper conventions and standards
  - Provide step-by-step guide for new wrappers
  - Include code template
  - Reference implementation plan

---

## Priority 10: Final Validation & Deployment

- [ ] **TODO-036**: Run all tests and validate backward compatibility
  - Execute pytest on all test files
  - Verify all tests pass
  - Check code coverage
  - Validate no regressions

- [ ] **TODO-037**: Final validation: Paper2Agent tool discovery and Five Laws validator
  - Manually test Paper2Agent can discover tools via manifest
  - Test Five Laws validator with multiple sample responses
  - Validate all tool wrappers work standalone
  - Test protocol composition chains
  - Verify documentation is complete and accurate

- [ ] **TODO-038**: Create release notes and update changelog
  - Document all new features
  - List all new tool wrappers
  - Explain naming clarification strategy
  - Note backward compatibility maintained

---

## Completion Checklist

### Documentation Complete
- [ ] All 8 documentation files created/updated
- [ ] Naming clarifications in all READMEs
- [ ] Migration plan documented
- [ ] Integration guide complete

### Tools Complete
- [ ] Five Laws validator working
- [ ] All core protocol wrappers functional
- [ ] Workflow orchestrators operational
- [ ] Manifest file accurate and complete

### Testing Complete
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Backward compatibility verified
- [ ] Zero regressions detected

### Quality Assurance
- [ ] All tools follow standard template
- [ ] JSON output validated
- [ ] Error handling robust
- [ ] Documentation accurate
- [ ] Examples tested and working

---

**Total Tasks**: 38
**Estimated Effort**: 20-25 hours
**Target Completion**: 2025-01-13

---

## Notes & Observations

*Use this section to track progress notes, blockers, or observations during implementation*

- [ ] Note: Check which protocols actually exist before creating wrappers (SEP, EEP, SP may need verification)
- [ ] Note: Ensure async handling is correct in all wrappers that use async protocols
- [ ] Note: Validate Five Laws validator modules exist and are importable
- [ ] Note: Consider adding --verbose flag to all wrappers for debugging

---

*Document Version: 1.0*
*Created: 2025-01-10*
*Status: Active - In Progress*
