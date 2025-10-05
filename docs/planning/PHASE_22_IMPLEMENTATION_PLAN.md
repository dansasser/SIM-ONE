# Phase 22: Paper2Agent Integration - Implementation Plan

## Overview
Transform the SIM-ONE repository to be fully indexable and discoverable by Paper2Agent while preserving all current functionality, maintaining the Five Laws of Cognitive Governance, and ensuring backward compatibility.

**Key Objectives:**
1. Address `mcp_server` naming confusion (documentation, future migration plan)
2. Create Paper2Agent-discoverable tool wrappers for all protocols
3. Build Five Laws governance tools for AI response validation
4. Enable Paper2Agent to self-govern using SIM-ONE principles

---

## PHASE 1: Repository Naming & Documentation Strategy

### 1.1 Add Clarification Notices (NO directory rename)
**Action**: Add prominent notices explaining `mcp_server` naming

**Files to Update**:
- ✅ Root `README.md` - Add bold warning at top
- ✅ `code/README.md` - Add clarification section
- ✅ Create `code/mcp_server/README.md` - Dedicated explanation

**Warning Text** (to add to each):
```markdown
⚠️ **Important Naming Note**: The `mcp_server` directory predates the industry-standard "Model Context Protocol" (MCP). In this repository, "mcp_server" refers to SIM-ONE's **"Multi-Protocol Cognitive Platform"** or **"Modular Cognitive Platform"** - the core orchestrator and agent system. This is NOT an MCP tool registry in the modern sense. Directory renaming is planned for future compatibility (see [MIGRATION_PLAN.md](MIGRATION_PLAN.md)), but unchanged now to preserve link/SEO integrity.
```

### 1.2 Prepare Future Migration Documentation
**Action**: Create migration planning documents

**Files to Create**:
- ✅ `MIGRATION_PLAN.md` (root directory)
  - Roadmap for `mcp_server` → `agent_core` transition
  - Symlink strategy for backward compatibility
  - URL preservation approach
  - Timeline (6-12 month window)

- ✅ `code/mcp_server/FUTURE_NAMING.md`
  - Internal reference for new naming
  - Docstring update strategy
  - Gradual transition plan

**Key Points**:
- Keep current directory name unchanged
- Document planned migration to `agent_core/`
- Plan for symlink or stub README when migration occurs
- Maintain all existing URLs/links for 6-12 months minimum

### 1.3 Update Internal Documentation
**Action**: Start using new terminology in new code/docs

**Changes**:
- New docstrings reference "agent_core" or "cognitive_orchestrator"
- New comments use updated terminology
- Existing code unchanged (backward compatibility)

---

## PHASE 2: Core Five Laws Governance Tools

### 2.1 Five Laws Validator Tool (HIGHEST PRIORITY)
**File**: `code/tools/run_five_laws_validator.py`

**Purpose**: Allow Paper2Agent to evaluate ANY AI response against Five Laws

**Implementation**:
```python
"""
Tool: Five Laws Cognitive Governance Validator
Input: Text response to validate, optional context
Output: Compliance assessment with scores and recommendations

Validates any AI-generated content against SIM-ONE's Five Laws:
1. Architectural Intelligence - Coordination vs brute force
2. Cognitive Governance - Governed processes
3. Truth Foundation - Factual accuracy
4. Energy Stewardship - Computational efficiency
5. Deterministic Reliability - Consistent outcomes

Usage:
  python run_five_laws_validator.py --text "response to validate"
  echo "response" | python run_five_laws_validator.py
  python run_five_laws_validator.py --file response.txt
"""
# MCP_TOOL_ENTRYPOINT
```

**Integrates**:
- `code/mcp_server/protocols/governance/five_laws_validator/law1_architectural_intelligence.py`
- `code/mcp_server/protocols/governance/five_laws_validator/law2_cognitive_governance.py`
- `code/mcp_server/protocols/governance/five_laws_validator/law3_truth_foundation.py`
- `code/mcp_server/protocols/governance/five_laws_validator/law4_energy_stewardship.py`
- `code/mcp_server/protocols/governance/five_laws_validator/law5_deterministic_reliability.py`

### 2.2 Governed Response Generator Tool
**File**: `code/tools/run_governed_response.py`

**Purpose**: Generate AI response using SIM-ONE governance

**Features**:
- Accept user prompt
- Execute through protocol stack (REP, ESL, VVP)
- Apply Five Laws validation
- Return governed response + compliance report

### 2.3 Cognitive Workflow Orchestrator
**File**: `code/tools/run_cognitive_workflow.py`

**Purpose**: Multi-step governed processing

**Workflows**:
- `reason_validate_govern`: REP → VVP → Five Laws
- `analyze_govern`: ESL + REP → Five Laws
- `full_governance`: Complete protocol stack with validation

---

## PHASE 3: Individual Protocol Tool Wrappers

### 3.1 Core Cognitive Protocol Wrappers

**Priority 1 Protocols**:
- ✅ `code/tools/run_rep_tool.py` - Reasoning & Explanation Protocol
- ✅ `code/tools/run_esl_tool.py` - Emotional State Layer Protocol
- ✅ `code/tools/run_vvp_tool.py` - Validation & Verification Protocol
- ✅ `code/tools/run_ccp_tool.py` - Cognitive Control Protocol
- ✅ `code/tools/run_sep_tool.py` - Semantic Encoding Protocol
- ✅ `code/tools/run_mtp_tool.py` - Memory Tagging Protocol

**Priority 2 Protocols**:
- ✅ `code/tools/run_eep_tool.py` - Energy Efficiency Protocol
- ✅ `code/tools/run_hip_tool.py` - Human Interaction Protocol
- ✅ `code/tools/run_sp_tool.py` - Security Protocol
- ✅ `code/tools/run_pocp_tool.py` - Protocol Orchestration Protocol

### 3.2 Multi-Agent Workflow Wrappers

**Files**:
- ✅ `code/tools/run_writing_team_workflow.py` - Ideator→Drafter→Critic→Revisor→Summarizer
- ✅ `code/tools/run_reasoning_workflow.py` - Full reasoning pipeline
- ✅ `code/tools/run_analysis_workflow.py` - Analysis-only workflow

### 3.3 Wrapper Implementation Standards

**Each wrapper must include**:
```python
"""
Tool: [Protocol Name]
Input: [Input specification]
Output: [Output specification]
Description: [Clear description]

Usage:
  python run_[name]_tool.py --input "data"
  cat input.txt | python run_[name]_tool.py
  python run_[name]_tool.py --file input.json
"""
# MCP_TOOL_ENTRYPOINT

import sys
import json
import asyncio
from pathlib import Path

# Import protocol from mcp_server
sys.path.insert(0, str(Path(__file__).parent.parent))
from mcp_server.protocols.[name].[name] import [ProtocolClass]

def main():
    # CLI argument parsing
    # Input handling (stdin, file, args)
    # Protocol execution
    # JSON output
    # Error handling

if __name__ == "__main__":
    main()
```

---

## PHASE 4: Discovery & Manifest System

### 4.1 Comprehensive Tools Manifest
**File**: `code/tools/tools_manifest.json`

```json
{
  "version": "1.0",
  "framework": "SIM-ONE",
  "governance_tools": {
    "five_laws_validator": {
      "wrapper": "run_five_laws_validator.py",
      "category": "governance",
      "description": "Evaluate text against Five Laws of Cognitive Governance",
      "input": "text, optional_context",
      "output": "compliance_report",
      "use_case": "Validate AI responses for governed cognition",
      "priority": "highest"
    }
  },
  "orchestration_tools": {
    "governed_response": {
      "wrapper": "run_governed_response.py",
      "description": "Generate governed AI response",
      "uses": ["REP", "ESL", "VVP", "five_laws_validator"]
    },
    "cognitive_workflow": {
      "wrapper": "run_cognitive_workflow.py",
      "description": "Multi-step governed workflow"
    }
  },
  "protocol_tools": {
    "REP": {
      "wrapper": "run_rep_tool.py",
      "category": "reasoning",
      "description": "Advanced reasoning and explanation",
      "composable": true
    },
    "ESL": {
      "wrapper": "run_esl_tool.py",
      "category": "emotional_intelligence",
      "composable": true
    },
    "VVP": {
      "wrapper": "run_vvp_tool.py",
      "category": "validation",
      "composable": true
    },
    "CCP": {
      "wrapper": "run_ccp_tool.py",
      "category": "cognitive_control",
      "composable": true
    },
    "SEP": {
      "wrapper": "run_sep_tool.py",
      "category": "semantic_encoding",
      "composable": true
    },
    "MTP": {
      "wrapper": "run_mtp_tool.py",
      "category": "memory",
      "composable": true
    }
  },
  "workflow_tools": {
    "writing_team": {
      "wrapper": "run_writing_team_workflow.py",
      "description": "Multi-agent collaborative writing"
    },
    "reasoning": {
      "wrapper": "run_reasoning_workflow.py",
      "description": "Full reasoning pipeline"
    }
  }
}
```

### 4.2 Paper2Agent Integration Guide
**File**: `PAPER2AGENT_INTEGRATION.md` (root directory)

**Sections**:
1. **Overview**: How Paper2Agent can use SIM-ONE
2. **Quick Start**: Validate a response in one command
3. **Five Laws Governance**: Apply cognitive governance to any AI output
4. **Protocol Composition**: Chain multiple tools
5. **Example Scenarios**:
   - Validate another AI's response
   - Generate self-governed response
   - Multi-step reasoning with validation
6. **Tool Discovery**: How to find and use tools
7. **API Reference**: All tool inputs/outputs

### 4.3 Tool Index Documentation
**File**: `code/tools/README.md`

**Content**:
- Complete tool catalog
- Usage examples for each wrapper
- Composition patterns
- Input/output specifications
- Error handling guide

---

## PHASE 5: Documentation Updates

### 5.1 Root README.md Updates

**Add after existing Case Studies section**:

```markdown
## 🛠️ Tool Entrypoints for AI Agent Integration

SIM-ONE protocols are available as standalone CLI tools for integration with autonomous agents like Paper2Agent.

### Five Laws Governance for Any AI Response

Validate any AI-generated response against the Five Laws of Cognitive Governance:

```bash
# Validate a response
echo "AI response text" | python code/tools/run_five_laws_validator.py

# Generate a governed response
python code/tools/run_governed_response.py --prompt "Explain quantum mechanics"

# Run complete governed workflow
python code/tools/run_cognitive_workflow.py --input query.txt --workflow full_governance
```

### Available Tools

- **Governance**: Five Laws Validator, Governed Response Generator
- **Protocols**: REP, ESL, VVP, CCP, SEP, MTP, EEP, HIP, SP, POCP
- **Workflows**: Writing Team, Reasoning Pipeline, Analysis Workflow

📖 **Full Integration Guide**: [PAPER2AGENT_INTEGRATION.md](PAPER2AGENT_INTEGRATION.md)
🔧 **Tool Catalog**: [code/tools/README.md](code/tools/README.md)
📋 **Tool Manifest**: [code/tools/tools_manifest.json](code/tools/tools_manifest.json)
```

### 5.2 code/README.md Updates

**Add section after Quick Start**:

```markdown
## Using SIM-ONE Protocols as Standalone Tools

Each protocol is available as a CLI tool in `/tools/` for integration with autonomous agents:

### Individual Protocol Tools
- `run_rep_tool.py` - Reasoning & Explanation
- `run_esl_tool.py` - Emotional State Analysis
- `run_vvp_tool.py` - Validation & Verification
- (see tools/README.md for complete list)

### Governance Tools
- `run_five_laws_validator.py` - Validate any response against Five Laws
- `run_governed_response.py` - Generate governed AI responses
- `run_cognitive_workflow.py` - Multi-step governed processing

### Example Usage
```bash
# Validate AI response
python tools/run_five_laws_validator.py --text "response to check"

# Use REP for reasoning
python tools/run_rep_tool.py --facts "fact1,fact2" --rules "rule1,rule2"

# Chain protocols
python tools/run_cognitive_workflow.py --protocols REP,VVP --validate
```

See [tools/README.md](tools/README.md) for detailed usage.
```

### 5.3 Add Naming Clarification to All READMEs

Insert warning notice at the top of:
- Root `README.md` (after badges, before "Core Philosophy")
- `code/README.md` (after title, before "Project Overview")
- Create `code/mcp_server/README.md` (new file, dedicated explanation)

---

## PHASE 6: Testing & Validation

### 6.1 Tool Wrapper Tests
**File**: `code/tests/test_tool_wrappers.py`

**Test Coverage**:
- ✅ Each wrapper executes successfully
- ✅ CLI argument parsing works
- ✅ JSON output is valid
- ✅ Error handling is robust
- ✅ Stdin/file input works
- ✅ Protocol integration correct

### 6.2 Five Laws Validator Tests
**File**: `code/tests/test_five_laws_tools.py`

**Test Scenarios**:
- ✅ Validate compliant response (high scores)
- ✅ Validate non-compliant response (low scores)
- ✅ Detect truth violations
- ✅ Detect governance failures
- ✅ Recommendations are actionable

### 6.3 Integration Tests
**File**: `code/tests/test_paper2agent_integration.py`

**Test Coverage**:
- ✅ Manifest file is valid JSON
- ✅ All listed tools exist
- ✅ All tools are executable
- ✅ Tool discovery works
- ✅ Protocol composition works

### 6.4 Backward Compatibility Tests
**File**: `code/tests/test_backward_compatibility.py`

**Validate**:
- ✅ Existing API endpoints still work
- ✅ FastAPI server functions unchanged
- ✅ All original workflows execute
- ✅ No breaking changes in core protocols

---

## PHASE 7: Examples & Use Cases

### 7.1 Example Scripts
**File**: `examples/paper2agent_usage_examples.sh`

```bash
#!/bin/bash

# Example 1: Validate external AI response
echo "Example 1: Validate ChatGPT response"
echo "The Earth is 6000 years old" | python code/tools/run_five_laws_validator.py

# Example 2: Generate governed response
echo "Example 2: Generate governed response"
python code/tools/run_governed_response.py \
  --prompt "Explain climate change" \
  --protocols REP,VVP \
  --min-compliance 85

# Example 3: Multi-step workflow
echo "Example 3: Complete governed workflow"
python code/tools/run_cognitive_workflow.py \
  --input examples/complex_query.txt \
  --workflow reason_validate_govern \
  --output report.json

# Example 4: Protocol composition
echo "Example 4: Chain protocols manually"
python code/tools/run_rep_tool.py --input data.json | \
  python code/tools/run_vvp_tool.py | \
  python code/tools/run_five_laws_validator.py
```

### 7.2 Use Case Documentation
**File**: `examples/USE_CASES.md`

**Scenarios**:
1. **Paper2Agent Self-Governance**: Validate own responses before returning to user
2. **Multi-Agent Validation**: One agent validates another's output
3. **Governed Content Generation**: Generate responses with built-in Five Laws compliance
4. **Reasoning Chain Validation**: Check logical consistency and truth foundation
5. **Emotional Intelligence Analysis**: Evaluate empathy and sentiment in responses

---

## PHASE 8: Deployment & Finalization

### 8.1 Update Requirements
**File**: `code/requirements.txt`

Add if needed:
- No new dependencies expected (using existing protocols)
- Verify all tool wrappers work with current dependencies

### 8.2 Create Migration Artifacts

**Files to Create**:
- ✅ `MIGRATION_PLAN.md` - Future renaming strategy
- ✅ `code/mcp_server/FUTURE_NAMING.md` - Internal transition plan
- ✅ `.github/ISSUE_TEMPLATE/naming_migration.md` - Template for tracking migration

### 8.3 Update CONTRIBUTING.md

**Add section**:
```markdown
## Creating Tool Wrappers

When adding new protocols, create corresponding tool wrappers:

1. Create `code/tools/run_[protocol]_tool.py`
2. Include required docstring with MCP_TOOL_ENTRYPOINT marker
3. Support CLI args, stdin, and file input
4. Output valid JSON
5. Add to `tools_manifest.json`
6. Update `code/tools/README.md`
7. Add tests in `code/tests/test_tool_wrappers.py`

See existing wrappers for examples.
```

---

## Summary of Deliverables

### Documentation (8 files)
1. ✅ `MIGRATION_PLAN.md` - Naming strategy
2. ✅ `PAPER2AGENT_INTEGRATION.md` - Integration guide
3. ✅ `code/mcp_server/README.md` - Naming explanation
4. ✅ `code/mcp_server/FUTURE_NAMING.md` - Transition plan
5. ✅ `code/tools/README.md` - Tool catalog
6. ✅ `examples/USE_CASES.md` - Use case scenarios
7. ✅ Updated `README.md` (root) - Naming notice + tools section
8. ✅ Updated `code/README.md` - Tools usage

### Tool Wrappers (~20 files)
- ✅ `run_five_laws_validator.py` (PRIORITY 1)
- ✅ `run_governed_response.py` (PRIORITY 1)
- ✅ `run_cognitive_workflow.py` (PRIORITY 1)
- ✅ 6 core protocol wrappers (PRIORITY 2)
- ✅ 4 additional protocol wrappers (PRIORITY 3)
- ✅ 3 workflow wrappers (PRIORITY 3)
- ✅ Supporting library files

### Manifest & Discovery (2 files)
- ✅ `code/tools/tools_manifest.json`
- ✅ `examples/paper2agent_usage_examples.sh`

### Tests (4 files)
- ✅ `test_tool_wrappers.py`
- ✅ `test_five_laws_tools.py`
- ✅ `test_paper2agent_integration.py`
- ✅ `test_backward_compatibility.py`

---

## Success Criteria

1. ✅ Clear `mcp_server` naming explanation in all key documentation
2. ✅ Future migration plan documented (no immediate changes)
3. ✅ Paper2Agent can discover all tools via manifest
4. ✅ Five Laws validator works on ANY text input
5. ✅ All protocols have working CLI wrappers
6. ✅ Protocol composition enables governed workflows
7. ✅ All existing functionality remains intact (zero breaking changes)
8. ✅ Comprehensive testing validates all components
9. ✅ Documentation enables Paper2Agent integration

---

## Key Design Principles

✅ **No Breaking Changes**: All existing code, APIs, and workflows remain functional
✅ **Backward Compatible**: New tool wrappers are *adapters*, not replacements
✅ **Five Laws Compliant**: All implementations follow SIM-ONE's governance principles
✅ **Energy Efficient**: Minimal overhead, simple wrappers with direct protocol access
✅ **Deterministic**: Consistent, predictable tool behavior
✅ **Discoverable**: Clear naming, comprehensive documentation, standardized structure

---

## Estimated Implementation Scope

- **New Files**: ~35-40
- **Modified Files**: ~5-8
- **Test Files**: ~4
- **Lines of Code**: ~2,500-3,000 (mostly wrapper boilerplate + docs)
- **Risk Level**: LOW (additive only, no core changes)
- **Timeline**: 2-3 days for full implementation + testing

---

*Document Version: 1.0*
*Created: 2025-01-10*
*Status: Approved - Ready for Implementation*
