# SIM-ONE mcp_server - Naming Explanation

## ⚠️ Important Notice About This Directory Name

The `mcp_server` directory name in this repository **predates** the now-standard industry term "Model Context Protocol" (MCP) that has since emerged in the AI tooling ecosystem.

### What `mcp_server` Means in SIM-ONE

In the SIM-ONE Framework, **"mcp_server"** historically refers to:

- **Multi-Protocol Cognitive Platform** - The modular, protocol-based cognitive architecture
- **Modular Cognitive Platform** - The core orchestrator for agent systems
- **Cognitive Orchestrator** - The central system coordinating all cognitive protocols

### What `mcp_server` Does NOT Mean

This directory is **NOT**:
- ❌ A Model Context Protocol (MCP) server in the Anthropic/industry sense
- ❌ An MCP tool registry or tool provider
- ❌ Related to the Claude Desktop MCP integration system
- ❌ A server implementing the MCP specification

### Why the Name Hasn't Changed

The `mcp_server` directory name has been preserved for:

1. **URL/Link Preservation**: Existing documentation links and references remain valid
2. **SEO/AEO Continuity**: Search engine optimization and AI Engine Optimization continuity
3. **Backward Compatibility**: Existing integrations, imports, and code references continue to work
4. **Historical Context**: The naming reflects the project's development timeline

### Future Migration Plan

A comprehensive migration plan is documented in the root-level [MIGRATION_PLAN.md](../../MIGRATION_PLAN.md) file, which outlines:

- Planned transition to `agent_core/` or `cognitive_orchestrator/`
- Symlink strategy for backward compatibility
- Timeline: 6-12 month gradual transition
- URL preservation and redirect strategy
- Import compatibility layer

### For New Development

When referencing this system in **new code or documentation**, consider using:

- `agent_core` - Preferred future name
- `cognitive_orchestrator` - Descriptive alternative
- `SIM-ONE orchestrator` - Framework-specific reference
- `cognitive platform` - Generic descriptive term

### For Existing Code

**No changes required** for existing code. All current imports, references, and integrations will continue to work throughout the migration period and beyond (via compatibility layers).

---

## Directory Contents

This directory contains the SIM-ONE Framework's core implementation:

```
mcp_server/
├── protocols/              # 18+ Cognitive Governance Protocols
│   ├── governance/         # Five Laws validators and governance systems
│   ├── monitoring/         # Real-time system monitoring
│   ├── compliance/         # Compliance reporting and assessment
│   ├── ccp/                # Cognitive Control Protocol
│   ├── esl/                # Emotional State Layer
│   ├── rep/                # Reasoning & Explanation Protocol
│   ├── vvp/                # Validation & Verification Protocol
│   └── [other protocols]/  # Additional cognitive protocols
├── cognitive_governance_engine/  # Central governance system
├── neural_engine/          # Efficient neural processing
├── orchestration_engine/   # Protocol coordination
├── protocol_manager/       # Dynamic protocol loading
├── memory_manager/         # Persistent cognitive memory
├── database/               # Database integrations
└── main.py                 # FastAPI server entry point
```

## What This System Does

The SIM-ONE cognitive platform (mcp_server) provides:

1. **Governed Cognition**: Implements the Five Laws of Cognitive Governance
2. **Protocol Orchestration**: Coordinates multiple specialized cognitive protocols
3. **Metacognitive Control**: Self-monitoring, adaptation, and quality assurance
4. **Energy Efficiency**: Resource-aware, optimized cognitive processing
5. **Deterministic Reliability**: Consistent, predictable AI behavior

## Getting Started

See the main [code/README.md](../README.md) for:
- Quick start guide
- Installation instructions
- Configuration options
- API documentation

## Questions?

If you're confused about whether this is related to the Model Context Protocol (MCP):

**Answer**: No, this is SIM-ONE's cognitive orchestration platform, unrelated to the modern MCP standard.

For questions about:
- **SIM-ONE Framework**: See root [README.md](../../README.md)
- **Migration Timeline**: See [MIGRATION_PLAN.md](../../MIGRATION_PLAN.md)
- **Technical Implementation**: See [code/README.md](../README.md)

---

*This naming clarification was added in Phase 22 (January 2025) to prevent confusion with the industry-standard Model Context Protocol.*
