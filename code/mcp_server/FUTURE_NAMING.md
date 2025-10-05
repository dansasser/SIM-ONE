# Internal Naming Transition Guide

**Audience**: SIM-ONE Framework Contributors & Developers
**Status**: Planning Phase
**Effective Date**: TBD (See [MIGRATION_PLAN.md](../../MIGRATION_PLAN.md))

---

## Quick Reference

### Current Name
`mcp_server` - Multi-Protocol Cognitive Platform

### Future Name
`agent_core` - SIM-ONE Agent Core System

### Timeline
6-12 months gradual transition (See root MIGRATION_PLAN.md)

---

## For New Code Development

When writing **new code**, prefer the future naming conventions:

### In Code Comments
```python
# ✅ GOOD - Use future terminology
"""
The agent_core orchestrates all cognitive protocols...
"""

# ❌ AVOID - Don't reinforce legacy naming
"""
The mcp_server orchestrates all cognitive protocols...
"""
```

### In Docstrings
```python
# ✅ GOOD
def execute_protocol(self, protocol_name: str):
    """
    Execute a cognitive protocol through the agent_core orchestrator.

    The agent_core system coordinates protocol execution...
    """

# ❌ AVOID
def execute_protocol(self, protocol_name: str):
    """
    Execute a cognitive protocol through the mcp_server orchestrator.
    ...
    """
```

### In Variable Names
```python
# ✅ GOOD - Use descriptive names that avoid "mcp"
orchestrator = OrchestrationEngine()
cognitive_platform = CognitivePlatform()
agent_system = AgentSystem()

# ❌ AVOID - Don't create new "mcp" variable names
mcp_orchestrator = OrchestrationEngine()
mcp_platform = CognitivePlatform()
```

---

## For Existing Code

**DO NOT** mass-rename existing code yet. The migration will happen gradually.

### Safe to Update Now
- ✅ Comments describing functionality
- ✅ Docstrings (non-breaking)
- ✅ Internal documentation
- ✅ README files (with compatibility notes)
- ✅ Variable names in new functions

### DO NOT Update Yet
- ❌ Module/package names (`mcp_server` directory)
- ❌ Import statements
- ❌ API endpoints
- ❌ Configuration variable names
- ❌ Database table/column names
- ❌ External-facing interfaces

---

## Terminology Guide

When describing the system, use this terminology:

| Context | Preferred Term | Acceptable Alternatives | Avoid |
|---------|---------------|------------------------|-------|
| Architecture docs | "agent core" | "cognitive orchestrator", "cognitive platform" | "mcp server" |
| Code comments | "orchestrator", "cognitive platform" | "agent system" | "mcp" |
| User-facing docs | "SIM-ONE agent system" | "cognitive framework" | "mcp server" |
| Technical specs | "agent_core" | "orchestration layer" | "mcp" |
| Blog posts | "SIM-ONE's agent core" | "cognitive orchestrator" | "mcp server" |

---

## Import Path Evolution

### Phase 1 (Current)
```python
from mcp_server.protocols.rep import REP
from mcp_server.orchestration_engine import OrchestrationEngine
```

### Phase 2 (Compatibility Layer)
Both work simultaneously:
```python
# Legacy (still works)
from mcp_server.protocols.rep import REP

# New (preferred)
from agent_core.protocols.rep import REP
```

### Phase 3 (Future)
```python
# Primary import path
from agent_core.protocols.rep import REP
from agent_core.orchestration_engine import OrchestrationEngine

# Legacy compatibility maintained via shim
```

---

## Naming Conventions for New Modules

When creating new modules/protocols:

### Module Names
```python
# ✅ GOOD
agent_core/protocols/new_protocol/
agent_core/utils/cognitive_helpers.py

# ⚠️ TRANSITIONAL (acceptable during migration)
mcp_server/protocols/new_protocol/  # will be migrated

# ❌ AVOID
mcp_server/new_mcp_feature/  # don't create new "mcp" names
```

### Class Names
```python
# ✅ GOOD
class CognitiveOrchestrator:
    pass

class AgentCoreSystem:
    pass

class ProtocolCoordinator:
    pass

# ❌ AVOID
class MCPOrchestrator:  # don't introduce new MCP naming
    pass
```

### Function Names
```python
# ✅ GOOD
def initialize_cognitive_platform():
    pass

def orchestrate_protocols():
    pass

# ❌ AVOID
def initialize_mcp_server():  # avoid reinforcing legacy naming
    pass
```

---

## Configuration Variables

### Current Environment Variables
```bash
# These remain unchanged for backward compatibility
MCP_API_KEY=...
MCP_SERVER_PORT=...
```

### Future Environment Variables (when migration occurs)
```bash
# New names (to be introduced during migration)
SIMONE_API_KEY=...  # or AGENT_CORE_API_KEY
SIMONE_SERVER_PORT=...

# Legacy names will be supported via compatibility layer
```

**Note**: Do NOT rename environment variables yet. This will happen in migration Phase 3.

---

## API Endpoints

### Current Endpoints
```
POST /execute
POST /protocols/list
GET /health
```

### Future Considerations
Endpoints remain unchanged (no "mcp" in URLs currently).
If new endpoints are added, avoid "mcp" terminology:

```
# ✅ GOOD
POST /agent/execute
GET /cognitive/status

# ❌ AVOID
POST /mcp/execute
```

---

## Documentation Updates

### README Files
When updating README files:

```markdown
# ✅ GOOD
This module provides the core agent orchestration system...

⚠️ Note: This directory is currently named `mcp_server` for historical reasons.
See [MIGRATION_PLAN.md](../../MIGRATION_PLAN.md) for renaming timeline.

# ❌ AVOID
This is the MCP server for SIM-ONE...
```

### Inline Documentation
```python
# ✅ GOOD
"""
Agent Core Orchestration System

This module coordinates cognitive protocols and manages the execution
of multi-step workflows through the SIM-ONE agent core.

Historical Note: Package currently located in `mcp_server/` directory.
"""

# ❌ AVOID
"""
MCP Server Orchestration

This module manages the MCP server operations...
"""
```

---

## Testing Strategy

### Test File Naming
```python
# ✅ GOOD - Function-focused names
tests/test_orchestration_engine.py
tests/test_protocol_coordination.py
tests/test_cognitive_governance.py

# ⚠️ ACCEPTABLE - Current structure
tests/test_mcp_server.py  # existing files, don't rename yet

# ❌ AVOID - New MCP naming
tests/test_mcp_new_feature.py  # use descriptive names instead
```

### Test Imports
For new tests, prefer future import structure:
```python
# ✅ GOOD (when compatibility layer is ready)
from agent_core.protocols.rep import REP

# ⚠️ ACCEPTABLE (current)
from mcp_server.protocols.rep import REP

# Add both to test compatibility layer when ready
```

---

## Git Commit Messages

### Commit Message Style
```bash
# ✅ GOOD
git commit -m "feat(orchestrator): Add adaptive protocol scheduling"
git commit -m "docs(agent-core): Update terminology to use 'agent core'"
git commit -m "refactor(protocols): Improve REP reasoning chain"

# ❌ AVOID
git commit -m "fix(mcp): Fix mcp server bug"  # use descriptive component names
```

---

## Code Review Checklist

When reviewing pull requests, check:

- [ ] New code comments use future terminology ("agent core", "cognitive platform")
- [ ] Docstrings avoid reinforcing "mcp" naming
- [ ] Variable names in new code are descriptive, not "mcp_*"
- [ ] No new configuration variables with "MCP_" prefix
- [ ] Documentation references future naming (with compatibility notes)
- [ ] No breaking changes to existing imports/APIs

---

## Migration Coordination

### Current Phase: Documentation & Clarification

**What to do now**:
- ✅ Update comments in new code to use "agent core" terminology
- ✅ Write new documentation with future naming
- ✅ Add compatibility notes where needed
- ✅ Use descriptive variable names (not "mcp_*")

**What NOT to do yet**:
- ❌ Don't rename the `mcp_server` directory
- ❌ Don't change import paths
- ❌ Don't modify existing configuration variables
- ❌ Don't break backward compatibility

### Future Phases

See [MIGRATION_PLAN.md](../../MIGRATION_PLAN.md) for:
- Phase 2: Preparation & Planning (Months 3-4)
- Phase 3: Infrastructure Preparation (Months 5-6)
- Phase 4: Gradual Code Migration (Months 7-9)
- Phase 5: External Communication (Months 10-11)
- Phase 6: Long-term Support (Month 12+)

---

## Questions & Clarifications

### Q: Can I create new files in `mcp_server/`?
**A**: Yes, but use descriptive names and future terminology in documentation.

### Q: Should I update existing docstrings?
**A**: Yes, you can update docstrings to use "agent core" terminology. This is non-breaking.

### Q: Can I change import statements?
**A**: Not yet. Wait until Phase 3 when the compatibility layer is ready.

### Q: What if external code imports from `mcp_server`?
**A**: That will continue to work indefinitely via the compatibility layer.

### Q: Should new protocols use future naming?
**A**: Yes, in documentation and comments. Directory structure stays under `mcp_server/` until migration.

---

## Communication

### Internal Team
- Slack channel: #sim-one-migration (if applicable)
- Monthly updates in team meetings
- Migration status in sprint planning

### External Contributors
- See root [MIGRATION_PLAN.md](../../MIGRATION_PLAN.md)
- Public announcement planned for Month 10
- Compatibility maintained throughout

---

## Quick Tips

**Do This** ✅:
```python
"""
The agent core orchestrates cognitive protocols through a modular
architecture. Each protocol is loaded dynamically by the orchestration
engine.

Note: Implementation currently in mcp_server/ directory pending migration.
"""
```

**Not This** ❌:
```python
"""
The MCP server manages protocol execution. The mcp orchestrator
loads protocols from the mcp_server directory.
"""
```

---

## References

- **Public Migration Plan**: [../../MIGRATION_PLAN.md](../../MIGRATION_PLAN.md)
- **mcp_server Naming Explanation**: [README.md](README.md)
- **Phase 22 Implementation**: [../../docs/planning/PHASE_22_IMPLEMENTATION_PLAN.md](../../docs/planning/PHASE_22_IMPLEMENTATION_PLAN.md)

---

## Change Log

| Date | Changes | Author |
|------|---------|--------|
| 2025-01-10 | Initial internal transition guide created | SIM-ONE Team |

---

*This document is for internal SIM-ONE contributors. For user-facing migration info, see [MIGRATION_PLAN.md](../../MIGRATION_PLAN.md).*
