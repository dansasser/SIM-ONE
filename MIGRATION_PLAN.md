# SIM-ONE mcp_server Migration Plan

**Status**: Planning Phase
**Created**: January 2025
**Target Timeline**: 6-12 months
**Priority**: Medium (No immediate action required)

---

## Executive Summary

The `mcp_server` directory name predates the now-standard "Model Context Protocol" (MCP) term in the AI industry. To prevent confusion and improve discoverability, this document outlines a gradual migration strategy to rename the directory to `agent_core` while maintaining complete backward compatibility and preserving all existing links, documentation, and integrations.

---

## Background

### Current Situation

The SIM-ONE Framework uses `mcp_server` to refer to:
- **Multi-Protocol Cognitive Platform**
- **Modular Cognitive Platform**
- The core orchestrator and agent system

### Industry Confusion

The term "MCP" now commonly refers to:
- **Model Context Protocol** - Anthropic's tool integration standard
- MCP servers - Tool providers for Claude Desktop and similar systems
- MCP registries - Collections of MCP-compliant tools

### Why This Matters

- **Developer confusion**: New users may expect MCP protocol compliance
- **SEO/AEO interference**: Search results may conflate SIM-ONE with MCP tools
- **Documentation clarity**: Need to explicitly clarify what "mcp_server" means
- **Future-proofing**: Prevent ongoing confusion as MCP adoption grows

---

## Migration Strategy

### Phase 1: Documentation & Clarification (CURRENT - Months 1-2)

**Status**: ✅ In Progress (Phase 22)

**Actions**:
- [x] Add prominent naming notices to all README files
- [x] Create `code/mcp_server/README.md` with detailed explanation
- [x] Create this `MIGRATION_PLAN.md` document
- [ ] Create `code/mcp_server/FUTURE_NAMING.md` for internal reference
- [ ] Update all new documentation to reference future naming
- [ ] Add naming clarification to API documentation

**Goal**: Ensure all users understand current naming context

---

### Phase 2: Preparation & Planning (Months 3-4)

**Actions**:
- [ ] Audit all internal and external references to `mcp_server`
- [ ] Identify all import statements across codebase
- [ ] Document all URL/link dependencies
- [ ] Create comprehensive import compatibility layer design
- [ ] Plan symlink strategy for backward compatibility
- [ ] Design redirect strategy for documentation links
- [ ] Identify potential breaking changes and mitigations

**Deliverables**:
- Complete reference audit spreadsheet
- Import compatibility layer specification
- Redirect mapping document
- Risk assessment report

---

### Phase 3: Infrastructure Preparation (Months 5-6)

**Actions**:
- [ ] Create `code/agent_core/` directory structure
- [ ] Implement import compatibility layer (allows both `mcp_server` and `agent_core` imports)
- [ ] Set up automated tests for import compatibility
- [ ] Create symbolic links or directory mirrors
- [ ] Implement deprecation warnings (configurable, off by default)
- [ ] Update CI/CD pipelines to test both import paths

**Deliverables**:
- Working `agent_core/` directory alongside `mcp_server/`
- Import compatibility module that supports both paths
- Test suite validating dual-import support
- CI/CD configuration for testing both namespaces

**Compatibility Layer Example**:
```python
# code/agent_core/__init__.py
"""
SIM-ONE Agent Core (formerly mcp_server)
Provides backward compatibility imports for mcp_server namespace
"""
import warnings
from pathlib import Path

# Re-export everything from agent_core as if it were mcp_server
from .protocols import *
from .orchestration_engine import *
from .cognitive_governance_engine import *

def _emit_deprecation_warning():
    """Optionally warn about legacy mcp_server imports"""
    if os.getenv("SIMONE_WARN_LEGACY_IMPORTS", "false").lower() == "true":
        warnings.warn(
            "Importing from 'mcp_server' is deprecated. "
            "Please update imports to 'agent_core'. "
            "See MIGRATION_PLAN.md for details.",
            DeprecationWarning,
            stacklevel=3
        )
```

---

### Phase 4: Gradual Code Migration (Months 7-9)

**Actions**:
- [ ] Begin migrating internal code to use `agent_core` imports
- [ ] Update new code to reference `agent_core` exclusively
- [ ] Maintain `mcp_server` as a compatibility alias
- [ ] Update example code and tutorials to use `agent_core`
- [ ] Update API responses to reference `agent_core` terminology
- [ ] Gradually update internal documentation

**Migration Priority**:
1. **New code**: All new modules use `agent_core`
2. **Core systems**: Orchestrator, governance engine
3. **Protocols**: Individual protocol modules
4. **Utilities**: Supporting modules and tools
5. **Tests**: Test suite references
6. **Documentation**: Internal docs and comments

**Principle**: No breaking changes - both imports work simultaneously

---

### Phase 5: External Communication (Months 10-11)

**Actions**:
- [ ] Announce migration plan to users/contributors
- [ ] Update main documentation to prefer `agent_core` references
- [ ] Create migration guide for external integrators
- [ ] Update tutorials and examples
- [ ] Publish blog post or changelog entry explaining migration
- [ ] Update GitHub repository description
- [ ] Notify any known external integrators

**Deliverables**:
- Migration announcement
- User-facing migration guide
- Updated "Getting Started" documentation
- External integrator notification template

---

### Phase 6: Legacy Support & Monitoring (Month 12+)

**Actions**:
- [ ] Monitor usage of `mcp_server` imports (telemetry if available)
- [ ] Maintain compatibility layer indefinitely (or minimum 12 months)
- [ ] Continue responding to both `mcp_server` and `agent_core` references
- [ ] Optionally enable deprecation warnings (configurable)
- [ ] Plan for eventual removal of compatibility layer (18-24 months post-migration)

**Long-term Strategy**:
- **Years 1-2**: Full compatibility, both imports work equally
- **Years 2-3**: Deprecation warnings enabled by default (configurable off)
- **Year 3+**: Consider removal of compatibility layer (with major version bump)

---

## Directory Renaming Options

### Option 1: `agent_core` (RECOMMENDED)

**Pros**:
- Clear, descriptive, and unambiguous
- Aligns with "core agent system" terminology
- No MCP confusion
- Professional and technical

**Cons**:
- Generic term

**Verdict**: ✅ Recommended

### Option 2: `cognitive_orchestrator`

**Pros**:
- Highly descriptive of function
- Unique to SIM-ONE
- Academic/professional tone

**Cons**:
- Longer path name
- May be too verbose

**Verdict**: ⚠️ Alternative option

### Option 3: `simone_core`

**Pros**:
- Framework-branded
- Clearly SIM-ONE specific

**Cons**:
- Less descriptive of function
- Redundant in SIM-ONE repository

**Verdict**: ⚠️ Fallback option

**Decision**: Proceed with `agent_core` unless stakeholder feedback suggests otherwise.

---

## URL & Link Preservation Strategy

### Documentation URLs

**Current**:
- `github.com/yourorg/SIM-ONE/tree/main/code/mcp_server`
- Documentation references to `/code/mcp_server/`

**Strategy**:
1. **GitHub**: Directory rename automatically preserves most GitHub links (redirects)
2. **Documentation**: Use relative links where possible (e.g., `../agent_core/`)
3. **Hardcoded URLs**: Create redirect/compatibility layer
4. **External links**: Cannot control, but GitHub handles redirects

**Action Items**:
- [ ] Audit all hardcoded URLs in documentation
- [ ] Replace absolute URLs with relative paths
- [ ] Test GitHub's automatic redirect behavior
- [ ] Create redirect mapping for any custom documentation hosting

---

## Import Compatibility Strategy

### Current Import Pattern
```python
from mcp_server.protocols.rep import REP
from mcp_server.orchestration_engine import OrchestrationEngine
```

### Future Import Pattern (Preferred)
```python
from agent_core.protocols.rep import REP
from agent_core.orchestration_engine import OrchestrationEngine
```

### Compatibility Layer Implementation

**Approach**: Create import aliasing at package level

```python
# code/mcp_server/__init__.py (becomes compatibility shim)
"""
LEGACY COMPATIBILITY LAYER
This module provides backward compatibility for code importing from 'mcp_server'.
New code should import from 'agent_core' instead.
See MIGRATION_PLAN.md for migration guide.
"""
import warnings
import os

# Re-export everything from agent_core
from agent_core import *
from agent_core.protocols import *
from agent_core.orchestration_engine import *
from agent_core.cognitive_governance_engine import *

# Optional deprecation warning
if os.getenv("SIMONE_WARN_LEGACY_IMPORTS", "false").lower() == "true":
    warnings.warn(
        "Importing from 'mcp_server' is deprecated. Update imports to 'agent_core'.",
        DeprecationWarning,
        stacklevel=2
    )
```

**Result**: Both import paths work identically, allowing gradual migration.

---

## SEO/AEO Preservation

### Current SEO Investment

The `mcp_server` directory has accumulated:
- Search engine indexing
- Documentation backlinks
- Tutorial references
- Blog post mentions
- Stack Overflow discussions (potentially)

### Preservation Strategy

1. **GitHub Redirects**: Leverage GitHub's automatic redirects for renamed files/directories
2. **Documentation Updates**: Update all controlled documentation gradually
3. **Redirect Notes**: Add "formerly mcp_server" references in meta descriptions
4. **Gradual Rollout**: Allows search engines to re-index gradually
5. **Canonical URLs**: Use canonical link tags to point to new paths

**Action Items**:
- [ ] Add "formerly mcp_server" to relevant page titles/descriptions
- [ ] Monitor search console for 404s post-migration
- [ ] Update sitemap to include both old and new paths initially
- [ ] Add schema markup indicating naming change

---

## Testing & Validation

### Pre-Migration Tests

- [ ] All imports from `mcp_server` work correctly (baseline)
- [ ] All tests pass with current naming
- [ ] All documentation links are valid
- [ ] All example code runs successfully

### During-Migration Tests

- [ ] Both `mcp_server` and `agent_core` imports work
- [ ] No import errors in compatibility layer
- [ ] All tests pass with dual imports
- [ ] Deprecation warnings trigger correctly (when enabled)
- [ ] No performance regression from compatibility layer

### Post-Migration Tests

- [ ] All `agent_core` imports work correctly
- [ ] Legacy `mcp_server` imports still work (compatibility)
- [ ] Documentation links resolve correctly
- [ ] Example code updated and functional
- [ ] External integrations notified and updated

---

## Rollback Plan

If migration encounters critical issues:

### Rollback Triggers
- Breaking changes in external integrations (>3 complaints)
- Performance regression >10%
- Critical bugs introduced by compatibility layer
- Community backlash or significant user confusion

### Rollback Process
1. Pause migration announcements
2. Revert documentation to prefer `mcp_server` terminology
3. Disable deprecation warnings
4. Investigate root cause
5. Revise migration plan
6. Re-communicate updated timeline

**Note**: Due to compatibility layer, rollback is low-risk and can happen at any time.

---

## Communication Timeline

### Internal Communication
- **Month 1**: Team briefing on migration plan
- **Month 3**: Quarterly update on preparation progress
- **Month 7**: Code migration kickoff announcement
- **Month 10**: Public migration announcement

### External Communication
- **Month 10**: Blog post announcing migration
- **Month 10**: GitHub discussion thread for feedback
- **Month 11**: Documentation updated to prefer `agent_core`
- **Month 12**: Migration completion announcement

---

## Success Criteria

Migration is considered successful when:

- [ ] ✅ `agent_core` directory exists and is primary
- [ ] ✅ All new code uses `agent_core` imports
- [ ] ✅ Both `mcp_server` and `agent_core` imports work (compatibility layer)
- [ ] ✅ All documentation references `agent_core` as primary (with compatibility notes)
- [ ] ✅ No breaking changes for existing users
- [ ] ✅ No performance regression
- [ ] ✅ All tests pass
- [ ] ✅ External integrators notified and updated
- [ ] ✅ SEO/AEO rankings maintained or improved
- [ ] ✅ Zero critical bugs from migration
- [ ] ✅ Positive or neutral community feedback

---

## Open Questions

- [ ] Should we maintain `mcp_server` compatibility indefinitely or plan eventual removal?
- [ ] What's the appropriate timeline for deprecation warnings (if ever)?
- [ ] Should we create a SIM-ONE package namespace (e.g., `simone.agent_core`)?
- [ ] Do we need a major version bump for this change?
- [ ] Should we coordinate with any major releases or milestones?

---

## Stakeholder Sign-Off

This migration plan requires approval from:

- [ ] Core maintainers
- [ ] Active contributors
- [ ] Known external integrators (if any)
- [ ] Project leadership

---

## Timeline Overview

| Phase | Months | Status | Key Deliverable |
|-------|--------|--------|----------------|
| Phase 1: Documentation | 1-2 | ✅ In Progress | Naming clarifications added |
| Phase 2: Planning | 3-4 | ⏳ Pending | Reference audit complete |
| Phase 3: Infrastructure | 5-6 | ⏳ Pending | Compatibility layer working |
| Phase 4: Code Migration | 7-9 | ⏳ Pending | Internal code migrated |
| Phase 5: External Comms | 10-11 | ⏳ Pending | Public announcement made |
| Phase 6: Long-term Support | 12+ | ⏳ Pending | Compatibility maintained |

**Total Timeline**: 12+ months
**Current Phase**: Phase 1 (Documentation & Clarification)

---

## References

- [Phase 22 Implementation Plan](docs/planning/PHASE_22_IMPLEMENTATION_PLAN.md)
- [Phase 22 TODO List](docs/planning/PHASE_22_TODO.md)
- [mcp_server Naming Explanation](code/mcp_server/README.md)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/) (industry MCP, for reference)

---

## Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-01-10 | 1.0 | Initial migration plan created | SIM-ONE Team |

---

*This is a living document and will be updated as the migration progresses.*
