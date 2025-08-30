# SIM-ONE Admin Panel – Architecture and Implementation Guide

This document outlines what we need and how to build a separate Admin Panel for operating and observing the SIM-ONE system (MCP server + Chat UI + persistence).

## Summary
- Purpose: Operational visibility, configuration, and troubleshooting for MCP + Chat.
- Scope (MVP): Conversations + session mappings, MCP health/metrics, protocol/template discovery, execution tracing, persistence management (SQLite/Redis), security + audit logs.
- Delivery: Separate web app (admin-only), backed by a dedicated Admin API (SSR/Server-only).

## Goals
- Deterministic, least-privilege control plane (Law 2 & Law 5).
- Read-first dashboards; write operations behind explicit role-gated actions and confirmations.
- No secrets in browser; all privileged calls from server only.

## Tech Stack
- Frontend: React + Vite (or Next.js/Remix), Typescript, Tailwind.
- Admin API: Node (Express/Fastify) or Python (FastAPI) – pick one; consistent with current infra.
- AuthN/Z: OAuth2/OIDC (IdP) or API key + TOTP for internal use; RBAC enforced server-side.
- Persistence: Redis (primary in prod), SQLite fallback for local dev; optional Postgres later.
- Telemetry: OpenTelemetry (traces, metrics), Prometheus scrape, Grafana dashboards.

## Integrations
- MCP Server (FastAPI):
  - Health: `/health`, `/health/detailed`
  - Discovery: `/protocols`, `/templates`
  - Orchestration: `/execute` (read-only testing in admin), rate-limit aware
  - Metrics (admin): `/metrics`, DB analytics endpoints
- Chat SSR API (this repo):
  - Conversations: `/api/chat/conversations/*`
  - Session mapping: `/api/chat/conversations/:id/session`, `/api/chat/conversations/session-mappings`
  - Messages: `/api/chat/messages/:conversationId`
  - Send (test): `/api/chat/send` (optional in admin)

## Authentication & Authorization
- Recommended: OIDC (Okta/Auth0/AzureAD) with server-side session.
- Alternative: Admin API key (header) + optional IP allowlist.
- Roles:
  - read-only: view dashboards & logs
  - operator: read + limited write (restart jobs, re-run execute in sandbox)
  - admin: full access (migrations, backups)
- CSRF: for any form POST; JWT only for server-to-server, not browser storage.

## MVP Features
- Conversations
  - List/search/filter conversations
  - View session mapping (conversationId → sessionId)
  - View reconstructed messages (via SSR API)
  - Delete conversation (server-side)
- Session Mappings
  - Table view of mappings; filter missing mappings
  - Copy sessionId; deep link to MCP session endpoint
- MCP Health & Info
  - `/health`, `/health/detailed` status cards
  - Protocols/templates inventory
- Metrics
  - High-level counters (requests, rate-limit hits)
  - DB analytics (if enabled on MCP)
- Persistence Controls
  - Backend switch indicator (SQLite/Redis)
  - Migration action (SQLite → Redis) with confirmation + admin token
- Audit & Security
  - Log privileged actions with timestamp, user, reason

## Optional Phase 2
- Workflow Explorer: show recent `/execute` requests, latency, error distribution
- Rate Limit Insights: current limits, offender IPs, exemptions
- Key Management Helper: rotate `VALID_API_KEYS` (if formalized)
- Backup/Restore UI (wrap MCP DB backup endpoints)

## Admin API Design (Server-Only)
- Base URL: `ADMIN_API_BASE_URL`
- Security: `ADMIN_API_TOKEN` (header: `x-admin-token`) or OIDC session cookie
- Endpoints (initial):
  - `GET /admin/mcp/health` → proxies MCP `/health(/detailed)`
  - `GET /admin/mcp/protocols` → proxies MCP `/protocols`
  - `GET /admin/mcp/templates` → proxies MCP `/templates`
  - `GET /admin/chat/conversations` → proxies SSR `/api/chat/conversations`
  - `GET /admin/chat/conversations/:id` → proxies SSR
  - `GET /admin/chat/conversations/:id/session` → proxies SSR
  - `GET /admin/chat/conversations/session-mappings` → proxies SSR
  - `POST /admin/chat/admin/migrate-sqlite-to-redis` → proxies SSR migration endpoint
- Notes:
  - All proxies add admin auth and rate limit.
  - Log inbound/outbound and redact secrets.

## UI Pages
- Dashboard
  - Overall health, MCP status, backend type (SQLite/Redis)
- Conversations
  - Table (id, title, updatedAt, messageCount)
  - Detail page with messages + mapping card
- Mappings
  - List of conversationId/sessionId; filter missing
- MCP
  - Protocols/templates listing
- Metrics (Phase 2+)
  - Status cards, charts (histories via Prometheus or MCP endpoints)
- Settings
  - Admin token status, connected backends, environment overview

## Environment & Config
- Frontend (Admin):
  - `VITE_ADMIN_API_BASE_URL`
- Admin API:
  - `ADMIN_API_TOKEN` (if key-based)
  - `MCP_SERVER_URL`
  - `CHAT_SSR_URL` (the Astro app base URL)
  - `LOG_LEVEL`, `PORT`
- Chat SSR (already in this repo):
  - `PERSISTENCE_BACKEND=sqlite|redis`
  - `REDIS_URL=redis://...`
  - `SIMONE_API_URL` + `SIMONE_API_KEY`

## Data Model (Admin-Side, read-mostly)
- Conversation:
  - id, title, timestamps, messageCount, flags, tags, metadata
- SessionMapping:
  - conversationId, sessionId
- MCP Health:
  - status, services map (database, redis)

## Security Checklist
- All admin routes behind auth (no public endpoints)
- Server-only secrets; never leak keys to browser
- Rate limit & audit log privileged actions
- CSRF protection for mutations
- CORS: allow only admin origin
- Headers: security headers (CSP, HSTS, Referrer-Policy, etc.)

## Observability
- Request logging with correlation IDs
- Traces for proxy calls (MCP, SSR)
- Metrics: p95 latency, error rate, rate-limit counters
- Dashboard: Grafana with Prometheus scraping admin API and MCP

## Testing
- Unit: adapters, auth guards, error mapping
- Integration: mock MCP/SSR endpoints; verify auth + proxy correctness
- E2E: admin user flows (Cypress/Playwright)

## Rollout Plan
- Stand up Admin API (dev): point at local MCP + SSR
- Build Admin UI (dev): connect to Admin API
- Secure staging with OIDC and real Redis
- Validate migration tool on staging; test rollback
- Production: deploy behind VPN or restricted SSO group

## Local Dev
- Pre-req:
  - MCP running (`SIMONE_API_URL`), SSR running (`PUBLIC_SDK_MODE=server`), Redis optional
- Start Admin API (dev): `pnpm dev` (or `uvicorn` if FastAPI)
- Start Admin UI: `pnpm dev` with `VITE_ADMIN_API_BASE_URL=http://localhost:<admin-api-port>`

## Future Enhancements
- Role-scoped UI elements
- Compare conversation vs. session integrity checks
- Export/import conversation data
- Key rotation workflows with approvals

## Action Items (MVP)
- Define Admin API (Node or FastAPI) with auth guard and proxy routes
- Build UI: Dashboard, Conversations (list/detail), Mappings, MCP Discovery
- Wire SSR endpoints already present in this repo
- Add audit log table (file or DB)
- Document deployment and operations

