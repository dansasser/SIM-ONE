# @simone-ai/mcp-sdk

TypeScript SDK for the SIM-ONE MCP Server (FastAPI). Provides a low-level client for MCP endpoints and an optional high-level chat helper for conversation/session mapping.

## Features

- Typed client for `/execute`, `/protocols`, `/templates`, `/session/{id}`, `/health`
- Optional `ChatClient` that maps conversation IDs to MCP session IDs
- Isomorphic (Node/SSR/Browser via injected fetch)
- Deterministic error types (AuthError, RateLimitError, TimeoutError, etc.)

## Installation

```bash
npm install @simone-ai/mcp-sdk
# or
pnpm add @simone-ai/mcp-sdk
```

## Quick Start (Astro SSR)

For security, keep your MCP API key server-side. In Astro SSR, use the SDK in server routes or server-side code only.

1) Configure environment (server-only variables):

```bash
# .env
SIMONE_API_URL=http://localhost:8000
SIMONE_API_KEY=your-api-key
```

2) Create a server module to initialize the client (e.g., `src/server/mcp.ts`):

```ts
import { McpClient, ChatClient, MemorySessionStore } from '@simone-ai/mcp-sdk';

const baseUrl = import.meta.env.SIMONE_API_URL;
const apiKey = process.env.SIMONE_API_KEY; // ensure not exposed to client

export const mcp = new McpClient({ baseUrl, apiKey });
export const chat = new ChatClient(mcp, { sessionStore: new MemorySessionStore() });
```

3) Use in an Astro API route (SSR):

```ts
// src/pages/api/send-message.ts
import type { APIRoute } from 'astro';
import { mcp, chat } from '../../server/mcp';

export const POST: APIRoute = async ({ request }) => {
  try {
    const { conversationId, text, style } = await request.json();
    // Send via high-level chat helper (maps conversation to session)
    const { assistant, sessionId } = await chat.sendMessage({ conversationId, text, style });
    return new Response(JSON.stringify({ assistant, sessionId }), { status: 200 });
  } catch (err: any) {
    return new Response(JSON.stringify({ error: err?.message || 'Unexpected error' }), { status: 500 });
  }
};
```

4) Or call the low-level client directly:

```ts
// src/pages/api/execute.ts
import type { APIRoute } from 'astro';
import { mcp } from '../../server/mcp';

export const POST: APIRoute = async ({ request }) => {
  const body = await request.json();
  // body: WorkflowRequest (template_name | protocol_names + initial_data, optional session_id)
  const result = await mcp.execute(body);
  return new Response(JSON.stringify(result), { status: 200 });
};
```

## API

### McpClient

```ts
new McpClient({ baseUrl, apiKey?, headers?, timeoutMs?, fetchImpl? })

client.execute(request: WorkflowRequest): Promise<WorkflowResponse>
client.getProtocols(): Promise<Record<string, any>>
client.getTemplates(): Promise<Record<string, any>>
client.getSession(sessionId: string): Promise<{ session_id: string; history: any[] }>
client.health(): Promise<{ status: string }>
```

- `baseUrl`: points to your MCP server (e.g., `http://localhost:8000`) or a proxy (`/api`).
- `apiKey`: only provide in SSR/server contexts; never expose in the browser.
- `fetchImpl`: optional custom fetch (Node <18 or testing).

### ChatClient (optional helper)

```ts
new ChatClient(client: McpClient, { sessionStore?, templateMapper? })

chat.createConversation(title?): Promise<Conversation>
chat.sendMessage({ conversationId, text, style?, priority? }): Promise<{ assistant: Message; sessionId: string }>
chat.getMessages({ conversationId }): Promise<Message[]>
```

- `sessionStore`: pluggable interface for storing conversationId ↔ sessionId mappings. Defaults to in-memory store. For persistence, implement your own (DB, Redis, cookies) and pass it in.
- `templateMapper`: lets you customize how styles map to MCP templates and initial_data.

Default mapping:
- `universal_chat` → `template_name: 'analyze_only'` with `initial_data: { text }`
- other styles → `template_name: 'writing_team'` with `initial_data: { topic: text }`

## Error Handling

Errors are thrown as typed subclasses of `HttpError`:
- `AuthError` (401/403)
- `NotFoundError` (404)
- `RateLimitError` (429) with optional `retryAfter`
- `TimeoutError` (408)
- `ServerError` (>=500)

Wrap calls in try/catch in your server code and return appropriate responses.

## Security Notes (Astro SSR)

- Never expose `SIMONE_API_KEY` to the client. Use it only in server-side code (API routes, server modules).
- If you need browser access, put a server proxy in front of MCP that injects the key.
- Ensure your backend CORS `ALLOWED_ORIGINS` includes your Astro origin.

## License

MIT

