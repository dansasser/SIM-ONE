import { McpClient, ChatClient, MemorySessionStore } from '@simone-ai/mcp-sdk';

// Read server-side environment
const baseUrl = (import.meta as any).env?.SIMONE_API_URL || process.env.SIMONE_API_URL || 'http://localhost:8000';
const apiKey = process.env.SIMONE_API_KEY || process.env.MCP_API_KEY;

export const mcp = new McpClient({ baseUrl, apiKey });
export const chat = new ChatClient(mcp, { sessionStore: new MemorySessionStore() });

