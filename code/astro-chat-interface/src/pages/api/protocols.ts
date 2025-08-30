import type { APIRoute } from 'astro';
import { mcp } from '../../server/mcp';

export const GET: APIRoute = async () => {
  try {
    const data = await mcp.getProtocols();
    return new Response(JSON.stringify(data), { status: 200, headers: { 'Content-Type': 'application/json' } });
  } catch (err: any) {
    return new Response(JSON.stringify({ error: err?.message || 'Failed to fetch protocols' }), { status: err?.status || 500 });
  }
};
