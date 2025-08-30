import type { APIRoute } from 'astro';
import { mcp } from '../../server/mcp';

export const GET: APIRoute = async () => {
  try {
    const data = await mcp.health();
    return new Response(JSON.stringify(data), { status: 200, headers: { 'Content-Type': 'application/json' } });
  } catch (err: any) {
    return new Response(JSON.stringify({ status: 'error', message: err?.message || 'Health check failed' }), { status: err?.status || 500 });
  }
};
