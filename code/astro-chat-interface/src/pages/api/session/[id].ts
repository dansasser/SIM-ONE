import type { APIRoute } from 'astro';
import { mcp } from '../../../server/mcp';

export const GET: APIRoute = async ({ params }) => {
  const id = params.id as string;
  if (!id) return new Response(JSON.stringify({ error: 'Missing session id' }), { status: 400 });
  try {
    const data = await mcp.getSession(id);
    return new Response(JSON.stringify(data), { status: 200, headers: { 'Content-Type': 'application/json' } });
  } catch (err: any) {
    return new Response(JSON.stringify({ error: err?.message || 'Failed to fetch session' }), { status: err?.status || 500 });
  }
};
