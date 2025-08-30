import type { APIRoute } from 'astro';
import { mcp } from '../../server/mcp';

export const POST: APIRoute = async ({ request }) => {
  try {
    const body = await request.json();
    const data = await mcp.execute(body);
    return new Response(JSON.stringify(data), { status: 200, headers: { 'Content-Type': 'application/json' } });
  } catch (err: any) {
    return new Response(JSON.stringify({ error: err?.message || 'Execute failed' }), { status: err?.status || 500 });
  }
};
