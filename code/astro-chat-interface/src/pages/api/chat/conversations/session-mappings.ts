import type { APIRoute } from 'astro';
import { conversations } from '../../../../server/conversations';

export const GET: APIRoute = async ({ request }) => {
  try {
    const url = new URL(request.url);
    const includeNull = url.searchParams.get('includeNull') === 'true';
    const list = await conversations.list();
    const mappings: Array<{ conversationId: string; sessionId: string | null }> = [];
    for (const c of list) {
      const sessionId = await conversations.getSessionMapping(c.id);
      if (includeNull || sessionId) {
        mappings.push({ conversationId: c.id, sessionId: sessionId || null });
      }
    }
    return new Response(JSON.stringify({ mappings, count: mappings.length }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  } catch (err: any) {
    return new Response(JSON.stringify({ error: err?.message || 'Failed to list mappings' }), { status: 500 });
  }
};

