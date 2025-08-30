import type { APIRoute } from 'astro';
import { conversations } from '../../../../../../server/conversations';

export const GET: APIRoute = async ({ params }) => {
  const id = params.id as string;
  if (!id) return new Response(JSON.stringify({ error: 'Missing conversation id' }), { status: 400 });
  const sessionId = await conversations.getSessionMapping(id);
  return new Response(JSON.stringify({ conversationId: id, sessionId: sessionId || null }), { status: 200, headers: { 'Content-Type': 'application/json' } });
};
