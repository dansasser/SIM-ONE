import type { APIRoute } from 'astro';
import { chat } from '../../../../server/mcp';

export const GET: APIRoute = async ({ params }) => {
  const conversationId = params.conversationId as string;
  if (!conversationId) return new Response(JSON.stringify({ error: 'Missing conversationId' }), { status: 400 });
  try {
    const messages = await chat.getMessages({ conversationId });
    return new Response(JSON.stringify({ messages }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  } catch (err: any) {
    return new Response(JSON.stringify({ error: err?.message || 'Failed to load messages' }), { status: err?.status || 500 });
  }
};

