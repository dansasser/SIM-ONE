import type { APIRoute } from 'astro';
import { chat } from '../../../server/mcp';
import { conversations } from '../../../server/conversations';

export const POST: APIRoute = async ({ request }) => {
  try {
    const { conversationId, text, style, priority } = await request.json();
    if (!conversationId || !text) {
      return new Response(JSON.stringify({ error: 'conversationId and text are required' }), { status: 400 });
    }
    // Ensure conversation exists
    if (!(await conversations.get(conversationId))) {
      await conversations.create('New Chat');
    }
    const { assistant, sessionId } = await chat.sendMessage({ conversationId, text, style, priority });
    if (sessionId) {
      conversations.setSessionMapping(conversationId, sessionId);
    }
    // Count both user and assistant messages
    await conversations.bumpMessageCount(conversationId, 2);
    return new Response(JSON.stringify({ assistant, sessionId }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  } catch (err: any) {
    return new Response(JSON.stringify({ error: err?.message || 'Chat send failed' }), { status: err?.status || 500 });
  }
};
