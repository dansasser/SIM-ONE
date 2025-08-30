import type { APIRoute } from 'astro';
import { conversations } from '../../../../server/conversations';

export const GET: APIRoute = async () => {
  const list = await conversations.list();
  return new Response(JSON.stringify({ conversations: list }), { status: 200, headers: { 'Content-Type': 'application/json' } });
};

export const POST: APIRoute = async ({ request }) => {
  const { title } = await request.json().catch(() => ({}));
  const rec = await conversations.create(title);
  return new Response(JSON.stringify(rec), { status: 201, headers: { 'Content-Type': 'application/json' } });
};
