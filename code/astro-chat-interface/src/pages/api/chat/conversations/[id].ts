import type { APIRoute } from 'astro';
import { conversations } from '../../../../../server/conversations';

export const GET: APIRoute = async ({ params }) => {
  const id = params.id as string;
  const rec = await conversations.get(id);
  if (!rec) return new Response(JSON.stringify({ error: 'Not found' }), { status: 404 });
  return new Response(JSON.stringify(rec), { status: 200, headers: { 'Content-Type': 'application/json' } });
};

export const DELETE: APIRoute = async ({ params }) => {
  const id = params.id as string;
  const rec = await conversations.get(id);
  if (!rec) return new Response(JSON.stringify({ error: 'Not found' }), { status: 404 });
  await conversations.delete(id);
  return new Response(null, { status: 204 });
};
