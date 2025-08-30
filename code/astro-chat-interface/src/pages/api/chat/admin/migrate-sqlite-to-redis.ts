import type { APIRoute } from 'astro';
import { repo as sqliteRepo } from '../../../../server/persistence/sqlite';

export const POST: APIRoute = async ({ request }) => {
  const ADMIN_TOKEN = process.env.ADMIN_API_TOKEN;
  if (ADMIN_TOKEN) {
    const token = request.headers.get('x-admin-token');
    if (token !== ADMIN_TOKEN) {
      return new Response(JSON.stringify({ error: 'Unauthorized' }), { status: 401 });
    }
  }

  try {
    const mod = await import('../../../../server/persistence/redis');
    const redisRepo = mod.repo;

    const convRows = sqliteRepo.list();
    let migrated = 0;
    for (const row of convRows) {
      const raw = {
        id: row.id,
        title: row.title,
        userId: row.userId ?? null,
        createdAt: row.createdAt,
        updatedAt: row.updatedAt,
        lastMessageAt: row.lastMessageAt,
        messageCount: Number(row.messageCount || 0),
        isArchived: Number(row.isArchived || 0),
        isPinned: Number(row.isPinned || 0),
        tags: row.tags ?? null,
        metadata: row.metadata ?? null
      } as any;
      await redisRepo.upsertRow(raw);
      const sessionId = sqliteRepo.getSession(row.id);
      if (sessionId) await redisRepo.setSession(row.id, sessionId);
      migrated++;
    }
    return new Response(JSON.stringify({ ok: true, migrated }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  } catch (err: any) {
    return new Response(JSON.stringify({ error: err?.message || 'Migration failed' }), { status: 500 });
  }
};

