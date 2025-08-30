import Redis from 'ioredis';

const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';
const redis = new Redis(REDIS_URL);

const H_CONVERSATIONS = 'conversations';
const Z_UPDATED = 'conversations:updatedAt';
const H_SESSIONS = 'session_mappings';

type ConvJSON = {
  id: string;
  title: string;
  userId?: string | null;
  createdAt: string;
  updatedAt: string;
  lastMessageAt: string;
  messageCount: number;
  isArchived: number;
  isPinned: number;
  tags: string | null;
  metadata: string | null;
};

function nowIso() { return new Date().toISOString(); }
function nowMs() { return Date.now(); }

export const repo = {
  async list() {
    const ids = await redis.zrevrange(Z_UPDATED, 0, -1);
    if (ids.length === 0) return [] as ConvJSON[];
    const res = await redis.hmget(H_CONVERSATIONS, ...ids);
    const rows: ConvJSON[] = [];
    res.forEach((s) => { if (s) rows.push(JSON.parse(s)); });
    return rows;
  },
  async get(id: string) {
    const s = await redis.hget(H_CONVERSATIONS, id);
    return s ? (JSON.parse(s) as ConvJSON) : undefined;
  },
  async create(input: { id: string; title: string; userId?: string }) {
    const row: ConvJSON = {
      id: input.id,
      title: input.title,
      userId: input.userId ?? null,
      createdAt: nowIso(),
      updatedAt: nowIso(),
      lastMessageAt: nowIso(),
      messageCount: 0,
      isArchived: 0,
      isPinned: 0,
      tags: null,
      metadata: null
    };
    await redis.hset(H_CONVERSATIONS, row.id, JSON.stringify(row));
    await redis.zadd(Z_UPDATED, nowMs(), row.id);
    return row;
  },
  async del(id: string) {
    await redis.hdel(H_CONVERSATIONS, id);
    await redis.zrem(Z_UPDATED, id);
    await redis.hdel(H_SESSIONS, id);
  },
  async bump(id: string, inc = 1) {
    const s = await redis.hget(H_CONVERSATIONS, id);
    if (!s) return;
    const row = JSON.parse(s) as ConvJSON;
    row.messageCount = (row.messageCount || 0) + inc;
    row.updatedAt = nowIso();
    row.lastMessageAt = row.updatedAt;
    await redis.hset(H_CONVERSATIONS, id, JSON.stringify(row));
    await redis.zadd(Z_UPDATED, nowMs(), id);
  },
  async setSession(conversationId: string, sessionId: string) {
    await redis.hset(H_SESSIONS, conversationId, sessionId);
  },
  async getSession(conversationId: string) {
    const v = await redis.hget(H_SESSIONS, conversationId);
    return v ?? undefined;
  },
  async upsertRow(row: ConvJSON) {
    await redis.hset(H_CONVERSATIONS, row.id, JSON.stringify(row));
    const score = Date.parse(row.updatedAt || new Date().toISOString()) || Date.now();
    await redis.zadd(Z_UPDATED, score, row.id);
  }
};
