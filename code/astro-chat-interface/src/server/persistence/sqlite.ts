import Database from 'better-sqlite3';
import fs from 'node:fs';
import path from 'node:path';

export interface ConversationRow {
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
}

const DATA_DIR = path.join(process.cwd(), 'astro-chat-interface', 'data');
const DB_PATH = path.join(DATA_DIR, 'conversations.db');

function ensureDir() {
  try { fs.mkdirSync(DATA_DIR, { recursive: true }); } catch {}
}

let db: Database.Database | null = null;

export function getDb() {
  if (db) return db;
  ensureDir();
  db = new Database(DB_PATH);
  db.pragma('journal_mode = WAL');
  db.exec(`
    CREATE TABLE IF NOT EXISTS conversations (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      userId TEXT,
      createdAt TEXT NOT NULL,
      updatedAt TEXT NOT NULL,
      lastMessageAt TEXT NOT NULL,
      messageCount INTEGER NOT NULL DEFAULT 0,
      isArchived INTEGER NOT NULL DEFAULT 0,
      isPinned INTEGER NOT NULL DEFAULT 0,
      tags TEXT,
      metadata TEXT
    );
    CREATE TABLE IF NOT EXISTS session_mappings (
      conversationId TEXT PRIMARY KEY,
      sessionId TEXT NOT NULL
    );
  `);
  return db;
}

export const repo = {
  list() {
    const rows = getDb().prepare('SELECT * FROM conversations ORDER BY updatedAt DESC').all() as ConversationRow[];
    return rows;
  },
  get(id: string) {
    const row = getDb().prepare('SELECT * FROM conversations WHERE id = ?').get(id) as ConversationRow | undefined;
    return row;
  },
  create(input: { id: string; title: string; userId?: string }) {
    const now = new Date().toISOString();
    getDb().prepare(`INSERT INTO conversations (id, title, userId, createdAt, updatedAt, lastMessageAt, messageCount, isArchived, isPinned, tags, metadata)
      VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, NULL, NULL)`
    ).run(input.id, input.title, input.userId ?? null, now, now, now);
    return this.get(input.id)!;
  },
  del(id: string) {
    getDb().prepare('DELETE FROM conversations WHERE id = ?').run(id);
    getDb().prepare('DELETE FROM session_mappings WHERE conversationId = ?').run(id);
  },
  bump(id: string, inc = 1) {
    const now = new Date().toISOString();
    getDb().prepare('UPDATE conversations SET messageCount = messageCount + ?, updatedAt = ?, lastMessageAt = ? WHERE id = ?')
      .run(inc, now, now, id);
  },
  setSession(conversationId: string, sessionId: string) {
    getDb().prepare('INSERT INTO session_mappings (conversationId, sessionId) VALUES (?, ?) ON CONFLICT(conversationId) DO UPDATE SET sessionId = excluded.sessionId')
      .run(conversationId, sessionId);
  },
  getSession(conversationId: string): string | undefined {
    const row = getDb().prepare('SELECT sessionId FROM session_mappings WHERE conversationId = ?').get(conversationId) as { sessionId: string } | undefined;
    return row?.sessionId;
  },
  delSession(conversationId: string) {
    getDb().prepare('DELETE FROM session_mappings WHERE conversationId = ?').run(conversationId);
  }
};

