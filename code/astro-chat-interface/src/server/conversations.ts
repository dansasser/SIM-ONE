// Simple in-memory conversation store for SSR endpoints
// For production, replace with a persistent store.

import { repo } from './persistence';

export interface ConversationRec {
  id: string;
  title: string;
  userId?: string;
  createdAt: Date;
  updatedAt: Date;
  lastMessageAt: Date;
  messageCount: number;
  isArchived: boolean;
  isPinned: boolean;
  tags: string[];
  metadata: Record<string, any>;
}

function rowToRec(row: any): ConversationRec {
  return {
    id: row.id,
    title: row.title,
    userId: row.userId || undefined,
    createdAt: new Date(row.createdAt),
    updatedAt: new Date(row.updatedAt),
    lastMessageAt: new Date(row.lastMessageAt),
    messageCount: Number(row.messageCount || 0),
    isArchived: !!row.isArchived,
    isPinned: !!row.isPinned,
    tags: row.tags ? JSON.parse(row.tags) : [],
    metadata: row.metadata ? JSON.parse(row.metadata) : { totalProcessingTime: 0, averageQuality: 0, topAgents: [], lastStyle: 'universal_chat', lastPriority: 'balanced' }
  };
}

class ConversationStore {
  async list(): Promise<ConversationRec[]> { const rows = await repo.list(); return rows.map(rowToRec); }
  async get(id: string): Promise<ConversationRec | undefined> { const row = await repo.get(id); return row ? rowToRec(row) : undefined; }
  async create(title?: string, userId?: string): Promise<ConversationRec> { const id = `conv-${Date.now()}-${Math.floor(Math.random() * 1000)}`; const row = await repo.create({ id, title: title || 'New Chat', userId }); return rowToRec(row); }
  async delete(id: string) { await repo.del(id); }
  async bumpMessageCount(id: string, increment = 1) { await repo.bump(id, increment); }
  async setSessionMapping(conversationId: string, sessionId: string) { await repo.setSession(conversationId, sessionId); }
  async getSessionMapping(conversationId: string) { return await repo.getSession(conversationId); }
}

export const conversations = new ConversationStore();
