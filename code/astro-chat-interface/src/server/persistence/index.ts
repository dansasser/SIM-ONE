import { repo as sqliteRepo } from './sqlite';

let backend = (process.env.PERSISTENCE_BACKEND || '').toLowerCase();

// Lazy import redis to avoid requiring it if not used
let redisRepo: any = null;
async function getRedisRepo() {
  if (!redisRepo) {
    const mod = await import('./redis');
    redisRepo = mod.repo;
  }
  return redisRepo;
}

export const repo = {
  async list() {
    if (backend === 'redis') {
      const r = await getRedisRepo();
      return r.list();
    }
    return sqliteRepo.list();
  },
  async get(id: string) {
    if (backend === 'redis') {
      const r = await getRedisRepo();
      return r.get(id);
    }
    return sqliteRepo.get(id);
  },
  async create(input: { id: string; title: string; userId?: string }) {
    if (backend === 'redis') {
      const r = await getRedisRepo();
      return r.create(input);
    }
    return sqliteRepo.create(input);
  },
  async del(id: string) {
    if (backend === 'redis') {
      const r = await getRedisRepo();
      return r.del(id);
    }
    return sqliteRepo.del(id);
  },
  async bump(id: string, inc = 1) {
    if (backend === 'redis') {
      const r = await getRedisRepo();
      return r.bump(id, inc);
    }
    return sqliteRepo.bump(id, inc);
  },
  async setSession(conversationId: string, sessionId: string) {
    if (backend === 'redis') {
      const r = await getRedisRepo();
      return r.setSession(conversationId, sessionId);
    }
    return sqliteRepo.setSession(conversationId, sessionId);
  },
  async getSession(conversationId: string) {
    if (backend === 'redis') {
      const r = await getRedisRepo();
      return r.getSession(conversationId);
    }
    return sqliteRepo.getSession(conversationId);
  }
};

