import type { McpClient } from '../client/McpClient';
import type { Conversation, Message, ProcessingStyle, Priority, WorkflowRequest } from '../types';

export interface SessionStore {
  get(conversationId: string): Promise<string | undefined> | string | undefined;
  set(conversationId: string, sessionId: string): Promise<void> | void;
  delete(conversationId: string): Promise<void> | void;
}

export class MemorySessionStore implements SessionStore {
  private map = new Map<string, string>();
  get(id: string) { return this.map.get(id); }
  set(id: string, sid: string) { this.map.set(id, sid); }
  delete(id: string) { this.map.delete(id); }
}

export interface ChatClientOptions {
  templateMapper?: (input: { text: string; style: ProcessingStyle }) => { template_name: string; initial_data: Record<string, any> };
  sessionStore?: SessionStore;
}

export class ChatClient {
  private client: McpClient;
  private sessionStore: SessionStore;
  private templateMapper: ChatClientOptions['templateMapper'];

  constructor(client: McpClient, opts: ChatClientOptions = {}) {
    this.client = client;
    this.sessionStore = opts.sessionStore ?? new MemorySessionStore();
    this.templateMapper = opts.templateMapper ?? defaultTemplateMapper;
  }

  async createConversation(title = 'New Chat'): Promise<Conversation> {
    const now = new Date();
    return { id: `conv-${now.getTime()}`, title, createdAt: now, updatedAt: now, lastMessageAt: now, messageCount: 0 };
  }

  async sendMessage(params: { conversationId: string; text: string; style?: ProcessingStyle; priority?: Priority }): Promise<{ assistant: Message; sessionId: string }> {
    const style = params.style ?? 'universal_chat';
    const { template_name, initial_data } = this.templateMapper({ text: params.text, style });

    const sessionId = await Promise.resolve(this.sessionStore.get(params.conversationId));
    const request: WorkflowRequest = { template_name, initial_data, session_id: sessionId };
    const response = await this.client.execute(request);

    if (!sessionId && response.session_id) {
      await Promise.resolve(this.sessionStore.set(params.conversationId, response.session_id));
    }

    const content = extractTextFromResults(response.results) ?? JSON.stringify(response.results);
    const assistant: Message = {
      id: `msg-${Date.now()}`,
      conversationId: params.conversationId,
      role: 'assistant',
      content,
      timestamp: new Date(),
      metadata: { executionTimeMs: response.execution_time_ms }
    };

    return { assistant, sessionId: response.session_id };
  }

  async getMessages(params: { conversationId: string }): Promise<Message[]> {
    const sessionId = await Promise.resolve(this.sessionStore.get(params.conversationId));
    if (!sessionId) return [];
    const history = await this.client.getSession(sessionId);
    const msgs: Message[] = [];
    history.history?.forEach((entry: any, idx: number) => {
      if (entry.user_request) {
        msgs.push({ id: `u-${idx}`, conversationId: params.conversationId, role: 'user', content: JSON.stringify(entry.user_request.initial_data ?? entry.user_request), timestamp: new Date() });
      }
      if (entry.server_response) {
        const content = extractTextFromResults(entry.server_response) ?? JSON.stringify(entry.server_response);
        msgs.push({ id: `a-${idx}`, conversationId: params.conversationId, role: 'assistant', content, timestamp: new Date() });
      }
    });
    return msgs;
  }
}

function defaultTemplateMapper({ text, style }: { text: string; style: ProcessingStyle }) {
  const isChat = style === 'universal_chat';
  const template_name = isChat ? 'analyze_only' : 'writing_team';
  const initial_data = isChat ? { text } : { topic: text };
  return { template_name, initial_data };
}

function extractTextFromResults(results: Record<string, any>): string | undefined {
  if (!results) return undefined;
  if (typeof results.text === 'string') return results.text;
  if (typeof results.output === 'string') return results.output;
  if (typeof results.content === 'string') return results.content;
  const strings: string[] = [];
  for (const v of Object.values(results)) if (typeof v === 'string') strings.push(v);
  return strings.length ? strings.join('\n\n') : undefined;
}

