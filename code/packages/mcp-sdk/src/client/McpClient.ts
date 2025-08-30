import { HttpClient, type HttpClientOptions } from '../http';
import type { WorkflowRequest, WorkflowResponse, SessionHistory, Health } from '../types';

export interface McpClientOptions extends Omit<HttpClientOptions, 'baseUrl'> {
  baseUrl: string; // e.g., http://localhost:8000 or /api (proxy)
  apiKey?: string; // optional; only use in server contexts
}

export class McpClient {
  private http: HttpClient;

  constructor(opts: McpClientOptions) {
    const headers = { ...(opts.headers ?? {}) };
    if (opts.apiKey) headers['X-API-Key'] = opts.apiKey;
    this.http = new HttpClient({ baseUrl: opts.baseUrl, headers, timeoutMs: opts.timeoutMs, fetchImpl: opts.fetchImpl });
  }

  execute(payload: WorkflowRequest) {
    return this.http.post<WorkflowResponse>('/execute', payload);
  }

  getProtocols() {
    return this.http.get<Record<string, any>>('/protocols');
  }

  getTemplates() {
    return this.http.get<Record<string, any>>('/templates');
  }

  getSession(sessionId: string) {
    return this.http.get<SessionHistory>(`/session/${encodeURIComponent(sessionId)}`);
  }

  health() {
    return this.http.get<Health>('/health');
  }
}

