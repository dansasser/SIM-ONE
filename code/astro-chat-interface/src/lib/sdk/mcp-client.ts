import { HttpClient } from './http';

export type CoordinationMode = 'Sequential' | 'Parallel';

export interface WorkflowRequest {
  template_name?: string;
  protocol_names?: string[];
  coordination_mode?: CoordinationMode;
  initial_data: Record<string, any>;
  session_id?: string;
}

export interface WorkflowResponse {
  session_id: string;
  results: Record<string, any>;
  error?: string | null;
  execution_time_ms: number;
}

export class McpClient {
  private http: HttpClient;

  constructor(baseUrl = '/api') {
    this.http = new HttpClient({ baseUrl });
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
    return this.http.get<{ session_id: string; history: any[] }>(`/session/${encodeURIComponent(sessionId)}`);
  }

  health() {
    return this.http.get<any>('/health');
  }
}

