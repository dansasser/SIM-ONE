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

export interface SessionHistory {
  session_id: string;
  history: any[];
}

export interface Health {
  status: string;
  services?: Record<string, string>;
}

// Optional higher-level chat types (UI-agnostic)
export type AgentType = 'ideator' | 'drafter' | 'reviser' | 'critic' | 'summarizer';
export type ProcessingStyle =
  | 'universal_chat'
  | 'analytical_article'
  | 'creative_writing'
  | 'academic_paper'
  | 'business_report'
  | 'technical_documentation';
export type Priority = 'fast' | 'balanced' | 'quality';

export interface Conversation {
  id: string;
  title: string;
  createdAt: Date;
  updatedAt: Date;
  lastMessageAt: Date;
  messageCount: number;
}

export interface Message {
  id: string;
  conversationId: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  metadata?: Record<string, any>;
}

