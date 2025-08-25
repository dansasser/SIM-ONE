// ===== USER TYPES =====
export interface User {
  id: string;
  email: string;
  displayName: string;
  avatar?: string;
  createdAt: Date;
  updatedAt: Date;
  lastLoginAt: Date;
  isEmailVerified: boolean;
  subscription: SubscriptionTier;
  usage: UsageStats;
}

export interface UserPreferences {
  theme: Theme;
  language: string;
  timezone: string;
  notifications: NotificationSettings;
  privacy: PrivacySettings;
  simone: SIMONESettings;
}

export interface SubscriptionTier {
  name: string;
  features: string[];
  limits: {
    messagesPerMonth: number;
    conversationsLimit: number;
    processingPriority: Priority[];
    storageLimit: number;
  };
}

export interface UsageStats {
  messagesThisMonth: number;
  conversationsCount: number;
  storageUsed: number;
  lastResetDate: Date;
}

// ===== CHAT TYPES =====
export interface Conversation {
  id: string;
  title: string;
  userId: string;
  createdAt: Date;
  updatedAt: Date;
  lastMessageAt: Date;
  messageCount: number;
  isArchived: boolean;
  isPinned: boolean;
  tags: string[];
  metadata: ConversationMetadata;
}

export interface ConversationMetadata {
  totalProcessingTime: number;
  averageQuality: number;
  topAgents: AgentType[];
  lastStyle: ProcessingStyle;
  lastPriority: Priority;
}

export interface Message {
  id: string;
  conversationId: string;
  userId?: string;
  content: string;
  type: 'user' | 'assistant' | 'system';
  timestamp: Date;
  status: 'sending' | 'sent' | 'delivered' | 'failed';
  attachments: Attachment[];
  reactions: Reaction[];
  metadata: MessageMetadata;
}

export interface MessageMetadata {
  jobId?: string;
  agentsUsed?: AgentType[];
  processingTime?: number;
  qualityMetrics?: QualityMetrics;
  style?: ProcessingStyle;
  priority?: Priority;
  model?: string;
}

export interface Attachment {
  id: string;
  name: string;
  size: number;
  type: string;
  url: string;
}

export interface Reaction {
  id: string;
  userId: string;
  emoji: string;
  timestamp: Date;
}

// ===== SIM-ONE TYPES =====
export type AgentType = 'ideator' | 'drafter' | 'reviser' | 'critic' | 'summarizer';

export type ProcessingStyle =
  | 'universal_chat'
  | 'analytical_article'
  | 'creative_writing'
  | 'academic_paper'
  | 'business_report'
  | 'technical_documentation';

export type Priority = 'fast' | 'balanced' | 'quality';

export interface ProcessingJob {
  id: string;
  input: string;
  style: ProcessingStyle;
  priority: Priority;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  currentAgent?: AgentType;
  progress: number;
  estimatedTimeRemaining: number;
  startedAt: Date;
  completedAt?: Date;
  result?: ProcessingResult;
  error?: ProcessingError;
  metrics: ProcessingMetrics;
}

export interface ProcessingResult {
  content: string;
  agentsUsed: AgentType[];
  processingTime: number;
  qualityMetrics: QualityMetrics;
  metadata: ProcessingMetadata;
}

export interface ProcessingError {
  code: string;
  message: string;
  details?: any;
  recoverable: boolean;
  agent?: AgentType;
}

export interface ProcessingMetrics {
  totalTime: number;
  agentTimes: Record<AgentType, number>;
  tokenUsage: {
    input: number;
    output: number;
    total: number;
  };
  qualityScore: number;
  coherenceScore: number;
  creativityScore: number;
}

export interface ProcessingMetadata {
  workflow: string;
  modelUsed: string;
  temperature: number;
  maxTokens: number;
}

export interface QualityMetrics {
  coherence: number;
  creativity: number;
  accuracy: number;
  relevance: number;
  overall: number;
}

export interface AIModel {
  id: string;
  name: string;
  description: string;
  capabilities: string[];
  maxTokens: number;
  costPerToken: number;
  responseTime: number;
  isAvailable: boolean;
}

// ===== SETTINGS TYPES =====
export interface Theme {
  name: string;
  colors: {
    primary: string;
    secondary: string;
    background: string;
    surface: string;
    text: string;
    textSecondary: string;
    border: string;
    accent: string;
    success: string;
    warning: string;
    error: string;
  };
  fonts: {
    primary: string;
    secondary: string;
    mono: string;
  };
  spacing: {
    xs: string;
    sm: string;
    md: string;
    lg: string;
    xl: string;
  };
}

export interface AppearanceSettings {
  theme: 'dark' | 'light' | 'auto';
  fontSize: 'small' | 'medium' | 'large';
  fontFamily: 'inter' | 'system' | 'serif';
  animationLevel: 'none' | 'reduced' | 'full';
  reducedMotion: boolean;
  highContrast: boolean;
  compactMode: boolean;
}

export interface NotificationSettings {
  desktop: boolean;
  sound: boolean;
  processingComplete: boolean;
  errors: boolean;
  systemUpdates: boolean;
  marketing: boolean;
  volume: number;
  soundTheme: string;
}

export interface PrivacySettings {
  dataRetention: number; // days
  shareUsageData: boolean;
  shareErrorReports: boolean;
  allowAnalytics: boolean;
  exportFormat: 'json' | 'csv' | 'markdown';
  autoDelete: boolean;
  encryptLocal: boolean;
}

export interface SIMONESettings {
  defaultStyle: ProcessingStyle;
  defaultPriority: Priority;
  showAgentPipeline: boolean;
  processingTimeout: number;
  autoRetry: boolean;
  retryAttempts: number;
  qualityThreshold: number;
  preferredModel: string;
  customPrompts: Record<string, string>;
}

export interface AdvancedSettings {
  debugMode: boolean;
  experimentalFeatures: boolean;
  apiEndpoint: string;
  requestTimeout: number;
  maxConcurrentRequests: number;
  cacheEnabled: boolean;
  logLevel: 'error' | 'warn' | 'info' | 'debug';
}

// ===== UI TYPES =====
export interface MessageFilter {
  search?: string;
  dateRange?: {
    start: Date;
    end: Date;
  };
  agents?: AgentType[];
  styles?: ProcessingStyle[];
  priorities?: Priority[];
  hasAttachments?: boolean;
}

export interface SearchResult {
  messageId: string;
  conversationId: string;
  snippet: string;
  highlights: string[];
  relevance: number;
}

export interface MessageOptions {
  style: ProcessingStyle;
  priority: Priority;
  attachments?: File[];
  customPrompt?: string;
}

export interface AgentStep {
  agent: AgentType;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  startTime?: Date;
  endTime?: Date;
  output?: string;
  metrics?: StepMetrics;
}

export interface StepMetrics {
  duration: number;
  tokenUsage: number;
  qualityScore: number;
  memoryUsage: number;
}

export interface SIMONEMetrics {
  totalProcessingTime: number;
  totalMessages: number;
  averageQuality: number;
  agentUsageStats: Record<AgentType, number>;
  styleUsageStats: Record<ProcessingStyle, number>;
  priorityUsageStats: Record<Priority, number>;
  errorRate: number;
  uptime: number;
}

// ===== AUTH TYPES =====
export interface LoginCredentials {
  email: string;
  password: string;
  authMethod: 'api-key' | 'jwt' | 'oauth2';
  rememberMe?: boolean;
}

export interface RegisterData {
  email: string;
  password: string;
  confirmPassword: string;
  displayName: string;
  acceptTerms: boolean;
}

export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
}

// ===== COMPONENT PROP TYPES =====
export interface BaseComponentProps {
  className?: string;
  children?: React.ReactNode;
}

export type SettingsSection = 
  | 'profile'
  | 'simone'
  | 'appearance'
  | 'notifications'
  | 'privacy'
  | 'advanced'
  | 'about';

export type MessageAction = 
  | 'copy'
  | 'share'
  | 'favorite'
  | 'delete'
  | 'edit'
  | 'retry';

// ===== UTILITY TYPES =====
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>;

export type OptionalFields<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

// ===== CONSTANTS =====
export const PROCESSING_STYLES: Record<ProcessingStyle, { name: string; description: string; icon: string }> = {
  universal_chat: {
    name: 'Universal Chat',
    description: 'General conversation and Q&A',
    icon: '💬'
  },
  analytical_article: {
    name: 'Analytical Article',
    description: 'Technical analysis and research',
    icon: '📊'
  },
  creative_writing: {
    name: 'Creative Writing',
    description: 'Stories, poems, creative content',
    icon: '✨'
  },
  academic_paper: {
    name: 'Academic Paper',
    description: 'Formal academic writing',
    icon: '🎓'
  },
  business_report: {
    name: 'Business Report',
    description: 'Professional business communication',
    icon: '📈'
  },
  technical_documentation: {
    name: 'Technical Documentation',
    description: 'Technical guides and manuals',
    icon: '📚'
  }
};

export const PRIORITIES: Record<Priority, { name: string; description: string; estimatedTime: string }> = {
  fast: {
    name: 'Fast',
    description: 'Optimized for speed',
    estimatedTime: '2-5 seconds'
  },
  balanced: {
    name: 'Balanced',
    description: 'Balanced speed and quality',
    estimatedTime: '3-10 seconds'
  },
  quality: {
    name: 'Quality',
    description: 'Maximum quality processing',
    estimatedTime: '5-20 seconds'
  }
};

export const AGENTS: Record<AgentType, { name: string; description: string; color: string }> = {
  ideator: {
    name: 'Ideator',
    description: 'Generates initial concepts and ideas',
    color: '#3b82f6'
  },
  drafter: {
    name: 'Drafter',
    description: 'Creates structured content drafts',
    color: '#10b981'
  },
  reviser: {
    name: 'Reviser',
    description: 'Refines and improves content',
    color: '#f59e0b'
  },
  critic: {
    name: 'Critic',
    description: 'Evaluates quality and coherence',
    color: '#ef4444'
  },
  summarizer: {
    name: 'Summarizer',
    description: 'Produces final polished output',
    color: '#8b5cf6'
  }
};