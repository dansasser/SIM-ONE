/**
 * React Hooks for SIM-ONE SDK Integration
 * 
 * These hooks provide a clean, reactive interface to the SIM-ONE SDK
 * and will work seamlessly when the actual SDK is integrated.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { SimoneSDK, SDKUtils } from '../sdk/simone-sdk';
const SERVER_MODE = (import.meta as any).env?.PUBLIC_SDK_MODE === 'server';
import type { 
  User, 
  Conversation, 
  Message, 
  ProcessingJob, 
  ProcessingStyle, 
  Priority,
  AIModel,
  MessageOptions,
  LoginCredentials,
  RegisterData
} from '../types/global';

// ===== AUTHENTICATION HOOKS =====

export function useAuth() {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Check if user is already authenticated on mount
    SimoneSDK.auth.getCurrentUser()
      .then(setUser)
      .catch(() => setUser(null))
      .finally(() => setIsLoading(false));
  }, []);

  const login = useCallback(async (credentials: LoginCredentials) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await SimoneSDK.auth.login(credentials);
      setUser(result.user);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Login failed';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const register = useCallback(async (data: RegisterData) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await SimoneSDK.auth.register(data);
      setUser(result.user);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Registration failed';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const logout = useCallback(async () => {
    setIsLoading(true);
    try {
      await SimoneSDK.auth.logout();
      setUser(null);
    } catch (err) {
      console.error('Logout error:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const updateProfile = useCallback(async (updates: Partial<User>) => {
    if (!user) return null;
    
    try {
      const updatedUser = await SimoneSDK.users.updateProfile(updates);
      setUser(updatedUser);
      return updatedUser;
    } catch (err) {
      console.error('Profile update error:', err);
      throw err;
    }
  }, [user]);

  return {
    user,
    isLoading,
    error,
    isAuthenticated: !!user,
    login,
    register,
    logout,
    updateProfile,
    clearError: () => setError(null)
  };
}

// ===== CHAT HOOKS =====

export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadConversations = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      if (SERVER_MODE) {
        const res = await fetch('/api/chat/conversations');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const convs = (data.conversations || []).map((c: any) => ({
          ...c,
          createdAt: c.createdAt ? new Date(c.createdAt) : new Date(),
          updatedAt: c.updatedAt ? new Date(c.updatedAt) : new Date(),
          lastMessageAt: c.lastMessageAt ? new Date(c.lastMessageAt) : new Date(),
          isArchived: !!c.isArchived,
          isPinned: !!c.isPinned,
          tags: c.tags || [],
          metadata: c.metadata || { totalProcessingTime: 0, averageQuality: 0, topAgents: [], lastStyle: 'universal_chat', lastPriority: 'balanced' }
        }));
        setConversations(convs);
      } else {
        const convs = await SimoneSDK.chat.getConversations();
        setConversations(convs);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load conversations';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const createConversation = useCallback(async (title?: string) => {
    try {
      if (SERVER_MODE) {
        const res = await fetch('/api/chat/conversations', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title })
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const c = await res.json();
        const newConv = {
          ...c,
          createdAt: c.createdAt ? new Date(c.createdAt) : new Date(),
          updatedAt: c.updatedAt ? new Date(c.updatedAt) : new Date(),
          lastMessageAt: c.lastMessageAt ? new Date(c.lastMessageAt) : new Date(),
          isArchived: !!c.isArchived,
          isPinned: !!c.isPinned,
          tags: c.tags || [],
          metadata: c.metadata || { totalProcessingTime: 0, averageQuality: 0, topAgents: [], lastStyle: 'universal_chat', lastPriority: 'balanced' }
        } as Conversation;
        setConversations(prev => [newConv, ...prev]);
        return newConv;
      } else {
        const newConv = await SimoneSDK.chat.createConversation(title);
        setConversations(prev => [newConv, ...prev]);
        return newConv;
      }
    } catch (err) {
      console.error('Create conversation error:', err);
      throw err;
    }
  }, []);

  const deleteConversation = useCallback(async (id: string) => {
    try {
      if (SERVER_MODE) {
        const res = await fetch(`/api/chat/conversations/${encodeURIComponent(id)}`, { method: 'DELETE' });
        if (!res.ok && res.status !== 204) throw new Error(`HTTP ${res.status}`);
        setConversations(prev => prev.filter(conv => conv.id !== id));
      } else {
        await SimoneSDK.chat.deleteConversation(id);
        setConversations(prev => prev.filter(conv => conv.id !== id));
      }
    } catch (err) {
      console.error('Delete conversation error:', err);
      throw err;
    }
  }, []);

  // Load conversations on mount
  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  return {
    conversations,
    isLoading,
    error,
    loadConversations,
    createConversation,
    deleteConversation,
    clearError: () => setError(null)
  };
}

export function useMessages(conversationId: string | null) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadMessages = useCallback(async (convId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      if (SERVER_MODE) {
        const res = await fetch(`/api/chat/messages/${encodeURIComponent(convId)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setMessages(data.messages || []);
      } else {
        const msgs = await SimoneSDK.chat.getMessages(convId);
        setMessages(msgs);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load messages';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const sendMessage = useCallback(async (content: string, options: MessageOptions) => {
    if (!conversationId) throw new Error('No conversation selected');
    try {
      if (SERVER_MODE) {
        // Optimistically add user message
        const userMessage: Message = {
          id: `msg-${Date.now()}`,
          conversationId,
          userId: undefined,
          content,
          type: 'user',
          timestamp: new Date(),
          status: 'sent',
          attachments: options.attachments || [],
          reactions: [],
          metadata: { style: options.style, priority: options.priority }
        };
        setMessages(prev => [...prev, userMessage]);
        const res = await fetch('/api/chat/send', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ conversationId, text: content, style: options.style, priority: options.priority })
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (data.assistant) setMessages(prev => [...prev, {
          id: data.assistant.id,
          conversationId,
          userId: undefined,
          content: data.assistant.content,
          type: 'assistant',
          timestamp: new Date(data.assistant.timestamp || Date.now()),
          status: 'delivered',
          attachments: [],
          reactions: [],
          metadata: data.assistant.metadata || {}
        }]);
        return { message: userMessage, jobId: data.assistant?.metadata?.jobId || '' };
      } else {
        const result = await SimoneSDK.chat.sendMessage(conversationId, content, options);
        setMessages(prev => [...prev, result.message]);
        return result;
      }
    } catch (err) {
      console.error('Send message error:', err);
      throw err;
    }
  }, [conversationId]);

  // Load messages when conversation changes
  useEffect(() => {
    if (conversationId) {
      loadMessages(conversationId);
    } else {
      setMessages([]);
    }
  }, [conversationId, loadMessages]);

  // Listen for new messages
  useEffect(() => {
    const handleNewMessage = (data: { message: Message }) => {
      if (data.message.conversationId === conversationId) {
        setMessages(prev => {
          // Avoid duplicates
          if (prev.some(msg => msg.id === data.message.id)) {
            return prev;
          }
          return [...prev, data.message];
        });
      }
    };

    SimoneSDK.events.on('message-received', handleNewMessage);
    
    return () => {
      SimoneSDK.events.off('message-received', handleNewMessage);
    };
  }, [conversationId]);

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    clearError: () => setError(null)
  };
}

// ===== PROCESSING HOOKS =====

export function useProcessing() {
  const [activeJobs, setActiveJobs] = useState<Map<string, ProcessingJob>>(new Map());
  const [isProcessing, setIsProcessing] = useState(false);

  const startProcessing = useCallback(async (
    input: string, 
    style: ProcessingStyle, 
    priority: Priority
  ) => {
    setIsProcessing(true);
    try {
      const job = await SimoneSDK.processing.startProcessing(input, style, priority);
      setActiveJobs(prev => new Map(prev).set(job.id, job));
      return job;
    } catch (err) {
      console.error('Processing error:', err);
      throw err;
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const getJob = useCallback(async (jobId: string) => {
    try {
      const job = await SimoneSDK.processing.getJob(jobId);
      setActiveJobs(prev => new Map(prev).set(job.id, job));
      return job;
    } catch (err) {
      console.error('Get job error:', err);
      throw err;
    }
  }, []);

  const cancelJob = useCallback(async (jobId: string) => {
    try {
      await SimoneSDK.processing.cancelJob(jobId);
      setActiveJobs(prev => {
        const newJobs = new Map(prev);
        const job = newJobs.get(jobId);
        if (job) {
          job.status = 'failed';
          newJobs.set(jobId, job);
        }
        return newJobs;
      });
    } catch (err) {
      console.error('Cancel job error:', err);
      throw err;
    }
  }, []);

  // Listen for processing updates
  useEffect(() => {
    const handleProcessingStep = (data: any) => {
      setActiveJobs(prev => {
        const newJobs = new Map(prev);
        const job = newJobs.get(data.jobId);
        if (job) {
          job.progress = data.progress;
          job.currentAgent = data.agent;
          newJobs.set(data.jobId, job);
        }
        return newJobs;
      });
    };

    const handleProcessingComplete = (data: any) => {
      setActiveJobs(prev => {
        const newJobs = new Map(prev);
        const job = newJobs.get(data.jobId);
        if (job) {
          job.status = 'completed';
          job.completedAt = new Date();
          job.progress = 100;
          newJobs.set(data.jobId, job);
        }
        return newJobs;
      });
    };

    SimoneSDK.events.on('processing-step', handleProcessingStep);
    SimoneSDK.events.on('processing-complete', handleProcessingComplete);

    return () => {
      SimoneSDK.events.off('processing-step', handleProcessingStep);
      SimoneSDK.events.off('processing-complete', handleProcessingComplete);
    };
  }, []);

  return {
    activeJobs: Array.from(activeJobs.values()),
    isProcessing,
    startProcessing,
    getJob,
    cancelJob,
    getActiveJob: (jobId: string) => activeJobs.get(jobId)
  };
}

// ===== MODELS HOOKS =====

export function useModels() {
  const [models, setModels] = useState<AIModel[]>([]);
  const [currentModel, setCurrentModel] = useState<AIModel | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const loadModels = useCallback(async () => {
    setIsLoading(true);
    try {
      const [availableModels, current] = await Promise.all([
        SimoneSDK.models.getAvailableModels(),
        SimoneSDK.models.getCurrentModel()
      ]);
      setModels(availableModels);
      setCurrentModel(current);
    } catch (err) {
      console.error('Load models error:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const switchModel = useCallback(async (modelId: string) => {
    try {
      await SimoneSDK.models.switchModel(modelId);
      const newCurrentModel = models.find(m => m.id === modelId);
      if (newCurrentModel) {
        setCurrentModel(newCurrentModel);
      }
    } catch (err) {
      console.error('Switch model error:', err);
      throw err;
    }
  }, [models]);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  return {
    models,
    currentModel,
    isLoading,
    switchModel,
    refreshModels: loadModels
  };
}

// ===== CONNECTION HOOKS =====

export function useSDKStatus() {
  const [isConnected, setIsConnected] = useState(false);
  const [status, setStatus] = useState(SDKUtils.getStatus());

  useEffect(() => {
    const handleConnectionStatus = (data: { connected: boolean }) => {
      setIsConnected(data.connected);
      setStatus(SDKUtils.getStatus());
    };

    SimoneSDK.events.on('connection-status', handleConnectionStatus);

    // Initial connection
    SDKUtils.init().then(success => {
      setIsConnected(success);
      setStatus(SDKUtils.getStatus());
    });

    return () => {
      SimoneSDK.events.off('connection-status', handleConnectionStatus);
    };
  }, []);

  return {
    isConnected,
    status,
    reconnect: () => SDKUtils.init()
  };
}

// ===== UTILITY HOOKS =====

export function useLocalStorage<T>(key: string, defaultValue: T) {
  const [value, setValue] = useState<T>(() => {
    if (typeof window === 'undefined') return defaultValue;
    
    try {
      const saved = localStorage.getItem(key);
      return saved ? JSON.parse(saved) : defaultValue;
    } catch {
      return defaultValue;
    }
  });

  const setStoredValue = useCallback((newValue: T | ((prev: T) => T)) => {
    setValue(prev => {
      const valueToStore = newValue instanceof Function ? newValue(prev) : newValue;
      
      if (typeof window !== 'undefined') {
        try {
          localStorage.setItem(key, JSON.stringify(valueToStore));
        } catch (err) {
          console.error('Failed to save to localStorage:', err);
        }
      }
      
      return valueToStore;
    });
  }, [key]);

  return [value, setStoredValue] as const;
}

export function useDebounce<T>(value: T, delay: number) {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}
