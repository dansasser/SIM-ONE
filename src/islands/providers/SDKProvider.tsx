/**
 * SIM-ONE SDK Provider
 * 
 * Provides SDK context and global state management for the entire application.
 * This ensures consistent SDK usage across all components.
 */

import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { useAuth, useSDKStatus, useConversations, useModels } from '../../lib/hooks/useSimoneSDK';
import type { User, Conversation, AIModel } from '../../lib/types/global';

interface SDKContextValue {
  // Authentication
  user: User | null;
  isAuthenticated: boolean;
  authLoading: boolean;
  authError: string | null;
  
  // Connection Status
  isConnected: boolean;
  sdkStatus: any;
  
  // Conversations
  conversations: Conversation[];
  conversationsLoading: boolean;
  currentConversationId: string | null;
  setCurrentConversationId: (id: string | null) => void;
  
  // Models
  models: AIModel[];
  currentModel: AIModel | null;
  modelsLoading: boolean;
  
  // Methods
  login: (credentials: any) => Promise<any>;
  register: (data: any) => Promise<any>;
  logout: () => Promise<void>;
  createConversation: (title?: string) => Promise<Conversation>;
  deleteConversation: (id: string) => Promise<void>;
  switchModel: (modelId: string) => Promise<void>;
  
  // UI State
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
  settingsOpen: boolean;
  setSettingsOpen: (open: boolean) => void;
  agentPipelineVisible: boolean;
  setAgentPipelineVisible: (visible: boolean) => void;
}

const SDKContext = createContext<SDKContextValue | null>(null);

interface SDKProviderProps {
  children: ReactNode;
}

export const SDKProvider: React.FC<SDKProviderProps> = ({ children }) => {
  // Authentication state
  const { 
    user, 
    isAuthenticated, 
    isLoading: authLoading, 
    error: authError, 
    login, 
    register, 
    logout 
  } = useAuth();
  
  // Connection state
  const { isConnected, status: sdkStatus } = useSDKStatus();
  
  // Conversations state
  const { 
    conversations, 
    isLoading: conversationsLoading, 
    createConversation, 
    deleteConversation 
  } = useConversations();
  
  // Models state
  const { 
    models, 
    currentModel, 
    isLoading: modelsLoading, 
    switchModel 
  } = useModels();
  
  // UI state
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [agentPipelineVisible, setAgentPipelineVisible] = useState(false);
  
  // Auto-select first conversation if none selected
  useEffect(() => {
    if (!currentConversationId && conversations.length > 0) {
      setCurrentConversationId(conversations[0].id);
    }
  }, [conversations, currentConversationId]);
  
  // Listen for global UI events
  useEffect(() => {
    const handleToggleAgentPipeline = () => {
      setAgentPipelineVisible(prev => !prev);
    };
    
    const handleOpenSettings = () => {
      setSettingsOpen(true);
    };
    
    const handleOpenAuth = () => {
      // Handle auth modal opening
      console.log('Auth modal requested');
    };
    
    // Listen for custom events from static components
    window.addEventListener('toggle-agent-pipeline', handleToggleAgentPipeline);
    window.addEventListener('open-settings-modal', handleOpenSettings);
    window.addEventListener('open-auth-modal', handleOpenAuth);
    
    return () => {
      window.removeEventListener('toggle-agent-pipeline', handleToggleAgentPipeline);
      window.removeEventListener('open-settings-modal', handleOpenSettings);
      window.removeEventListener('open-auth-modal', handleOpenAuth);
    };
  }, []);
  
  // Handle mobile sidebar
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth > 768) {
        setSidebarOpen(false);
      }
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  const value: SDKContextValue = {
    // Authentication
    user,
    isAuthenticated,
    authLoading,
    authError,
    
    // Connection
    isConnected,
    sdkStatus,
    
    // Conversations
    conversations,
    conversationsLoading,
    currentConversationId,
    setCurrentConversationId,
    
    // Models
    models,
    currentModel,
    modelsLoading,
    
    // Methods
    login,
    register,
    logout,
    createConversation,
    deleteConversation,
    switchModel,
    
    // UI State
    sidebarOpen,
    setSidebarOpen,
    settingsOpen,
    setSettingsOpen,
    agentPipelineVisible,
    setAgentPipelineVisible
  };
  
  return (
    <SDKContext.Provider value={value}>
      {children}
    </SDKContext.Provider>
  );
};

export const useSDKContext = () => {
  const context = useContext(SDKContext);
  if (!context) {
    throw new Error('useSDKContext must be used within an SDKProvider');
  }
  return context;
};

// ===== SDK STATUS INDICATOR =====

export const SDKStatusIndicator: React.FC = () => {
  const { isConnected, sdkStatus } = useSDKContext();
  
  if (!isConnected) {
    return (
      <div className="flex items-center space-x-2 px-2 py-1 bg-red-500/20 text-red-400 rounded-md text-xs">
        <div className="w-2 h-2 bg-red-400 rounded-full animate-pulse"></div>
        <span>SDK Disconnected</span>
      </div>
    );
  }
  
  return (
    <div className="flex items-center space-x-2 px-2 py-1 bg-green-500/20 text-green-400 rounded-md text-xs">
      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
      <span>SDK v{sdkStatus.version}</span>
    </div>
  );
};

// ===== CONVERSATION SELECTOR =====

export const ConversationSelector: React.FC = () => {
  const { 
    conversations, 
    currentConversationId, 
    setCurrentConversationId,
    createConversation,
    deleteConversation
  } = useSDKContext();
  
  const handleNewChat = async () => {
    try {
      const newConv = await createConversation();
      setCurrentConversationId(newConv.id);
    } catch (err) {
      console.error('Failed to create conversation:', err);
    }
  };
  
  return (
    <div className="conversation-selector">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-text-primary">Conversations</h3>
        <button
          onClick={handleNewChat}
          className="p-1 hover:bg-bg-tertiary rounded text-text-secondary hover:text-text-primary transition-colors"
          title="New Conversation"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 5v14M5 12h14"/>
          </svg>
        </button>
      </div>
      
      <div className="space-y-1">
        {conversations.map((conv) => (
          <button
            key={conv.id}
            onClick={() => setCurrentConversationId(conv.id)}
            className={`w-full text-left p-2 rounded-md transition-colors ${
              currentConversationId === conv.id
                ? 'bg-bg-tertiary text-text-primary'
                : 'text-text-secondary hover:text-text-primary hover:bg-bg-tertiary'
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium truncate">{conv.title}</div>
                <div className="text-xs text-text-tertiary">
                  {conv.messageCount} messages
                </div>
              </div>
              
              {conv.isPinned && (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" className="text-accent-primary">
                  <path d="M16 4v4h4l-4 4-4-4h4V4zm-4 12H8v-4H4l4-4 4 4h-4v4z"/>
                </svg>
              )}
            </div>
          </button>
        ))}
      </div>
      
      {conversations.length === 0 && (
        <div className="text-center py-8">
          <div className="text-text-tertiary text-sm mb-2">No conversations yet</div>
          <button
            onClick={handleNewChat}
            className="text-accent-primary hover:text-accent-secondary text-sm transition-colors"
          >
            Start your first chat
          </button>
        </div>
      )}
    </div>
  );
};

export default SDKProvider;