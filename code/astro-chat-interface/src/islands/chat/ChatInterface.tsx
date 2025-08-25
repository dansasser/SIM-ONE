/**
 * Main Chat Interface
 * 
 * This is the complete chat interface that showcases the SIM-ONE SDK capabilities.
 * It's designed to be a compelling demo of what developers can build with the SDK.
 */

import React from 'react';
import SDKProvider, { useSDKContext, SDKStatusIndicator, ConversationSelector } from '../providers/SDKProvider';
import ChatMessages from './ChatMessages';
import ChatInput from './ChatInput';
import AgentPipeline from './AgentPipeline';
import { useProcessing } from '../../lib/hooks/useSimoneSDK';

// Internal Chat Interface Component (wrapped by SDK Provider)
const InternalChatInterface: React.FC = () => {
  const { 
    currentConversationId, 
    agentPipelineVisible,
    sidebarOpen,
    setSidebarOpen,
    user,
    isAuthenticated
  } = useSDKContext();
  
  const { activeJobs } = useProcessing();
  const currentJob = activeJobs.find(job => job.status === 'in_progress');

  return (
    <div className="chat-interface h-screen flex bg-bg-primary">
      {/* Sidebar */}
      <div className={`sidebar bg-bg-secondary border-r border-border-primary transition-transform duration-300 ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      } md:translate-x-0 fixed md:static z-50 md:z-auto w-64 h-full`}>
        
        {/* Sidebar Header */}
        <div className="p-4 border-b border-border-primary">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-lg font-semibold text-text-primary">SIM-ONE Chat</h1>
            <button
              onClick={() => setSidebarOpen(false)}
              className="md:hidden p-1 hover:bg-bg-tertiary rounded text-text-secondary hover:text-text-primary"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6L6 18M6 6l12 12"/>
              </svg>
            </button>
          </div>
          
          <SDKStatusIndicator />
        </div>

        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto p-4">
          <ConversationSelector />
        </div>

        {/* User Section */}
        <div className="border-t border-border-primary p-4">
          {isAuthenticated ? (
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-accent-primary rounded-full flex items-center justify-center">
                <span className="text-white text-sm font-medium">
                  {user?.displayName?.[0]?.toUpperCase() || 'U'}
                </span>
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-text-primary truncate">
                  {user?.displayName || 'User'}
                </div>
                <div className="text-xs text-text-tertiary">
                  {user?.subscription.name} Plan
                </div>
              </div>
            </div>
          ) : (
            <button className="w-full p-3 bg-accent-primary hover:bg-accent-secondary text-white rounded-lg transition-colors">
              Sign In to SIM-ONE
            </button>
          )}
        </div>
      </div>

      {/* Sidebar Overlay for Mobile */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Chat Header */}
        <div className="flex-shrink-0 bg-bg-primary border-b border-border-primary px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              {/* Mobile Menu Button */}
              <button
                onClick={() => setSidebarOpen(true)}
                className="md:hidden p-2 hover:bg-bg-tertiary rounded text-text-secondary hover:text-text-primary"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 12h18M3 6h18M3 18h18"/>
                </svg>
              </button>

              <div>
                <h2 className="text-sm font-medium text-text-primary">
                  {currentConversationId ? 'SIM-ONE Chat' : 'Welcome to SIM-ONE'}
                </h2>
                {currentJob && (
                  <div className="text-xs text-accent-primary">
                    Processing with {currentJob.currentAgent}...
                  </div>
                )}
              </div>
            </div>

            <div className="flex items-center space-x-2">
              {/* Model Indicator */}
              <div className="hidden sm:flex items-center px-2 py-1 bg-bg-tertiary rounded-md">
                <div className="w-2 h-2 bg-accent-primary rounded-full mr-2"></div>
                <span className="text-xs text-text-secondary">SIM-ONE v1.2</span>
              </div>

              {/* Pipeline Toggle */}
              <button
                onClick={() => window.dispatchEvent(new CustomEvent('toggle-agent-pipeline'))}
                className="p-2 hover:bg-bg-tertiary rounded text-text-secondary hover:text-text-primary"
                title="Toggle Agent Pipeline"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>
                </svg>
              </button>
            </div>
          </div>
        </div>

        {/* Agent Pipeline (Conditional) */}
        {agentPipelineVisible && currentJob && (
          <div className="flex-shrink-0">
            <AgentPipeline 
              jobId={currentJob.id}
              isVisible={true}
            />
          </div>
        )}

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto">
          <ChatMessages />
        </div>

        {/* Input Area */}
        <div className="flex-shrink-0 border-t border-border-primary p-4">
          <ChatInput conversationId={currentConversationId} />
        </div>
      </div>
    </div>
  );
};

// Main exported component with SDK Provider
const ChatInterface: React.FC = () => {
  return (
    <SDKProvider>
      <InternalChatInterface />
    </SDKProvider>
  );
};

export default ChatInterface;