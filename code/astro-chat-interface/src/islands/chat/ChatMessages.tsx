/**
 * ChatMessages Component
 * 
 * Real-time message display with SDK integration
 * Showcases SIM-ONE's processing capabilities and agent pipeline
 */

import React, { useRef, useEffect } from 'react';
import { useMessages, useProcessing } from '../../lib/hooks/useSimoneSDK';
import { useSDKContext } from '../providers/SDKProvider';
import type { Message, AgentType } from '../../lib/types/global';

interface ChatMessagesProps {
  className?: string;
}

const ChatMessages: React.FC<ChatMessagesProps> = ({ className = '' }) => {
  const { currentConversationId } = useSDKContext();
  const { messages, isLoading } = useMessages(currentConversationId);
  const { activeJobs } = useProcessing();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const getAgentIcon = (agent: AgentType): string => {
    const icons = {
      ideator: '💡',
      drafter: '✏️', 
      reviser: '🔄',
      critic: '🔍',
      summarizer: '✨'
    };
    return icons[agent] || '🤖';
  };

  const getAgentColor = (agent: AgentType): string => {
    const colors = {
      ideator: 'bg-blue-500/20 text-blue-400',
      drafter: 'bg-green-500/20 text-green-400',
      reviser: 'bg-yellow-500/20 text-yellow-400',
      critic: 'bg-red-500/20 text-red-400',
      summarizer: 'bg-purple-500/20 text-purple-400'
    };
    return colors[agent] || 'bg-gray-500/20 text-gray-400';
  };

  const formatProcessingTime = (ms: number): string => {
    return `${(ms / 1000).toFixed(1)}s`;
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      // Could show success toast here
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  };

  if (isLoading) {
    return (
      <div className={`flex items-center justify-center h-full ${className}`}>
        <div className="flex items-center space-x-2 text-text-secondary">
          <div className="animate-spin w-5 h-5 border-2 border-accent-primary border-t-transparent rounded-full"></div>
          <span>Loading conversation...</span>
        </div>
      </div>
    );
  }

  if (!currentConversationId) {
    return (
      <div className={`flex items-center justify-center h-full ${className}`}>
        <div className="text-center">
          <div className="text-4xl mb-4">🤖</div>
          <h3 className="text-lg font-medium text-text-primary mb-2">
            Welcome to SIM-ONE
          </h3>
          <p className="text-text-secondary">
            Select a conversation to start chatting with our governed AI system
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={`chat-messages ${className}`}>
      <div className="messages-container space-y-6 p-4 max-w-4xl mx-auto">
        {messages.map((message) => (
          <div key={message.id} className="message-bubble">
            {message.type === 'user' ? (
              // User Message
              <div className="flex justify-end">
                <div className="user-message max-w-xs md:max-w-md lg:max-w-lg">
                  <div className="bg-accent-primary text-white rounded-2xl px-4 py-3">
                    <div className="text-sm whitespace-pre-wrap">{message.content}</div>
                  </div>
                  <div className="text-xs text-text-tertiary mt-1 text-right">
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ) : (
              // Assistant Message
              <div className="assistant-message">
                <div className="flex items-start space-x-3">
                  {/* SIM-ONE Avatar */}
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 bg-gradient-to-br from-accent-primary to-accent-secondary rounded-full flex items-center justify-center">
                      <span className="text-white text-xs font-bold">S1</span>
                    </div>
                  </div>

                  {/* Message Content */}
                  <div className="flex-1 min-w-0">
                    <div className="bg-bg-surface border border-border-primary rounded-2xl px-4 py-3">
                      {/* Message Text */}
                      <div className="prose prose-sm max-w-none text-text-primary">
                        <div className="whitespace-pre-wrap">{message.content}</div>
                      </div>

                      {/* SDK Processing Metadata */}
                      {message.metadata && (
                        <div className="mt-4 pt-3 border-t border-border-primary">
                          <div className="space-y-2">
                            {/* Agents Used */}
                            {message.metadata.agentsUsed && (
                              <div className="flex flex-wrap items-center gap-2">
                                <span className="text-xs text-text-tertiary">Processed by:</span>
                                {message.metadata.agentsUsed.map((agent) => (
                                  <span 
                                    key={agent}
                                    className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getAgentColor(agent)}`}
                                    title={`${agent.charAt(0).toUpperCase()}${agent.slice(1)} Agent`}
                                  >
                                    <span className="mr-1">{getAgentIcon(agent)}</span>
                                    {agent}
                                  </span>
                                ))}
                              </div>
                            )}

                            {/* Processing Stats */}
                            <div className="flex flex-wrap items-center gap-4 text-xs text-text-tertiary">
                              {message.metadata.processingTime && (
                                <span>⏱️ {formatProcessingTime(message.metadata.processingTime)}</span>
                              )}
                              
                              {message.metadata.qualityMetrics?.overall && (
                                <span>✨ Quality: {(message.metadata.qualityMetrics.overall * 100).toFixed(0)}%</span>
                              )}
                              
                              {message.metadata.style && (
                                <span>🎨 {message.metadata.style.replace('_', ' ')}</span>
                              )}
                              
                              {message.metadata.priority && (
                                <span>⚡ {message.metadata.priority}</span>
                              )}
                            </div>

                            {/* Quality Breakdown */}
                            {message.metadata.qualityMetrics && (
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-2">
                                {Object.entries(message.metadata.qualityMetrics).map(([metric, score]) => {
                                  if (metric === 'overall') return null;
                                  return (
                                    <div key={metric} className="text-xs">
                                      <div className="text-text-tertiary capitalize">{metric}</div>
                                      <div className="text-text-primary font-medium">
                                        {(score * 100).toFixed(0)}%
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Message Actions */}
                    <div className="flex items-center space-x-2 mt-2 ml-1">
                      <button
                        onClick={() => copyToClipboard(message.content)}
                        className="p-1 hover:bg-bg-tertiary rounded text-text-tertiary hover:text-text-primary transition-colors"
                        title="Copy message"
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                        </svg>
                      </button>

                      <div className="text-xs text-text-tertiary">
                        {message.timestamp.toLocaleTimeString()}
                      </div>

                      {message.metadata?.jobId && (
                        <div className="text-xs text-text-tertiary">
                          Job: {message.metadata.jobId.slice(-8)}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}

        {/* Active Processing Indicator */}
        {activeJobs.filter(job => job.status === 'in_progress').map((job) => (
          <div key={job.id} className="processing-indicator">
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-accent-primary to-accent-secondary rounded-full flex items-center justify-center animate-pulse">
                <span className="text-white text-xs font-bold">S1</span>
              </div>

              <div className="flex-1 min-w-0">
                <div className="bg-bg-surface border border-border-primary rounded-2xl px-4 py-3">
                  <div className="flex items-center space-x-2 mb-2">
                    <div className="animate-spin w-4 h-4 border-2 border-accent-primary border-t-transparent rounded-full"></div>
                    <span className="text-sm text-text-primary">
                      {job.currentAgent ? (
                        <>
                          {getAgentIcon(job.currentAgent)} {job.currentAgent} is processing...
                        </>
                      ) : (
                        'Processing your message...'
                      )}
                    </span>
                  </div>

                  {/* Progress Bar */}
                  <div className="w-full bg-bg-tertiary rounded-full h-2 mb-2">
                    <div 
                      className="bg-accent-primary h-2 rounded-full transition-all duration-500"
                      style={{ width: `${job.progress}%` }}
                    ></div>
                  </div>

                  <div className="flex justify-between text-xs text-text-tertiary">
                    <span>{job.progress.toFixed(0)}% complete</span>
                    <span>ETA: {formatProcessingTime(job.estimatedTimeRemaining)}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}

        {/* Empty State */}
        {messages.length === 0 && (
          <div className="empty-state text-center py-12">
            <div className="text-6xl mb-4">🤖</div>
            <h3 className="text-xl font-semibold text-text-primary mb-2">
              Start a conversation with SIM-ONE
            </h3>
            <p className="text-text-secondary mb-6">
              Experience governed cognition with our five-agent AI system
            </p>
            
            <div className="flex justify-center mb-6">
              <div className="flex items-center space-x-2">
                {['ideator', 'drafter', 'reviser', 'critic', 'summarizer'].map((agent, index) => (
                  <React.Fragment key={agent}>
                    <div className={`px-2 py-1 rounded-full text-xs font-medium ${getAgentColor(agent as AgentType)}`}>
                      {getAgentIcon(agent as AgentType)} {agent}
                    </div>
                    {index < 4 && <span className="text-text-tertiary">→</span>}
                  </React.Fragment>
                ))}
              </div>
            </div>

            <div className="text-sm text-text-tertiary">
              Each message is processed through our complete cognitive governance pipeline
            </div>
          </div>
        )}
      </div>
      
      {/* Scroll anchor */}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default ChatMessages;