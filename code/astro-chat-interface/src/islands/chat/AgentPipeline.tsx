import React, { useState, useEffect } from 'react';
import type { AgentType, ProcessingJob, AgentStep } from '../../lib/types/global';

interface AgentPipelineProps {
  jobId: string;
  isVisible: boolean;
  onComplete?: (result: any) => void;
  onError?: (error: any) => void;
}

const AgentPipeline: React.FC<AgentPipelineProps> = ({
  jobId,
  isVisible,
  onComplete,
  onError
}) => {
  const [currentJob, setCurrentJob] = useState<ProcessingJob | null>(null);
  const [agentSteps, setAgentSteps] = useState<AgentStep[]>([]);
  const [isExpanded, setIsExpanded] = useState(false);

  // Agent configuration with colors and descriptions
  const agents: Record<AgentType, { name: string; description: string; color: string; icon: string }> = {
    ideator: {
      name: 'Ideator',
      description: 'Generates initial concepts and creative ideas',
      color: 'text-blue-400 bg-blue-500/20',
      icon: '💡'
    },
    drafter: {
      name: 'Drafter', 
      description: 'Creates structured content drafts',
      color: 'text-green-400 bg-green-500/20',
      icon: '✏️'
    },
    reviser: {
      name: 'Reviser',
      description: 'Refines and improves content quality',
      color: 'text-yellow-400 bg-yellow-500/20', 
      icon: '🔄'
    },
    critic: {
      name: 'Critic',
      description: 'Evaluates quality and coherence',
      color: 'text-red-400 bg-red-500/20',
      icon: '🔍'
    },
    summarizer: {
      name: 'Summarizer',
      description: 'Produces final polished output',
      color: 'text-purple-400 bg-purple-500/20',
      icon: '✨'
    }
  };

  // Mock agent steps for demonstration
  useEffect(() => {
    if (!jobId || !isVisible) return;

    // Initialize mock agent steps
    const mockSteps: AgentStep[] = [
      {
        agent: 'ideator',
        name: 'Concept Generation',
        status: 'completed',
        progress: 100,
        startTime: new Date(Date.now() - 8000),
        endTime: new Date(Date.now() - 6000),
        output: 'Generated core concepts and creative framework',
        metrics: {
          duration: 2000,
          tokenUsage: 150,
          qualityScore: 0.92,
          memoryUsage: 45
        }
      },
      {
        agent: 'drafter',
        name: 'Content Drafting',
        status: 'completed', 
        progress: 100,
        startTime: new Date(Date.now() - 6000),
        endTime: new Date(Date.now() - 4000),
        output: 'Created structured content draft with key points',
        metrics: {
          duration: 2000,
          tokenUsage: 280,
          qualityScore: 0.88,
          memoryUsage: 67
        }
      },
      {
        agent: 'reviser',
        name: 'Content Revision',
        status: 'running',
        progress: 65,
        startTime: new Date(Date.now() - 4000),
        output: 'Enhancing clarity and flow...',
        metrics: {
          duration: 4000,
          tokenUsage: 210,
          qualityScore: 0.91,
          memoryUsage: 52
        }
      },
      {
        agent: 'critic',
        name: 'Quality Analysis',
        status: 'pending',
        progress: 0,
        metrics: {
          duration: 0,
          tokenUsage: 0,
          qualityScore: 0,
          memoryUsage: 0
        }
      },
      {
        agent: 'summarizer',
        name: 'Final Processing',
        status: 'pending',
        progress: 0,
        metrics: {
          duration: 0,
          tokenUsage: 0,
          qualityScore: 0,
          memoryUsage: 0
        }
      }
    ];

    setAgentSteps(mockSteps);

    // Mock job data
    setCurrentJob({
      id: jobId,
      input: 'Example user input',
      style: 'universal_chat',
      priority: 'balanced',
      status: 'in_progress',
      currentAgent: 'reviser',
      progress: 40,
      estimatedTimeRemaining: 8000,
      startedAt: new Date(Date.now() - 8000),
      metrics: {
        totalTime: 8000,
        agentTimes: {
          ideator: 2000,
          drafter: 2000,
          reviser: 4000,
          critic: 0,
          summarizer: 0
        },
        tokenUsage: {
          input: 50,
          output: 640,
          total: 690
        },
        qualityScore: 0.90,
        coherenceScore: 0.88,
        creativityScore: 0.85
      }
    });

  }, [jobId, isVisible]);

  const getStepStatus = (step: AgentStep) => {
    switch (step.status) {
      case 'completed':
        return '✅';
      case 'running':
        return '🔄';
      case 'failed':
        return '❌';
      default:
        return '⏳';
    }
  };

  const formatDuration = (ms: number) => {
    return `${(ms / 1000).toFixed(1)}s`;
  };

  const formatTime = (date: Date | undefined) => {
    if (!date) return '';
    return date.toLocaleTimeString();
  };

  if (!isVisible || !currentJob) {
    return null;
  }

  return (
    <div className="agent-pipeline bg-bg-surface border-b border-border-primary">
      {/* Pipeline Header */}
      <div className="pipeline-header p-4 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <h3 className="text-sm font-semibold text-text-primary">
            Agent Pipeline - {currentJob.style.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
          </h3>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-accent-primary rounded-full animate-pulse"></div>
            <span className="text-xs text-text-secondary">
              Processing ({currentJob.progress}%)
            </span>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <span className="text-xs text-text-tertiary">
            ETA: {formatDuration(currentJob.estimatedTimeRemaining)}
          </span>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1 hover:bg-bg-tertiary rounded text-text-secondary hover:text-text-primary transition-colors"
          >
            <svg 
              width="16" 
              height="16" 
              viewBox="0 0 24 24" 
              fill="none" 
              stroke="currentColor"
              className={`transform transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            >
              <path d="m6 9 6 6 6-6"/>
            </svg>
          </button>
        </div>
      </div>

      {/* Pipeline Progress Bar */}
      <div className="px-4 pb-2">
        <div className="w-full bg-bg-tertiary rounded-full h-2">
          <div 
            className="bg-accent-primary h-2 rounded-full transition-all duration-500"
            style={{ width: `${currentJob.progress}%` }}
          ></div>
        </div>
      </div>

      {/* Agent Steps */}
      <div className="agent-steps px-4 pb-4">
        <div className="flex items-center justify-between space-x-2 overflow-x-auto">
          {agentSteps.map((step, index) => {
            const agentConfig = agents[step.agent];
            const isActive = step.status === 'running';
            const isCompleted = step.status === 'completed';
            
            return (
              <div key={step.agent} className="flex items-center">
                <div 
                  className={`agent-step relative flex flex-col items-center min-w-0 ${
                    isActive ? 'scale-110' : ''
                  } transition-transform duration-200`}
                >
                  {/* Agent Icon */}
                  <div 
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium ${
                      agentConfig.color
                    } ${isActive ? 'animate-pulse' : ''}`}
                    title={agentConfig.description}
                  >
                    {agentConfig.icon}
                  </div>
                  
                  {/* Agent Name */}
                  <div className="mt-1 text-xs font-medium text-text-primary">
                    {agentConfig.name}
                  </div>
                  
                  {/* Status */}
                  <div className="mt-1 text-xs">
                    {getStepStatus(step)}
                  </div>
                  
                  {/* Progress for Running Agent */}
                  {step.status === 'running' && (
                    <div className="mt-1 w-12 bg-bg-tertiary rounded-full h-1">
                      <div 
                        className="bg-accent-primary h-1 rounded-full transition-all duration-300"
                        style={{ width: `${step.progress}%` }}
                      ></div>
                    </div>
                  )}
                </div>
                
                {/* Arrow */}
                {index < agentSteps.length - 1 && (
                  <div className="mx-2 text-text-tertiary">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path d="m9 18 6-6-6-6"/>
                    </svg>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="expanded-details border-t border-border-primary p-4 bg-bg-primary">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Current Agent Details */}
            <div>
              <h4 className="text-sm font-semibold text-text-primary mb-2">Current Agent</h4>
              {currentJob.currentAgent && (
                <div className="bg-bg-surface p-3 rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    <span className="text-lg">{agents[currentJob.currentAgent].icon}</span>
                    <span className="font-medium text-text-primary">
                      {agents[currentJob.currentAgent].name}
                    </span>
                  </div>
                  <p className="text-xs text-text-secondary mb-2">
                    {agents[currentJob.currentAgent].description}
                  </p>
                  <div className="text-xs text-text-tertiary">
                    {agentSteps.find(s => s.agent === currentJob.currentAgent)?.output}
                  </div>
                </div>
              )}
            </div>

            {/* Metrics */}
            <div>
              <h4 className="text-sm font-semibold text-text-primary mb-2">Processing Metrics</h4>
              <div className="bg-bg-surface p-3 rounded-lg space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-text-secondary">Total Time:</span>
                  <span className="text-text-primary">{formatDuration(currentJob.metrics.totalTime)}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-text-secondary">Tokens Used:</span>
                  <span className="text-text-primary">{currentJob.metrics.tokenUsage.total}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-text-secondary">Quality Score:</span>
                  <span className="text-text-primary">{(currentJob.metrics.qualityScore * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-text-secondary">Coherence:</span>
                  <span className="text-text-primary">{(currentJob.metrics.coherenceScore * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AgentPipeline;