import React, { useState, useRef, useEffect } from 'react';
import type { ProcessingStyle, Priority, MessageOptions } from '../../lib/types/global';
import { useMessages, useProcessing } from '../../lib/hooks/useSimoneSDK';

interface ChatInputProps {
  conversationId?: string | null;
  placeholder?: string;
  maxLength?: number;
}

const ChatInput: React.FC<ChatInputProps> = ({
  conversationId,
  placeholder = "Message SIM-ONE...",
  maxLength = 4000
}) => {
  const { sendMessage, error } = useMessages(conversationId);
  const { isProcessing } = useProcessing();
  const [message, setMessage] = useState('');
  const [selectedStyle, setSelectedStyle] = useState<ProcessingStyle>('universal_chat');
  const [selectedPriority, setSelectedPriority] = useState<Priority>('balanced');
  const [showOptions, setShowOptions] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
    }
  }, [message]);

  // Listen for example prompt events
  useEffect(() => {
    const handleExamplePrompt = (event: CustomEvent) => {
      setMessage(event.detail.prompt);
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    };

    window.addEventListener('use-example-prompt', handleExamplePrompt as EventListener);
    return () => {
      window.removeEventListener('use-example-prompt', handleExamplePrompt as EventListener);
    };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!message.trim() || isProcessing || !conversationId) return;
    
    const options: MessageOptions = {
      style: selectedStyle,
      priority: selectedPriority
    };
    
    try {
      await sendMessage(message.trim(), options);
      setMessage('');
      setShowOptions(false);
    } catch (err) {
      console.error('Failed to send message:', err);
      // Error handling could show a toast notification
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleStyleChange = (style: ProcessingStyle) => {
    setSelectedStyle(style);
    setShowOptions(false);
  };

  const handlePriorityChange = (priority: Priority) => {
    setSelectedPriority(priority);
    setShowOptions(false);
  };

  const processingStyles: { value: ProcessingStyle; label: string; description: string }[] = [
    { value: 'universal_chat', label: 'Universal Chat', description: 'General conversation and Q&A' },
    { value: 'analytical_article', label: 'Analytical Article', description: 'Technical analysis and research' },
    { value: 'creative_writing', label: 'Creative Writing', description: 'Stories, poems, creative content' },
    { value: 'academic_paper', label: 'Academic Paper', description: 'Formal academic writing' },
    { value: 'business_report', label: 'Business Report', description: 'Professional business communication' },
    { value: 'technical_documentation', label: 'Technical Documentation', description: 'Technical guides and manuals' }
  ];

  const priorities: { value: Priority; label: string; description: string; time: string }[] = [
    { value: 'fast', label: 'Fast', description: 'Optimized for speed', time: '2-5s' },
    { value: 'balanced', label: 'Balanced', description: 'Balanced speed and quality', time: '3-10s' },
    { value: 'quality', label: 'Quality', description: 'Maximum quality processing', time: '5-20s' }
  ];

  const currentStyleLabel = processingStyles.find(s => s.value === selectedStyle)?.label || 'Universal Chat';
  const currentPriorityLabel = priorities.find(p => p.value === selectedPriority)?.label || 'Balanced';

  return (
    <div className="chat-input-component">
      <form onSubmit={handleSubmit} className="flex flex-col">
        {/* Options Panel */}
        {showOptions && (
          <div className="options-panel bg-bg-surface border border-border-primary rounded-lg mb-3 p-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Processing Style */}
              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">
                  Processing Style
                </label>
                <div className="space-y-2">
                  {processingStyles.map((style) => (
                    <button
                      key={style.value}
                      type="button"
                      onClick={() => handleStyleChange(style.value)}
                      className={`w-full text-left p-2 rounded-md transition-colors ${
                        selectedStyle === style.value
                          ? 'bg-accent-primary text-white'
                          : 'bg-bg-tertiary hover:bg-bg-secondary text-text-primary'
                      }`}
                    >
                      <div className="font-medium text-sm">{style.label}</div>
                      <div className="text-xs opacity-75">{style.description}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Priority */}
              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">
                  Processing Priority
                </label>
                <div className="space-y-2">
                  {priorities.map((priority) => (
                    <button
                      key={priority.value}
                      type="button"
                      onClick={() => handlePriorityChange(priority.value)}
                      className={`w-full text-left p-2 rounded-md transition-colors ${
                        selectedPriority === priority.value
                          ? 'bg-accent-primary text-white'
                          : 'bg-bg-tertiary hover:bg-bg-secondary text-text-primary'
                      }`}
                    >
                      <div className="flex justify-between items-center">
                        <div className="font-medium text-sm">{priority.label}</div>
                        <div className="text-xs opacity-75">{priority.time}</div>
                      </div>
                      <div className="text-xs opacity-75">{priority.description}</div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Input Area */}
        <div className="input-container flex items-end space-x-2">
          {/* Settings Button */}
          <button
            type="button"
            onClick={() => setShowOptions(!showOptions)}
            className={`flex-shrink-0 p-2 rounded-md transition-colors ${
              showOptions 
                ? 'bg-accent-primary text-white' 
                : 'bg-bg-tertiary hover:bg-bg-secondary text-text-secondary hover:text-text-primary'
            }`}
            title="Processing options"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <circle cx="12" cy="12" r="3"></circle>
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1 1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
            </svg>
          </button>

          {/* Textarea Container */}
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              disabled={isProcessing}
              maxLength={maxLength}
              className="w-full px-3 py-2 pr-12 bg-bg-surface border border-border-primary rounded-lg text-text-primary placeholder-text-tertiary resize-none focus:outline-none focus:border-border-focus transition-colors min-h-[44px] max-h-[120px]"
              rows={1}
            />
            
            {/* Character Count */}
            <div className="absolute bottom-1 right-12 text-xs text-text-tertiary">
              {message.length}/{maxLength}
            </div>
          </div>

          {/* Send Button */}
          <button
            type="submit"
            disabled={!message.trim() || isProcessing}
            className="flex-shrink-0 p-2 bg-accent-primary hover:bg-accent-secondary disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg transition-colors"
            title="Send message"
          >
            {isProcessing ? (
              <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            ) : (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="m22 2-7 20-4-9-9-4 20-7z"/>
              </svg>
            )}
          </button>
        </div>

        {/* Quick Settings Display */}
        {!showOptions && (
          <div className="flex items-center justify-between mt-2 text-xs text-text-tertiary">
            <div className="flex items-center space-x-4">
              <span>Style: <strong>{currentStyleLabel}</strong></span>
              <span>Priority: <strong>{currentPriorityLabel}</strong></span>
            </div>
            <div>Press Enter to send, Shift+Enter for new line</div>
          </div>
        )}
      </form>
    </div>
  );
};

export default ChatInput;