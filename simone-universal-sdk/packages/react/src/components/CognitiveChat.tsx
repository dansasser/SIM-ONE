import React, { useState } from 'react';
import { useCognitiveWorkflow } from '../hooks/useCognitiveWorkflow';
import { SimoneClient } from '@simone/core';

interface CognitiveChatProps {
  client: SimoneClient | null;
  workflowTemplate: string;
  sessionId?: string;
}

export const CognitiveChat: React.FC<CognitiveChatProps> = ({ client, workflowTemplate, sessionId }) => {
  const [inputValue, setInputValue] = useState('');
  const { result, loading, error, execute } = useCognitiveWorkflow(client, sessionId);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    execute({
      template_name: workflowTemplate,
      initial_data: { user_input: inputValue },
    });

    setInputValue('');
  };

  return (
    <div style={{ fontFamily: 'sans-serif', maxWidth: '600px', margin: 'auto' }}>
      <div style={{ border: '1px solid #ccc', padding: '10px', minHeight: '300px', marginBottom: '10px' }}>
        {/* This is a very basic display. A real app would map over a list of messages. */}
        {loading && <p><i>Thinking...</i></p>}
        {error && <p style={{ color: 'red' }}>Error: {error.message}</p>}
        {result && (
          <div>
            <p><b>You:</b> {result.results.user_input}</p>
            <p><b>Simone:</b> {result.results.REP?.conclusion || 'No conclusion found.'}</p>
          </div>
        )}
      </div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Ask Simone..."
          disabled={loading}
          style={{ width: '80%', padding: '8px' }}
        />
        <button type="submit" disabled={loading} style={{ width: '18%', padding: '8px' }}>
          {loading ? '...' : 'Send'}
        </button>
      </form>
    </div>
  );
};
