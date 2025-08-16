import React, { useState } from 'react';
import { useCognitiveWorkflow } from '../hooks/useCognitiveWorkflow';
import { SimoneClient } from '@simone/core';
import { WorkflowRequest } from '@simone/core/api';

interface WorkflowExecutorProps {
  client: SimoneClient | null;
  sessionId?: string;
}

export const WorkflowExecutor: React.FC<WorkflowExecutorProps> = ({ client, sessionId }) => {
  const [workflowRequest, setWorkflowRequest] = useState<Omit<WorkflowRequest, 'session_id'>>({
    template_name: 'StandardReasoningWorkflow',
    initial_data: { user_input: 'Tell me about photosynthesis.' },
  });

  const { result, loading, error, execute } = useCognitiveWorkflow(client, sessionId);

  const handleExecute = () => {
    execute(workflowRequest);
  };

  const handleRequestChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    try {
      const parsed = JSON.parse(e.target.value);
      setWorkflowRequest(parsed);
    } catch (err) {
      // Ignore parse errors while typing
    }
  };

  return (
    <div style={{ fontFamily: 'sans-serif', display: 'flex', gap: '20px' }}>
      <div style={{ flex: 1 }}>
        <h3>Workflow Request</h3>
        <textarea
          value={JSON.stringify(workflowRequest, null, 2)}
          onChange={handleRequestChange}
          rows={15}
          style={{ width: '100%', fontFamily: 'monospace' }}
        />
        <button onClick={handleExecute} disabled={loading} style={{ marginTop: '10px', padding: '10px' }}>
          {loading ? 'Executing...' : 'Execute Workflow'}
        </button>
      </div>
      <div style={{ flex: 1 }}>
        <h3>Workflow Response</h3>
        {loading && <p><i>Loading...</i></p>}
        {error && <pre style={{ color: 'red', whiteSpace: 'pre-wrap' }}>{error.stack}</pre>}
        {result && (
          <pre style={{ backgroundColor: '#f0f0f0', padding: '10px', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
            {JSON.stringify(result, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
};
