import { useState, useCallback } from 'react';
import { SimoneClient } from '@simone/core'; // Assuming a monorepo setup with path mapping
import { WorkflowRequest, WorkflowResponse } from '@simone/core/api';

export interface UseCognitiveWorkflowResult {
  result: WorkflowResponse | null;
  loading: boolean;
  error: Error | null;
  execute: (request: Omit<WorkflowRequest, 'session_id'>) => Promise<void>;
}

/**
 * A React hook for executing cognitive workflows with the SimoneClient.
 *
 * @param client - An instance of the SimoneClient.
 * @param sessionId - An optional session ID to maintain conversational context.
 * @returns An object with the workflow result, loading state, error state, and an execute function.
 */
export const useCognitiveWorkflow = (
  client: SimoneClient | null,
  sessionId?: string
): UseCognitiveWorkflowResult => {
  const [result, setResult] = useState<WorkflowResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);

  const execute = useCallback(
    async (request: Omit<WorkflowRequest, 'session_id'>) => {
      if (!client) {
        setError(new Error('SimoneClient is not initialized.'));
        return;
      }

      setLoading(true);
      setError(null);
      setResult(null);

      try {
        const fullRequest: WorkflowRequest = {
          ...request,
          session_id: sessionId,
        };
        const response = await client.executeWorkflow(fullRequest);
        setResult(response);
      } catch (e: any) {
        setError(e);
      } finally {
        setLoading(false);
      }
    },
    [client, sessionId]
  );

  return { result, loading, error, execute };
};
