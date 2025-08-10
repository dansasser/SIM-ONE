/**
 * @file This file contains the TypeScript interfaces that directly correspond to the
 * API data structures of the MCP server. They ensure type safety and exact
 * compatibility between the SDK and the server.
 */

/**
 * Defines the structure for a request to the `/execute` endpoint of the MCP server.
 */
export interface WorkflowRequest {
  /**
   * The name of a predefined workflow template to execute.
   * @optional
   */
  template_name?: string;

  /**
   * A list of protocol names to execute dynamically.
   * @optional
   */
  protocol_names?: string[];

  /**
   * The coordination mode for dynamic protocols.
   * @default 'Sequential'
   * @optional
   */
  coordination_mode?: 'Sequential' | 'Parallel';

  /**
   * The initial data payload for the workflow. Must contain at least a 'user_input' field.
   */
  initial_data: Record<string, any> & { user_input: string };

  /**
   * An existing session ID to continue a conversation. If omitted, a new session
   * will be created by the server.
   * @optional
   */
  session_id?: string;
}

/**
 * Defines the structure of a successful response from the `/execute` endpoint.
 */
export interface WorkflowResponse {
  /**
   * The session ID for the conversation, either continued or newly created.
   */
  session_id: string;

  /**
   * The final, aggregated data context after the workflow has completed.
   * The contents of this object will vary depending on the protocols executed.
   */
  results: Record<string, any>;

  /**
   * If an error occurred during execution, this field will contain a
   * descriptive error message. It is `null` on success.
   */
  error: string | null;

  /**
   * The total time taken for the workflow execution on the server, in milliseconds.
   */
  execution_time_ms: number;
}
