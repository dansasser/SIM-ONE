/**
 * @file This file contains the core SDK client for interacting with the MCP server.
 */

import { WorkflowRequest, WorkflowResponse } from './api/McpServerApi';
import { handleHttpError, SdkError, AuthenticationError } from './http/ErrorHandler';
import { SecurityMiddleware } from './http/SecurityMiddleware';
import { TokenManager } from './auth/TokenManager';

export interface SimoneClientConfig {
  mcpServerUrl: string;
  apiKey: string;
  // OAuth config will be added here in a future step
}

/**
 * The main client for interacting with the SIM-ONE mCP Server.
 */
export class SimoneClient {
  private config: SimoneClientConfig;
  private securityMiddleware: SecurityMiddleware;
  private tokenManager: TokenManager;

  constructor(config: SimoneClientConfig) {
    if (!config.mcpServerUrl || !config.apiKey) {
      throw new Error('mcpServerUrl and apiKey are required configuration properties.');
    }
    this.config = config;
    this.tokenManager = new TokenManager();
    this.securityMiddleware = new SecurityMiddleware(this.config, this.tokenManager);
  }

  /**
   * A private helper method to handle fetch requests with security and retries.
   */
  private async _fetch(endpoint: string, options: RequestInit = {}, retry: boolean = true): Promise<any> {
    const url = `${this.config.mcpServerUrl}${endpoint}`;

    // Use the middleware to process the request and add auth headers
    const processedOptions = await this.securityMiddleware.processRequest(options);

    try {
      const response = await fetch(url, processedOptions);
      await handleHttpError(response);
      const text = await response.text();
      return text ? JSON.parse(text) : {};
    } catch (error) {
      // If it's an auth error, try to refresh the token and retry ONCE
      if (error instanceof AuthenticationError && retry) {
        const refreshed = await this.securityMiddleware.handleAuthError(error);
        if (refreshed) {
          console.log('Retrying request with new token...');
          return this._fetch(endpoint, options, false); // Set retry to false to prevent infinite loops
        }
      }

      if (error instanceof SdkError) {
        throw error;
      }
      throw new SdkError(`Network request failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  public async executeWorkflow(request: WorkflowRequest): Promise<WorkflowResponse> {
    return this._fetch('/execute', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  public async getProtocols(): Promise<any> {
    return this._fetch('/protocols', { method: 'GET' });
  }

  public async getTemplates(): Promise<any> {
    return this._fetch('/templates', { method: 'GET' });
  }

  public async getSession(sessionId: string): Promise<any> {
    return this._fetch(`/session/${sessionId}`, { method: 'GET' });
  }
}
