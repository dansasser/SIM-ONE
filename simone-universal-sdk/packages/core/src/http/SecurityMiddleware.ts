/**
 * @file Implements a security middleware/interceptor for handling authentication.
 */

import { TokenManager } from '../auth/TokenManager';
import { SimoneClientConfig } from '../SimoneClient';

export class SecurityMiddleware {
  private tokenManager: TokenManager;
  private config: SimoneClientConfig;

  constructor(config: SimoneClientConfig, tokenManager: TokenManager) {
    this.config = config;
    this.tokenManager = tokenManager;
  }

  /**
   * Processes a request to add the appropriate authentication header.
   * This acts as a request interceptor.
   * @param options The initial RequestInit options.
   * @returns The updated RequestInit options with the Authorization header.
   */
  public async processRequest(options: RequestInit): Promise<RequestInit> {
    const newOptions = { ...options };
    if (!newOptions.headers) {
      newOptions.headers = {};
    }

    // Prioritize OAuth token if it exists
    const oauthTokens = this.tokenManager.getTokens();
    if (oauthTokens && oauthTokens.accessToken) {
      (newOptions.headers as Record<string, string>)['Authorization'] = `Bearer ${oauthTokens.accessToken}`;
      console.log('Using OAuth token for authentication.');
    } else {
      // Fallback to API key
      (newOptions.headers as Record<string, string>)['Authorization'] = `Bearer ${this.config.apiKey}`;
      console.log('Using API key for authentication.');
    }

    return newOptions;
  }

  /**
   * Handles an authentication error, attempting to refresh the token if possible.
   * This acts as a response interceptor for errors.
   * @param error The error received from the initial request.
   * @returns A boolean indicating if the token was successfully refreshed (and the request can be retried).
   */
  public async handleAuthError(error: any): Promise<boolean> {
    // Check if it's an auth error and if a refresh token is available
    const tokens = this.tokenManager.getTokens();
    if (error.name === 'AuthenticationError' && tokens && tokens.refreshToken) {
      console.log('Authentication failed. Attempting to refresh token...');

      // In a real implementation, you would call your OAuth server's refresh endpoint here.
      // For now, we'll simulate a successful refresh for demonstration purposes.
      try {
        // const newTokens = await this.refreshToken(tokens.refreshToken);
        // this.tokenManager.saveTokens(newTokens);
        console.log('Token refresh simulation successful.');
        return true; // Indicate that the request should be retried
      } catch (refreshError) {
        console.error('Failed to refresh token:', refreshError);
        this.tokenManager.clearTokens(); // Clear invalid tokens
        return false;
      }
    }

    return false; // Cannot handle the error
  }
}
