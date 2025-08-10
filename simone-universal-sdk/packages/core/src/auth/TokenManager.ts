/**
 * @file Manages the secure storage and retrieval of authentication tokens.
 */

export interface AuthTokens {
  accessToken: string;
  refreshToken?: string;
  expiresAt: number; // Timestamp in milliseconds
}

/**
 * Handles the storage and retrieval of authentication tokens in a secure manner.
 * This implementation uses browser localStorage. For production, consider
 * more secure alternatives depending on the environment.
 */
export class TokenManager {
  private storageKey: string;

  constructor(storageKey: string = 'simone_sdk_auth_tokens') {
    this.storageKey = storageKey;
  }

  /**
   * Saves the authentication tokens to storage.
   * @param tokens The tokens to save.
   */
  public saveTokens(tokens: AuthTokens): void {
    try {
      const tokenData = JSON.stringify(tokens);
      window.localStorage.setItem(this.storageKey, tokenData);
    } catch (error) {
      console.error('Error saving tokens to localStorage:', error);
    }
  }

  /**
   * Retrieves the authentication tokens from storage.
   * @returns The stored tokens, or null if not found or expired.
   */
  public getTokens(): AuthTokens | null {
    try {
      const tokenData = window.localStorage.getItem(this.storageKey);
      if (!tokenData) {
        return null;
      }

      const tokens: AuthTokens = JSON.parse(tokenData);

      // Check for expiration
      if (Date.now() >= tokens.expiresAt) {
        this.clearTokens();
        console.log('Tokens have expired and were cleared.');
        return null;
      }

      return tokens;
    } catch (error) {
      console.error('Error retrieving tokens from localStorage:', error);
      return null;
    }
  }

  /**
   * Clears all authentication tokens from storage.
   */
  public clearTokens(): void {
    try {
      window.localStorage.removeItem(this.storageKey);
    } catch (error) {
      console.error('Error clearing tokens from localStorage:', error);
    }
  }
}
