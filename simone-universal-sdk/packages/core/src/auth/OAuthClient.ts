/**
 * @file Implements the client-side logic for the OAuth 2.0 Authorization Code Flow with PKCE.
 */

import { TokenManager, AuthTokens } from './TokenManager';

export interface OAuthClientConfig {
  authorizationUrl: string;
  tokenUrl: string;
  clientId: string;
  redirectUri: string;
  scopes: string[];
}

export class OAuthClient {
  private config: OAuthClientConfig;
  private tokenManager: TokenManager;
  private pkceState: { verifier: string; challenge: string; state: string } | null = null;

  constructor(config: OAuthClientConfig, tokenManager: TokenManager) {
    this.config = config;
    this.tokenManager = tokenManager;
  }

  /**
   * Generates the PKCE code verifier, challenge, and a state parameter.
   */
  private async generatePkce(): Promise<void> {
    const verifier = this.generateRandomString(128);
    const challenge = await this.sha256(verifier);
    const challengeBase64 = this.base64urlencode(challenge);

    this.pkceState = {
      verifier,
      challenge: challengeBase64,
      state: this.generateRandomString(32),
    };

    // Store PKCE state temporarily to survive the redirect
    sessionStorage.setItem('pkce_verifier', this.pkceState.verifier);
    sessionStorage.setItem('oauth_state', this.pkceState.state);
  }

  /**
   * Constructs and redirects the user to the authorization URL.
   */
  public async redirectToAuthorizationUrl(): Promise<void> {
    await this.generatePkce();
    if (!this.pkceState) {
      throw new Error('Failed to generate PKCE state.');
    }

    const params = new URLSearchParams({
      response_type: 'code',
      client_id: this.config.clientId,
      redirect_uri: this.config.redirectUri,
      scope: this.config.scopes.join(' '),
      state: this.pkceState.state,
      code_challenge: this.pkceState.challenge,
      code_challenge_method: 'S256',
    });

    window.location.assign(`${this.config.authorizationUrl}?${params.toString()}`);
  }

  /**
   * Handles the redirect back from the authorization server, exchanges the code for tokens.
   * @param url The full redirect URL containing the code and state.
   */
  public async handleRedirect(url: string): Promise<AuthTokens> {
    const params = new URLSearchParams(new URL(url).search);
    const code = params.get('code');
    const state = params.get('state');

    const storedState = sessionStorage.getItem('oauth_state');
    const storedVerifier = sessionStorage.getItem('pkce_verifier');

    sessionStorage.removeItem('oauth_state');
    sessionStorage.removeItem('pkce_verifier');

    if (!code || !state || !storedState || !storedVerifier) {
      throw new Error('Invalid redirect: missing code, state, or verifier.');
    }
    if (state !== storedState) {
      throw new Error('Invalid state parameter. Possible CSRF attack.');
    }

    const tokenResponse = await fetch(this.config.tokenUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'authorization_code',
        code: code,
        redirect_uri: this.config.redirectUri,
        client_id: this.config.clientId,
        code_verifier: storedVerifier,
      }),
    });

    if (!tokenResponse.ok) {
      throw new Error(`Failed to exchange token: ${await tokenResponse.text()}`);
    }

    const responseJson = await tokenResponse.json();
    const tokens: AuthTokens = {
      accessToken: responseJson.access_token,
      refreshToken: responseJson.refresh_token,
      expiresAt: Date.now() + responseJson.expires_in * 1000,
    };

    this.tokenManager.saveTokens(tokens);
    return tokens;
  }

  // --- PKCE Helper Functions ---

  private generateRandomString(length: number): string {
    const array = new Uint32Array(length / 2);
    window.crypto.getRandomValues(array);
    return Array.from(array, (dec) => ('0' + dec.toString(16)).substr(-2)).join('');
  }

  private async sha256(plain: string): Promise<ArrayBuffer> {
    const encoder = new TextEncoder();
    const data = encoder.encode(plain);
    return window.crypto.subtle.digest('SHA-256', data);
  }

  private base64urlencode(buffer: ArrayBuffer): string {
    return btoa(String.fromCharCode.apply(null, Array.from(new Uint8Array(buffer))))
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');
  }
}
