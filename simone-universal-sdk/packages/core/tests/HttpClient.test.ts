/**
 * @file Tests for the SimoneClient, covering HTTP logic, error handling, and security middleware.
 * This test suite is designed for a Jest-like environment.
 */

import { SimoneClient, SimoneClientConfig } from '../src/SimoneClient';
import { AuthenticationError, ServerError } from '../src/http/ErrorHandler';

// Mock the global fetch API before each test
const mockFetch = jest.fn();
global.fetch = mockFetch;

// Mock the TokenManager to control token state
jest.mock('../src/auth/TokenManager', () => {
  return {
    TokenManager: jest.fn().mockImplementation(() => ({
      getTokens: jest.fn().mockReturnValue(null),
      saveTokens: jest.fn(),
      clearTokens: jest.fn(),
    })),
  };
});

describe('SimoneClient', () => {
  let client: SimoneClient;
  const config: SimoneClientConfig = {
    mcpServerUrl: 'http://localhost:8000',
    apiKey: 'test-api-key',
  };

  beforeEach(() => {
    // Reset mocks before each test
    mockFetch.mockClear();
    client = new SimoneClient(config);
  });

  it('should execute a workflow successfully with an API key', async () => {
    const mockResponse = { session_id: '123', results: {}, error: null, execution_time_ms: 100 };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockResponse)),
      json: () => Promise.resolve(mockResponse),
    });

    const request = { initial_data: { user_input: 'Hello' } };
    const response = await client.executeWorkflow(request);

    expect(mockFetch).toHaveBeenCalledWith('http://localhost:8000/execute', {
      method: 'POST',
      body: JSON.stringify(request),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer test-api-key',
      },
    });
    expect(response).toEqual(mockResponse);
  });

  it('should throw a ServerError on a 500 response', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      text: () => Promise.resolve(JSON.stringify({ error: 'Server exploded' })),
      json: () => Promise.resolve({ error: 'Server exploded' }),
    });

    await expect(client.getProtocols()).rejects.toThrow(ServerError);
    await expect(client.getProtocols()).rejects.toThrow('Server exploded');
  });

  it('should throw an AuthenticationError on a 401 response', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      statusText: 'Unauthorized',
      text: () => Promise.resolve(JSON.stringify({ error: 'Invalid token' })),
      json: () => Promise.resolve({ error: 'Invalid token' }),
    });

    // Mock the security middleware to indicate refresh failed
    const securityMiddleware = (client as any).securityMiddleware;
    jest.spyOn(securityMiddleware, 'handleAuthError').mockResolvedValueOnce(false);

    await expect(client.getTemplates()).rejects.toThrow(AuthenticationError);
    await expect(client.getTemplates()).rejects.toThrow('Invalid token');
  });

  it('should not retry a request if the first attempt succeeds', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({ data: 'success' })),
      json: () => Promise.resolve({ data: 'success' }),
    });

    await client.getProtocols();
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  // This test simulates the token refresh and retry logic, which is a key feature.
  // Note: The handleAuthError is mocked to simulate a successful refresh.
  it('should retry the request once if an auth error occurs and token is refreshed', async () => {
    // First call fails with 401
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      statusText: 'Unauthorized',
      text: () => Promise.resolve(JSON.stringify({ error: 'Token expired' })),
      json: () => Promise.resolve({ error: 'Token expired' }),
    });

    // Second call (the retry) succeeds
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({ data: 'success after retry' })),
      json: () => Promise.resolve({ data: 'success after retry' }),
    });

    // Mock the security middleware to indicate refresh was successful
    const securityMiddleware = (client as any).securityMiddleware;
    jest.spyOn(securityMiddleware, 'handleAuthError').mockResolvedValueOnce(true);

    const response = await client.getProtocols();

    // Check that fetch was called twice
    expect(mockFetch).toHaveBeenCalledTimes(2);
    // Check that the final response is from the successful second call
    expect(response).toEqual({ data: 'success after retry' });
  });
});
