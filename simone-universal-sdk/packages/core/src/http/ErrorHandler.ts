/**
 * @file Defines custom error types and an error handler for API responses.
 */

// Base error for all SDK-specific issues
export class SdkError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SdkError';
  }
}

// Specific error types for different failure scenarios
export class AuthenticationError extends SdkError {
  constructor(message: string = 'Authentication failed. Please check your API key or token.') {
    super(message);
    this.name = 'AuthenticationError';
  }
}

export class RateLimitError extends SdkError {
  constructor(message: string = 'Too many requests. Please try again later.') {
    super(message);
    this.name = 'RateLimitError';
  }
}

export class ServerError extends SdkError {
  constructor(message: string = 'An unexpected server error occurred.') {
    super(message);
    this.name = 'ServerError';
  }
}

export class NotFoundError extends SdkError {
  constructor(message: string = 'The requested resource was not found.') {
    super(message);
    this.name = 'NotFoundError';
  }
}

/**
 * Handles an HTTP response and throws a corresponding custom error if the
 * request was not successful.
 * @param response The raw HTTP response from the fetch call.
 */
export async function handleHttpError(response: Response): Promise<void> {
  if (response.ok) {
    return;
  }

  let errorMessage = `HTTP Error: ${response.status} ${response.statusText}`;
  try {
    const errorBody = await response.json();
    if (errorBody && errorBody.error) {
      errorMessage = errorBody.error;
    }
  } catch (e) {
    // Ignore if the error body is not valid JSON
  }

  switch (response.status) {
    case 401:
    case 403:
      throw new AuthenticationError(errorMessage);
    case 404:
      throw new NotFoundError(errorMessage);
    case 429:
      throw new RateLimitError(errorMessage);
    case 500:
    case 502:
    case 503:
    case 504:
      throw new ServerError(errorMessage);
    default:
      throw new SdkError(`An unexpected error occurred: ${errorMessage}`);
  }
}
