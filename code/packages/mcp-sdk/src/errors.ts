export class HttpError extends Error {
  status: number;
  data?: any;
  code: string;
  constructor(status: number, message: string, data?: any, code = 'HTTP_ERROR') {
    super(message);
    this.name = 'HttpError';
    this.status = status;
    this.data = data;
    this.code = code;
  }
}

export class TimeoutError extends HttpError {
  constructor(message = 'Request timeout') { super(408, message, undefined, 'TIMEOUT'); }
}

export class AuthError extends HttpError {
  constructor(status = 403, message = 'Not authorized', data?: any) { super(status, message, data, 'AUTH'); }
}

export class NotFoundError extends HttpError {
  constructor(message = 'Not found', data?: any) { super(404, message, data, 'NOT_FOUND'); }
}

export class RateLimitError extends HttpError {
  retryAfter?: number;
  constructor(message = 'Rate limited', data?: any, retryAfter?: number) { super(429, message, data, 'RATE_LIMIT'); this.retryAfter = retryAfter; }
}

export class ServerError extends HttpError {
  constructor(status = 500, message = 'Server error', data?: any) { super(status, message, data, 'SERVER'); }
}

