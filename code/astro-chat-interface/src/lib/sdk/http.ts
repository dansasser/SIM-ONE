export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE';

export interface HttpClientOptions {
  baseUrl?: string;
  timeoutMs?: number;
  defaultHeaders?: Record<string, string>;
}

export class HttpError extends Error {
  status: number;
  data: any;
  constructor(status: number, message: string, data?: any) {
    super(message);
    this.name = 'HttpError';
    this.status = status;
    this.data = data;
  }
}

export class HttpClient {
  private baseUrl: string;
  private timeoutMs: number;
  private defaultHeaders: Record<string, string>;

  constructor(opts: HttpClientOptions = {}) {
    this.baseUrl = opts.baseUrl ?? '/api';
    this.timeoutMs = opts.timeoutMs ?? 20000;
    this.defaultHeaders = opts.defaultHeaders ?? { 'Content-Type': 'application/json' };
  }

  async request<T>(method: HttpMethod, path: string, body?: any, headers: Record<string, string> = {}): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        method,
        headers: { ...this.defaultHeaders, ...headers },
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      const text = await res.text();
      const contentType = res.headers.get('content-type') || '';
      const data = contentType.includes('application/json') ? (text ? JSON.parse(text) : undefined) : text;

      if (!res.ok) {
        throw new HttpError(res.status, (data && (data.message || data.detail || data.error)) || res.statusText, data);
      }
      return data as T;
    } catch (err: any) {
      if (err.name === 'AbortError') {
        throw new HttpError(408, 'Request timeout');
      }
      if (err instanceof HttpError) throw err;
      throw new HttpError(500, err?.message || 'Network error');
    } finally {
      clearTimeout(timer);
    }
  }

  get<T>(path: string, headers?: Record<string, string>) { return this.request<T>('GET', path, undefined, headers); }
  post<T>(path: string, body?: any, headers?: Record<string, string>) { return this.request<T>('POST', path, body, headers); }
}

