import { AuthError, HttpError, NotFoundError, RateLimitError, ServerError, TimeoutError } from './errors';

export interface HttpClientOptions {
  baseUrl: string;
  headers?: Record<string, string>;
  timeoutMs?: number;
  fetchImpl?: typeof fetch;
}

export class HttpClient {
  private baseUrl: string;
  private headers: Record<string, string>;
  private timeoutMs: number;
  private fetchImpl: typeof fetch;

  constructor(opts: HttpClientOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/$/, '');
    this.headers = opts.headers ?? {};
    this.timeoutMs = opts.timeoutMs ?? 20000;
    this.fetchImpl = opts.fetchImpl ?? (globalThis.fetch as any);
    if (!this.fetchImpl) throw new Error('Fetch implementation not available. Provide fetchImpl in HttpClientOptions.');
  }

  async request<T>(method: string, path: string, body?: any, headers?: Record<string, string>): Promise<T> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const res = await this.fetchImpl(`${this.baseUrl}${path}`, {
        method,
        headers: { 'Content-Type': 'application/json', ...this.headers, ...(headers ?? {}) },
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: controller.signal
      } as any);

      const contentType = res.headers.get('content-type') || '';
      const text = await res.text();
      const data = contentType.includes('application/json') && text ? JSON.parse(text) : text;

      if (!res.ok) {
        const msg = (data && (data.message || data.detail || data.error)) || res.statusText;
        if (res.status === 403 || res.status === 401) throw new AuthError(res.status, msg, data);
        if (res.status === 404) throw new NotFoundError(msg, data);
        if (res.status === 429) {
          const ra = res.headers.get('retry-after');
          throw new RateLimitError(msg, data, ra ? parseInt(ra, 10) : undefined);
        }
        if (res.status >= 500) throw new ServerError(res.status, msg, data);
        throw new HttpError(res.status, msg, data);
      }
      return data as T;
    } catch (err: any) {
      if (err?.name === 'AbortError') throw new TimeoutError();
      if (err instanceof HttpError) throw err;
      throw new ServerError(500, err?.message || 'Network error');
    } finally { clearTimeout(timeout); }
  }

  get<T>(path: string, headers?: Record<string, string>) { return this.request<T>('GET', path, undefined, headers); }
  post<T>(path: string, body?: any, headers?: Record<string, string>) { return this.request<T>('POST', path, body, headers); }
}

