import type { NodeApp } from 'astro/app/node';
import type { RequestHandler } from './types.js';
/**
 * Creates a middleware that can be used with Express, Connect, etc.
 *
 * Similar to `createAppHandler` but can additionally be placed in the express
 * chain as an error middleware.
 *
 * https://expressjs.com/en/guide/using-middleware.html#middleware.error-handling
 */
export default function createMiddleware(app: NodeApp): RequestHandler;
