import type { IncomingMessage, ServerResponse } from 'node:http';
import type { NodeApp } from 'astro/app/node';
import type { Options } from './types.js';
/**
 * Creates a Node.js http listener for static files and prerendered pages.
 * In standalone mode, the static handler is queried first for the static files.
 * If one matching the request path is not found, it relegates to the SSR handler.
 * Intended to be used only in the standalone mode.
 */
export declare function createStaticHandler(app: NodeApp, options: Options): (req: IncomingMessage, res: ServerResponse, ssr: () => unknown) => ServerResponse<IncomingMessage> | undefined;
