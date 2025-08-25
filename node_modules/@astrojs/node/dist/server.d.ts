import type { SSRManifest } from 'astro';
import type { Options } from './types.js';
export declare function createExports(manifest: SSRManifest, options: Options): {
    options: Options;
    handler: import("./types.js").RequestHandler;
    startServer: () => {
        server: {
            host: string;
            port: number;
            closed(): Promise<void>;
            stop(): Promise<void>;
            server: import("http").Server<typeof import("http").IncomingMessage, typeof import("http").ServerResponse> | import("https").Server<typeof import("http").IncomingMessage, typeof import("http").ServerResponse>;
        };
        done: Promise<void>;
    };
};
export declare function start(manifest: SSRManifest, options: Options): void;
