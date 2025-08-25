import http from 'node:http';
import https from 'node:https';
import type { NodeApp } from 'astro/app/node';
import type { Options } from './types.js';
export declare const hostOptions: (host: Options["host"]) => string;
export default function standalone(app: NodeApp, options: Options): {
    server: {
        host: string;
        port: number;
        closed(): Promise<void>;
        stop(): Promise<void>;
        server: http.Server<typeof http.IncomingMessage, typeof http.ServerResponse> | https.Server<typeof http.IncomingMessage, typeof http.ServerResponse>;
    };
    done: Promise<void>;
};
export declare function createStandaloneHandler(app: NodeApp, options: Options): (req: http.IncomingMessage, res: http.ServerResponse) => void;
export declare function createServer(listener: http.RequestListener, host: string, port: number): {
    host: string;
    port: number;
    closed(): Promise<void>;
    stop(): Promise<void>;
    server: http.Server<typeof http.IncomingMessage, typeof http.ServerResponse> | https.Server<typeof http.IncomingMessage, typeof http.ServerResponse>;
};
