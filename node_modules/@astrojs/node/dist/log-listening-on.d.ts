import type http from 'node:http';
import https from 'node:https';
import type { AstroIntegrationLogger } from 'astro';
import type { Options } from './types.js';
export declare function logListeningOn(logger: AstroIntegrationLogger, server: http.Server | https.Server, options: Pick<Options, 'host'>): Promise<void>;
interface NetworkAddressOpt {
    local: string[];
    network: string[];
}
export declare function getNetworkAddress(protocol: ("http" | "https") | undefined, hostname: string | undefined, port: number, base?: string): NetworkAddressOpt;
export {};
