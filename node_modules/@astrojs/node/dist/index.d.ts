import type { AstroAdapter, AstroIntegration } from 'astro';
import type { Options, UserOptions } from './types.js';
export declare function getAdapter(options: Options): AstroAdapter;
export default function createIntegration(userOptions: UserOptions): AstroIntegration;
