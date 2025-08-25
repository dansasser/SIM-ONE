import { fileURLToPath } from 'node:url';
import { AstroError } from 'astro/errors';
import { logListeningOn } from './log-listening-on.js';
import { createServer } from './standalone.js';
const createPreviewServer = async (preview) => {
    let ssrHandler;
    let options;
    try {
        process.env.ASTRO_NODE_AUTOSTART = 'disabled';
        const ssrModule = await import(preview.serverEntrypoint.toString());
        if (typeof ssrModule.handler === 'function') {
            ssrHandler = ssrModule.handler;
            // biome-ignore lint/style/noNonNullAssertion: <explanation>
            options = ssrModule.options;
        }
        else {
            throw new AstroError(`The server entrypoint doesn't have a handler. Are you sure this is the right file?`);
        }
    }
    catch (err) {
        if (err.code === 'ERR_MODULE_NOT_FOUND') {
            throw new AstroError(`The server entrypoint ${fileURLToPath(preview.serverEntrypoint)} does not exist. Have you ran a build yet?`);
            // biome-ignore lint/style/noUselessElse: <explanation>
        }
        else {
            throw err;
        }
    }
    const host = preview.host ?? 'localhost';
    const port = preview.port ?? 4321;
    const server = createServer(ssrHandler, host, port);
    // If user specified custom headers append a listener
    // to the server to add those headers to response
    if (preview.headers) {
        server.server.addListener('request', (_, res) => {
            if (res.statusCode === 200) {
                for (const [name, value] of Object.entries(preview.headers ?? {})) {
                    if (value)
                        res.setHeader(name, value);
                }
            }
        });
    }
    logListeningOn(preview.logger, server.server, options);
    await new Promise((resolve, reject) => {
        server.server.once('listening', resolve);
        server.server.once('error', reject);
        server.server.listen(port, host);
    });
    return server;
};
export { createPreviewServer as default };
