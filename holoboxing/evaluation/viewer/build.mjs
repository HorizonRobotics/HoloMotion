import { build } from 'esbuild'
import { mkdir, writeFile, cp } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const root = path.dirname(fileURLToPath(import.meta.url))
const out = process.env.BOXING_VIEWER_OUTPUT || path.join(root, 'dist')
await mkdir(out, { recursive: true })
await build({
  absWorkingDir: root,
  entryPoints: ['src/index.tsx'],
  outdir: out,
  bundle: true,
  format: 'esm',
  splitting: true,
  platform: 'browser',
  target: 'es2020',
  minify: true,
  loader: { '.wasm': 'file' },
  define: { 'process.env.NODE_ENV': '"production"' },
  plugins: [{
    name: 'browser-only-wasm',
    setup(builder) {
      builder.onResolve({ filter: /^(node:)?(module|fs|path|url|crypto)$/ }, args => ({
        path: args.path, namespace: 'node-only'
      }))
      builder.onLoad({ filter: /.*/, namespace: 'node-only' }, () => ({
        contents: 'export default {}; export const createRequire = () => { throw new Error("Node modules are unavailable in the replay browser") }',
        loader: 'js'
      }))
    }
  }]
})
await writeFile(path.join(out, 'index.html'), `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HoloBoxing Replay</title><link rel="stylesheet" href="./index.css"></head>
<body><div id="root"></div><script type="module" src="./index.js"></script></body></html>`)

// Retain upstream notices alongside the redistributed browser bundle.
await cp(path.join(root, 'notices'), path.join(out, 'notices'), { recursive: true })
