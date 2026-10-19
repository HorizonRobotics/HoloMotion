// @ts-nocheck
import { REPLAY_ASSETS_BASE } from './assetPaths'
import loadMujoco from '@mujoco/mujoco'
import wasmUrl from '@mujoco/mujoco/mujoco.wasm'

interface IPackedFile {
  file: string
  sha256: string
  bytes: number
  decoded_bytes: number
  decoded_sha256: string
}
interface IManifest {
  entry: string
  entries?: Record<string, string>
  files: Array<string | { path: string }>
  shared_files?: Record<string, string>
  sha256?: Record<string, string>
  runtime_bundle?: IPackedFile & { wasm: IPackedFile }
}
let enginePromise: Promise<any> | undefined
const pending = new Map<string, Promise<{ mj: any; manifest: IManifest }>>()
async function verify(bytes: Uint8Array, expected?: string) {
  if (globalThis.crypto?.subtle && expected) {
    const digest = await crypto.subtle.digest('SHA-256', bytes as BufferSource)
    const actual = Array.from(new Uint8Array(digest), byte =>
      byte.toString(16).padStart(2, '0')
    ).join('')
    if (actual !== expected) {
      throw new Error('G1 资源哈希不匹配')
    }
  }
}
function safePath(path: string) {
  if (path.startsWith('/') || path.split('/').includes('..')) {
    throw new Error('G1 资源路径无效')
  }
  return path
}
async function unpack(base: string, description: IPackedFile) {
  const response = await fetch(base + safePath(description.file), {
    cache: 'force-cache'
  })
  if (!response.ok) {
    throw new Error('G1 压缩资源加载失败')
  }
  const bytes = new Uint8Array(await response.arrayBuffer())
  // Some static servers automatically decode .gz responses before fetch exposes
  // the body. Accept both representations; never decompress the same body twice.
  const compressed = bytes[0] === 0x1f && bytes[1] === 0x8b
  if (compressed) {
    await verify(bytes, description.sha256)
  }
  const decoded = compressed
    ? new Uint8Array(
        await new Response(
          new Blob([bytes])
            .stream()
            .pipeThrough(new DecompressionStream('gzip'))
        ).arrayBuffer()
      )
    : bytes
  await verify(decoded, description.decoded_sha256)
  if (decoded.length !== description.decoded_bytes) {
    throw new Error('G1 压缩资源长度不一致')
  }
  return decoded
}
export async function loadG1Assets(variant: string) {
  const base = `${REPLAY_ASSETS_BASE}${variant}/`
  if (!pending.has(variant)) {
    pending.set(
      variant,
      (async () => {
        const response = await fetch(`${base}manifest.json`)
        if (!response.ok) {
          throw new Error('G1 模型资源清单加载失败')
        }
        const manifest: IManifest = await response.json()
        const packed =
          typeof DecompressionStream !== 'undefined'
            ? manifest.runtime_bundle
            : undefined
        // Public content-addressed bundles only; private recordings retain their own authorization.
        const mjPromise =
          enginePromise ||
          (enginePromise = (async () =>
            loadMujoco(
              packed
                ? { wasmBinary: await unpack(base, packed.wasm) }
                : {
                    locateFile: path =>
                      path.endsWith('.wasm') ? wasmUrl : path
                  }
            ))().catch(error => {
            enginePromise = undefined
            throw error
          }))
        const filesPromise = (async () => {
          if (packed) {
            const bytes = await unpack(base, packed)
            const headerSize = new DataView(bytes.buffer).getUint32(0, true)
            const entries: Array<[string, number]> = JSON.parse(
              new TextDecoder().decode(bytes.subarray(4, 4 + headerSize))
            )
            let offset = 4 + headerSize
            const unpackedFiles = []
            for (const [path, size] of entries) {
              safePath(path)
              if (
                !manifest.sha256?.[path] ||
                !Number.isSafeInteger(size) ||
                size < 0 ||
                offset + size > bytes.length
              ) {
                throw new Error('G1 压缩资源索引无效')
              }
              const content = bytes.subarray(offset, offset + size)
              await verify(content, manifest.sha256[path])
              unpackedFiles.push({ path, bytes: content })
              offset += size
            }
            if (offset !== bytes.length) {
              throw new Error('G1 压缩资源索引长度不一致')
            }
            return unpackedFiles
          }
          return Promise.all(
            manifest.files.map(async item => {
              const path = safePath(typeof item === 'string' ? item : item.path)
              const shared = manifest.shared_files?.[path]
              const url = shared
                ? `${REPLAY_ASSETS_BASE}${safePath(shared)}`
                : `${base}${path}`
              const file = await fetch(url)
              if (!file.ok) {
                throw new Error(`G1 资源加载失败：${path}`)
              }
              const bytes = new Uint8Array(await file.arrayBuffer())
              await verify(bytes, manifest.sha256?.[path])
              return { path, bytes }
            })
          )
        })()
        const [mj, files] = await Promise.all([mjPromise, filesPromise])
        for (const file of files) {
          const path = `/${variant}-assets/${file.path}`
          mj.FS.mkdirTree(path.slice(0, path.lastIndexOf('/')))
          mj.FS.writeFile(path, file.bytes)
        }
        return { mj, manifest }
      })().catch(error => {
        pending.delete(variant)
        throw error
      })
    )
  }
  return pending.get(variant)!
}
