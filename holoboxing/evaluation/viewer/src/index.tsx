import React, { useEffect, useState } from 'react'
import ReactDOM from 'react-dom'
import { readHdf5Replay, ReplayData } from './replay/Hdf5ReplayReader'
import { ReplayPlayer } from './replay/ReplayPlayer'
import './style.css'

function App() {
  const [data, setData] = useState<ReplayData>()
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [name, setName] = useState('')
  const [downloadUrl, setDownloadUrl] = useState<string>()
  async function open(bytes: ArrayBuffer, label: string, url?: string) {
    setError('')
    setLoading(true)
    try {
      const replay = await readHdf5Replay(bytes)
      setData(replay)
      setName(label)
      setDownloadUrl(url)
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure))
    } finally {
      setLoading(false)
    }
  }
  useEffect(() => {
    const controller = new AbortController()
    fetch('/replay.hdf5', { signal: controller.signal })
      .then(response => response.ok ? response.arrayBuffer() : null)
      .then(bytes => bytes && !controller.signal.aborted && open(bytes, 'replay.hdf5', '/replay.hdf5'))
      .catch(failure => !controller.signal.aborted && setError(String(failure)))
    return () => controller.abort()
  }, [])
  return <main className="hb-studio">
    <header className="viewer-header"><h1>HoloBoxing Replay</h1>
      <p>只读取已保存的轨迹与事件，不运行物理仿真。</p>
      <label>打开回放 <input aria-label="打开HDF5回放" type="file" accept=".hdf5,.h5" disabled={loading}
        onChange={async event => {
          const file = event.target.files?.[0]
          if (file) await open(await file.arrayBuffer(), file.name)
        }} /></label>
      {loading && <span role="status">正在读取回放…</span>}
      {error && <p role="alert">{error}</p>}
    </header>
    {data && <ReplayPlayer key={name + data.meta.match_id} data={data}
      downloadUrl={downloadUrl} title={name} notice="MP4导出取决于浏览器编码能力。" />}
  </main>
}

ReactDOM.render(<App />, document.getElementById('root'))
