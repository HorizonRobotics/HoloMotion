// @ts-nocheck
import React, { useEffect, useRef, useState } from 'react'
import { ReplayRendererProps } from './ReplayStateView'
import { ReplayData } from './Hdf5ReplayReader'
import { LoadingStage } from './LoadingStage'
import { ReplayCameraTools, observeReplayViewport } from './ReplayViewportTools'
import { G1Camera, G1Runtime } from './g1Runtime'

export function G1ReplayCanvas({ data, frame }: ReplayRendererProps) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const runtime = useRef<G1Runtime | null>(null)
  const latestFrame = useRef(frame)
  latestFrame.current = frame
  const [status, setStatus] = useState('正在加载 G1 模型与 MuJoCo…')
  const [ready, setReady] = useState(false)
  const [cameraPreset, setCameraPreset] = useState<G1Camera>('broadcast')
  const [error, setError] = useState('')
  const update = (active: G1Runtime, value: number) => {
    active.render(value)
    const diag = active.diagnostics()
    if (canvas.current) {
      canvas.current.dataset.frame = String(diag.frame)
      canvas.current.dataset.qposRed = JSON.stringify(diag.qposRed)
      canvas.current.dataset.qposBlue = JSON.stringify(diag.qposBlue)
      canvas.current.dataset.meshCount = String(diag.meshCount)
      canvas.current.dataset.jointAddresses = JSON.stringify(
        diag.jointAddresses
      )
      canvas.current.dataset.rootAddress = String(diag.rootAddress)
      canvas.current.dataset.backend = 'mujoco-mj_forward'
    }
  }
  useEffect(() => {
    let disposed = false
    const controller = new AbortController()
    let disconnectResize: (() => void) | undefined
    setReady(false)
    setCameraPreset('broadcast')
    setError('')
    setStatus('正在加载 G1 模型与 MuJoCo…')
    import('./g1Runtime')
      .then(module =>
        disposed
          ? null
          : module.createG1Runtime(
              data,
              canvas.current!,
              controller.signal,
              label => {
                if (!disposed) {
                  setStatus(label)
                }
              }
            )
      )
      .then(active => {
        if (!active) {
          return
        }
        if (disposed) {
          active.dispose()
          return
        }
        runtime.current = active
        disconnectResize = observeReplayViewport(canvas.current!, active)
        update(active, latestFrame.current)
        setReady(true)
        setStatus('G1 · MuJoCo WASM · 真实关节状态回放')
      })
      .catch(error => {
        if (!disposed) {
          setError(String(error.message || error))
          setStatus('三维场景加载失败')
        }
      })
    return () => {
      disposed = true
      controller.abort()
      disconnectResize?.()
      runtime.current?.dispose()
      runtime.current = null
    }
  }, [data])
  useEffect(() => {
    if (runtime.current) {
      update(runtime.current, frame)
    }
  }, [frame, data])
  return (
    <div className="hb-g1-viewport">
      <div className="hb-render-surface">
        {!ready && !error && (
          <div className="hb-scene-loading">
            <LoadingStage label={status} detail="首次打开需要下载模型资源" />
          </div>
        )}
        <canvas
          ref={canvas}
          width={1280}
          height={720}
          aria-label="G1 三维回放"
          data-testid="g1-canvas"
          style={{ width: '100%', display: 'block' }}
        />
        <span className="hb-render-status" data-testid="g1-status">
          {status}
        </span>
      </div>
      <ReplayCameraTools>
        {(['broadcast', 'ringside', 'overhead'] as G1Camera[]).map(
          (preset, i) => (
            <button
              key={preset}
              disabled={!ready}
              aria-pressed={cameraPreset === preset}
              onClick={() => {
                runtime.current?.setCamera(preset)
                setCameraPreset(preset)
              }}
            >
              {['主视角', '场边', '俯视'][i]}
            </button>
          )
        )}
      </ReplayCameraTools>
      {error && <p role="alert">{error}</p>}
    </div>
  )
}

export async function createG1ExportRenderer(data: ReplayData) {
  const { createG1Runtime } = await import('./g1Runtime')
  const runtime = await createG1Runtime(data)
  return {
    drawFrame(ctx: CanvasRenderingContext2D, frame: number) {
      runtime.render(frame)
      ctx.drawImage(runtime.canvas, 0, 0, ctx.canvas.width, ctx.canvas.height)
      const w = ctx.canvas.width,
        h = ctx.canvas.height
      ctx.fillStyle = 'rgba(255, 255, 255, .92)'
      ctx.fillRect(w * 0.08, 20, w * 0.84, 54)
      ctx.textAlign = 'center'
      ctx.font = '600 23px sans-serif'
      const redName =
        data.displayParticipants?.red || data.meta.red?.name || '红方'
      const blueName =
        data.displayParticipants?.blue || data.meta.blue?.name || '蓝方'
      ctx.fillStyle = '#cf334b'
      ctx.fillText(
        `${redName} ${data.score[frame * 2].toFixed(1)}`,
        w * 0.26,
        55,
        w * 0.32
      )
      ctx.fillStyle = '#1677ff'
      ctx.fillText(
        `${data.score[frame * 2 + 1].toFixed(1)} ${blueName}`,
        w * 0.74,
        55,
        w * 0.32
      )
      ctx.fillStyle = '#253858'
      ctx.font = '14px sans-serif'
      ctx.fillText(`${data.time[frame].toFixed(2)} s`, w * 0.5, 54)
      ctx.fillStyle = 'rgba(255, 255, 255, .92)'
      ctx.fillRect(16, h - 40, 390, 26)
      ctx.textAlign = 'left'
      ctx.fillStyle = '#526987'
      ctx.font = '13px sans-serif'
      ctx.fillText(
        'HOLOBOXING  /  G1 RECORDED STATE  /  EXPERIMENT',
        28,
        h - 22
      )
    },
    dispose() {
      runtime.dispose()
    }
  }
}
