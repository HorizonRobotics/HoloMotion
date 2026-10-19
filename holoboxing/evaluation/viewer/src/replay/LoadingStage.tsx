// @ts-nocheck
import React, { useEffect, useState } from 'react'
import './loadingStage.css'

/** Indeterminate progress: module loading and GPU setup expose no reliable percentage. */
export function LoadingStage({
  label,
  detail
}: {
  label: string
  detail?: string
}) {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    const started = performance.now()
    setElapsed(0)
    const timer = window.setInterval(
      () => setElapsed(Math.floor((performance.now() - started) / 1000)),
      1000
    )
    return () => window.clearInterval(timer)
  }, [label])
  return (
    <div className="hb-loading-stage" role="status">
      <strong>{label}</strong>
      <progress aria-label={`${label}进度`} />
      {detail && <span>{detail}</span>}
      <small aria-live="off">
        {elapsed > 0 ? `已等待 ${elapsed} 秒` : '正在准备资源'}
        {elapsed >= 15 ? ' · 资源加载较慢，请稍候' : ''}
      </small>
    </div>
  )
}
