// @ts-nocheck
import React, { useEffect, useMemo, useRef, useState } from 'react'
import { ReplayTransferActions } from './ReplayTransferActions'
import { downloadReplayData } from '../services/boxing'
import { ReplayData } from './Hdf5ReplayReader'
import './replay.css'
import { ReplayStateView, getReplayRenderer } from './ReplayStateView'
import { downloadMp4, exportReplayMp4, mp4MimeType } from './exportMp4'
import { replayVideoDuration, scheduleReplaySounds } from './replayAudio'
import { replayOutcome, ReplaySide } from './replayPerspective'
import { ReplayCameraToolbarContext } from './ReplayViewportTools'

export function ReplayPlayer({
  data: sourceData,
  downloadUrl,
  title = '比赛回放',
  notice,
  participants,
  viewerSide
}: {
  data: ReplayData
  downloadUrl?: string
  title?: string
  notice?: string
  participants?: { red: string; blue: string }
  viewerSide?: ReplaySide
}) {
  const data = useMemo(
    () => ({ ...sourceData, displayParticipants: participants }),
    [sourceData, participants?.red, participants?.blue]
  )
  const side = viewerSide
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [soundEnabled, setSoundEnabled] = useState(true)
  const [audioError, setAudioError] = useState('')
  const audioContext = useRef<AudioContext | null>(null)
  const [speed, setSpeed] = useState(1)
  const [downloading, setDownloading] = useState(false)
  const [dataError, setDataError] = useState('')
  const [exporting, setExporting] = useState(false)
  const [progress, setProgress] = useState(0)
  const [exportMessage, setExportMessage] = useState('')
  const [cameraToolbar, setCameraToolbar] = useState<HTMLDivElement | null>(
    null
  )
  const exportController = useRef<AbortController | null>(null)
  useEffect(() => {
    setExportMessage('')
    return () => {
      exportController.current?.abort()
      exportController.current = null
    }
  }, [data])
  const exportData = async (event: React.MouseEvent) => {
    event.preventDefault()
    if (!downloadUrl || downloading) {
      return
    }
    setDownloading(true)
    setDataError('')
    try {
      await downloadReplayData(downloadUrl, data.meta.match_id)
    } catch (error) {
      setDataError((error as Error).message)
    } finally {
      setDownloading(false)
    }
  }
  const exportVideo = async () => {
    if (exportController.current) {
      return
    }
    const adapter = getReplayRenderer(data)
    if (!adapter?.drawFrame && !adapter?.createExportRenderer) {
      setExportMessage('当前回放渲染器尚未提供视频导出。')
      return
    }
    const controller = new AbortController()
    exportController.current = controller
    setPlaying(false)
    setExporting(true)
    setProgress(0)
    setExportMessage('')
    try {
      const blob = await exportReplayMp4(
        data,
        adapter,
        controller.signal,
        setProgress,
        soundEnabled
      )
      if (exportController.current !== controller) {
        return
      }
      downloadMp4(blob, data.meta.match_id)
      setExportMessage('MP4 已生成并开始下载。')
    } catch (error) {
      if (exportController.current === controller) {
        setExportMessage(
          error.name === 'AbortError'
            ? '已取消导出。'
            : String(error.message || error)
        )
      }
    } finally {
      if (exportController.current === controller) {
        exportController.current = null
        setExporting(false)
      }
    }
  }
  useEffect(() => {
    setFrame(0)
    setPlaying(false)
    setExporting(false)
    setProgress(0)
  }, [data])
  useEffect(() => {
    if (!playing) {
      return
    }
    const start = performance.now(),
      origin = data.time[frame]
    let raf = 0,
      active = true
    const tick = (now: number) => {
      if (!active) {
        return
      }
      const time = origin + ((now - start) / 1000) * speed
      let low = 0,
        high = data.time.length - 1
      while (low < high) {
        const mid = Math.ceil((low + high) / 2)
        if (data.time[mid] <= time) {
          low = mid
        } else {
          high = mid - 1
        }
      }
      setFrame(low)
      if (time >= data.time[0] + replayVideoDuration(data, soundEnabled)) {
        setPlaying(false)
      } else if (active) {
        raf = requestAnimationFrame(tick)
      }
    }
    raf = requestAnimationFrame(tick)
    return () => {
      active = false
      cancelAnimationFrame(raf)
    }
    // frame is intentionally captured only at playback start, seek and speed changes pause playback.
  }, [playing, speed, data, soundEnabled])
  useEffect(() => {
    setAudioError('')
    return () => {
      audioContext.current?.close()
      audioContext.current = null
    }
  }, [data])
  useEffect(() => {
    if (!playing || !soundEnabled || !audioContext.current) return
    // Pause/resume skips the event already reached; seeking alone stays silent.
    return scheduleReplaySounds(audioContext.current, data, data.time[frame], speed, frame === 0)
  }, [playing, soundEnabled, speed, data])
  const enableAudio = async () => {
    try {
      if (!audioContext.current) audioContext.current = new AudioContext()
      await audioContext.current.resume()
      setAudioError('')
    } catch (error) {
      setSoundEnabled(false)
      setAudioError('浏览器无法启用音效，仍可无声播放。')
    }
  }
  const seek = (value: number) => {
    setPlaying(false)
    setFrame(value)
  }
  const adapter = getReplayRenderer(data)
  const canExport = !!(adapter?.drawFrame || adapter?.createExportRenderer)
  const isG1 = data.meta.state_schema?.id === 'g1_kinematic_v1'
  const isArticulated =
    data.meta.state_schema?.id === 'generic_articulated_replay_v1'
  const resultLabel = replayOutcome(data.meta.result?.winner, side, {
    red: participants?.red || data.meta.red?.name,
    blue: participants?.blue || data.meta.blue?.name
  })
  const endReason =
    data.meta.scoring_version === 'sword_pair_hits_v2'
      ? {
          elimination: '倒地判定',
          out_of_bounds: '出界判定',
          time_limit: '时间到'
        }[data.meta.result?.termination_reason] || ''
      : ''
  const scoreText = (value: number) =>
    value.toLocaleString('en-US', { maximumFractionDigits: 1 })
  const eventName = (side: string) =>
    side === 'red' || side === 'blue'
      ? participants?.[side] ||
        data.meta[side]?.name ||
        (side === 'red' ? '红方' : '蓝方')
      : side === 'system'
      ? '裁判'
      : side === 'both'
      ? '双方'
      : side
  const labels: Record<string, string> = {
    KO:
      ['hands_only_normal_impulse100ms_5Ns_ko1000_v3', 'hands_only_normal_impulse100ms_5Ns_cooldown100ms_fall030_ko1000_v4'].includes(data.meta.scoring_version)
        ? 'KO（胜方 +1000）'
        : data.meta.scoring_version === 'smpl_boxing_no_out_ko100_v1'
        ? 'KO（胜方 +100）'
        : 'KO',
    HIT: '命中',
    ROUND_END: '回合结束',
    MATCH_END: '比赛结束',
    valid_hit: '有效命中',
    contact: '身体接触',
    fall: '失稳',
    knockdown: '倒地',
    termination: '结束',
    contact_evidence: '交互',
    low_root_height: '低位姿态',
    DAMAGE: '伤害',
    damage: '伤害',
    WIN: '胜负',
    APPROACH: '接近',
    MATCH_START: '开始',
    CONTACT: '身体接触',
    FALL: '失稳',
    KNOCKDOWN: '倒地',
    DEV_HIT: '开发计分'
  }
  return (
    <ReplayCameraToolbarContext.Provider value={cameraToolbar}>
      <section
        className="hb-replay"
        aria-label="比赛回放"
        data-match-id={data.meta.match_id}
      >
        <div className="hb-player-topbar hb-player-top">
          <div className="hb-player-title">
            <h2>{title}</h2>
            {notice && <span>{notice}</span>}
          </div>
          <div className="hb-player-actions">
            <div className="hb-player-camera-slot" ref={setCameraToolbar} />
            <ReplayTransferActions
              downloadUrl={downloadUrl}
              matchId={data.meta.match_id}
              onExportData={exportData}
              downloading={downloading}
              onExportVideo={exportVideo}
              exporting={exporting}
              progress={progress}
              canExportVideo={Boolean(mp4MimeType()) && canExport}
            />
          </div>
        </div>
        {dataError && <p role="alert">{dataError}</p>}
        {audioError && <p role="status">{audioError}</p>}
        {(exporting || exportMessage) && (
          <div
            className={`hb-export-panel ${exporting ? 'is-running' : ''}`}
            role="region"
            aria-label="MP4 导出状态"
            data-testid="mp4-export-panel"
          >
            {exporting ? (
              <>
                <div className="hb-export-panel-head">
                  <div>
                    <strong>
                      {progress > 0 ? '正在生成 MP4' : '正在准备视频场景'}
                    </strong>
                    <span>本机导出 · 保持页面在前台，完成后自动下载</span>
                  </div>
                  <b data-testid="mp4-export-percent">
                    {Math.floor(progress * 100)}
                    <small>%</small>
                  </b>
                  <button onClick={() => exportController.current?.abort()}>
                    取消导出
                  </button>
                </div>
                <progress aria-label="MP4 导出进度" max={1} value={progress} />
              </>
            ) : (
              <div className="hb-export-panel-head">
                <p role="status">{exportMessage}</p>
                <button
                  className="hb-export-dismiss"
                  aria-label="关闭导出提示"
                  onClick={() => setExportMessage('')}
                >
                  ×
                </button>
              </div>
            )}
          </div>
        )}
        <div className="hb-match-strip" data-testid="replay-score">
          <div className="hb-fighter-label hb-fighter-red">
            <i />
            <span>
              <strong title={participants?.red}>
                {participants?.red ||
                  data.meta.red?.name ||
                  data.meta.red?.model_id ||
                  '红方'}
              </strong>
            </span>
          </div>
          <div className="hb-match-score">
            <strong>
              {scoreText(data.score[frame * 2])} <span>:</span>{' '}
              {scoreText(data.score[frame * 2 + 1])}
            </strong>
            <small>
              {frame === data.time.length - 1 ? (
                <strong
                  className="hb-replay-result"
                  data-testid="replay-result"
                >
                  {resultLabel}
                  {endReason && <span> · {endReason}</span>}
                </strong>
              ) : data.meta.scoring_version === 'sword_pair_hits_v2' ? (
                '有效命中'
              ) : isArticulated ? (
                '累计伤害'
              ) : (
                '比分'
              )}
            </small>
          </div>
          <div className="hb-fighter-label hb-fighter-blue">
            <span>
              <strong title={participants?.blue}>
                {participants?.blue ||
                  data.meta.blue?.name ||
                  data.meta.blue?.model_id ||
                  '蓝方'}
              </strong>
            </span>
            <i />
          </div>
        </div>
        <div className="hb-replay-viewport hb-viewport-wrap">
          <ReplayStateView data={data} frame={frame} />
        </div>
        <div className="hb-replay-controls hb-player-transport">
          <button
            className="hb-play-button"
            aria-label={playing ? '暂停' : '播放'}
            onClick={async () => {
              if (!playing && soundEnabled && isG1) await enableAudio()
              if (!playing && frame === data.time.length - 1) {
                setFrame(0)
              }
              setPlaying(!playing)
            }}
          >
            <span aria-hidden="true">{playing ? 'Ⅱ' : '▶'}</span>
          </button>
          <input
            aria-label="回放进度"
            type="range"
            min={0}
            max={data.time.length - 1}
            value={frame}
            style={
              {
                '--replay-progress': `${(frame /
                  Math.max(1, data.time.length - 1)) *
                  100}%`
              } as React.CSSProperties
            }
            onChange={e => seek(Number(e.target.value))}
          />
          <output data-testid="replay-time">
            {data.time[frame].toFixed(2)}{' '}
            <span>/ {data.time[data.time.length - 1].toFixed(2)} s</span>
          </output>
          {isG1 && <button aria-label="回放音效" aria-pressed={soundEnabled}
            onClick={async () => {
              if (soundEnabled) setSoundEnabled(false)
              else {
                setSoundEnabled(true)
                await enableAudio()
              }
            }}>{soundEnabled ? '音效开' : '音效关'}</button>}
          <select
            aria-label="播放速度"
            value={speed}
            onChange={e => {
              setPlaying(false)
              setSpeed(Number(e.target.value))
            }}
          >
            <option value={0.5}>0.5×</option>
            <option value={1}>1×</option>
            <option value={2}>2×</option>
          </select>
        </div>
        <span className="hb-visually-hidden" data-testid="replay-frame-count">
          {data.time.length} 帧 · 第 {data.round[frame]} 回合
        </span>
        <details className="hb-event-details">
          <summary>
            关键事件 <span>{data.events.length}</span>
          </summary>
          <div className="hb-replay-events" aria-label="事件标记">
            {data.events.map((event, index) => (
              <button
                key={index}
                title={`${eventName(event.actor)} → ${eventName(
                  event.target
                )}: ${event.value}`}
                onClick={() => seek(event.frame)}
              >
                <span>{data.time[event.frame].toFixed(2)}s</span>{' '}
                {labels[event.type] || event.type}
              </button>
            ))}
          </div>
        </details>
        <div className="hb-export-status">
          {!mp4MimeType() && (
            <p>
              {window.isSecureContext
                ? '此浏览器未提供 MP4 编码，请使用新版 Chrome / Edge。'
                : '当前使用 HTTP 地址，浏览器禁止 MP4 编码。请通过 HTTPS 或 localhost 打开播放器。'}
            </p>
          )}
          {!canExport && <p>当前状态渲染器尚未提供 MP4 导出。</p>}
        </div>
      </section>
    </ReplayCameraToolbarContext.Provider>
  )
}
