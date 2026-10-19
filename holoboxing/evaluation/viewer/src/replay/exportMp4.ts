// @ts-nocheck
import {
  AudioBufferSource, BlobSource, BufferTarget, CanvasSource, Input, MP4,
  Mp4OutputFormat, Output, canEncodeVideo, getFirstEncodableAudioCodec
} from 'mediabunny'
import { ReplayData } from './Hdf5ReplayReader'
import { ReplayRendererAdapter } from './ReplayStateView'
import { renderReplayAudio, replaySounds, replayVideoDuration } from './replayAudio'

export function mp4MimeType(): string | undefined {
  return typeof VideoEncoder !== 'undefined' && typeof VideoFrame !== 'undefined'
    ? 'video/mp4'
    : undefined
}

/** Encode saved states with explicit timestamps, independent of rendering speed. */
export async function exportReplayMp4(
  data: ReplayData,
  adapter: ReplayRendererAdapter,
  signal: AbortSignal,
  onProgress: (value: number) => void,
  soundEnabled = true
): Promise<Blob> {
  if (!mp4MimeType()) {
    throw new Error('当前浏览器不支持 MP4 视频编码，请使用新版 Chrome / Edge，通过 HTTPS 或 localhost 打开播放器。')
  }
  if (!adapter.drawFrame && !adapter.createExportRenderer) {
    throw new Error('当前回放渲染器尚未提供视频导出。')
  }
  const check = () => {
    if (signal.aborted) throw new DOMException('已取消导出', 'AbortError')
    if (document.hidden) throw new Error('请保持标签页可见后重试导出。')
  }
  check()
  const renderer = adapter.createExportRenderer
    ? await adapter.createExportRenderer(data)
    : undefined
  let output: Output | undefined
  try {
    check()
    const canvas = document.createElement('canvas')
    canvas.width = renderer ? 1280 : 720
    canvas.height = renderer ? 720 : 380
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('无法创建视频画布')
    const fps = renderer ? 30 : Math.min(60, Math.max(1, Number(data.meta.control_fps) || 50))
    const bitrate = renderer ? 5_000_000 : 2_500_000
    if (!await canEncodeVideo('avc', { width: canvas.width, height: canvas.height, bitrate })) {
      throw new Error('当前浏览器没有可用的 MP4 / H.264 编码器，请更换新版 Chrome / Edge。')
    }
    const times = data.time
    const duration = replayVideoDuration(data, soundEnabled)
    const count = Math.max(1, Math.ceil(duration * fps - 1e-6))
    const target = new BufferTarget()
    output = new Output({ format: new Mp4OutputFormat(), target })
    const source = new CanvasSource(canvas, { codec: 'avc', bitrate })
    output.addVideoTrack(source)
    const hasSounds = soundEnabled && replaySounds(data).length > 0
    const audioCodec = hasSounds ? await getFirstEncodableAudioCodec(['aac', 'opus'], {
      numberOfChannels: 1, sampleRate: 48_000, bitrate: 128_000
    }) : undefined
    if (hasSounds && !audioCodec) throw new Error('当前浏览器没有可用的音频编码器，请更换新版 Chrome / Edge，或关闭音效后导出。')
    const audio = audioCodec ? new AudioBufferSource({ codec: audioCodec, bitrate: 128_000 }) : undefined
    if (audio) output.addAudioTrack(audio)
    const audioBuffer = audio ? await renderReplayAudio(data, duration) : undefined
    check()
    await output.start()
    // The entire short mono track fits in memory; encode it alongside video.
    const audioTask = audio ? audio.add(audioBuffer!) : Promise.resolve()
    audioTask.catch(() => {})
    let frame = 0
    for (let index = 0; index < count; index++) {
      check()
      const timestamp = index / fps
      while (frame + 1 < times.length && times[frame + 1] - times[0] <= timestamp + 1e-9) frame++
      // Include the recorded terminal pose and score within the final video sample.
      if (index === count - 1) frame = times.length - 1
      if (renderer) renderer.drawFrame(ctx, frame)
      else adapter.drawFrame!(ctx, data, frame)
      // Await encoding backpressure: slow rendering cannot skip time or lose the tail.
      await source.add(timestamp, Math.min(1 / fps, duration - timestamp))
      onProgress(Math.min(0.99, (index + 1) / count))
      await new Promise(resolve => window.setTimeout(resolve, 0))
    }
    check()
    await audioTask
    await output.finalize()
    check()
    const blob = new Blob([target.buffer!], { type: 'video/mp4' })
    const input = new Input({ source: new BlobSource(blob), formats: [MP4] })
    try {
      const track = await input.getPrimaryVideoTrack()
      const actualDuration = await track!.computeDuration()
      if (Math.abs(actualDuration - duration) > 1 / fps) {
        throw new Error('视频时长与记录不符，未保存不完整文件。请重试。')
      }
    } finally {
      input.dispose()
    }
    check()
    onProgress(1)
    return blob
  } finally {
    if (output && output.state !== 'finalized' && output.state !== 'canceled') await output.cancel()
    renderer?.dispose()
  }
}

export function downloadMp4(blob: Blob, matchId: string) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `${String(matchId || 'holoboxing-replay').replace(/[^a-zA-Z0-9_-]/g, '_')}.mp4`
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 60_000)
}
