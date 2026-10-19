import { exportReplayMp4 } from '../src/replay/exportMp4'
import * as THREE from 'three'
import { createHitEffects } from '../src/replay/hitEffects'
import { renderReplayAudio, replaySounds, replayVideoDuration, scheduleReplaySounds } from '../src/replay/replayAudio'

const assert = (condition: unknown, message: string) => { if (!condition) throw Error(message) }
const makeData = (count = 51, events: any[] = [], boxing = false): any => ({
  time: Array.from({ length: count }, (_, i) => 10 + i / 50),
  meta: { control_fps: 50, state_schema: { id: boxing ? 'g1_kinematic_v1' : 'generic_articulated_replay_v1' } }, events
})
const bytes = async (blob: Blob) => Array.from(new Uint8Array(await blob.arrayBuffer()))
async function metadata(blob: Blob) {
  const url = URL.createObjectURL(blob), video = document.createElement('video')
  video.muted = true
  video.src = url
  try {
    await new Promise<void>((resolve, reject) => { video.onloadedmetadata = () => resolve(); video.onerror = () => reject(Error('Video decode failed')) })
    const duration = video.duration
    const sample = async (time: number) => {
      const seek = new Promise<void>(resolve => { video.onseeked = () => resolve() })
      video.currentTime = time
      await seek
      const canvas = document.createElement('canvas'); canvas.width = 720; canvas.height = 380
      const context = canvas.getContext('2d')!
      context.drawImage(video, 0, 0, 720, 380)
      return Array.from(context.getImageData(30, 30, 1, 1).data)
    }
    const first = await sample(0.001), last = await sample(duration - 0.005)
    return { duration, first, last }
  } finally { video.removeAttribute('src'); video.load(); URL.revokeObjectURL(url) }
}
;(window as any).runExportTests = async () => {
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.1, 100)
  camera.position.set(0, -4, 3); camera.lookAt(0, 0, 1)
  const canvas = document.createElement('canvas'), effects = createHitEffects(scene, camera, [], canvas)
  for (const actor of ['red', 'blue']) {
    effects.render([{frame: 1, type: 'KO', actor}], 1, 1, [0, 1], {winner: actor})
    assert(canvas.dataset.resultEffect === 'K.O.', 'Winner KO label absent')
    assert(canvas.dataset.koEffectActors === (actor === 'red' ? '[0]' : '[1]'), 'KO label colored for loser')
    const bounds = JSON.parse(canvas.dataset.resultTextBounds!)
    assert(Math.abs(bounds.title[0] - bounds.subtitle[0]) < 2 && Math.abs(bounds.title[1] - bounds.subtitle[1]) < 2, 'Winner subtitle edges differ from KO title')
  }
  effects.render([{frame: 1, type: 'KO', actor: 'red'}, {frame: 1, type: 'KO', actor: 'blue'}], 1, 1, [0, 1], {winner: 'draw'})
  assert(canvas.dataset.resultEffect === 'DRAW' && canvas.dataset.activeKoEffects === '1' && canvas.dataset.koEffectActors === '[-1]', 'Simultaneous fall displayed two winners')
  effects.render([], 1, 1, [0, 1], {winner: 'draw'})
  assert(canvas.dataset.resultEffect === 'DRAW', 'Time-limit draw absent')
  effects.render([], 0, 0, [0, 1], {winner: 'draw'})
  assert(canvas.dataset.activeKoEffects === '0', 'Seeking before result left a stale label')
  effects.dispose(); assert(scene.children.length === 0, 'Result effects leaked scene objects')

  const progress: number[] = [], frames: number[] = []
  const data = makeData()
  const blob = await exportReplayMp4(data, { drawFrame(context: any, _data: any, frame: number) {
    // Force rendering slower than the 50 Hz sample interval.
    const end = performance.now() + 30; while (performance.now() < end) {}
    frames.push(frame)
    context.fillStyle = frame === 50 ? '#00ff00' : '#ff0000'; context.fillRect(0, 0, 720, 380)
  } } as any, new AbortController().signal, value => progress.push(value))
  const video = await metadata(blob)
  assert(Math.abs(video.duration - 1.02) < 0.02, 'Slow rendering truncated or stretched the video')
  assert(video.first[0] > 220 && video.last[1] > 220, 'Encoded first or terminal state missing')
  assert(frames[0] === 0 && frames.at(-1) === 50 && progress.at(-1) === 1, 'Incomplete frame sequence')
  const fullRound = await exportReplayMp4(makeData(3001), { drawFrame(context: any, _data: any, frame: number) {
    context.fillStyle = frame === 3000 ? '#00ff00' : '#ff0000'; context.fillRect(0, 0, 720, 380)
  } } as any, new AbortController().signal, () => {})
  const fullRoundMeta = await metadata(fullRound)
  assert(Math.abs(fullRoundMeta.duration - 60.02) < 0.02 && fullRoundMeta.last[1] > 220, 'Full 60-second round or terminal frame was truncated')
  // The first timestamp is 10 seconds; the exported file must start at zero.
  const timeline = makeData(201, [{ frame: 50, type: 'HIT' }, { frame: 150, type: 'KO' }], true)
  const duration = replayVideoDuration(timeline)
  const sound = await renderReplayAudio(timeline, duration), pcm = sound.getChannelData(0)
  const energy = (start: number, end: number) => {
    const part = pcm.slice(start * sound.sampleRate, end * sound.sampleRate)
    return Math.sqrt(part.reduce((sum, value) => sum + value * value, 0) / part.length)
  }
  assert(energy(0.1, 0.8) === 0 && energy(1.0, 1.1) > 0.01 && energy(3.0, 3.5) > 0.005, 'Event sound alignment failed')
  assert(replaySounds({ ...timeline, meta: data.meta }).length === 0, 'Other track acquired boxing sounds')
  const simultaneous = makeData(51, [
    {frame: 50, type: 'KO', actor: 'red'},
    {frame: 50, type: 'KO', actor: 'blue'}
  ], true)
  assert(replaySounds(simultaneous).length === 0, 'Simultaneous-fall draw announced two KO winners')
  const duplicate = makeData(51, [{frame: 50, type: 'KO', actor: 'red'}, {frame: 50, type: 'KO', actor: 'red'}], true)
  assert(replaySounds(duplicate).length === 1, 'Duplicate KO event stacked the voice recording')
  const tail = makeData(51, [{ frame: 50, type: 'KO' }], true)
  assert(Math.abs(replayVideoDuration(tail) - 2.1) < 1e-9, 'Terminal KO audio was truncated')
  const context = new AudioContext(); await context.resume()
  const scheduled: number[] = [], stopped: number[] = []
  const original = context.createBufferSource.bind(context)
  context.createBufferSource = () => {
    const oscillator = original(), start = oscillator.start.bind(oscillator), stop = oscillator.stop.bind(oscillator)
    oscillator.start = time => { scheduled.push(time); start(time) }
    oscillator.stop = time => { stopped.push(time); stop(time) }
    return oscillator
  }
  const cancelSounds = scheduleReplaySounds(context, timeline, 10, 2)
  assert(scheduled.length === 2 && Math.abs(scheduled[0] - context.currentTime - 0.5) < 0.1, 'Live hit or KO schedule failed')
  cancelSounds(); assert(stopped.length === 2, 'Pause did not stop scheduled sound')
  const later: number[] = []; scheduled.length = 0
  const stopLater = scheduleReplaySounds(context, timeline, 12, 1)
  later.push(...scheduled); assert(later.length === 1, 'Seek replayed an old hit')
  stopLater(); await context.close()
  const sample = await exportReplayMp4(timeline, { drawFrame(ctx: any, _data: any, frame: number) {
    ctx.fillStyle = '#101b2a'; ctx.fillRect(0, 0, 720, 380)
    ctx.fillStyle = '#50a4cf'; ctx.font = '24px sans-serif'
    ctx.fillText('SYNTHETIC EVENT SOUND TEST', 60, 100)
    ctx.fillStyle = 'white'; ctx.fillText('HIT at 1s / KO at 3s', 60, 150)
    ctx.fillText(((frame / 50).toFixed(2)) + ' s', 60, 210)
  } } as any, new AbortController().signal, () => {})
  const sampleMeta = await metadata(sample)
  assert(Math.abs(sampleMeta.duration - duration) < 0.03, 'Audiovisual duration changed')
  const muted = await exportReplayMp4(tail, { drawFrame(ctx: any) { ctx.fillRect(0, 0, 720, 380) } } as any, new AbortController().signal, () => {}, false)
  const mutedMeta = await metadata(muted)
  assert(Math.abs(mutedMeta.duration - 1.02) < 0.02, 'Mute retained sound tail')
  const controller = new AbortController(); let cancelled = false
  try {
    await exportReplayMp4(data, { drawFrame() {} } as any, controller.signal, () => controller.abort())
  } catch (error: any) { cancelled = error.name === 'AbortError' }
  assert(cancelled, 'Cancellation did not reject the export')
  return { status: 'PASS', video, fullRoundMeta, frameCount: frames.length, duration, hitEnergy: energy(1, 1.1), koEnergy: energy(3, 3.5), sampleMeta,
    files: { 'slow-render.mp4': await bytes(blob), 'full-round.mp4': await bytes(fullRound), 'sound-test.mp4': await bytes(sample), 'muted.mp4': await bytes(muted) } }
}
