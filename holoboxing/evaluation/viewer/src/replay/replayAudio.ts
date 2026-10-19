import { ReplayData } from './Hdf5ReplayReader'
import { recordedEffects } from './recordedEffects'

export type ReplaySound = { time: number; kind: 'hit' | 'ko'; duration: number }

/** Recorded effects are bundled locally and require no audio downloads. */
export function replaySounds(data: ReplayData): ReplaySound[] {
  if (data.meta.state_schema?.id !== 'g1_kinematic_v1') return []
  const koActors = new Map<number, Set<unknown>>()
  for (const event of data.events) if (event.type === 'KO') {
    const actors = koActors.get(event.frame) || new Set()
    const actor = event.actor === 'red' ? 0 : event.actor === 'blue' ? 1 : event.actor
    actors.add(actor); koActors.set(event.frame, actors)
  }
  const voicedKoFrames = new Set<number>()
  return data.events.flatMap(event => {
    if (event.type === 'KO') {
      // Simultaneous knockouts have no winner; do not stack two KO shouts.
      if (koActors.get(event.frame)!.size > 1 || voicedKoFrames.has(event.frame)) return []
      voicedKoFrames.add(event.frame)
    }
    const kind = event.type === 'HIT' ? 'hit' : event.type === 'KO' ? 'ko' : undefined
    const time = data.time[event.frame]
    return kind && Number.isFinite(time)
      ? [{ time, kind, duration: recordedEffects[kind].duration }]
      : []
  })
}

/** Play the exact user-selected preview PCM, shared by live playback and MP4. */
const buffers = new WeakMap<BaseAudioContext, Map<string, AudioBuffer>>()
function effect(context: BaseAudioContext, start: number, sound: ReplaySound, speed = 1) {
  let cache = buffers.get(context)
  if (!cache) { cache = new Map(); buffers.set(context, cache) }
  let buffer = cache.get(sound.kind)
  if (!buffer) {
    const recording = recordedEffects[sound.kind]
    const raw = atob(recording.pcm)
    buffer = context.createBuffer(1, raw.length / 2, recording.rate)
    const samples = buffer.getChannelData(0)
    for (let i = 0; i < samples.length; i++) {
      const value = raw.charCodeAt(i * 2) | (raw.charCodeAt(i * 2 + 1) << 8)
      samples[i] = (value >= 32768 ? value - 65536 : value) / 32768
    }
    cache.set(sound.kind, buffer)
  }
  const source = context.createBufferSource()
  source.buffer = buffer
  source.playbackRate.value = speed
  source.connect(context.destination)
  source.start(start)
  source.onended = () => source.disconnect()
  return () => { try { source.stop() } catch {} source.disconnect() }
}

export function scheduleReplaySounds(context: AudioContext, data: ReplayData, origin: number, speed: number, includeOrigin = true) {
  const start = context.currentTime
  const stops = replaySounds(data)
    .filter(sound => includeOrigin ? sound.time >= origin : sound.time > origin + 1e-6)
    .map(sound => effect(context, start + (sound.time - origin) / speed, sound, speed))
  return () => stops.forEach(stop => stop())
}

export function replayVideoDuration(data: ReplayData, soundEnabled = true) {
  const times = data.time
  if (!times.length || !times.every((time, index) => Number.isFinite(time) && (!index || time > times[index - 1]))) {
    throw new Error('回放时间必须为有限、严格递增的记录。')
  }
  const step = times.length > 1 ? times[times.length - 1] - times[times.length - 2] : 1 / (Number(data.meta.control_fps) || 50)
  const recorded = times[times.length - 1] - times[0] + step
  return soundEnabled ? replaySounds(data).reduce((end, sound) => Math.max(end, sound.time - times[0] + sound.duration), recorded) : recorded
}

export async function renderReplayAudio(data: ReplayData, duration: number) {
  const rate = 48_000
  const context = new OfflineAudioContext(1, Math.ceil(duration * rate), rate)
  for (const sound of replaySounds(data)) effect(context, sound.time - data.time[0], sound)
  return context.startRendering()
}
