import React from 'react'
import ReactDOM from 'react-dom'
import { ReplayPlayer } from '../src/replay/ReplayPlayer'
import { registerReplayRenderer } from '../src/replay/ReplayStateView'
import { renderReplayAudio } from '../src/replay/replayAudio'

const calls: Array<{kind: string; at?: number}> = []
const create = AudioContext.prototype.createBufferSource
AudioContext.prototype.createBufferSource = function () {
  const oscillator = create.call(this)
  const start = oscillator.start.bind(oscillator), stop = oscillator.stop.bind(oscillator)
  oscillator.start = (at = 0) => { calls.push({kind: 'start', at}); start(at) }
  oscillator.stop = at => { calls.push({kind: 'stop', at}); stop(at) }
  return oscillator
}
;(window as any).audioCalls = calls
const data: any = {
  time: Array.from({length: 201}, (_, index) => index / 50),
  score: new Float64Array(402), round: new Uint16Array(201).fill(1),
  meta: {match_id: 'audio-fixture', control_fps: 50, state_schema: {id: 'g1_kinematic_v1'}},
  events: [{frame: 50, type: 'HIT', actor: 'red', target: 'blue', value: 1},
    {frame: 150, type: 'KO', actor: 'blue', target: 'red', value: 1000}]
}
registerReplayRenderer({supports: replay => replay.meta.match_id === 'audio-fixture',
  Component: ({frame}) => <div data-testid="audio-fixture-frame">{frame}</div>})
ReactDOM.render(<ReplayPlayer data={data} />, document.getElementById('root'))
;(window as any).checkPreviewParity = async () => {
  const errors: Record<string, number> = {}
  for (const kind of ['hit', 'ko']) {
    const bytes = await (await fetch('/'+kind+'.wav')).arrayBuffer()
    const reference = await new OfflineAudioContext(1, 1, 48000).decodeAudioData(bytes)
    const fixture: any = {time: [0, 0.05], meta: data.meta,
      events: [{frame: 0, type: kind === 'hit' ? 'HIT' : 'KO'}]}
    const actual = await renderReplayAudio(fixture, reference.duration)
    const expected = reference.getChannelData(0), pcm = actual.getChannelData(0)
    let error = 0
    for (let index = 0; index < expected.length; index++) error = Math.max(error, Math.abs(expected[index] - pcm[index]))
    if (error > 2 / 32767) throw Error(kind+' differs from approved preview: '+error)
    errors[kind] = error
  }
  return errors
}
