// @ts-nocheck
export type ReplaySide = 'red' | 'blue'
export function replayOutcome(
  winner: string | null | undefined,
  side?: ReplaySide,
  names?: { red?: string; blue?: string }
) {
  if (winner === 'draw') {
    return '平局'
  }
  if (winner !== 'red' && winner !== 'blue') {
    return '对局结束'
  }
  if (side) {
    return winner === side ? '胜利' : '失败'
  }
  return names?.[winner] ? `${names[winner]} 获胜` : '对局结束'
}
