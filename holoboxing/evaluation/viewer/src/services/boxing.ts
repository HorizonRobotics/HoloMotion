// @ts-nocheck
export async function downloadReplayData(url: string, matchId: string) {
  const response = await fetch(url, { credentials: 'same-origin' })
  if (!response.ok) {
    throw new Error(`下载比赛数据失败 (${response.status})`)
  }
  const blob = await response.blob(),
    objectUrl = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = objectUrl
  link.download = `${matchId}.hdf5`
  document.body.appendChild(link)
  link.click()
  link.remove()
  setTimeout(() => URL.revokeObjectURL(objectUrl), 1000)
}
