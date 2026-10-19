// @ts-nocheck
import React from 'react'

export function ReplayTransferActions({
  downloadUrl,
  matchId,
  onExportData,
  downloading = false,
  onExportVideo,
  exporting = false,
  progress = 0,
  canExportVideo = false
}: {
  downloadUrl?: string
  matchId?: string
  onExportData?: (event: React.MouseEvent) => void
  downloading?: boolean
  onExportVideo?: () => void
  exporting?: boolean
  progress?: number
  canExportVideo?: boolean
}) {
  return (
    <div className="hb-replay-export" role="group" aria-label="比赛数据与视频">
      {downloadUrl ? (
        <a
          className="hb-state-download"
          onClick={onExportData}
          aria-disabled={downloading}
          title="下载 HDF5 比赛数据"
          href={downloadUrl}
          download={`${matchId}.hdf5`}
        >
          <span aria-hidden="true">↓</span>{' '}
          {downloading ? '导出中…' : '导出比赛数据'}
        </a>
      ) : (
        <button
          className="hb-state-download"
          disabled={true}
          title="请先选择比赛录像"
        >
          <span aria-hidden="true">↓</span> 导出比赛数据
        </button>
      )}
      <button
        className="hb-export-button"
        title={canExportVideo ? "MP4 格式，在本机生成" : !window.isSecureContext ? "请通过 HTTPS 或 localhost 打开后导出 MP4" : "当前浏览器或渲染器不支持 MP4 导出"}
        onClick={onExportVideo}
        disabled={exporting || !canExportVideo}
      >
        <span aria-hidden="true">{exporting ? '◌' : '↓'}</span>{' '}
        {exporting ? `导出中 ${Math.floor(progress * 100)}%` : '导出比赛视频'}
      </button>
    </div>
  )
}
