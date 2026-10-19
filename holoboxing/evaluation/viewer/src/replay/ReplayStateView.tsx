// @ts-nocheck
import React from 'react'
import { ReplayData } from './Hdf5ReplayReader'
import { G1ReplayCanvas, createG1ExportRenderer } from './G1ReplayCanvas'

export interface ReplayRendererProps {
  data: ReplayData
  frame: number
}
export interface ReplayRendererAdapter {
  Component: React.ComponentType<ReplayRendererProps>
  drawFrame?: (
    context: CanvasRenderingContext2D,
    data: ReplayData,
    frame: number
  ) => void
  createExportRenderer?: (
    data: ReplayData
  ) => Promise<{
    drawFrame(context: CanvasRenderingContext2D, frame: number): void
    dispose(): void
  }>
  supports(data: ReplayData): boolean
}
const adapters: ReplayRendererAdapter[] = []
export function registerReplayRenderer(adapter: ReplayRendererAdapter) {
  adapters.unshift(adapter)
}
export function getReplayRenderer(data: ReplayData) {
  return adapters.find(candidate => candidate.supports(data))
}

registerReplayRenderer({
  supports: data =>
    data.stateDataset === 'state' &&
    data.meta.state_schema?.id === 'g1_kinematic_v1',
  Component: G1ReplayCanvas,
  createExportRenderer: createG1ExportRenderer
})

export function ReplayStateView(props: ReplayRendererProps) {
  const adapter = getReplayRenderer(props.data)
  if (!adapter) {
    return (
      <p role="status">
        当前状态 schema 尚无渲染 adapter；时间轴、事件与计分可读。
      </p>
    )
  }
  const Renderer = adapter.Component
  return <Renderer {...props} />
}
