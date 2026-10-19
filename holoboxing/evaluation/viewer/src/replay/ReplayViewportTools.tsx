// @ts-nocheck
import React, { createContext, useContext } from 'react'
import { createPortal } from 'react-dom'

/** Camera tools belong to the player toolbar, independently of its renderer implementation. */
export const ReplayCameraToolbarContext = createContext<HTMLElement | null>(
  null
)
export function ReplayCameraTools({ children }: { children: React.ReactNode }) {
  const toolbar = useContext(ReplayCameraToolbarContext)
  return toolbar
    ? createPortal(
        <div className="hb-camera-tools" role="group" aria-label="视角选择">
          {children}
        </div>,
        toolbar
      )
    : null
}

/** Resize the actual drawing buffer and projection, not a letterboxed or stretched bitmap. */
export function observeReplayViewport(
  canvas: HTMLCanvasElement,
  runtime: { resize(width: number, height: number): void }
) {
  const surface = canvas.parentElement!
  let width = 0,
    height = 0
  const resize = () => {
    const rect = surface.getBoundingClientRect()
    const nextWidth = Math.round(rect.width),
      nextHeight = Math.round(rect.height)
    if (
      nextWidth < 1 ||
      nextHeight < 1 ||
      (width === nextWidth && height === nextHeight)
    ) {
      return
    }
    width = nextWidth
    height = nextHeight
    runtime.resize(width, height)
  }
  const observer = new ResizeObserver(resize)
  observer.observe(surface)
  resize()
  return () => observer.disconnect()
}
