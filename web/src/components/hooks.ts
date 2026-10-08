import { useLayoutEffect, useRef, useState } from 'react'

/** Width of a container, tracked with ResizeObserver (for responsive SVG charts). */
export function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(0)
  useLayoutEffect(() => {
    if (!ref.current) return
    setWidth(ref.current.getBoundingClientRect().width)
    const ro = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])
  return [ref, width] as const
}

export const linear = (d0: number, d1: number, r0: number, r1: number) =>
  (v: number) => (d1 === d0 ? r0 : r0 + ((v - d0) / (d1 - d0)) * (r1 - r0))
