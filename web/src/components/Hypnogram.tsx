import { useMemo, useState, type MouseEvent } from 'react'
import type { Night } from '../api'
import { HYPNO_ORDER, STAGES, clockTime, hypnoRow, pct, stageColor } from '../stages'
import { linear, useWidth } from './hooks'

const LEFT = 76
const RIGHT = 12
const ROW = 22
const LANE = ROW * 5
const DENSITY = 64

type Run = { start: number; end: number; stage: number }

function runsOf(stages: number[], from: number, to: number): Run[] {
  const runs: Run[] = []
  for (let i = from; i < to; i++) {
    const s = stages[i]
    const last = runs[runs.length - 1]
    if (last && last.stage === s && last.end === i) last.end = i + 1
    else runs.push({ start: i, end: i + 1, stage: s })
  }
  return runs
}

function Lane({ runs, y0, x, title }: { runs: Run[]; y0: number; x: (v: number) => number; title: string }) {
  const rowY = (stage: number) => y0 + hypnoRow(stage) * ROW + ROW / 2
  const scored = runs.filter((r) => r.stage >= 0)
  return (
    <g>
      <text x={0} y={y0 - 8} style={{ fill: 'var(--ink)', fontWeight: 600, fontSize: 12 }}>{title}</text>
      {HYPNO_ORDER.map((s, i) => (
        <g key={s}>
          <line x1={LEFT} x2={x(Infinity)} y1={y0 + i * ROW + ROW / 2} y2={y0 + i * ROW + ROW / 2}
            stroke="var(--grid)" strokeWidth={1} />
          <text x={LEFT - 10} y={y0 + i * ROW + ROW / 2 + 4} textAnchor="end">{STAGES[s]}</text>
        </g>
      ))}
      {/* vertical transitions, recessive */}
      {scored.slice(1).map((r, i) => {
        const prev = scored[i]
        if (prev.end !== r.start) return null
        return <line key={`v${r.start}`} x1={x(r.start)} x2={x(r.start)} y1={rowY(prev.stage)} y2={rowY(r.stage)}
          stroke="var(--axis)" strokeWidth={1} />
      })}
      {/* stage segments carry the colour */}
      {scored.map((r) => (
        <line key={r.start} x1={x(r.start)} x2={Math.max(x(r.end), x(r.start) + 1)} y1={rowY(r.stage)} y2={rowY(r.stage)}
          stroke={stageColor(r.stage)} strokeWidth={4} strokeLinecap="butt" />
      ))}
    </g>
  )
}

export function Hypnogram({ night, range, playhead, selected, onSelect, onZoom }: {
  night: Night
  range: [number, number]
  playhead: number | null
  selected: number | null
  onSelect: (epoch: number) => void
  onZoom?: (range: [number, number]) => void
}) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [hover, setHover] = useState<number | null>(null)
  const [drag, setDrag] = useState<{ x0: number; x1: number } | null>(null)
  const [a, b] = range
  const hasExpert = night.expert !== null
  const W = Math.max(width, 320)
  const xs = linear(a, b, LEFT, W - RIGHT)
  const x = (v: number) => (v === Infinity ? W - RIGHT : xs(v))

  const expertTop = 22
  const strip = hasExpert ? expertTop + LANE + 10 : 0
  const predTop = hasExpert ? strip + 50 : 22
  const densTop = predTop + LANE + 34
  const axisTop = densTop + DENSITY + 8
  const H = axisTop + 22

  const layers = useMemo(() => {
    const sx = linear(a, b, LEFT, W - RIGHT)
    const predRuns = runsOf(night.predicted, a, b)
    const expRuns = night.expert ? runsOf(night.expert, a, b) : []
    const mismatch: number[] = []
    if (night.expert) {
      for (let i = a; i < b; i++) {
        if (night.expert[i] >= 0 && night.expert[i] !== night.predicted[i]) mismatch.push(i)
      }
    }
    // Hypnodensity: stacked probabilities, Wake on top down to N3 (matches the hypnogram rows).
    const stackOrder = [3, 2, 1, 4, 0]
    const areas = stackOrder.map((stage, k) => {
      const top: string[] = []
      const bottom: string[] = []
      for (let i = a; i < b; i++) {
        const p = night.probs[i]
        let below = 0
        for (let j = 0; j < k; j++) below += p[stackOrder[j]]
        const above = below + p[stage]
        const y1 = densTop + DENSITY * (1 - above)
        const y0 = densTop + DENSITY * (1 - below)
        top.push(`${sx(i).toFixed(1)},${y1.toFixed(1)} ${sx(i + 1).toFixed(1)},${y1.toFixed(1)}`)
        bottom.push(`${sx(i + 1).toFixed(1)},${y0.toFixed(1)} ${sx(i).toFixed(1)},${y0.toFixed(1)}`)
      }
      return { stage, d: `M${top.join(' L')} L${bottom.reverse().join(' L')}Z` }
    })
    return { predRuns, expRuns, mismatch, areas }
  }, [night, a, b, W, densTop])

  const ticks = useMemo(() => {
    const hours = ((b - a) * night.epoch_sec) / 3600
    const stepH = hours > 12 ? 2 : hours > 5 ? 1 : hours > 2 ? 0.5 : hours > 0.75 ? 0.25 : 1 / 12
    const stepE = (stepH * 3600) / night.epoch_sec
    const first = Math.ceil(a / stepE) * stepE
    const out: number[] = []
    for (let e = first; e <= b; e += stepE) out.push(e)
    return out
  }, [a, b, night.epoch_sec])

  const clipX = playhead === null ? W : x(Math.min(playhead + 1, b))

  const clampX = (px: number) => Math.min(Math.max(px, LEFT), W - RIGHT)
  const epochAt = (px: number) =>
    Math.min(Math.max(Math.floor(a + ((clampX(px) - LEFT) / (W - RIGHT - LEFT)) * (b - a)), a), b - 1)
  const localX = (e: MouseEvent<SVGSVGElement>) => e.clientX - e.currentTarget.getBoundingClientRect().left

  const onMove = (e: MouseEvent<SVGSVGElement>) => {
    const px = localX(e)
    if (drag) setDrag({ ...drag, x1: clampX(px) })
    if (px < LEFT || px > W - RIGHT) return setHover(null)
    setHover(epochAt(px))
  }
  const onDown = (e: MouseEvent<SVGSVGElement>) => {
    const px = localX(e)
    if (px >= LEFT && px <= W - RIGHT) setDrag({ x0: px, x1: px })
  }
  const onUp = (e: MouseEvent<SVGSVGElement>) => {
    if (!drag) return
    const px = clampX(localX(e))
    const e0 = epochAt(Math.min(drag.x0, px))
    const e1 = epochAt(Math.max(drag.x0, px))
    setDrag(null)
    if (Math.abs(px - drag.x0) > 6 && e1 - e0 >= 4 && onZoom) onZoom([e0, e1 + 1])
    else if (tip !== null) onSelect(tip)
  }

  const tip = hover !== null && (playhead === null || hover <= playhead) ? hover : null

  return (
    <div className="chart" ref={ref}>
      {width > 0 && (
        <svg width={W} height={H} onMouseMove={onMove} onMouseLeave={() => { setHover(null); setDrag(null) }}
          onMouseDown={onDown} onMouseUp={onUp} style={{ cursor: 'crosshair', userSelect: 'none' }}
          role="img" aria-label="Hypnogram: expert scoring and CNN prediction across the night">
          <defs>
            <clipPath id="played"><rect x={0} y={0} width={clipX} height={H} /></clipPath>
          </defs>

          <g clipPath="url(#played)">
            {hasExpert && <Lane runs={layers.expRuns} y0={expertTop} x={x} title="Expert scoring" />}
            {hasExpert && (
              <g>
                <text x={LEFT - 10} y={strip + 12} textAnchor="end">Disagree</text>
                <rect x={LEFT} y={strip + 4} width={W - RIGHT - LEFT} height={10} rx={2} fill="var(--surface-2)" />
                {layers.mismatch.map((i) => (
                  <rect key={i} x={xs(i)} y={strip + 4} width={Math.max(xs(i + 1) - xs(i), 1)} height={10}
                    fill="var(--critical)" opacity={0.85} />
                ))}
              </g>
            )}
            <Lane runs={layers.predRuns} y0={predTop} x={x} title="CNN prediction" />
            <text x={0} y={densTop - 8} style={{ fill: 'var(--ink)', fontWeight: 600, fontSize: 12 }}>Hypnodensity</text>
            <text x={LEFT - 10} y={densTop + DENSITY / 2 + 4} textAnchor="end">p(stage)</text>
            {layers.areas.map((ar) => (
              <path key={ar.stage} d={ar.d} fill={stageColor(ar.stage)} opacity={0.9} />
            ))}
          </g>

          {/* time axis */}
          <line x1={LEFT} x2={W - RIGHT} y1={axisTop} y2={axisTop} stroke="var(--axis)" />
          {ticks.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={axisTop} y2={axisTop + 4} stroke="var(--axis)" />
              <text className="tick-label" x={x(t)} y={axisTop + 16} textAnchor="middle">{clockTime(t, night.epoch_sec)}</text>
            </g>
          ))}

          {selected !== null && selected >= a && selected < b && (
            <rect x={xs(selected)} y={expertTop - 4} width={Math.max(xs(selected + 1) - xs(selected), 2)}
              height={axisTop - expertTop + 4} fill="var(--ink)" opacity={0.12} />
          )}
          {playhead !== null && playhead < b - 1 && (
            <g>
              <line x1={clipX} x2={clipX} y1={expertTop - 12} y2={axisTop} stroke="var(--ink)" strokeWidth={2} />
              <circle cx={clipX} cy={expertTop - 12} r={4} fill="var(--ink)" stroke="var(--surface)" strokeWidth={2} />
            </g>
          )}
          {drag && Math.abs(drag.x1 - drag.x0) > 2 && (
            <rect x={Math.min(drag.x0, drag.x1)} y={expertTop - 4} width={Math.abs(drag.x1 - drag.x0)}
              height={axisTop - expertTop + 4} fill="var(--seq-4)" opacity={0.18} stroke="var(--seq-5)" />
          )}
          {tip !== null && !drag && (
            <line x1={xs(tip + 0.5)} x2={xs(tip + 0.5)} y1={expertTop - 4} y2={axisTop}
              stroke="var(--ink-2)" strokeWidth={1} />
          )}
        </svg>
      )}
      {tip !== null && !drag && (
        <div className="tooltip" style={{ left: Math.min(Math.max(xs(tip + 0.5), 100), W - 100), top: expertTop }}>
          <div className="tooltip-title">Epoch {tip} · {clockTime(tip, night.epoch_sec)}</div>
          {night.expert && (
            <div className="tooltip-row">
              <span><span className="dot" style={{ background: stageColor(night.expert[tip]) }} />Expert</span>
              <strong>{night.expert[tip] >= 0 ? STAGES[night.expert[tip]] : 'Unscored'}</strong>
            </div>
          )}
          <div className="tooltip-row">
            <span><span className="dot" style={{ background: stageColor(night.predicted[tip]) }} />CNN</span>
            <strong>{STAGES[night.predicted[tip]]} · {pct(night.confidence[tip], 0)}</strong>
          </div>
          <div className="muted" style={{ marginTop: 6 }}>Click to inspect · drag to zoom</div>
        </div>
      )}
    </div>
  )
}
