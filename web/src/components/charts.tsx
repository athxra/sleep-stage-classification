import { useMemo, useState, type ReactNode } from 'react'
import { STAGES, pct, stageColor } from '../stages'
import { linear, useWidth } from './hooks'

/* ------------------------------------------------------------------
   EEG trace: one 30-s epoch, µV, with 1-s gridlines and hover readout.
   ------------------------------------------------------------------ */
export function EegTrace({ signal, sfreq, startSec }: { signal: number[]; sfreq: number; startSec: number }) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [hover, setHover] = useState<number | null>(null)
  const H = 330
  const L = 52, R = 8, T = 20, B = 26
  const W = Math.max(width, 300)
  const peak = useMemo(() => {
    const sorted = signal.map(Math.abs).sort((p, q) => p - q)
    const p99 = sorted[Math.floor(sorted.length * 0.995)] || 1
    const nice = [25, 50, 75, 100, 150, 200, 300, 400, 600, 800]
    return nice.find((n) => n >= p99) ?? Math.ceil(p99 / 100) * 100
  }, [signal])
  const x = linear(0, signal.length, L, W - R)
  const y = linear(-peak, peak, H - B, T)
  const d = useMemo(() => {
    const sx = linear(0, signal.length, L, W - R)
    const sy = linear(-peak, peak, H - B, T)
    return 'M' + signal.map((v, i) => `${sx(i).toFixed(1)},${sy(Math.max(-peak, Math.min(peak, v))).toFixed(1)}`).join('L')
  }, [signal, W, peak])
  const secs = Array.from({ length: 31 }, (_, i) => i)

  return (
    <div className="chart" ref={ref}>
      {width > 0 && (
        <svg width={W} height={H} role="img" aria-label="EEG waveform of the selected epoch"
          onMouseMove={(e) => {
            const px = e.clientX - e.currentTarget.getBoundingClientRect().left
            const i = Math.round(((px - L) / (W - R - L)) * signal.length)
            setHover(i >= 0 && i < signal.length ? i : null)
          }}
          onMouseLeave={() => setHover(null)}>
          {secs.map((s) => (
            <line key={s} x1={x(s * sfreq)} x2={x(s * sfreq)} y1={T} y2={H - B}
              stroke="var(--grid)" strokeWidth={s % 5 === 0 ? 1 : 0.5} />
          ))}
          {[-peak, -peak / 2, 0, peak / 2, peak].map((v) => (
            <g key={v}>
              <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke={v === 0 ? 'var(--axis)' : 'var(--grid)'} />
              <text className="tick-label" x={L - 8} y={y(v) + 4} textAnchor="end">{v}</text>
            </g>
          ))}
          <text x={L - 8} y={T - 10} textAnchor="end">µV</text>
          <path d={d} fill="none" stroke="var(--ink)" strokeWidth={1} strokeLinejoin="round" />
          {secs.filter((s) => s % 5 === 0).map((s) => (
            <text key={s} className="tick-label" x={x(s * sfreq)} y={H - 8} textAnchor="middle">{s} s</text>
          ))}
          {hover !== null && (
            <g>
              <line x1={x(hover)} x2={x(hover)} y1={T} y2={H - B} stroke="var(--ink-2)" />
              <circle cx={x(hover)} cy={y(Math.max(-peak, Math.min(peak, signal[hover])))} r={4}
                fill="var(--ink)" stroke="var(--surface)" strokeWidth={2} />
            </g>
          )}
        </svg>
      )}
      {hover !== null && (
        <div className="tooltip" style={{ left: Math.min(Math.max(x(hover), 90), W - 90), top: T + 4 }}>
          <div className="tooltip-row"><span>Time</span><strong className="tnum">{(hover / sfreq).toFixed(2)} s <span className="muted">({(startSec + hover / sfreq).toFixed(0)} s into recording)</span></strong></div>
          <div className="tooltip-row"><span>Amplitude</span><strong className="tnum">{signal[hover].toFixed(1)} µV</strong></div>
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------
   Horizontal bars with value at the tip (optionally ± error whisker).
   ------------------------------------------------------------------ */
export type BarRow = { label: ReactNode; value: number; color: string; err?: number; note?: ReactNode }

export function HBars({ rows, max = 1, format = (v: number) => pct(v), labelWidth = 64 }: {
  rows: BarRow[]; max?: number; format?: (v: number) => string; labelWidth?: number
}) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [hover, setHover] = useState<number | null>(null)
  const rowH = 30, bar = 14, valueW = 64
  const W = Math.max(width, 240)
  const x = linear(0, max, labelWidth, W - valueW)
  return (
    <div className="chart" ref={ref}>
      {width > 0 && (
        <svg width={W} height={rows.length * rowH} role="img" aria-label="Bar chart">
          {rows.map((r, i) => {
            const y = i * rowH + (rowH - bar) / 2
            const w = Math.max(x(Math.min(r.value, max)) - labelWidth, 0)
            const end = r.err ? x(Math.min(r.value + r.err, max)) : x(Math.min(r.value, max))
            return (
              <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
                <rect x={0} y={i * rowH} width={W} height={rowH} fill={hover === i ? 'var(--surface-2)' : 'transparent'} rx={4} />
                <text x={0} y={y + bar / 2 + 4} style={{ fill: 'var(--ink-2)', fontSize: 12 }}>{r.label}</text>
                <rect x={labelWidth} y={y} width={W - valueW - labelWidth} height={bar} rx={4} fill="var(--surface-2)" />
                {w > 0 && (
                  <path d={`M${labelWidth},${y} h${Math.max(w - 4, 0)} a4,4 0 0 1 4,4 v${bar - 8} a4,4 0 0 1 -4,4 h${-Math.max(w - 4, 0)} Z`}
                    fill={r.color} />
                )}
                {r.err !== undefined && r.err > 0 && (
                  <g stroke="var(--ink)" strokeWidth={1.5}>
                    <line x1={x(Math.max(r.value - r.err, 0))} x2={end} y1={y + bar / 2} y2={y + bar / 2} />
                    <line x1={x(Math.max(r.value - r.err, 0))} x2={x(Math.max(r.value - r.err, 0))} y1={y + 3} y2={y + bar - 3} />
                    <line x1={end} x2={end} y1={y + 3} y2={y + bar - 3} />
                  </g>
                )}
                <text className="tick-label" x={W - valueW + 10} y={y + bar / 2 + 4} style={{ fill: 'var(--ink)', fontSize: 12 }}>
                  {format(r.value)}{r.err ? ` ±${format(r.err).replace('%', '')}` : ''}
                </text>
              </g>
            )
          })}
        </svg>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------
   Confusion matrix on a sequential (single-hue) ramp.
   ------------------------------------------------------------------ */
const SEQ = 8
export function ConfusionMatrix({ counts, normalized, mode, picked, onPick }: {
  counts: number[][]; normalized: number[][]; mode: 'pct' | 'count'
  picked?: [number, number] | null; onPick?: (cell: [number, number]) => void
}) {
  const [hover, setHover] = useState<[number, number] | null>(null)
  const cell = 'minmax(0, 1fr)'
  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: `56px repeat(5, ${cell})`, gap: 2 }}>
        <div />
        {STAGES.map((s) => <div key={s} className="muted" style={{ fontSize: 12, textAlign: 'center', paddingBottom: 4 }}>{s}</div>)}
        {counts.map((row, i) => (
          <Row key={i} i={i} row={row} norm={normalized[i]} mode={mode} hover={hover} setHover={setHover}
            picked={picked ?? null} onPick={onPick} />
        ))}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 10 }} className="muted">
        <span style={{ fontSize: 12 }}>Rows: expert stage · Columns: CNN prediction</span>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
          0%
          <span style={{ display: 'inline-flex' }}>
            {Array.from({ length: SEQ }, (_, k) => <span key={k} style={{ width: 12, height: 8, background: `var(--seq-${k})` }} />)}
          </span>
          100%
        </span>
      </div>
    </div>
  )
}

function Row({ i, row, norm, mode, hover, setHover, picked, onPick }: {
  i: number; row: number[]; norm: number[]; mode: 'pct' | 'count'
  hover: [number, number] | null; setHover: (v: [number, number] | null) => void
  picked: [number, number] | null; onPick?: (cell: [number, number]) => void
}) {
  return (
    <>
      <div className="muted" style={{ fontSize: 12, display: 'flex', alignItems: 'center' }}>{STAGES[i]}</div>
      {row.map((c, j) => {
        const r = norm[j]
        const step = Math.min(SEQ - 1, Math.round(r * (SEQ - 1)))
        const active = (hover && hover[0] === i && hover[1] === j) || (picked && picked[0] === i && picked[1] === j)
        const clickable = onPick && i !== j && c > 0
        return (
          <div key={j} onMouseEnter={() => setHover([i, j])} onMouseLeave={() => setHover(null)}
            role={clickable ? 'button' : undefined} tabIndex={clickable ? 0 : undefined}
            onClick={clickable ? () => onPick!([i, j]) : undefined}
            onKeyDown={clickable ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onPick!([i, j]) } } : undefined}
            title={`Expert ${STAGES[i]} → predicted ${STAGES[j]}: ${c.toLocaleString()} epochs (${pct(r)} of ${STAGES[i]})`}
            style={{
              aspectRatio: '1.35', borderRadius: 4, background: `var(--seq-${step})`,
              display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column',
              color: `var(--seq-text-${step})`, fontVariantNumeric: 'tabular-nums',
              outline: active ? '2px solid var(--ink)' : i === j ? '1px solid var(--border)' : 'none',
              outlineOffset: -2, cursor: clickable ? 'pointer' : 'default', transition: 'outline .1s',
            }}>
            <span style={{ fontWeight: 600, fontSize: 14 }}>{mode === 'pct' ? pct(r, 0) : c.toLocaleString()}</span>
            <span style={{ fontSize: 10.5, opacity: 0.75 }}>{mode === 'pct' ? c.toLocaleString() : pct(r, 0)}</span>
          </div>
        )
      })}
    </>
  )
}

/* ------------------------------------------------------------------
   Per-fold dot plot with mean line and ±1 std band (single series).
   ------------------------------------------------------------------ */
export function FoldDots({ folds, mean, std, label }: {
  folds: { fold: number; value: number; subjects: number[] }[]; mean: number; std: number; label: string
}) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [hover, setHover] = useState<number | null>(null)
  const H = 200, L = 40, R = 12, T = 12, B = 28
  const W = Math.max(width, 260)
  const vals = folds.map((f) => f.value)
  const lo = Math.max(0, Math.floor((Math.min(...vals, mean - std) - 0.05) * 20) / 20)
  const hi = Math.min(1, Math.ceil((Math.max(...vals, mean + std) + 0.05) * 20) / 20)
  const x = (i: number) => L + ((i + 0.5) / folds.length) * (W - L - R)
  const y = linear(lo, hi, H - B, T)
  const ticks = Array.from({ length: Math.round((hi - lo) / 0.05) + 1 }, (_, k) => +(lo + k * 0.05).toFixed(2))
  return (
    <div className="chart" ref={ref}>
      {width > 0 && (
        <svg width={W} height={H} role="img" aria-label={`${label} per cross-validation fold`}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
              <text className="tick-label" x={L - 8} y={y(t) + 4} textAnchor="end">{t.toFixed(2)}</text>
            </g>
          ))}
          <rect x={L} width={W - L - R} y={y(mean + std)} height={y(mean - std) - y(mean + std)} fill="var(--seq-4)" opacity={0.12} />
          <line x1={L} x2={W - R} y1={y(mean)} y2={y(mean)} stroke="var(--seq-5)" strokeWidth={2} />
          <text x={W - R} y={y(mean) - 6} textAnchor="end" style={{ fill: 'var(--ink-2)', fontSize: 11 }}>
            mean {mean.toFixed(3)} ± {std.toFixed(3)}
          </text>
          {folds.map((f, i) => (
            <g key={f.fold} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect x={x(i) - 20} y={T} width={40} height={H - B - T} fill="transparent" />
              <circle cx={x(i)} cy={y(f.value)} r={hover === i ? 7 : 6} fill="var(--seq-5)" stroke="var(--surface)" strokeWidth={2} />
              <text className="tick-label" x={x(i)} y={H - 8} textAnchor="middle">Fold {f.fold}</text>
            </g>
          ))}
        </svg>
      )}
      {hover !== null && (
        <div className="tooltip" style={{ left: x(hover), top: y(folds[hover].value) }}>
          <div className="tooltip-title">Fold {folds[hover].fold}</div>
          <div className="tooltip-row"><span>{label}</span><strong className="tnum">{folds[hover].value.toFixed(3)}</strong></div>
          <div className="tooltip-row"><span>Test subjects</span><strong>{folds[hover].subjects.join(', ')}</strong></div>
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------
   100% stacked stage bars (e.g. sleep architecture, expert vs CNN).
   ------------------------------------------------------------------ */
export function StageStack({ rows }: { rows: { label: string; minutes: Record<string, number> }[] }) {
  const [hover, setHover] = useState<string | null>(null)
  const [ref, width] = useWidth<HTMLDivElement>()
  const order = [0, 4, 1, 2, 3]
  return (
    <div ref={ref} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {rows.map((row) => {
        const total = order.reduce((s, i) => s + (row.minutes[STAGES[i]] ?? 0), 0) || 1
        return (
          <div key={row.label}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 6 }}>
              <span className="ink2">{row.label}</span>
              <span className="muted tnum">{Math.round(total)} min</span>
            </div>
            <div style={{ display: 'flex', gap: 2, height: 22 }}>
              {order.map((i) => {
                const m = row.minutes[STAGES[i]] ?? 0
                if (m <= 0) return null
                const share = m / total
                const key = `${row.label}-${i}`
                const text = `${STAGES[i]} ${Math.round(share * 100)}%`
                const fits = share * width >= text.length * 6.5 + 12
                return (
                  <div key={i} onMouseEnter={() => setHover(key)} onMouseLeave={() => setHover(null)}
                    title={`${STAGES[i]}: ${Math.round(m)} min (${pct(share)})`}
                    style={{
                      flex: `${share} 0 0`, background: stageColor(i), borderRadius: 4, minWidth: 2,
                      opacity: hover && hover !== key ? 0.55 : 1, transition: 'opacity .15s, flex .4s ease',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      color: '#0b0b0b', fontSize: 11, fontWeight: 600, whiteSpace: 'nowrap',
                    }}>
                    {fits ? text : ''}
                  </div>
                )
              })}
            </div>
          </div>
        )
      })}
    </div>
  )
}

/* ------------------------------------------------------------------
   Power spectrum of one epoch (dB), with the classic EEG bands shaded.
   ------------------------------------------------------------------ */
const BAND_RANGES: [string, number, number][] = [
  ['Delta', 0.5, 4], ['Theta', 4, 8], ['Alpha', 8, 12], ['Sigma', 12, 15], ['Beta', 15, 30],
]

export function Spectrum({ freqs, db }: { freqs: number[]; db: number[] }) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [hover, setHover] = useState<number | null>(null)
  const H = 210, L = 44, R = 8, T = 22, B = 26
  const W = Math.max(width, 260)
  const lo = Math.floor(Math.min(...db) / 10) * 10
  const hi = Math.ceil(Math.max(...db) / 10) * 10
  const x = linear(0, 30, L, W - R)
  const y = linear(lo, hi, H - B, T)
  const d = 'M' + freqs.map((f, i) => `${x(f).toFixed(1)},${y(db[i]).toFixed(1)}`).join('L')
  const area = `${d}L${x(freqs[freqs.length - 1]).toFixed(1)},${H - B}L${x(freqs[0]).toFixed(1)},${H - B}Z`
  const yTicks = Array.from({ length: Math.round((hi - lo) / 10) + 1 }, (_, k) => lo + k * 10)
  return (
    <div className="chart" ref={ref}>
      {width > 0 && (
        <svg width={W} height={H} role="img" aria-label="Power spectrum of the selected epoch"
          onMouseMove={(e) => {
            const px = e.clientX - e.currentTarget.getBoundingClientRect().left
            const f = ((px - L) / (W - R - L)) * 30
            let best = 0
            freqs.forEach((q, i) => { if (Math.abs(q - f) < Math.abs(freqs[best] - f)) best = i })
            setHover(px >= L && px <= W - R ? best : null)
          }}
          onMouseLeave={() => setHover(null)}>
          {BAND_RANGES.map(([name, f0, f1], i) => (
            <g key={name}>
              <rect x={x(f0)} y={T} width={x(f1) - x(f0)} height={H - B - T}
                fill={i % 2 ? 'var(--surface-2)' : 'transparent'} />
              <text x={(x(f0) + x(f1)) / 2} y={T - 8} textAnchor="middle" style={{ fontSize: 10.5 }}>{name}</text>
            </g>
          ))}
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
              <text className="tick-label" x={L - 8} y={y(t) + 4} textAnchor="end">{t}</text>
            </g>
          ))}
          <text x={L - 8} y={T - 8} textAnchor="end" style={{ fontSize: 10 }}>dB</text>
          <path d={area} fill="var(--ink)" opacity={0.06} />
          <path d={d} fill="none" stroke="var(--ink)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          {[0, 5, 10, 15, 20, 25, 30].map((f) => (
            <text key={f} className="tick-label" x={x(f)} y={H - 8} textAnchor="middle">{f} Hz</text>
          ))}
          {hover !== null && (
            <g>
              <line x1={x(freqs[hover])} x2={x(freqs[hover])} y1={T} y2={H - B} stroke="var(--ink-2)" />
              <circle cx={x(freqs[hover])} cy={y(db[hover])} r={4} fill="var(--ink)" stroke="var(--surface)" strokeWidth={2} />
            </g>
          )}
        </svg>
      )}
      {hover !== null && (
        <div className="tooltip" style={{ left: Math.min(Math.max(x(freqs[hover]), 80), W - 80), top: y(db[hover]) }}>
          <div className="tooltip-row"><span>Frequency</span><strong className="tnum">{freqs[hover].toFixed(2)} Hz</strong></div>
          <div className="tooltip-row"><span>Power</span><strong className="tnum">{db[hover].toFixed(1)} dB</strong></div>
        </div>
      )}
    </div>
  )
}
