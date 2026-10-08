export const STAGES = ['Wake', 'N1', 'N2', 'N3', 'REM'] as const

export const stageColor = (i: number) => `var(--stage-${i})`

/** Hypnogram rows, top to bottom, following the clinical convention. */
export const HYPNO_ORDER = [0, 4, 1, 2, 3] // Wake, REM, N1, N2, N3
export const hypnoRow = (stage: number) => HYPNO_ORDER.indexOf(stage)

export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`
export const fixed = (v: number, digits = 3) => v.toFixed(digits)
export const compact = (v: number) =>
  v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `${(v / 1e3).toFixed(1)}K` : `${v}`
export const minutes = (v: number | null) => (v === null ? '—' : `${Math.round(v)} min`)

export function clockTime(epoch: number, epochSec = 30) {
  const s = epoch * epochSec
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  return `${h}h ${String(m).padStart(2, '0')}m`
}

export function kappaWord(k: number) {
  if (k > 0.8) return 'Almost perfect agreement'
  if (k > 0.6) return 'Substantial agreement'
  if (k > 0.4) return 'Moderate agreement'
  return 'Fair agreement'
}
