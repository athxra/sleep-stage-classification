import type { ReactNode } from 'react'
import { STAGES, stageColor } from '../stages'
import type { Role } from '../api'

export function Card({ title, sub, action, children, footer, flush = false, className = '' }: {
  title?: ReactNode; sub?: ReactNode; action?: ReactNode; children: ReactNode; footer?: ReactNode
  flush?: boolean; className?: string
}) {
  return (
    <section className={`card panel ${className}`}>
      {(title || action) && (
        <div className="card-head">
          <div style={{ minWidth: 0 }}>
            {title && <h3 className="card-title">{title}</h3>}
            {sub && <p className="card-sub">{sub}</p>}
          </div>
          {action}
        </div>
      )}
      <div className={`card-body ${flush ? 'flush' : ''}`}>{children}</div>
      {footer && <div className="card-foot">{footer}</div>}
    </section>
  )
}

export type Kpi = { label: ReactNode; value: ReactNode; unit?: ReactNode; note?: ReactNode }

/** A single bordered strip of headline numbers, as on an analytics dashboard. */
export function KpiStrip({ items }: { items: Kpi[] }) {
  return (
    <div className="kpis">
      {items.map((k, i) => (
        <div className="kpi" key={i}>
          <span className="stat-label">{k.label}</span>
          <span className="stat-value">{k.value}{k.unit && <small>{k.unit}</small>}</span>
          {k.note && <span className="stat-note">{k.note}</span>}
        </div>
      ))}
    </div>
  )
}

/** Inline table-cell bar for a 0–1 score. */
export function Meter({ value, color, digits = 3 }: { value: number; color: string; digits?: number }) {
  return (
    <span className="cell-bar">
      <span className="track"><span className="fill" style={{ display: 'block', width: `${value * 100}%`, background: color }} /></span>
      {value.toFixed(digits)}
    </span>
  )
}

export function StageLegend({ only }: { only?: number[] }) {
  return (
    <div className="legend">
      {STAGES.map((s, i) => (only && !only.includes(i) ? null : (
        <span className="legend-item" key={s}>
          <span className="swatch" style={{ background: stageColor(i) }} />{s}
        </span>
      )))}
    </div>
  )
}

const ROLE_TEXT: Record<Role, string> = { test: 'Test', val: 'Val', train: 'Train', unseen: 'Unseen' }
export function RoleBadge({ role }: { role: Role | null }) {
  if (!role) return null
  return <span className={`role role-${role}`}>{ROLE_TEXT[role]}</span>
}

export function Segmented<T extends string>({ value, options, onChange, label }: {
  value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; label: string
}) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} aria-pressed={o.value === value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Skeleton({ height = 200 }: { height?: number }) {
  return <div className="skeleton" style={{ height }} />
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="notice">
      <Icon name="alert" />
      <div><strong>Could not load data.</strong> {message}. Start the API with <span className="kbd">python -m uvicorn api.server:app --port 8000</span></div>
    </div>
  )
}

const ICONS: Record<string, ReactNode> = {
  overview: <><rect x="3" y="3" width="7" height="9" rx="1.5" /><rect x="14" y="3" width="7" height="5" rx="1.5" /><rect x="14" y="12" width="7" height="9" rx="1.5" /><rect x="3" y="16" width="7" height="5" rx="1.5" /></>,
  night: <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z" />,
  chart: <><path d="M4 20V10" /><path d="M10 20V4" /><path d="M16 20v-7" /><path d="M22 20H2" /></>,
  cpu: <><rect x="6" y="6" width="12" height="12" rx="2" /><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" /></>,
  play: <path d="M7 4.5v15l12-7.5-12-7.5Z" />,
  pause: <><path d="M8 5v14" /><path d="M16 5v14" /></>,
  reset: <><path d="M3 12a9 9 0 1 0 3-6.7" /><path d="M3 4v5h5" /></>,
  alert: <><circle cx="12" cy="12" r="9" /><path d="M12 8v4.5M12 16h.01" /></>,
  check: <path d="m5 12.5 4.5 4.5L19 7.5" />,
  x: <path d="M6 6l12 12M18 6 6 18" />,
  prev: <path d="m15 6-6 6 6 6" />,
  next: <path d="m9 6 6 6-6 6" />,
  book: <><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15H6.5A2.5 2.5 0 0 0 4 20.5v-15Z" /><path d="M4 20.5A2.5 2.5 0 0 1 6.5 18H20v3H6.5" /></>,
  up: <path d="m6 15 6-6 6 6" />,
  down: <path d="m6 9 6 6 6-6" />,
  upload: <><path d="M12 16V4" /><path d="m7 9 5-5 5 5" /><path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" /></>,
  download: <><path d="M12 4v12" /><path d="m7 11 5 5 5-5" /><path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" /></>,
  search: <><circle cx="11" cy="11" r="6.5" /><path d="m20 20-4.2-4.2" /></>,
  zoomOut: <><circle cx="11" cy="11" r="6.5" /><path d="m20 20-4.2-4.2M8 11h6" /></>,
  flag: <><path d="M5 21V4" /><path d="M5 4h11l-2 4 2 4H5" /></>,
  link: <><path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1" /><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1" /></>,
  keyboard: <><rect x="2.5" y="6" width="19" height="12" rx="2" /><path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M7 14h10" /></>,
  arrowRight: <><path d="M5 12h14" /><path d="m13 6 6 6-6 6" /></>,
  sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>,
  moon: <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z" />,
  monitor: <><rect x="3" y="4" width="18" height="12" rx="2" /><path d="M8 20h8M12 16v4" /></>,
  database: <><ellipse cx="12" cy="5.5" rx="7.5" ry="2.5" /><path d="M4.5 5.5v13c0 1.4 3.4 2.5 7.5 2.5s7.5-1.1 7.5-2.5v-13" /><path d="M4.5 12c0 1.4 3.4 2.5 7.5 2.5s7.5-1.1 7.5-2.5" /></>,
  layers: <><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 13 9 5 9-5" /></>,
  file: <><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5Z" /><path d="M14 3v5h5" /></>,
}

export function Icon({ name }: { name: keyof typeof ICONS | string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {ICONS[name]}
    </svg>
  )
}
