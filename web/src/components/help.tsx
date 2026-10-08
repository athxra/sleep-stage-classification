import { useEffect, useId, useRef, useState, type ReactNode } from 'react'
import { GLOSSARY } from '../glossary'
import { Icon } from './ui'

export function InfoTip({ term, children }: { term: keyof typeof GLOSSARY | string; children?: ReactNode }) {
  const entry = GLOSSARY[term]
  const [open, setOpen] = useState(false)
  const id = useId()
  if (!entry) return <>{children}</>
  return (
    <span className="infotip" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      {children}
      <button type="button" className="infotip-btn" aria-label={`What is ${entry.title}?`} aria-describedby={open ? id : undefined}
        onFocus={() => setOpen(true)} onBlur={() => setOpen(false)} onClick={() => setOpen((o) => !o)}>
        <svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.3" /><path d="M8 7.2v4M8 4.9v.1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" /></svg>
      </button>
      {open && (
        <span role="tooltip" id={id} className="infotip-pop">
          <strong>{entry.title}</strong>
          <span>{entry.text}</span>
        </span>
      )}
    </span>
  )
}

export function Dialog({ open, onClose, title, children, width = 520 }: {
  open: boolean; onClose: () => void; title: string; children: ReactNode; width?: number
}) {
  const ref = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    const d = ref.current
    if (!d) return
    if (open && !d.open) d.showModal()
    if (!open && d.open) d.close()
  }, [open])
  return (
    <dialog ref={ref} className="dialog" style={{ width }} onClose={onClose}
      onClick={(e) => { if (e.target === ref.current) onClose() }} aria-label={title}>
      <div className="dialog-head">
        <h2 className="card-title" style={{ fontSize: 16 }}>{title}</h2>
        <button className="btn btn-ghost" onClick={onClose} aria-label="Close"><Icon name="x" /></button>
      </div>
      <div className="dialog-body">{children}</div>
    </dialog>
  )
}

const SHORTCUTS: [string[], string][] = [
  [['←', '→'], 'Previous / next epoch'],
  [['Shift', '←/→'], 'Jump 10 epochs'],
  [['D'], 'Next disagreement with the expert'],
  [['Shift', 'D'], 'Previous disagreement'],
  [['Space'], 'Play / pause the overnight replay'],
  [['Z'], 'Reset hypnogram zoom'],
  [['?'], 'Show this help'],
]

export function ShortcutsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <Dialog open={open} onClose={onClose} title="Keyboard shortcuts" width={440}>
      <p className="ink2" style={{ fontSize: 13, marginBottom: 12 }}>Available on the Night explorer page.</p>
      <table className="table">
        <tbody>
          {SHORTCUTS.map(([keys, text]) => (
            <tr key={text}>
              <td style={{ whiteSpace: 'nowrap' }}>{keys.map((k) => <span key={k} className="kbd" style={{ marginRight: 4 }}>{k}</span>)}</td>
              <td className="ink2">{text}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted" style={{ fontSize: 12, marginTop: 12 }}>On the hypnogram: drag across a stretch of the night to zoom in; click an epoch to inspect it.</p>
    </Dialog>
  )
}
