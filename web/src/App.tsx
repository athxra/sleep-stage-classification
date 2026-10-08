import { useEffect, useState } from 'react'
import { api, useFetch } from './api'
import { ShortcutsDialog } from './components/help'
import { ErrorNote, Icon, Skeleton } from './components/ui'
import { Model } from './pages/Model'
import { NightExplorer } from './pages/NightExplorer'
import { Overview } from './pages/Overview'
import { Performance } from './pages/Performance'
import { navigate, useRoute } from './router'

const PAGES = [
  { id: 'overview', label: 'Overview', icon: 'overview' },
  { id: 'night', label: 'Night explorer', icon: 'night' },
  { id: 'performance', label: 'Performance', icon: 'chart' },
  { id: 'model', label: 'Model', icon: 'cpu' },
] as const
type PageId = (typeof PAGES)[number]['id']
type Theme = 'system' | 'light' | 'dark'

function readTheme(): Theme {
  try { return (localStorage.getItem('theme') as Theme) || 'system' } catch { return 'system' }
}

export default function App() {
  const route = useRoute()
  const page = (PAGES.find((p) => p.id === route.page)?.id ?? 'overview') as PageId
  const [theme, setTheme] = useState<Theme>(readTheme)
  const [shortcuts, setShortcuts] = useState(false)
  const summary = useFetch('summary', api.summary)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName
      if (e.key === '?' && tag !== 'INPUT' && tag !== 'TEXTAREA') setShortcuts(true)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => {
    const label = PAGES.find((p) => p.id === page)?.label ?? 'Overview'
    document.title = `${label} · SleepNet`
  }, [page])

  useEffect(() => {
    const root = document.documentElement
    if (theme === 'system') root.removeAttribute('data-theme')
    else root.setAttribute('data-theme', theme)
    try { localStorage.setItem('theme', theme) } catch { /* storage unavailable */ }
  }, [theme])

  const go = (id: string) => navigate(id)
  const current = PAGES.find((p) => p.id === page)!
  const nextTheme: Record<Theme, Theme> = { system: 'light', light: 'dark', dark: 'system' }
  const themeIcon = theme === 'light' ? 'sun' : theme === 'dark' ? 'moon' : 'monitor'
  const status = summary.error ? 'is-down' : summary.loading ? 'is-wait' : ''
  const m = summary.data?.model
  const split = summary.data?.split

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
            <rect width="32" height="32" rx="8" fill="var(--accent)" />
            <path d="M5 12h5v4h5v6h5v-8h7" fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <div>
            <div className="brand-name">SleepNet</div>
            <div className="brand-sub">EEG sleep staging</div>
          </div>
        </div>
        <div>
          <div className="nav-section">Analytics</div>
          <nav className="nav" aria-label="Pages">
            {PAGES.map((p) => (
              <button key={p.id} aria-current={page === p.id ? 'page' : undefined} onClick={() => go(p.id)}>
                <Icon name={p.icon} />{p.label}
              </button>
            ))}
          </nav>
        </div>
        <div className="sidebar-foot">
          <div className="nav-section">Deployment</div>
          <div className="meta-list">
            <div className="meta-row"><span>Dataset</span><span>Sleep-EDF-20</span></div>
            <div className="meta-row"><span>Channel</span><span>Fpz-Cz · 100 Hz</span></div>
            <div className="meta-row"><span>Subjects</span><span>{split?.n_subjects ?? '—'}</span></div>
            <div className="meta-row"><span>Parameters</span><span>{m ? m.parameters.toLocaleString() : '—'}</span></div>
            <div className="meta-row"><span>Checkpoint</span><span>{m ? `epoch ${m.checkpoint_epoch}` : '—'}</span></div>
          </div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="crumbs">
            <span>SleepNet</span><span aria-hidden="true">/</span><strong>{current.label}</strong>
          </div>
          <div className="topbar-right">
            <span className="chip optional"><Icon name="database" /><strong>Sleep-EDF</strong> SC · 5-class</span>
            <span className="chip"><span className={`status-dot ${status}`} />
              {summary.error ? 'API offline' : summary.loading ? 'Connecting' : 'API online'}</span>
            <button className="icon-btn" onClick={() => setShortcuts(true)} aria-label="Keyboard shortcuts" title="Keyboard shortcuts (?)">
              <Icon name="keyboard" />
            </button>
            <button className="icon-btn" onClick={() => setTheme(nextTheme[theme])}
              aria-label={`Theme: ${theme}. Switch to ${nextTheme[theme]}`} title={`Theme: ${theme}`}>
              <Icon name={themeIcon} />
            </button>
          </div>
        </header>

        <div className="content">
          {summary.error && <div className="page"><ErrorNote message={summary.error} /></div>}
          {summary.loading && (
            <div className="page"><Skeleton height={48} /><Skeleton height={92} /><div className="grid g3">{[0, 1, 2].map((i) => <Skeleton key={i} height={300} />)}</div></div>
          )}
          {summary.data && page === 'overview' && <Overview summary={summary.data} onNavigate={go} />}
          {page === 'night' && <NightExplorer key={route.params.get('rec') ?? 'default'} params={route.params} />}
          {summary.data && page === 'performance' && <Performance summary={summary.data} />}
          {summary.data && page === 'model' && <Model summary={summary.data} />}
        </div>
      </main>
      <ShortcutsDialog open={shortcuts} onClose={() => setShortcuts(false)} />
    </div>
  )
}
