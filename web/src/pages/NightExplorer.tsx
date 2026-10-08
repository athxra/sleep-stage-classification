import { useEffect, useMemo, useRef, useState, type DragEvent, type ReactNode } from 'react'
import { api, useFetch, type Night, type Recording, type Role } from '../api'
import { EegTrace, HBars, Spectrum, StageStack } from '../components/charts'
import { Dialog, InfoTip } from '../components/help'
import { Hypnogram } from '../components/Hypnogram'
import { Card, ErrorNote, Icon, KpiStrip, RoleBadge, Segmented, Skeleton, StageLegend } from '../components/ui'
import { replaceParams } from '../router'
import { STAGES, clockTime, kappaWord, minutes, pct, stageColor } from '../stages'

type Filter = 'all' | Role

export function NightExplorer({ params }: { params: URLSearchParams }) {
  const recs = useFetch('recordings', api.recordings)
  const [filter, setFilter] = useState<Filter>('test')
  const [query, setQuery] = useState('')
  const [picked, setPicked] = useState<string | null>(params.get('rec'))
  const [uploadOpen, setUploadOpen] = useState(false)
  const initialEpoch = params.get('epoch') !== null ? Number(params.get('epoch')) : null

  const list = recs.data?.recordings ?? []
  // Default to the first held-out test night until the user picks one.
  const key = picked ?? (list.find((r) => r.role === 'test') ?? list[0])?.key ?? null
  const pickedRec = list.find((r) => r.key === key)

  const q = query.trim().toLowerCase()
  const shown = list.filter((r) => (filter === 'all' || r.role === filter || r.uploaded)
    && (!q || `${r.label} subject ${r.subject ?? ''}`.toLowerCase().includes(q)))
    .sort((x, y) => Number(Boolean(y.uploaded)) - Number(Boolean(x.uploaded)))   // your uploads first
  const night = useFetch(key, () => api.night(key!))

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1 className="page-title">Night explorer</h1>
          <p className="page-sub">Epoch-by-epoch CNN scoring against the expert hypnogram, with overnight replay and signal inspection</p>
        </div>
        <button className="btn" onClick={() => setUploadOpen(true)}><Icon name="upload" />Analyze EDF</button>
      </header>

      {recs.error && <ErrorNote message={recs.error} />}
      {recs.data?.demo_mode && (
        <div className="notice is-warn"><Icon name="alert" /><div><strong>Demo mode.</strong> No raw EDF files were found,
          so this uses the bundled excerpt: {list.find((r) => r.demo)?.source}.</div></div>
      )}

      <div className="grid explorer-grid">
        <Card title="Recordings" sub={`${list.length} nights · ${list.filter((r) => r.role === 'test').length} held-out`}>
          <div className="search">
            <Icon name="search" />
            <input type="search" placeholder="Search SC4071 or subject 7" value={query}
              onChange={(e) => setQuery(e.target.value)} aria-label="Search recordings" />
          </div>
          <div style={{ margin: '10px 0 12px' }}>
            <Segmented<Filter> label="Filter by split" value={filter} onChange={setFilter}
              options={[{ value: 'test', label: 'Test' }, { value: 'val', label: 'Val' },
                { value: 'train', label: 'Train' }, { value: 'all', label: 'All' }]} />
          </div>
          {recs.loading && !recs.data ? <Skeleton height={300} /> : (
            <div className="rec-list" role="listbox" aria-label="Recordings">
              {shown.map((r) => <RecordingItem key={r.key} rec={r} active={r.key === key} onPick={() => setPicked(r.key)} />)}
              {!shown.length && <p className="muted" style={{ fontSize: 13, padding: 8 }}>No recordings match.</p>}
            </div>
          )}
          <div className="legend" style={{ marginTop: 12, gap: 10, fontSize: 11.5 }}>
            <span className="legend-item"><span className="role role-test">Test</span>held out</span>
            <span className="legend-item"><span className="role role-val">Val</span>selection</span>
            <span className="legend-item"><span className="role role-train">Train</span>fit</span>
          </div>
        </Card>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, minWidth: 0 }}>
          {night.error && <ErrorNote message={night.error} />}
          {night.loading && !night.data && <LoadingNight name={pickedRec?.label} hours={pickedRec?.hours ?? null} />}
          {night.data && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16, opacity: night.loading ? 0.55 : 1, transition: 'opacity .2s' }}>
              {night.loading && <div className="notice"><span className="spinner" />Loading {pickedRec?.label}…</div>}
              <NightView key={night.data.key} night={night.data}
                initialEpoch={night.data.key === params.get('rec') ? initialEpoch : null} />
            </div>
          )}
        </div>
      </div>

      <UploadDialog open={uploadOpen} onClose={() => setUploadOpen(false)}
        onDone={(k) => { setUploadOpen(false); setPicked(k); recs.reload() }} />
    </div>
  )
}

function LoadingNight({ name, hours }: { name?: string; hours: number | null }) {
  return (
    <>
      <div className="notice"><span className="spinner" />
        <div><strong>Preparing {name ?? 'recording'}.</strong> Reading the EDF file and running the CNN on
          {hours ? ` about ${Math.round((hours * 3600) / 30).toLocaleString()}` : ' every'} epochs. This takes a second or two.</div>
      </div>
      <div className="grid g4">{[0, 1, 2, 3].map((i) => <Skeleton key={i} height={92} />)}</div>
      <Skeleton height={460} />
    </>
  )
}

function RecordingItem({ rec, active, onPick }: { rec: Recording; active: boolean; onPick: () => void }) {
  return (
    <button className="rec-item" aria-current={active} onClick={onPick} role="option" aria-selected={active}>
      <div style={{ minWidth: 0 }}>
        <div className="rec-name" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{rec.label}</div>
        <div className="rec-meta">
          {rec.uploaded ? `Uploaded · ${rec.channel ?? 'EEG'}` : rec.subject !== null ? `Subject ${rec.subject}` : 'Sample'}
          {rec.night ? ` · night ${rec.night}` : ''}{rec.hours ? ` · ${rec.hours} h` : ''}
        </div>
      </div>
      {rec.uploaded ? <span className="role">New</span> : <RoleBadge role={rec.role} />}
    </button>
  )
}

/* ------------------------------------------------------------------ */

function NightView({ night, initialEpoch }: { night: Night; initialEpoch: number | null }) {
  const [view, setView] = useState<'sleep' | 'full'>('sleep')
  const [zoom, setZoom] = useState<[number, number] | null>(null)
  const base: [number, number] = view === 'sleep' ? night.sleep_window : [0, night.n_epochs]
  const [r0, r1] = zoom ?? base
  const [playhead, setPlayhead] = useState<number | null>(null)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(4)
  const [selected, setSelected] = useState<number>(() =>
    initialEpoch !== null && initialEpoch >= 0 && initialEpoch < night.n_epochs
      ? initialEpoch : Math.min(night.sleep_window[0] + 90, night.n_epochs - 1))
  const [copied, setCopied] = useState(false)

  // Keep the URL shareable: #night?rec=...&epoch=...
  useEffect(() => { replaceParams('night', { rec: night.key, epoch: selected }) }, [night.key, selected])

  const disagreements = useMemo(() => {
    if (!night.expert) return []
    const out: number[] = []
    for (let i = 0; i < night.n_epochs; i++) {
      if (night.expert[i] >= 0 && night.expert[i] !== night.predicted[i]) out.push(i)
    }
    return out
  }, [night])
  const inRange = disagreements.filter((i) => i >= r0 && i < r1)

  const jumpDisagreement = (dir: 1 | -1) => {
    const list = inRange.length ? inRange : disagreements
    if (!list.length) return
    const next = dir === 1
      ? list.find((i) => i > selected) ?? list[0]
      : [...list].reverse().find((i) => i < selected) ?? list[list.length - 1]
    setPlaying(false); setPlayhead(null)
    if (next < r0 || next >= r1) setZoom(null)
    setSelected(next)
  }

  // Overnight replay: advance the playhead `speed` epochs per frame.
  useEffect(() => {
    if (!playing) return
    const id = window.setInterval(() => {
      setPlayhead((p) => {
        const next = (p ?? r0) + speed
        if (next >= r1 - 1) { setPlaying(false); return null }
        return next
      })
    }, 60)
    return () => window.clearInterval(id)
  }, [playing, speed, r0, r1])

  const togglePlay = () => {
    if (playing) {
      setPlaying(false)
      if (playhead !== null) setSelected(playhead)
    } else {
      if (playhead === null) setPlayhead(r0)
      setPlaying(true)
    }
  }
  const reset = () => { setPlaying(false); setPlayhead(null) }

  // Keyboard shortcuts read the latest handlers through a ref.
  const actions = useRef({ togglePlay, jumpDisagreement })
  useEffect(() => { actions.current = { togglePlay, jumpDisagreement } })
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || e.metaKey || e.ctrlKey || e.altKey) return
      if (document.querySelector('dialog[open]')) return
      const step = e.shiftKey ? 10 : 1
      if (e.key === 'ArrowRight') { e.preventDefault(); setSelected((s) => Math.min(s + step, night.n_epochs - 1)) }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); setSelected((s) => Math.max(s - step, 0)) }
      else if (e.key === ' ') { e.preventDefault(); actions.current.togglePlay() }
      else if (e.key.toLowerCase() === 'd') actions.current.jumpDisagreement(e.shiftKey ? -1 : 1)
      else if (e.key.toLowerCase() === 'z') setZoom(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [night.n_epochs])

  const exportCsv = () => {
    const header = ['epoch', 'start_seconds', 'clock', 'expert', 'predicted', 'confidence', ...STAGES.map((s) => `p_${s}`)]
    const rows = night.predicted.map((p, i) => [
      i, i * night.epoch_sec, clockTime(i, night.epoch_sec),
      night.expert ? (night.expert[i] >= 0 ? STAGES[night.expert[i]] : 'unscored') : '',
      STAGES[p], night.confidence[i], ...night.probs[i],
    ].join(','))
    const blob = new Blob([[header.join(','), ...rows].join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${(night.info.recording ?? night.key).replace(/\.edf$/i, '')}_hypnogram.csv`
    link.click()
    URL.revokeObjectURL(url)
  }

  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href)
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    } catch { /* clipboard blocked */ }
  }

  const ag = night.agreement
  const scrubValue = playhead ?? Math.min(Math.max(selected, r0), r1 - 1)
  const roleNote = night.info.uploaded
    ? { strong: 'Your recording.', text: `Scored from the ${night.info.channel ?? 'EEG'} channel${night.expert ? ' and compared with the hypnogram you uploaded' : ''}. The model was trained on Fpz-Cz, so other channels may score less accurately.` }
    : night.role === 'test'
      ? { strong: 'Unseen subject.', text: 'This person was held out of training and model selection, so agreement here is an honest estimate.' }
      : night.role === 'train' || night.role === 'val'
        ? { strong: `${night.role === 'train' ? 'Training' : 'Validation'} subject.`, text: 'The model has seen this person, so agreement here is optimistic. Choose a test night for an honest view.' }
        : null

  return (
    <>
      {ag && (
        <KpiStrip items={[
          { label: <InfoTip term="kappa">Cohen’s κ</InfoTip>, value: ag.cohen_kappa.toFixed(3), note: kappaWord(ag.cohen_kappa) },
          { label: <InfoTip term="accuracy">Epoch agreement</InfoTip>, value: pct(ag.accuracy), note: `${disagreements.length.toLocaleString()} epochs differ` },
          { label: <InfoTip term="macroF1">Macro-F1</InfoTip>, value: ag.macro_f1.toFixed(3), note: 'Over 5 stages' },
          { label: 'Total sleep time', value: minutes(night.summary_predicted.total_sleep_time_min),
            note: night.summary_expert ? `Expert ${minutes(night.summary_expert.total_sleep_time_min)}` : 'CNN estimate' },
          { label: <InfoTip term="epoch">Scored epochs</InfoTip>, value: ag.n_epochs.toLocaleString(), note: 'Sleep period' },
        ]} />
      )}
      {roleNote && (
        <div className={`notice ${night.role === 'test' && !night.info.uploaded ? 'is-good' : 'is-warn'}`}><Icon name={night.role === 'test' && !night.info.uploaded ? 'check' : 'alert'} />
          <div><strong>{roleNote.strong}</strong> {roleNote.text}</div></div>
      )}

      <Card
        title={<span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <InfoTip term="hypnogram">Hypnogram</InfoTip>
          <span className="rec-name" style={{ fontWeight: 500 }}>{night.info.recording ?? night.key}</span></span>}
        sub={zoom
          ? `Zoomed to ${clockTime(zoom[0], night.epoch_sec)} – ${clockTime(zoom[1], night.epoch_sec)} (${Math.round(((zoom[1] - zoom[0]) * night.epoch_sec) / 60)} min) · press Z to reset`
          : `${view === 'sleep' ? `Sleep period ± ${night.wake_edge_min} min of Wake` : 'Entire recording'} · drag across the chart to zoom`}
        action={
          <div className="toolbar">
            {zoom && <button className="btn" onClick={() => setZoom(null)}><Icon name="zoomOut" />Reset zoom</button>}
            <Segmented label="Time range" value={view} onChange={(v) => { reset(); setZoom(null); setView(v) }}
              options={[{ value: 'sleep', label: 'Sleep period' }, { value: 'full', label: 'Full recording' }]} />
          </div>
        }>
        <div className="toolbar" style={{ marginBottom: 14, justifyContent: 'space-between' }}>
          <div className="toolbar">
            <button className="btn btn-primary" onClick={togglePlay} aria-label={playing ? 'Pause replay' : 'Play overnight replay'}>
              <Icon name={playing ? 'pause' : 'play'} />{playing ? 'Pause' : playhead !== null ? 'Resume' : 'Replay night'}
            </button>
            <button className="btn" onClick={reset} disabled={playhead === null}><Icon name="reset" />Reset</button>
            <Segmented label="Replay speed" value={String(speed)} onChange={(v) => setSpeed(Number(v))}
              options={[{ value: '1', label: '1×' }, { value: '4', label: '4×' }, { value: '10', label: '10×' }]} />
          </div>
          {playhead !== null ? (
            <div className="badge live-badge" aria-live="polite">
              <span className="dot pulse" />
              <span className="tnum">{clockTime(playhead, night.epoch_sec)}</span>
              <span className="dot" style={{ background: stageColor(night.predicted[playhead]) }} />
              <strong style={{ color: 'var(--ink)' }}>{STAGES[night.predicted[playhead]]}</strong>
              <span className="tnum">{pct(night.confidence[playhead], 0)}</span>
            </div>
          ) : <StageLegend />}
        </div>

        <Hypnogram night={night} range={[r0, r1]} playhead={playhead} selected={selected}
          onSelect={(e) => { setSelected(e); if (playing) setPlaying(false) }}
          onZoom={(z) => { reset(); setZoom(z); setSelected((s) => (s >= z[0] && s < z[1] ? s : z[0])) }} />

        <div className="scrubber">
          <span className="muted tnum">{clockTime(r0, night.epoch_sec)}</span>
          <input type="range" min={r0} max={r1 - 1} value={scrubValue} aria-label="Timeline position"
            onChange={(e) => {
              const v = Number(e.target.value)
              if (playhead !== null) setPlayhead(v)
              else setSelected(v)
            }} />
          <span className="muted tnum" style={{ textAlign: 'right' }}>{clockTime(r1, night.epoch_sec)}</span>
        </div>

        <div className="toolbar" style={{ marginTop: 12, justifyContent: 'space-between' }}>
          {night.expert ? (
            <div className="toolbar">
              <span className="legend-item" style={{ fontSize: 12.5 }}>
                <span className="swatch" style={{ background: 'var(--critical)' }} />
                <strong className="tnum">{inRange.length.toLocaleString()}</strong><span className="ink2">disagreements in view</span>
              </span>
              <button className="btn btn-sm" onClick={() => jumpDisagreement(-1)} disabled={!disagreements.length}><Icon name="prev" />Previous</button>
              <button className="btn btn-sm" onClick={() => jumpDisagreement(1)} disabled={!disagreements.length}>Next<Icon name="next" /></button>
            </div>
          ) : <span className="muted" style={{ fontSize: 12 }}>No expert hypnogram for this recording.</span>}
          <div className="toolbar">
            <button className="btn btn-sm" onClick={copyLink}><Icon name={copied ? 'check' : 'link'} />{copied ? 'Link copied' : 'Copy link'}</button>
            <button className="btn btn-sm" onClick={exportCsv}><Icon name="download" />Export CSV</button>
          </div>
        </div>
      </Card>

      <EpochInspector night={night} index={selected}
        onPick={(e) => setSelected(Math.min(Math.max(e, 0), night.n_epochs - 1))} />

      <div className="grid g2">
        <Card title="Sleep architecture" sub="Time in each stage during the sleep period">
          <StageStack rows={[
            ...(night.summary_expert ? [{ label: 'Expert', minutes: night.summary_expert.stage_minutes }] : []),
            { label: 'CNN prediction', minutes: night.summary_predicted.stage_minutes },
          ]} />
          <div style={{ marginTop: 14 }}><StageLegend /></div>
          {ag && (
            <>
              <div className="card-sub section-label">F1 by stage on this night</div>
              <HBars labelWidth={48} format={(v) => v.toFixed(2)}
                rows={STAGES.map((s, i) => ({ label: s, value: ag.per_class_f1[s], color: stageColor(i) }))} />
            </>
          )}
        </Card>
        <Card title="Sleep report" sub="Clinical measures derived from each hypnogram" flush>
          <SleepTable night={night} />
        </Card>
      </div>
    </>
  )
}

/* ------------------------------------------------------------------ */

function ContextStrip({ night, index, onPick }: { night: Night; index: number; onPick: (e: number) => void }) {
  const span = 12
  const cells = Array.from({ length: span * 2 + 1 }, (_, k) => index - span + k)
  const row = (label: string, stages: number[] | null) => stages && (
    <div className="ctx-row">
      <span className="ctx-label">{label}</span>
      <div className="ctx-cells">
        {cells.map((e) => {
          if (e < 0 || e >= night.n_epochs) return <span key={e} className="ctx-cell is-empty" />
          const s = stages[e]
          return (
            <button key={e} className={`ctx-cell ${e === index ? 'is-current' : ''}`} onClick={() => onPick(e)}
              title={`Epoch ${e} · ${clockTime(e, night.epoch_sec)} · ${label}: ${s >= 0 ? STAGES[s] : 'unscored'}`}
              style={{ background: s >= 0 ? stageColor(s) : 'var(--surface-2)' }} aria-label={`Epoch ${e}, ${label} ${s >= 0 ? STAGES[s] : 'unscored'}`} />
          )
        })}
      </div>
    </div>
  )
  return (
    <div className="ctx" aria-label="Neighbouring epochs">
      {row('Expert', night.expert)}
      {row('CNN', night.predicted)}
      <div className="ctx-row"><span className="ctx-label" />
        <div className="ctx-axis muted"><span>−6 min</span><span>this epoch</span><span>+6 min</span></div>
      </div>
    </div>
  )
}

function EpochInspector({ night, index, onPick }: { night: Night; index: number; onPick: (e: number) => void }) {
  // useFetch keeps the previous epoch on screen while the next one loads.
  const epoch = useFetch(`${night.key}:${index}`, () => api.epoch(night.key, index))
  const e = epoch.data
  const expert = night.expert ? night.expert[index] : null
  const pred = night.predicted[index]
  const hasExpert = expert !== null && expert >= 0

  return (
    <div className="grid g3">
      <Card className="span2"
        title={`Epoch ${index} · ${clockTime(index, night.epoch_sec)}`}
        sub="Fpz-Cz · 30 s · ← → to step"
        action={
          <div className="toolbar">
            <button className="btn" onClick={() => onPick(index - 1)} aria-label="Previous epoch"><Icon name="prev" /></button>
            <button className="btn" onClick={() => onPick(index + 1)} aria-label="Next epoch"><Icon name="next" /></button>
          </div>
        }>
        <ContextStrip night={night} index={index} onPick={onPick} />
        {e ? <div style={{ opacity: epoch.loading ? 0.6 : 1, transition: 'opacity .15s', marginTop: 12 }}>
          <EegTrace signal={e.signal_uv} sfreq={e.sfreq} startSec={e.start_sec} />
        </div> : <Skeleton height={330} />}
        {(!e || e.psd) && <div className="card-sub section-label"><InfoTip term="spectrum">Power spectrum</InfoTip></div>}
        {!e ? <Skeleton height={210} /> : e.psd && <Spectrum freqs={e.psd.freqs} db={e.psd.db} />}
      </Card>
      <Card title="Classifier output" sub={hasExpert ? (expert === pred ? 'The CNN agrees with the expert' : 'The CNN disagrees with the expert') : 'No expert label for this epoch'}>
        <div style={{ display: 'flex', gap: 10, marginBottom: 14, flexWrap: 'wrap' }}>
          <span className="badge" style={{ height: 28 }}>
            <span className="dot" style={{ background: stageColor(pred) }} />CNN: <strong style={{ color: 'var(--ink)' }}>{STAGES[pred]}</strong>
          </span>
          {hasExpert && (
            <span className="badge" style={{ height: 28 }}>
              <span className="dot" style={{ background: stageColor(expert) }} />Expert: <strong style={{ color: 'var(--ink)' }}>{STAGES[expert]}</strong>
              <span style={{ display: 'inline-flex', width: 16, color: expert === pred ? 'var(--good)' : 'var(--critical)' }}>
                <Icon name={expert === pred ? 'check' : 'x'} />
              </span>
            </span>
          )}
        </div>
        {e ? <>
          <div className="card-sub section-label" style={{ marginTop: 0 }}><InfoTip term="confidence">Stage probabilities</InfoTip></div>
          <HBars rows={STAGES.map((s, i) => ({ label: s, value: e.probs[i], color: stageColor(i) }))} labelWidth={48} />
          <div className="card-sub section-label"><InfoTip term="bandPower">Relative band power</InfoTip></div>
          <HBars labelWidth={48}
            rows={Object.entries(e.bands).map(([band, v]) => ({ label: band, value: v / 100, color: 'var(--ink-2)' }))} />
        </> : <Skeleton height={380} />}
      </Card>
    </div>
  )
}

function SleepTable({ night }: { night: Night }) {
  const p = night.summary_predicted
  const x = night.summary_expert
  const rows: [ReactNode, number | null, number | null, string][] = [
    ['Total sleep time', p.total_sleep_time_min, x?.total_sleep_time_min ?? null, 'min'],
    [<InfoTip key="se" term="sleepEfficiency">Sleep efficiency</InfoTip>, p.sleep_efficiency_pct, x?.sleep_efficiency_pct ?? null, '%'],
    ['Sleep onset latency', p.sleep_onset_latency_min, x?.sleep_onset_latency_min ?? null, 'min'],
    [<InfoTip key="waso" term="waso">Wake after sleep onset</InfoTip>, p.waso_min, x?.waso_min ?? null, 'min'],
    ['REM latency', p.rem_latency_min, x?.rem_latency_min ?? null, 'min'],
    ...STAGES.slice(1).map((s): [ReactNode, number | null, number | null, string] =>
      [`Time in ${s}`, p.stage_minutes[s], x?.stage_minutes[s] ?? null, 'min']),
    ['Time in bed (window)', p.time_in_bed_min, x?.time_in_bed_min ?? null, 'min'],
  ]
  const fmt = (v: number | null, unit: string) => (v === null ? '—' : unit === '%' ? `${v.toFixed(1)}%` : minutes(v))
  return (
    <table className="table">
      <thead><tr><th>Measure</th>{x && <th className="num">Expert</th>}<th className="num">CNN</th>{x && <th className="num">Δ</th>}</tr></thead>
      <tbody>
        {rows.map(([label, pv, xv, unit], i) => {
          const diff = x && pv !== null && xv !== null ? pv - xv : null
          return (
            <tr key={i}>
              <td>{label}</td>{x && <td className="num">{fmt(xv, unit)}</td>}<td className="num">{fmt(pv, unit)}</td>
              {x && <td className="num muted">{diff === null ? '—' : `${diff > 0 ? '+' : ''}${unit === '%' ? diff.toFixed(1) : Math.round(diff)}`}</td>}
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

/* ------------------------------------------------------------------ */

function UploadDialog({ open, onClose, onDone }: { open: boolean; onClose: () => void; onDone: (key: string) => void }) {
  const [psg, setPsg] = useState<File | null>(null)
  const [hyp, setHyp] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [over, setOver] = useState(false)

  const take = (files: FileList | null) => {
    if (!files) return
    for (const f of Array.from(files)) {
      if (/hypnogram/i.test(f.name)) setHyp(f)
      else setPsg(f)
    }
    setError(null)
  }
  const submit = async () => {
    if (!psg) return
    setBusy(true)
    setError(null)
    try {
      const res = await api.upload(psg, hyp)
      setPsg(null)
      setHyp(null)
      onDone(res.key)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }
  const onDrop = (e: DragEvent) => { e.preventDefault(); setOver(false); take(e.dataTransfer.files) }

  return (
    <Dialog open={open} onClose={onClose} title="Analyze your own recording" width={560}>
      <p className="ink2" style={{ fontSize: 13 }}>
        Upload an overnight <strong>EDF</strong> file. The model uses the Fpz-Cz channel if present, otherwise the
        first EEG channel, resampled to 100 Hz. Add a Sleep-EDF style hypnogram to compare with expert scoring.
      </p>
      <label className={`dropzone ${over ? 'is-over' : ''}`} onDragOver={(e) => { e.preventDefault(); setOver(true) }}
        onDragLeave={() => setOver(false)} onDrop={onDrop}>
        <Icon name="upload" />
        <span><strong>Drop EDF files here</strong> or click to browse</span>
        <span className="muted" style={{ fontSize: 12 }}>A file with “Hypnogram” in its name is used as the expert scoring</span>
        <input type="file" accept=".edf" multiple onChange={(e) => take(e.target.files)} style={{ display: 'none' }} />
      </label>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 14 }}>
        <FileRow label="Recording (PSG)" file={psg} required onClear={() => setPsg(null)} />
        <FileRow label="Hypnogram (optional)" file={hyp} onClear={() => setHyp(null)} />
      </div>
      {error && <div className="notice" style={{ marginTop: 14 }}><Icon name="alert" /><div><strong>Upload failed.</strong> {error}</div></div>}
      <div className="toolbar" style={{ justifyContent: 'flex-end', marginTop: 18 }}>
        <button className="btn" onClick={onClose} disabled={busy}>Cancel</button>
        <button className="btn btn-primary" onClick={submit} disabled={!psg || busy}>
          {busy ? <><span className="spinner" />Analyzing…</> : <><Icon name="arrowRight" />Analyze recording</>}
        </button>
      </div>
      <p className="muted" style={{ fontSize: 11.5, marginTop: 12 }}>
        Files stay on this computer: the local API processes them and keeps a copy in <span className="kbd">outputs/uploads</span>.
      </p>
    </Dialog>
  )
}

function FileRow({ label, file, required, onClear }: { label: string; file: File | null; required?: boolean; onClear: () => void }) {
  return (
    <div className="file-row">
      <Icon name="file" />
      <div style={{ minWidth: 0, flex: 1 }}>
        <div style={{ fontSize: 12, color: 'var(--ink-muted)' }}>{label}{required ? ' · required' : ''}</div>
        <div style={{ fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {file ? `${file.name} · ${(file.size / 1e6).toFixed(1)} MB` : <span className="muted">No file selected</span>}
        </div>
      </div>
      {file && <button className="btn btn-ghost btn-sm" onClick={onClear} aria-label={`Remove ${label}`}><Icon name="x" /></button>}
    </div>
  )
}
