import type { ReactNode } from 'react'
import type { Summary } from '../api'
import { InfoTip } from '../components/help'
import { Card, KpiStrip } from '../components/ui'
import { compact } from '../stages'

const PIPELINE = [
  ['Signal', 'Sleep-EDF PSG via MNE · Fpz-Cz · 100 Hz'],
  ['Epoching', '30-s epochs · S3+S4 merged to N3 · movement / unscored dropped'],
  ['Window', 'Sleep period ± 30 min of Wake'],
  ['Normalization', 'Per-epoch Z-score (train = inference)'],
  ['Split', 'Subject-grouped; both nights of a person in one split'],
  ['Evaluation', 'Macro-F1, Cohen’s κ, per-stage and transition analysis'],
]

const LAYERS = [
  { layer: 'Input', detail: 'One Fpz-Cz epoch, per-epoch Z-scored', shape: '1 × 3000' },
  { layer: 'Conv1d k50 s6', detail: '32 filters of 0.5 s · BatchNorm · ReLU', shape: '32 × 500' },
  { layer: 'MaxPool 8', detail: 'Dropout 0.25', shape: '32 × 62' },
  { layer: 'Conv1d k7 ×2', detail: '64 filters each · BatchNorm · ReLU', shape: '64 × 62' },
  { layer: 'MaxPool 4', detail: 'Dropout 0.25', shape: '64 × 15' },
  { layer: 'Conv1d k7', detail: '128 filters · BatchNorm · ReLU · ~17 s receptive field', shape: '128 × 15' },
  { layer: 'Global avg pool', detail: 'Collapses time, keeps the model tiny', shape: '128' },
  { layer: 'Dropout · Dense', detail: 'Dropout 0.5 → 5 stage logits', shape: '5' },
]

export function Model({ summary }: { summary: Summary }) {
  const { model, split } = summary
  const hp = model.hyperparameters
  const factor = model.baseline_macs_per_epoch / model.macs_per_epoch
  const folds = split?.folds ?? []

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1 className="page-title">Model</h1>
          <p className="page-sub">Lightweight 1D CNN on raw single-channel EEG · architecture, footprint and training configuration</p>
        </div>
      </header>

      <KpiStrip items={[
        { label: 'Parameters', value: model.parameters.toLocaleString(), note: `${model.size_mb.toFixed(2)} MB float32 weights` },
        { label: <InfoTip term="macs">Compute / epoch</InfoTip>, value: compact(model.macs_per_epoch), unit: 'MACs', note: 'Per 30 s of EEG' },
        { label: 'Compute reduction', value: `${Math.round(factor)}×`, note: `vs. baseline ${compact(model.baseline_macs_per_epoch)} MACs` },
        { label: 'Receptive field', value: '≈ 17', unit: 's', note: 'Baseline: 0.13 s' },
        { label: 'Checkpoint', value: `${model.checkpoint_epoch}`, unit: 'epoch', note: `Validation F1 ${model.val_f1.toFixed(3)}` },
      ]} />

      <div className="grid g3">
        <Card title="Architecture" sub="Input 1 × 3000 samples → 5 stage logits" className="span2" flush>
          <div className="arch">
            {LAYERS.map((l, i) => (
              <div className="arch-row" key={l.layer}>
                <span className="idx">{String(i).padStart(2, '0')}</span>
                <span className="layer">{l.layer}</span>
                <span className="detail">{l.detail}</span>
                <span className="shape">{l.shape}</span>
              </div>
            ))}
          </div>
        </Card>
        <Card title="Training configuration" flush>
          <table className="table">
            <tbody>
              <tr><td>Optimizer</td><td className="num">Adam, lr {String(hp.lr ?? '1e-3')}</td></tr>
              <tr><td>Weight decay</td><td className="num">{String(hp.weight_decay ?? '1e-4')}</td></tr>
              <tr><td>Batch size</td><td className="num">{String(hp.batch_size ?? 128)}</td></tr>
              <tr><td>Class weights</td><td className="num">{String(hp.class_weight ?? 'inverse')} frequency</td></tr>
              <tr><td>Augmentation</td><td className="num">{hp.augment === false ? 'off' : 'shift · scale · noise'}</td></tr>
              <tr><td>LR schedule</td><td className="num">halve on plateau</td></tr>
              <tr><td>Early stopping</td><td className="num">patience {String(hp.patience ?? 10)}</td></tr>
              <tr><td>Selected epoch</td><td className="num">{model.checkpoint_epoch} (val F1 {model.val_f1.toFixed(3)})</td></tr>
              <tr><td>Seed</td><td className="num">{String(hp.seed ?? 42)}</td></tr>
            </tbody>
          </table>
        </Card>
      </div>

      <Card title="Data pipeline" flush>
        <table className="table">
          <tbody>
            {PIPELINE.map(([k, v]) => <tr key={k}><td className="ink2" style={{ width: 160 }}>{k}</td><td>{v}</td></tr>)}
          </tbody>
        </table>
      </Card>

      {folds.length > 0 && (
        <Card title="Subject-grouped folds" sub="Columns are subjects · fold 0 is the deployed model">
          <FoldGrid folds={folds} />
        </Card>
      )}
    </div>
  )
}

function FoldGrid({ folds }: { folds: { train: number[]; val: number[]; test: number[] }[] }) {
  const subjects = Array.from(new Set(folds.flatMap((f) => [...f.train, ...f.val, ...f.test]))).sort((a, b) => a - b)
  const role = (f: { train: number[]; val: number[] }, s: number) =>
    f.train.includes(s) ? 'train' : f.val.includes(s) ? 'val' : 'test'
  const style = {
    train: { background: 'var(--surface-2)', color: 'var(--ink-muted)', border: '1px solid var(--border)' },
    val: { background: 'var(--seq-2)', color: 'var(--ink)', border: '1px solid transparent' },
    test: { background: 'var(--accent)', color: 'var(--accent-ink)', border: '1px solid var(--accent)' },
  }
  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{ display: 'grid', gridTemplateColumns: `64px repeat(${subjects.length}, minmax(26px, 1fr))`, gap: 3, minWidth: 640 }}>
        <div />
        {subjects.map((s) => <div key={s} className="muted tnum" style={{ fontSize: 11, textAlign: 'center' }}>{s}</div>)}
        {folds.map((f, k) => (
          <FoldRow key={k} k={k} subjects={subjects} cell={(s) => {
            const r = role(f, s)
            return <div title={`Subject ${s}: ${r}`} style={{ ...style[r], height: 26, borderRadius: 4, fontSize: 10.5, fontWeight: 600,
              display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{r === 'train' ? '' : r === 'val' ? 'V' : 'T'}</div>
          }} />
        ))}
      </div>
      <div className="legend" style={{ marginTop: 12 }}>
        <span className="legend-item"><span className="swatch" style={style.test} />Test</span>
        <span className="legend-item"><span className="swatch" style={style.val} />Validation (checkpoint selection)</span>
        <span className="legend-item"><span className="swatch" style={style.train} />Train</span>
      </div>
    </div>
  )
}

function FoldRow({ k, subjects, cell }: { k: number; subjects: number[]; cell: (s: number) => ReactNode }) {
  return (
    <>
      <div className="muted" style={{ fontSize: 12, display: 'flex', alignItems: 'center' }}>Fold {k}</div>
      {subjects.map((s) => <div key={s}>{cell(s)}</div>)}
    </>
  )
}
