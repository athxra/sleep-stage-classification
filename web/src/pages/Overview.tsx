import type { Summary } from '../api'
import { FoldDots, HBars } from '../components/charts'
import { InfoTip } from '../components/help'
import { Card, Icon, KpiStrip, Meter } from '../components/ui'
import { navigate } from '../router'
import { STAGES, compact, kappaWord, pct, stageColor } from '../stages'

export function Overview({ summary, onNavigate }: { summary: Summary; onNavigate: (page: string) => void }) {
  const { cv, test, split, model } = summary
  const counts = split?.class_counts ?? {}
  const totalEpochs = split?.n_epochs ?? 1
  const pooled = cv?.pooled ?? test
  const acc = cv?.accuracy.mean ?? test?.accuracy ?? 0
  const baseline = pooled?.majority_class_baseline ?? 0

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1 className="page-title">Overview</h1>
          <p className="page-sub">
            5-class sleep staging from single-channel Fpz-Cz EEG · {cv ? `${cv.n_folds}-fold subject-grouped cross-validation` : 'held-out test subjects'}
          </p>
        </div>
        <div className="page-actions">
          <button className="btn" onClick={() => onNavigate('performance')}><Icon name="chart" />Performance</button>
          <button className="btn btn-primary" onClick={() => onNavigate('night')}><Icon name="night" />Open night explorer</button>
        </div>
      </header>

      <KpiStrip items={[
        { label: <InfoTip term="macroF1">Macro-F1</InfoTip>, value: (cv?.macro_f1.mean ?? test?.macro_f1 ?? 0).toFixed(3),
          unit: cv && `± ${cv.macro_f1.std.toFixed(3)}`, note: cv ? `Mean over ${cv.n_folds} folds` : 'Test subjects' },
        { label: <InfoTip term="kappa">Cohen’s κ</InfoTip>, value: (cv?.cohen_kappa.mean ?? test?.cohen_kappa ?? 0).toFixed(3),
          unit: cv && `± ${cv.cohen_kappa.std.toFixed(3)}`, note: kappaWord(cv?.cohen_kappa.mean ?? test?.cohen_kappa ?? 0) },
        { label: <InfoTip term="accuracy">Accuracy</InfoTip>, value: pct(acc),
          note: <><span className="delta up">+{((acc - baseline) * 100).toFixed(1)} pts</span> vs majority class</> },
        { label: 'Subjects', value: split?.n_subjects ?? '—', unit: split && `${split.n_recordings} nights`, note: 'Sleep-EDF-20 · SC cohort' },
        { label: <InfoTip term="epoch">Scored epochs</InfoTip>, value: compact(totalEpochs), note: '30 s · AASM stages' },
        { label: <InfoTip term="macs">Model footprint</InfoTip>, value: compact(model.parameters), unit: 'params',
          note: `${(model.macs_per_epoch / 1e6).toFixed(1)}M MACs / epoch` },
      ]} />

      <div className="grid g3">
        <Card className="span2" title="Per-stage F1" sub={cv ? 'Cross-validation mean ± std' : 'Held-out test subjects'}>
          <HBars labelWidth={56} format={(v) => v.toFixed(3)}
            rows={STAGES.map((s, i) => cv
              ? { label: s, value: cv.per_class_f1[s].mean, err: cv.per_class_f1[s].std, color: stageColor(i) }
              : { label: s, value: test?.per_class[s].f1 ?? 0, color: stageColor(i) })} />
        </Card>
        <Card title="Class distribution" sub={`${totalEpochs.toLocaleString()} expert-scored epochs`}>
          <HBars labelWidth={56} format={(v) => pct(v)}
            rows={STAGES.map((s, i) => ({ label: s, value: (counts[s] ?? 0) / totalEpochs, color: stageColor(i) }))}
            max={Math.max(...STAGES.map((s) => (counts[s] ?? 0) / totalEpochs)) * 1.1} />
        </Card>
      </div>

      {pooled && (
        <div className="grid g3">
          <Card className="span2" flush title="Stage breakdown" sub={cv ? 'Pooled predictions, all folds' : 'Held-out test subjects'}>
            <table className="table">
              <thead><tr><th>Stage</th><th className="num">Share</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th><th className="num">Support</th></tr></thead>
              <tbody>
                {STAGES.map((s, i) => {
                  const c = pooled.per_class[s]
                  return (
                    <tr key={s}>
                      <td><span className="legend-item"><span className="swatch" style={{ background: stageColor(i) }} />{s}</span></td>
                      <td className="num">{pct((counts[s] ?? 0) / totalEpochs)}</td>
                      <td className="num">{c.precision.toFixed(3)}</td>
                      <td className="num">{c.recall.toFixed(3)}</td>
                      <td className="num"><Meter value={c.f1} color={stageColor(i)} /></td>
                      <td className="num">{c.support.toLocaleString()}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </Card>
          <Card flush title="Top confusions" sub="Share of the expert stage"
            action={<button className="btn btn-sm btn-ghost" onClick={() => onNavigate('performance')}>View all<Icon name="arrowRight" /></button>}>
            <table className="table">
              <thead><tr><th>Expert → CNN</th><th className="num">Epochs</th><th className="num">Rate</th></tr></thead>
              <tbody>
                {pooled.top_confusions.slice(0, 6).map((c) => {
                  const i = STAGES.indexOf(c.true as never)
                  const j = STAGES.indexOf(c.predicted as never)
                  return (
                    <tr key={`${c.true}-${c.predicted}`}>
                      <td>
                        <span className="legend-item"><span className="swatch" style={{ background: stageColor(i) }} />{c.true}</span>
                        <span className="muted"> → </span>
                        <span className="legend-item"><span className="swatch" style={{ background: stageColor(j) }} />{c.predicted}</span>
                      </td>
                      <td className="num">{c.count.toLocaleString()}</td>
                      <td className="num">{pct(c.rate)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      <div className="grid g3">
        {cv && (
          <Card className="span2" title={<InfoTip term="crossValidation">Macro-F1 by fold</InfoTip>} sub="Each fold tests unseen subjects · hover for details">
            <FoldDots label="Macro-F1" mean={cv.macro_f1.mean} std={cv.macro_f1.std}
              folds={cv.folds.map((f) => ({ fold: f.fold, value: f.macro_f1, subjects: f.test_subjects }))} />
          </Card>
        )}
        <Card flush title="Deployed model" sub="Fold 0 checkpoint"
          action={<button className="btn btn-sm btn-ghost" onClick={() => navigate('model')}>Details<Icon name="arrowRight" /></button>}>
          <table className="table">
            <tbody>
              <tr><td className="ink2">Architecture</td><td className="num">1D CNN · 4 conv</td></tr>
              <tr><td className="ink2">Parameters</td><td className="num">{model.parameters.toLocaleString()}</td></tr>
              <tr><td className="ink2">Weights</td><td className="num">{model.size_mb.toFixed(2)} MB</td></tr>
              <tr><td className="ink2">Compute / epoch</td><td className="num">{compact(model.macs_per_epoch)} MACs</td></tr>
              <tr><td className="ink2">Receptive field</td><td className="num">≈ 17 s</td></tr>
              <tr><td className="ink2">Checkpoint</td><td className="num">epoch {model.checkpoint_epoch}</td></tr>
              <tr><td className="ink2">Validation F1</td><td className="num">{model.val_f1.toFixed(3)}</td></tr>
              {test && <tr><td className="ink2">Test F1 / κ</td><td className="num">{test.macro_f1.toFixed(3)} / {test.cohen_kappa.toFixed(3)}</td></tr>}
            </tbody>
          </table>
        </Card>
      </div>
    </div>
  )
}
