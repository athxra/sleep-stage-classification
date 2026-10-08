import { useState } from 'react'
import { api, useFetch, type Summary, type TestMetrics } from '../api'
import { ConfusionMatrix, FoldDots, HBars } from '../components/charts'
import { InfoTip } from '../components/help'
import { Card, Icon, KpiStrip, Meter, Segmented, Skeleton } from '../components/ui'
import { navigate } from '../router'
import { STAGES, clockTime, kappaWord, pct, stageColor } from '../stages'

type Source = 'cv' | 'test'

export function Performance({ summary }: { summary: Summary }) {
  const { cv, test } = summary
  const [source, setSource] = useState<Source>(cv ? 'cv' : 'test')
  const [cmMode, setCmMode] = useState<'pct' | 'count'>('pct')
  const [cell, setCell] = useState<[number, number] | null>(null)
  const m: TestMetrics | null = source === 'cv' ? cv?.pooled ?? null : test

  if (!m) {
    return <div className="page"><div className="notice">No evaluation results yet. Run <span className="kbd">python src/evaluate.py</span>.</div></div>
  }
  const t = test?.transition_analysis
  const isCv = source === 'cv' && cv

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1 className="page-title">Performance</h1>
          <p className="page-sub">
            {isCv
              ? `Pooled over ${cv.n_folds} subject-grouped folds · ${cv.n_subjects} subjects, each scored by a model that never saw them`
              : `Deployed model · held-out test subjects ${test!.test_subjects.join(', ')}`}
          </p>
        </div>
        {cv && test && (
          <Segmented<Source> label="Evaluation set" value={source} onChange={setSource}
            options={[{ value: 'cv', label: `All folds · ${cv.n_subjects} subjects` },
              { value: 'test', label: 'Deployed model' }]} />
        )}
      </header>

      <KpiStrip items={[
        { label: <InfoTip term="macroF1">Macro-F1</InfoTip>, value: isCv ? cv.macro_f1.mean.toFixed(3) : m.macro_f1.toFixed(3),
          unit: isCv ? `± ${cv.macro_f1.std.toFixed(3)}` : undefined, note: isCv ? 'Mean across folds' : 'Unweighted mean over stages' },
        { label: <InfoTip term="kappa">Cohen’s κ</InfoTip>, value: isCv ? cv.cohen_kappa.mean.toFixed(3) : m.cohen_kappa.toFixed(3),
          unit: isCv ? `± ${cv.cohen_kappa.std.toFixed(3)}` : undefined, note: kappaWord(isCv ? cv.cohen_kappa.mean : m.cohen_kappa) },
        { label: <InfoTip term="accuracy">Accuracy</InfoTip>, value: pct(isCv ? cv.accuracy.mean : m.accuracy),
          note: <InfoTip term="baseline">{`Majority baseline ${pct(m.majority_class_baseline, 0)}`}</InfoTip> },
        { label: 'N1 recall', value: pct(m.n1_rem_focus.n1_recall), note: 'Hardest stage' },
        { label: 'REM recall', value: pct(m.n1_rem_focus.rem_recall), note: 'Most confused with N1' },
        { label: <InfoTip term="epoch">Epochs evaluated</InfoTip>, value: m.n_epochs.toLocaleString(), note: '30-s expert-scored' },
      ]} />

      <div className="grid g2">
        <Card title="Confusion matrix" sub={cmMode === 'pct' ? 'Rows: expert · columns: CNN · row-normalized · click a cell for examples' : 'Rows: expert · columns: CNN · epoch counts'}
          action={<Segmented label="Cell values" value={cmMode} onChange={setCmMode}
            options={[{ value: 'pct', label: '% of row' }, { value: 'count', label: 'Counts' }]} />}>
          <ConfusionMatrix counts={m.confusion_matrix} normalized={m.confusion_matrix_normalized} mode={cmMode}
            picked={cell} onPick={setCell} />
        </Card>

        {cell ? (
          <ExamplesCard trueStage={STAGES[cell[0]]} predStage={STAGES[cell[1]]} fromCv={Boolean(isCv)} onClose={() => setCell(null)} />
        ) : (
          <Card title="Per-stage metrics" sub={isCv ? 'F1 mean ± std across folds' : 'Deployed model, test subjects'}>
            <HBars labelWidth={56}
              rows={STAGES.map((s, i) => isCv
                ? { label: s, value: cv.per_class_f1[s].mean, err: cv.per_class_f1[s].std, color: stageColor(i) }
                : { label: s, value: m.per_class[s].f1, color: stageColor(i) })}
              format={(v) => v.toFixed(2)} />
            <table className="table" style={{ marginTop: 16 }}>
              <thead><tr><th>Stage</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th><th className="num">Support</th></tr></thead>
              <tbody>
                {STAGES.map((s, i) => (
                  <tr key={s}>
                    <td><span className="legend-item"><span className="swatch" style={{ background: stageColor(i) }} />{s}</span></td>
                    <td className="num">{m.per_class[s].precision.toFixed(3)}</td>
                    <td className="num">{m.per_class[s].recall.toFixed(3)}</td>
                    <td className="num"><Meter value={m.per_class[s].f1} color={stageColor(i)} /></td>
                    <td className="num">{m.per_class[s].support.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        )}
      </div>

      <div className="grid g3">
        <Card title="N1 / REM error distribution" sub="Predicted stage for expert N1 and REM epochs">
          <FocusBlock title="Expert N1 predicted as" data={m.n1_rem_focus.n1_predicted_as} recall={m.n1_rem_focus.n1_recall} name="N1" />
          <div style={{ height: 14 }} />
          <FocusBlock title="Expert REM predicted as" data={m.n1_rem_focus.rem_predicted_as} recall={m.n1_rem_focus.rem_recall} name="REM" />
        </Card>

        <Card title={<InfoTip term="transition">Stage transitions</InfoTip>} sub="Deployed model: stable runs vs epochs next to an expert stage change">
          {t && <>
            <HBars labelWidth={118} rows={[
              { label: 'Stable epochs', value: t.stable.accuracy ?? 0, color: 'var(--seq-5)' },
              { label: 'Near a change', value: t.near_transition.accuracy ?? 0, color: 'var(--seq-3)' },
            ]} />
            <table className="table" style={{ marginTop: 12 }}>
              <thead><tr><th /><th className="num">Epochs</th><th className="num">Accuracy</th><th className="num">Macro-F1</th></tr></thead>
              <tbody>
                <tr><td>Stable</td><td className="num">{t.stable.n_epochs.toLocaleString()}</td><td className="num">{pct(t.stable.accuracy ?? 0)}</td><td className="num">{(t.stable.macro_f1 ?? 0).toFixed(3)}</td></tr>
                <tr><td>Near a change</td><td className="num">{t.near_transition.n_epochs.toLocaleString()}</td><td className="num">{pct(t.near_transition.accuracy ?? 0)}</td><td className="num">{(t.near_transition.macro_f1 ?? 0).toFixed(3)}</td></tr>
              </tbody>
            </table>
            <p className="muted tnum" style={{ fontSize: 12, marginTop: 10 }}>
              {pct(t.near_transition.n_epochs / (t.near_transition.n_epochs + t.stable.n_epochs), 1)} of epochs sit next to an expert stage change.
            </p>
          </>}
        </Card>

        <Card flush title="Most frequent confusions" sub="Share of the expert stage · click for examples">
          <table className="table">
            <tbody>
              {m.top_confusions.map((c) => {
                const i = STAGES.indexOf(c.true as never)
                const j = STAGES.indexOf(c.predicted as never)
                return (
                  <tr key={`${c.true}-${c.predicted}`} className="row-link" onClick={() => setCell([i, j])} tabIndex={0}
                    onKeyDown={(e) => { if (e.key === 'Enter') setCell([i, j]) }}>
                    <td>
                      <span className="legend-item"><span className="swatch" style={{ background: stageColor(i) }} />{c.true}</span>
                      <span className="muted"> → </span>
                      <span className="legend-item"><span className="swatch" style={{ background: stageColor(j) }} />{c.predicted}</span>
                    </td>
                    <td className="num">{c.count.toLocaleString()}</td>
                    <td className="num muted">{pct(c.rate)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </Card>
      </div>

      {cv && (
        <div className="grid g3">
          <Card title={<InfoTip term="crossValidation">Macro-F1 by fold</InfoTip>} sub="Each fold tests unseen subjects · hover a dot" className="span2">
            <FoldDots label="Macro-F1" mean={cv.macro_f1.mean} std={cv.macro_f1.std}
              folds={cv.folds.map((f) => ({ fold: f.fold, value: f.macro_f1, subjects: f.test_subjects }))} />
          </Card>
          <Card flush title="Fold details">
            <table className="table">
              <thead><tr><th>Fold</th><th>Test subjects</th><th className="num">F1</th><th className="num">κ</th></tr></thead>
              <tbody>
                {cv.folds.map((f) => (
                  <tr key={f.fold}><td>{f.fold}{f.fold === 0 && <span className="role" style={{ marginLeft: 6 }}>Deployed</span>}</td><td className="tnum">{f.test_subjects.join(', ')}</td>
                    <td className="num">{f.macro_f1.toFixed(3)}</td><td className="num">{f.cohen_kappa.toFixed(3)}</td></tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}
    </div>
  )
}

function ExamplesCard({ trueStage, predStage, fromCv, onClose }: {
  trueStage: string; predStage: string; fromCv: boolean; onClose: () => void
}) {
  const ex = useFetch(`${trueStage}>${predStage}`, () => api.examples(trueStage, predStage))
  const ti = STAGES.indexOf(trueStage as never)
  const pi = STAGES.indexOf(predStage as never)
  return (
    <Card
      title={<span className="legend-item" style={{ fontSize: 15, gap: 8 }}>
        Expert <span className="swatch" style={{ background: stageColor(ti) }} />{trueStage}
        <span className="muted">→</span> CNN <span className="swatch" style={{ background: stageColor(pi) }} />{predStage}
      </span>}
      sub={ex.data ? `${ex.data.total.toLocaleString()} such epochs in the deployed model’s test set${fromCv ? ' (the matrix shows all folds)' : ''}` : 'Finding examples…'}
      action={<button className="btn btn-sm" onClick={onClose}><Icon name="x" />Close</button>}>
      {ex.loading && !ex.data && <Skeleton height={280} />}
      {ex.error && <p className="muted">{ex.error}</p>}
      {ex.data && ex.data.total === 0 && <p className="muted">The deployed model made no such mistakes on its test subjects.</p>}
      {ex.data && ex.data.total > 0 && (
        <>
          <p className="ink2" style={{ fontSize: 13, marginBottom: 10 }}>
            Open an example to inspect its EEG, spectrum and neighbouring epochs.
          </p>
          <div className="example-list">
            {ex.data.examples.map((e) => (
              <button key={`${e.key}-${e.epoch}`} className="example" onClick={() => navigate('night', { rec: e.key, epoch: e.epoch })}>
                <span className="rec-name">{e.key}</span>
                <span className="muted tnum">epoch {e.epoch} · {clockTime(e.epoch)}</span>
                <span className="example-go"><Icon name="arrowRight" /></span>
              </button>
            ))}
          </div>
        </>
      )}
    </Card>
  )
}

function FocusBlock({ title, data, recall, name }: { title: string; data: Record<string, number>; recall: number; name: string }) {
  const rows = [{ label: `${name} ✓`, value: recall, color: stageColor(STAGES.indexOf(name as never)) },
    ...Object.entries(data).map(([s, v]) => ({ label: s, value: v, color: stageColor(STAGES.indexOf(s as never)) }))]
  return (
    <div>
      <div style={{ fontSize: 12.5, fontWeight: 600, marginBottom: 4 }}>{title}</div>
      <HBars labelWidth={52} rows={rows} max={1} />
    </div>
  )
}
