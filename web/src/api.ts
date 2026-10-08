import { useEffect, useState } from 'react'

export type PerClass = { precision: number; recall: number; f1: number; support: number }
export type Split = { n_epochs: number; accuracy?: number; macro_f1?: number }
export type Confusion = { true: string; predicted: string; count: number; rate: number }

export type TestMetrics = {
  n_epochs: number
  accuracy: number
  macro_f1: number
  cohen_kappa: number
  majority_class_baseline: number
  per_class: Record<string, PerClass>
  confusion_matrix: number[][]
  confusion_matrix_normalized: number[][]
  top_confusions: Confusion[]
  n1_rem_focus: {
    n1_recall: number
    rem_recall: number
    n1_predicted_as: Record<string, number>
    rem_predicted_as: Record<string, number>
  }
  transition_analysis: { near_transition: Split; stable: Split }
  test_subjects: number[]
  test_recordings: string[]
}

export type MeanStd = { mean: number; std: number }
export type CvFold = {
  fold: number
  test_subjects: number[]
  accuracy: number
  macro_f1: number
  cohen_kappa: number
  n_epochs: number
  per_class_f1: Record<string, number>
}
export type CvMetrics = {
  n_folds: number
  n_subjects: number
  accuracy: MeanStd
  macro_f1: MeanStd
  cohen_kappa: MeanStd
  per_class_f1: Record<string, MeanStd>
  pooled: TestMetrics
  folds: CvFold[]
}

export type Summary = {
  classes: string[]
  test: TestMetrics | null
  cv: CvMetrics | null
  split: {
    n_epochs: number
    n_subjects: number
    n_recordings: number
    class_counts: Record<string, number>
    wake_edge_min: number
    folds: { train: number[]; val: number[]; test: number[] }[]
  } | null
  model: {
    parameters: number
    size_mb: number
    macs_per_epoch: number
    baseline_macs_per_epoch: number
    checkpoint_epoch: number
    val_f1: number
    train_subjects: number[]
    val_subjects: number[]
    test_subjects: number[]
    hyperparameters: Record<string, string | number | boolean>
  }
}

export type Role = 'train' | 'val' | 'test' | 'unseen'
export type Recording = {
  key: string
  label: string
  subject: number | null
  night: number | null
  role: Role | null
  has_labels: boolean
  hours: number | null
  demo?: boolean
  source?: string
  uploaded?: boolean
  channel?: string
}

export type SleepSummary = {
  time_in_bed_min: number
  total_sleep_time_min: number
  sleep_efficiency_pct: number
  sleep_onset_latency_min: number | null
  rem_latency_min: number | null
  waso_min: number | null
  stage_minutes: Record<string, number>
}

export type Night = {
  key: string
  info: { subject?: number; night?: number; recording?: string; source?: string; channel?: string; uploaded?: boolean }
  role: Role | null
  epoch_sec: number
  n_epochs: number
  sleep_window: [number, number]
  wake_edge_min: number
  predicted: number[]
  confidence: number[]
  probs: number[][]
  expert: number[] | null
  summary_predicted: SleepSummary
  summary_expert: SleepSummary | null
  agreement: {
    accuracy: number
    macro_f1: number
    cohen_kappa: number
    n_epochs: number
    per_class_f1: Record<string, number>
  } | null
}

export type Epoch = {
  index: number
  start_sec: number
  sfreq: number
  signal_uv: number[]
  probs: number[]
  predicted: number
  expert: number | null
  bands: Record<string, number>
  psd?: { freqs: number[]; db: number[] }   // absent on older API versions
}

export type Examples = {
  true: string
  pred: string
  total: number
  examples: { key: string; epoch: number }[]
}

export type UploadResult = { key: string; name: string; channel: string; hours: number; has_labels: boolean }

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) {
    let detail = res.statusText
    try { detail = (await res.json()).detail ?? detail } catch { /* not JSON */ }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  summary: () => getJson<Summary>('/api/summary'),
  recordings: () => getJson<{ demo_mode: boolean; recordings: Recording[] }>('/api/recordings'),
  night: (key: string) => getJson<Night>(`/api/recordings/${key}/night`),
  epoch: (key: string, index: number) => getJson<Epoch>(`/api/recordings/${key}/epoch/${index}`),
  examples: (trueStage: string, pred: string) =>
    getJson<Examples>(`/api/examples?true=${trueStage}&pred=${pred}`),
  upload: async (psg: File, hypnogram: File | null): Promise<UploadResult> => {
    const form = new FormData()
    form.append('psg', psg)
    if (hypnogram) form.append('hypnogram', hypnogram)
    const res = await fetch('/api/upload', { method: 'POST', body: form })
    const body = await res.json().catch(() => ({}))
    if (!res.ok) throw new Error(body.detail ?? res.statusText)
    return body as UploadResult
  },
}

/** Fetch with loading / error state; refetches when `key` changes (null = skip). */
export function useFetch<T>(key: string | null, fetcher: () => Promise<T>) {
  const [tick, setTick] = useState(0)
  const [state, setState] = useState<{ data: T | null; error: string | null; loading: boolean }>({
    data: null, error: null, loading: key !== null,
  })
  useEffect(() => {
    if (key === null) return
    let alive = true
    setState((s) => ({ ...s, loading: true, error: null }))
    fetcher()
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((e: Error) => alive && setState({ data: null, error: e.message, loading: false }))
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, tick])
  return { ...state, reload: () => setTick((t) => t + 1) }
}
