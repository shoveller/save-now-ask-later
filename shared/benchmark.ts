export type BenchmarkModel = 'embedModel' | 'embedModel2'

export interface BenchmarkSample {
  model: BenchmarkModel
  durationMs: number
  dimensions: number | null
  error: string | null
}

export interface BenchmarkRow extends BenchmarkSample {
  phase: 'warmup' | 'measurement'
  round: number
}

export const benchmarkModels: BenchmarkModel[] = ['embedModel', 'embedModel2']

export function summarize(samples: BenchmarkSample[]) {
  const times = samples.filter(sample => !sample.error).map(sample => sample.durationMs).sort((a, b) => a - b)
  const count = times.length
  return {
    count,
    failures: samples.length - count,
    mean: count ? times.reduce((sum, time) => sum + time, 0) / count : null,
    median: count ? (times[Math.floor((count - 1) / 2)] + times[Math.floor(count / 2)]) / 2 : null,
    min: count ? times[0] : null,
    max: count ? times[count - 1] : null,
  }
}
