import {useEffect, useRef, useState} from 'react'
import {useAgent} from 'agents/react'
import {benchmarkModels, summarize} from '../shared/benchmark'
import type {BenchmarkRow, BenchmarkSample} from '../shared/benchmark'
import './Bench.css'
import {initialText} from "./initialText.ts";

const ms = (value: number | null) => value === null ? '—' : `${value.toFixed(1)} ms`

export default function Bench() {
  const [name] = useState(() => `bench-${crypto.randomUUID()}`)
  const [connected, setConnected] = useState(false)
  const agent = useAgent({
    agent: 'RagAgent', name,
    onOpen: () => setConnected(true),
    onClose: () => setConnected(false),
  })
  const [text, setText] = useState(() => initialText.slice(0, 1000))
  const [repetitions, setRepetitions] = useState(5)
  const [rows, setRows] = useState<BenchmarkRow[]>([])
  const [running, setRunning] = useState(false)
  const [status, setStatus] = useState('실행을 누르면 모델별 워밍업 1회 후 본 측정을 시작합니다.')
  const stop = useRef(false)
  const busy = useRef(false)
  useEffect(() => () => { stop.current = true }, [])
  const inputError = text.length > 8000
    ? `입력이 ${text.length.toLocaleString()}자입니다. 8,000자 이하로 줄여 주세요.`
    : !text.trim() ? '측정할 텍스트를 입력해 주세요.' : null

  async function run() {
    if (busy.current || !connected) return
    if (inputError) {
      setStatus(inputError)
      return
    }
    busy.current = true
    stop.current = false
    setRunning(true)
    setRows([])
    const first = Math.random() < 0.5 ? [...benchmarkModels] : [...benchmarkModels].reverse()
    try {
      for (let round = 0; round <= repetitions; round++) {
        const phase = round === 0 ? 'warmup' : 'measurement'
        const order = round % 2 ? [...first].reverse() : first
        for (const model of order) {
          if (stop.current) break
          setStatus(`${round === 0 ? '워밍업' : `측정 ${round}/${repetitions}`} · ${model} 실행 중`)
          const sample = await agent.call<BenchmarkSample>('benchmarkEmbedding', [{model, text}], {timeout: 75_000})
          setRows(previous => [...previous, {...sample, phase, round}])
        }
        if (stop.current) break
      }
      setStatus(stop.current ? '중지됨 · 완료된 호출까지의 결과입니다.' : '완료 · 실패한 호출은 시간 통계에서 제외됩니다.')
    } catch {
      setStatus('연결 또는 RPC 오류로 중단되었습니다. 연결 상태와 서버 로그를 확인하세요.')
    } finally {
      busy.current = false
      setRunning(false)
    }
  }

  const stats = benchmarkModels.map(model => summarize(rows.filter(row => row.model === model && row.phase === 'measurement')))
  const [a, b] = stats.map(stat => stat.median)
  const comparison = a !== null && b !== null && a > 0 && b > 0
    ? a === b ? '두 모델의 중앙값이 같습니다.' : `${a < b ? 'embedModel' : 'embedModel2'}의 중앙값 지연시간이 ${(100 * (1 - Math.min(a, b) / Math.max(a, b))).toFixed(1)}% 짧습니다.`
    : '두 모델의 본 측정이 완료되면 중앙값을 비교합니다.'

  return <main className="bench">
    <header><a href="/">← 홈</a><span>{connected ? '● 연결됨' : '○ 연결 대기 중'}</span></header>
    <p className="bench-eyebrow">EMBEDDING BENCHMARK</p>
    <h1>임베딩 속도 비교</h1>
    <p>동일한 텍스트를 두 모델에 순차 호출합니다. 최초 순서는 무작위이며, 매 회차 순서를 교대합니다.</p>
    <form onSubmit={event => { event.preventDefault(); void run() }}>
      <label htmlFor="bench-text">입력 텍스트 <small>({text.length.toLocaleString()} / 8,000자)</small></label>
      <textarea id="bench-text" value={text} onChange={event => setText(event.target.value)} aria-invalid={!!inputError} aria-describedby="bench-input-help" required disabled={running} rows={5}/>
      <p id="bench-input-help"><small>{inputError ?? '기본 입력은 원문의 앞 1,000자입니다. 8,000자는 화면 입력 제한이며, 모델의 토큰 한도와는 다릅니다.'}</small></p>
      {text.length > 8000 && <button type="button" disabled={running} onClick={() => setText(previous => previous.slice(0, 1000))}>앞 1,000자만 사용</button>}
      <div className="bench-controls">
        <label htmlFor="bench-count">모델별 측정 횟수
          <input id="bench-count" type="number" min={1} max={20} required value={repetitions} disabled={running} onChange={event => setRepetitions(event.target.valueAsNumber)}/>
        </label>
        <button type="submit" disabled={running || !connected || !!inputError}>벤치마크 실행</button>
        {running && <button type="button" onClick={() => { stop.current = true; setStatus('중지 요청됨 · 진행 중인 호출이 끝나면 멈춥니다.') }}>중지</button>}
      </div>
      <small>워밍업 포함 총 {Number.isFinite(repetitions) ? 2 * (repetitions + 1) : '—'}회 호출 · 실제 모델 사용 비용 발생 · 호출당 최대 60초 · SDK 재시도 없음</small>
    </form>
    <p role="status" className="bench-status">{status} ({rows.length}회 완료)</p>
    <section className="bench-cards" aria-label="모델별 통계">
      {benchmarkModels.map((model, index) => {
        const stat = stats[index]
        const warmup = rows.find(row => row.model === model && row.phase === 'warmup')
        const dimensions = [...new Set(rows
          .filter(row => row.model === model && !row.error && row.dimensions !== null)
          .map(row => row.dimensions))]
        return <article key={model}>
          <h2>{model}</h2>
          <code>{index === 0 ? '@cf/baai/bge-base-en-v1.5' : 'embeddinggemma:300m'}</code>
          <strong>{ms(stat.median)}</strong><p>본 측정 중앙값</p>
          <dl>
            <dt>임베딩 차원</dt><dd>{dimensions.length ? dimensions.map(value => `${value}차원`).join(' / ') : '—'}</dd>
            <dt>평균</dt><dd>{ms(stat.mean)}</dd>
            <dt>최솟값 / 최댓값</dt><dd>{ms(stat.min)} / {ms(stat.max)}</dd>
            <dt>성공 / 실패</dt><dd>{stat.count} / {stat.failures}</dd>
            <dt>첫 호출 (워밍업)</dt><dd>{warmup ? warmup.error ? '실패' : ms(warmup.durationMs) : '—'}</dd>
          </dl>
        </article>
      })}
    </section>
    <p className="bench-comparison">{comparison}</p>
    <p className="bench-note">서버의 embed() 호출 구간만 측정합니다. 제공자 네트워크 지연은 포함하고 브라우저 통신·DB·벡터 저장은 제외합니다. 첫 호출이 실제 콜드 스타트임을 보장하지 않으며, 제공자 캐시·부하의 영향을 받을 수 있습니다. 속도 비교이며 검색 품질 비교는 아닙니다.</p>
    <h2>호출별 기록</h2>
    <div className="bench-table"><table>
      <thead><tr><th>순서</th><th>단계</th><th>모델</th><th>소요 시간</th><th>차원</th><th>결과</th></tr></thead>
      <tbody>{rows.map((row, index) => <tr key={index}>
        <td>{index + 1}</td><td>{row.phase === 'warmup' ? '워밍업' : `측정 ${row.round}`}</td><td>{row.model}</td>
        <td>{ms(row.durationMs)}</td><td>{row.dimensions ?? '—'}</td><td>{row.error ?? '성공'}</td>
      </tr>)}</tbody>
    </table>{!rows.length && <p className="bench-empty">아직 측정 결과가 없습니다.</p>}</div>
  </main>
}
