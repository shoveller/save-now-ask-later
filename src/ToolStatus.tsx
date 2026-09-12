import {getToolName} from 'ai'
import type {DynamicToolUIPart, ToolUIPart} from 'ai'

export default function ToolStatus({part}: {part: ToolUIPart | DynamicToolUIPart}) {
  const name = getToolName(part)
  const isEmbedding = name === 'saveUrl' || name === 'recall'
  const output: unknown = part.state === 'output-available' ? part.output : null
  const duration = output && typeof output === 'object' && 'embeddingDurationMs' in output
    && typeof output.embeddingDurationMs === 'number' && Number.isFinite(output.embeddingDurationMs)
    && output.embeddingDurationMs >= 0 ? output.embeddingDurationMs : null

  return <div className="tool-status">
    <small>Tool: {name}</small>
    {isEmbedding && <small className="embedding-time" title="서버 임베딩 처리 시간입니다. 제공자 통신과 SDK 재시도는 포함하며, 문서 수집·벡터 검색·저장·답변 생성은 제외합니다.">
      {name === 'saveUrl' ? '문서 임베딩' : '질문 임베딩'} · {part.state === 'output-available'
        ? duration === null ? '시간 기록 없음' : `${duration.toLocaleString('ko-KR', {minimumFractionDigits: 1, maximumFractionDigits: 1})} ms`
        : part.state === 'output-error' || part.state === 'output-denied'
          ? '작업 실패 · 시간 기록 없음' : '작업 진행 중…'}
    </small>}
  </div>
}
