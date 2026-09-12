import {createElement} from 'react'
import {renderToStaticMarkup} from 'react-dom/server'
import {expect, test} from 'vitest'
import type {DynamicToolUIPart} from 'ai'
import ToolStatus from '../src/ToolStatus'

function render(part: DynamicToolUIPart) {
  return renderToStaticMarkup(createElement(ToolStatus, {part}))
}

test('shows measured timing for document and query embeddings, including zero and no matches', () => {
  for (const [toolName, label] of [['saveUrl', '문서 임베딩'], ['recall', '질문 임베딩']]) {
    const html = render({type: 'dynamic-tool', toolName, toolCallId: '1', state: 'output-available', input: {},
      output: {matches: [], embeddingDurationMs: 1234.5}})
    expect(html).toContain(label)
    expect(html).toContain('1,234.5 ms')
  }
  expect(render({type: 'dynamic-tool', toolName: 'recall', toolCallId: '1', state: 'output-available', input: {},
    output: {embeddingDurationMs: 0}})).toContain('0.0 ms')
})

test('does not fabricate timings for old messages, errors or pending calls', () => {
  const base = {type: 'dynamic-tool', toolName: 'recall', toolCallId: '1'} as const
  for (const output of [[], null, {}, {embeddingDurationMs: -1}, {embeddingDurationMs: NaN}]) {
    expect(render({...base, state: 'output-available', input: {}, output})).toContain('시간 기록 없음')
  }
  expect(render({...base, state: 'input-available', input: {}})).toContain('작업 진행 중')
  expect(render({...base, state: 'output-error', input: {}, errorText: 'failed'})).toContain('작업 실패')
  expect(render({...base, toolName: 'listSources', state: 'output-available', input: {}, output: []}))
    .not.toContain('embedding-time')
})
