import {createElement} from 'react'
import {renderToStaticMarkup} from 'react-dom/server'
import {expect, test, vi} from 'vitest'
import App from '../src/App.tsx'

const chat = vi.hoisted(() => ({
  messages: [{id: 'answer', role: 'assistant', parts: [
    {type: 'text', text: 'Evidence: https://example.com'},
    {type: 'tool-recall', toolCallId: 'recall-1', state: 'output-available', input: {question: 'Question'},
      output: {matches: [], embeddingDurationMs: 125.5}},
  ]}],
  sendMessage: vi.fn(), status: 'ready', error: null,
  connectionError: null as {message: string} | null,
}))
vi.mock('agents/react', () => ({useAgent: () => ({})}))
vi.mock('agents/chat/react', () => ({useAgentChat: () => chat}))

test('chat renders text parts, tool activity and a labeled message form', () => {
  const html = renderToStaticMarkup(createElement(App))
  expect(html).toContain('Evidence: https://example.com')
  expect(html).toContain('Tool: recall')
  expect(html).toContain('질문 임베딩 · 125.5 ms')
  expect(html).toContain('for="message"')
  expect(html).toContain('<textarea')
  expect(html).toContain('type="submit"')
})

test('chat shows connection errors and disables input while streaming', () => {
  chat.connectionError = {message: 'Connection unavailable'}
  chat.status = 'streaming'
  const html = renderToStaticMarkup(createElement(App))
  expect(html).toContain('role="alert">Connection unavailable')
  expect(html).toContain('role="status">Working...')
  expect(html).toMatch(/<textarea[^>]*disabled/)
  chat.connectionError = null
  chat.status = 'ready'
})
