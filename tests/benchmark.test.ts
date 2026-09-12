import {afterEach, expect, test, vi} from 'vitest'
import {embed} from 'ai'
import {RagAgent} from '../worker/RagAgent'
import {summarize} from '../shared/benchmark'
import type {BenchmarkSample} from '../shared/benchmark'

vi.mock('@cloudflare/ai-chat', () => ({ AIChatAgent: class {} }))
vi.mock('agents', () => ({ callable: () => () => undefined }))
vi.mock('../worker/createAI.ts', () => ({createAI: vi.fn()}))
vi.mock('../drizzle/migrations.js', () => ({default: {}}))
vi.mock('ai', () => ({embed: vi.fn()}))
afterEach(() => vi.restoreAllMocks())

function makeAgent() {
  const agent = Object.create(RagAgent.prototype) as RagAgent
  Object.defineProperty(agent, 'embedModel', {value: {modelId: 'first'}})
  Object.defineProperty(agent, 'embedModel2', {value: {modelId: 'second'}})
  return agent
}

test('measures the selected model, disables retries and returns metadata without vectors', async () => {
  vi.mocked(embed).mockResolvedValue({embedding: [1, 2, 3]} as Awaited<ReturnType<typeof embed>>)
  vi.spyOn(performance, 'now').mockReturnValueOnce(100).mockReturnValueOnce(125)
  const agent = makeAgent()
  const result = await agent.benchmarkEmbedding({model: 'embedModel2', text: 'hello'})
  expect(embed).toHaveBeenCalledWith({model: agent.embedModel2, value: 'hello', maxRetries: 0, abortSignal: expect.any(AbortSignal)})
  expect(result).toEqual({model: 'embedModel2', durationMs: 25, dimensions: 3, error: null})
})

test('provider failure is recorded without exposing provider error details', async () => {
  vi.mocked(embed).mockRejectedValue(new Error('private provider details'))
  vi.spyOn(console, 'error').mockImplementation(() => {})
  const result = await makeAgent().benchmarkEmbedding({model: 'embedModel', text: 'hello'})
  expect(result.error).toBeTruthy()
  expect(result.error).not.toContain('private provider details')
  expect(result.dimensions).toBeNull()
})

test('rejects invalid models and empty or oversized inputs before calling the provider', async () => {
  vi.mocked(embed).mockClear()
  for (const input of [
    {model: 'llm', text: 'hello'},
    {model: 'embedModel', text: '   '},
    {model: 'embedModel', text: 'x'.repeat(8001)},
  ]) await expect(makeAgent().benchmarkEmbedding(input)).rejects.toThrow()
  expect(embed).not.toHaveBeenCalled()
})

test('statistics exclude failures and handle even, odd and empty samples', () => {
  const sample = (durationMs: number, error: string | null = null): BenchmarkSample => ({model: 'embedModel', durationMs, dimensions: 3, error})
  expect(summarize([sample(40), sample(10), sample(30), sample(20), sample(999, 'failed')])).toEqual({count: 4, failures: 1, mean: 25, median: 25, min: 10, max: 40})
  expect(summarize([sample(30), sample(10), sample(20)]).median).toBe(20)
  expect(summarize([sample(999, 'failed')])).toEqual({count: 0, failures: 1, mean: null, median: null, min: null, max: null})
  expect(summarize([]).median).toBeNull()
})
