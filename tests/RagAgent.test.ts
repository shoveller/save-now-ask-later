import { expect, test, vi } from 'vitest'
import { RagAgent } from '../worker/RagAgent.ts'

// Only stub the Durable Object shell; use the real embedding method and provider.
vi.mock('@cloudflare/ai-chat', () => ({ AIChatAgent: class {} }))
vi.mock('agents', () => ({ callable: () => () => undefined }))

const values = ['Korea is in East Asia.', 'Seoul is the capital of South Korea.']

function checkDimensions(embeddings: number[][]) {
  expect(embeddings).toHaveLength(values.length)
  for (const embedding of embeddings) {
    expect(embedding).toHaveLength(768)
    expect(embedding.every(value => Number.isFinite(value))).toBe(true)
  }
}

test('toEmbeddings preserves vector count and dimensions from a mocked AI binding', async () => {
  const data = values.map(() => Array.from({ length: 768 }, (_, i) => i / 768))
  const run = vi.fn().mockResolvedValue({ shape: [values.length, 768], data })
  const agent = Object.create(RagAgent.prototype) as RagAgent
  Object.defineProperty(agent, 'env', { value: { AI: { run } } })

  const embeddings = await agent.toEmbeddings(values)

  expect(run).toHaveBeenCalledWith(
    '@cf/google/embeddinggemma-300m',
    expect.objectContaining({ text: values }),
    expect.anything(),
  )
  expect(embeddings).toEqual(data)
  checkDimensions(embeddings)
})

// Opt in explicitly: this test uses Cloudflare authentication and billable AI usage.
test.skipIf(process.env.RUN_REMOTE_EMBEDDINGS !== '1')(
  'toEmbeddings returns 768-dimensional vectors from the real model',
  async () => {
    const { getPlatformProxy } = await import('wrangler')
    const platform = await getPlatformProxy({
      configPath: 'tests/wrangler.embeddings.jsonc',
      persist: false,
      remoteBindings: true,
    })

    try {
      const agent = Object.create(RagAgent.prototype) as RagAgent
      Object.defineProperty(agent, 'env', { value: platform.env })
      const embeddings = await agent.toEmbeddings(values)

      console.log('Actual embedding dimensions:', embeddings.map(vector => vector.length))
      checkDimensions(embeddings)
    } finally {
      await platform.dispose()
    }
  },
  120_000,
)
