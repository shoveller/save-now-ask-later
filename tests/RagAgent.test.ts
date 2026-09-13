import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { DatabaseSync } from 'node:sqlite'
import { RagAgent } from '../worker/RagAgent.ts'
import { embedMany, streamText } from 'ai'
import {drizzle} from 'drizzle-orm/node-sqlite'
import {migrate} from 'drizzle-orm/durable-sqlite/migrator'
import {eq} from 'drizzle-orm'
import {chunksTable, sourcesTable} from '../worker/schema'
import migrations from '../drizzle/migrations.js'

// Only stub the Durable Object shell; use the real embedding method and provider.
vi.mock('@cloudflare/ai-chat', () => ({ AIChatAgent: class {} }))
vi.mock('agents', () => ({ callable: () => () => undefined }))
vi.mock('../worker/createAI.ts', async () => {
  const {createOpenAICompatible} = await import('@ai-sdk/openai-compatible')
  return {createAI: () => createOpenAICompatible({
    name: 'test-proxy',
    baseURL: 'https://embedding.test/v1',
    fetch: async (_url, init) => {
      const request = JSON.parse(String(init?.body))
      const result = await run(request.model, {text: request.input}, {})
      return Response.json({
        data: result.data.map((embedding: number[], index: number) => ({embedding, index})),
        usage: {prompt_tokens: 1, total_tokens: 1},
      })
    },
  })}
})
vi.mock('ai', async importOriginal => ({
  ...await importOriginal<typeof import('ai')>(),
  streamText: vi.fn(),
}))

let database: DatabaseSync
let db: ReturnType<typeof drizzle>
let exec: ReturnType<typeof vi.fn>
let agent: RagAgent
let run: ReturnType<typeof vi.fn>
let upsert: ReturnType<typeof vi.fn>
let query: ReturnType<typeof vi.fn>
let browserFetch: ReturnType<typeof vi.fn>

beforeEach(() => {
  database = new DatabaseSync(':memory:')
  db = drizzle({client: database})
  agent = Object.create(RagAgent.prototype) as RagAgent
  run = vi.fn().mockImplementation(async (_model, input: {text: string[]}) => ({
    shape: [input.text.length, 384],
    data: input.text.map(() => Array.from({length: 384}, () => 0.1)),
  }))
  upsert = vi.fn().mockResolvedValue({mutationId: 'queued'})
  query = vi.fn().mockResolvedValue({matches: []})
  browserFetch = vi.fn().mockResolvedValue(Response.json({success: true, result: '# Example\n' + 'a'.repeat(1500)}))
  vi.stubGlobal('fetch', browserFetch)
  exec = vi.fn((sql: string, ...bindings: (string | number | null)[]) => {
    const statement = database.prepare(sql)
    const rows = statement.all(...bindings)
    return {toArray: () => rows, one: () => rows[0], raw: () => ({
      toArray: () => rows.map(row => Object.values(row)),
      next: () => ({value: rows[0] && Object.values(rows[0])}),
    })}
  })
  Object.defineProperties(agent, {
    env: {value: {AI: {run}, VECTORIZE: {upsert, query}, ACCOUNT_ID: 'account', CF_API_KEY: 'test-token'}},
    ctx: {value: {id: {toString: () => 'agent-1'}, storage: {
      sql: {exec},
      transactionSync: (callback: () => void) => db.transaction(callback),
    }}},
    messages: {value: []},
  })
  agent.onStart()
})

afterEach(() => {
  database.close()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

test('save uses only Browser Run Markdown JSON and persists matching chunk/vector IDs', async () => {
  const result = await agent.saveUrl('https://example.com')
  expect(browserFetch).toHaveBeenCalledExactlyOnceWith(
    'https://api.cloudflare.com/client/v4/accounts/account/browser-rendering/markdown',
    {method: 'post', headers: {Authorization: 'Bearer test-token', 'Content-Type': 'application/json'},
      body: JSON.stringify({url: 'https://example.com'})},
  )
  const rows = db.select().from(chunksTable).all()
  const vectors = upsert.mock.calls[0][0]
  expect(rows).toHaveLength(4)
  expect(vectors.map((vector: {id: string}) => vector.id)).toEqual(rows.map(row => row.id))
  expect(vectors.every((vector: {namespace: string, values: number[]}) =>
    vector.namespace === 'agent-1' && vector.values.length === 384)).toBe(true)
  expect(rows.every(row => new TextEncoder().encode(String(row.text).normalize('NFKC')).length <= 480)).toBe(true)
  expect(agent.listSources()).toEqual([{url: result.url, title: 'Example', savedAt: result.savedAt}])
  expect(Number.isNaN(Date.parse(result.savedAt))).toBe(false)
})

test('recall embeds the question and resolves topK5 matches through SQL in rank order', async () => {
  await agent.saveUrl('https://example.com')
  const rows = db.select().from(chunksTable).all()
  query.mockResolvedValue({matches: [{id: rows[1].id}, {id: 'missing'}, {id: rows[0].id}]})
  expect((await agent.recall('What was saved?')).matches).toEqual([rows[1], rows[0]].map(row => ({
    id: row.id, text: row.text, url: row.source,
  })))
  expect(query).toHaveBeenCalledWith(expect.any(Array), {topK: 5, namespace: 'agent-1'})
  expect(run).toHaveBeenLastCalledWith('multilingual-e5-small-fp32',
    expect.objectContaining({text: ['query: What was saved?']}), expect.anything())
})

test('empty recall and source list return no fabricated evidence', async () => {
  expect((await agent.recall('Unknown?')).matches).toEqual([])
  expect(agent.listSources()).toEqual([])
})

test('embedding timings exclude document fetch, vector storage and search, including empty results', async () => {
  let clock = 0
  vi.spyOn(performance, 'now').mockImplementation(() => clock)
  browserFetch.mockImplementation(async () => {
    clock += 1000
    return Response.json({success: true, result: '# Timing test'})
  })
  run.mockImplementation(async (_model, input: {text: string[]}) => {
    clock += 125
    return {data: input.text.map(() => Array.from({length: 384}, () => 0.1))}
  })
  upsert.mockImplementation(async () => { clock += 2000 })
  query.mockImplementation(async () => { clock += 3000; return {matches: []} })

  expect((await agent.saveUrl('https://example.com')).embeddingDurationMs).toBe(125)
  expect(await agent.recall('What was saved?')).toEqual({matches: [], embeddingDurationMs: 125})
  expect(clock).toBe(6250)
})

test.each([false, true])('startup preserves pre-migration data (existing sources: %s)', existingSources => {
  database.close()
  database = new DatabaseSync(':memory:')
  db = drizzle({client: database})
  const [chunksMigration, sourcesMigration] = Object.entries(migrations.migrations)
  migrate(agent.db, {migrations: Object.fromEntries([chunksMigration])})
  db.insert(chunksTable).values([
    {id: 'old', source: 'https://old.example', text: 'Legacy'},
    {id: 'old-2', source: 'https://old.example', text: 'Legacy 2'},
  ]).run()
  const recorded = {url: 'https://saved.example', title: 'Recorded title', savedAt: '2026-09-01T00:00:00.000Z'}
  if (existingSources) {
    // Reproduce the unmanaged table using generated DDL, not handwritten SQL.
    database.exec(sourcesMigration[1])
    db.insert(sourcesTable).values(recorded).run()
    db.insert(chunksTable).values({id: 'saved', source: recorded.url, text: 'Saved'}).run()
  }
  agent.onStart()
  agent.onStart()
  expect(db.select().from(chunksTable).where(eq(chunksTable.id, 'old')).get()?.text).toBe('Legacy')
  expect(db.select().from(chunksTable).all()).toHaveLength(existingSources ? 3 : 2)
  expect(agent.listSources()).toEqual([
    ...(existingSources ? [recorded] : []),
    {url: 'https://old.example', title: 'https://old.example', savedAt: null},
  ])
})

test('repeat saves retain old chunks but list one source; missing heading falls back to URL', async () => {
  browserFetch.mockImplementation(async () => Response.json({success: true, result: 'Plain text'}))
  await agent.saveUrl('https://example.com')
  expect(agent.listSources()[0].title).toBe('https://example.com')
  db.update(sourcesTable).set({savedAt: '2020-01-01T00:00:00.000Z'}).run()
  browserFetch.mockResolvedValue(Response.json({success: true, result: '# Updated title'}))
  await agent.saveUrl('https://example.com')
  expect(agent.listSources()).toHaveLength(1)
  expect(agent.listSources()[0].title).toBe('Updated title')
  expect(agent.listSources()[0].savedAt).not.toBe('2020-01-01T00:00:00.000Z')
  agent.onStart()
  expect(agent.listSources()[0].title).toBe('Updated title')
  expect(db.select().from(chunksTable).all()).toHaveLength(2)
})

test('sources sort newest first, then URL, with unknown legacy times last', () => {
  const sources = [
    {url: 'https://legacy.example', title: 'Legacy', savedAt: null},
    {url: 'https://b.example', title: 'B', savedAt: '2026-09-02T00:00:00.000Z'},
    {url: 'https://old.example', title: 'Old', savedAt: '2026-09-01T00:00:00.000Z'},
    {url: 'https://a.example', title: 'A', savedAt: '2026-09-02T00:00:00.000Z'},
  ]
  db.insert(sourcesTable).values(sources).run()
  expect(agent.listSources()).toEqual([sources[3], sources[1], sources[2], sources[0]])
})

test.each([
  [503, {success: false}],
  [200, {success: false, result: 'error'}],
  [200, {success: true, result: '   '}],
  [200, {success: true, result: {markdown: 'wrong shape'}}],
])('Browser Run failure %s does not write data', async (status, body) => {
  browserFetch.mockResolvedValue(Response.json(body, {status}))
  await expect(agent.saveUrl('https://example.com')).rejects.toThrow('Browser Run')
  expect(run).not.toHaveBeenCalled()
  expect(upsert).not.toHaveBeenCalled()
  expect(agent.listSources()).toEqual([])
})

test('invalid URLs and missing credentials fail before making a request', async () => {
  await expect(agent.saveUrl('file:///secret')).rejects.toThrow('HTTP(S)')
  await expect(agent.saveUrl('https://user:pass@example.com')).rejects.toThrow('HTTP(S)')
  Object.assign(agent.env, {CF_API_KEY: ''})
  await expect(agent.saveUrl('https://example.com')).rejects.toThrow('credentials')
  expect(browserFetch).not.toHaveBeenCalled()
})

test('network and embedding failures leave SQL untouched', async () => {
  browserFetch.mockRejectedValueOnce(new Error('network unavailable'))
  await expect(agent.saveUrl('https://example.com')).rejects.toThrow('network unavailable')
  run.mockRejectedValue(new Error('AI unavailable'))
  await expect(agent.saveUrl('https://example.com')).rejects.toThrow()
  expect(agent.listSources()).toEqual([])
  expect(upsert).not.toHaveBeenCalled()
})

test('wrong vector dimensions fail before storage or query', async () => {
  run.mockImplementation(async (_model, input: {text: string[]}) => ({
    data: input.text.map(() => [0.1, 0.2]),
  }))
  await expect(agent.saveUrl('https://example.com')).rejects.toThrow('384-dimensional')
  await expect(agent.recall('Question')).rejects.toThrow('384-dimensional')
  expect(upsert).not.toHaveBeenCalled()
  expect(query).not.toHaveBeenCalled()
})

test('Vectorize failures propagate without publishing a saved source', async () => {
  upsert.mockRejectedValue(new Error('index unavailable'))
  await expect(agent.saveUrl('https://example.com')).rejects.toThrow('index unavailable')
  expect(db.select().from(chunksTable).all()).toEqual([])
  expect(agent.listSources()).toEqual([])
  query.mockRejectedValue(new Error('query unavailable'))
  await expect(agent.recall('Question')).rejects.toThrow('query unavailable')
  await expect(agent.recall(' ')).rejects.toThrow('empty')
})

test('SQL write failures roll back chunks and sources; list errors are not hidden', async () => {
  const sourceInsert = db.insert(sourcesTable).values({url: '', title: '', savedAt: ''})
    .onConflictDoUpdate({target: sourcesTable.url, set: {title: '', savedAt: ''}}).toSQL().sql
  const execute = exec.getMockImplementation()!
  exec.mockImplementation((sql, ...bindings) => {
    if (sql === sourceInsert) throw new Error('SQL unavailable')
    return execute(sql, ...bindings)
  })
  await expect(agent.saveUrl('https://example.com')).rejects.toThrow()
  expect(upsert).toHaveBeenCalledOnce()
  expect(db.select().from(chunksTable).all()).toEqual([])
  expect(agent.listSources()).toEqual([])
  exec.mockImplementationOnce(() => {throw new Error('SQL unavailable')})
  expect(() => agent.listSources()).toThrow()
})

test('Unicode chunking keeps conservative normalized byte limits and preserves overlap', () => {
  const chunks = agent.toChunks('\u{1F600}'.repeat(1400))
  expect(chunks).toHaveLength(15)
  expect(Array.from(chunks[0])).toHaveLength(120)
  expect(chunks.every(chunk => new TextEncoder().encode(chunk).length <= 480)).toBe(true)
  const expanded = agent.toChunks('ﷺ'.repeat(100))
  expect(expanded.every(chunk => new TextEncoder().encode(chunk.normalize('NFKC')).length <= 480)).toBe(true)
  expect(agent.toChunks('  ')).toEqual([])
})

test('E5 document requests are prefixed and sent in bounded batches', async () => {
  const values = Array.from({length:130}, (_,i) => `chunk ${i}`)
  expect(await agent.toEmbeddings(values)).toHaveLength(130)
  expect(run.mock.calls.map(call => call[1].text.length)).toEqual([64,64,2])
  expect(run.mock.calls.flatMap(call => call[1].text)).toEqual(values.map(value => `passage: ${value}`))
  run.mockClear()
  await expect(agent.toEmbeddings(['가'.repeat(161)])).rejects.toThrow('input budget')
  await expect(agent.recall('가'.repeat(161))).rejects.toThrow('too long')
  expect(run).not.toHaveBeenCalled()
})

test('large saves split Vectorize writes and retain matching chunk IDs', async () => {
  browserFetch.mockResolvedValue(Response.json({success:true,result:'# Large\n'+'a'.repeat(40000)}))
  const result = await agent.saveUrl('https://example.com/large')
  expect(upsert.mock.calls.length).toBeGreaterThan(1)
  expect(upsert.mock.calls.every(call => call[0].length <= 100)).toBe(true)
  const vectors = upsert.mock.calls.flatMap(call => call[0])
  expect(vectors).toHaveLength(result.chunks)
  expect(db.select().from(chunksTable).all().map(row=>row.id)).toEqual(vectors.map(vector=>vector.id))
})

test('chat exposes all three executable tools with evidence instructions and a multi-step loop', async () => {
  vi.mocked(streamText).mockImplementation(() => {throw new Error('capture options')})
  await expect(agent.onChatMessage()).rejects.toThrow('capture options')
  const options = vi.mocked(streamText).mock.calls.at(-1)![0]
  expect(Object.keys(options.tools!)).toEqual(['saveUrl', 'recall', 'listSources'])
  expect(options.instructions).toContain('ONLY from relevant recall results')
  expect(options.instructions).toContain('cite their exact source URLs')
  expect(options.instructions).toContain('no supporting saved evidence')
  expect(options.instructions).toContain('even without an explicit save request')
  expect(options.stopWhen).toBeTypeOf('function')
  const save = vi.spyOn(agent, 'saveUrl').mockResolvedValue({} as Awaited<ReturnType<RagAgent['saveUrl']>>)
  const recall = vi.spyOn(agent, 'recall').mockResolvedValue({matches: [], embeddingDurationMs: 0})
  const list = vi.spyOn(agent, 'listSources').mockReturnValue([])
  const execution = {toolCallId: 'test', messages: []}
  await options.tools!.saveUrl.execute!({url: 'https://example.com'}, execution)
  await options.tools!.recall.execute!({question: 'Question'}, execution)
  await options.tools!.listSources.execute!({}, execution)
  expect(save).toHaveBeenCalledWith('https://example.com')
  expect(recall).toHaveBeenCalledWith('Question')
  expect(list).toHaveBeenCalled()
})

const values = ['Korea is in East Asia.', 'Seoul is the capital of South Korea.']

function checkDimensions(embeddings: number[][], dimensions = 384) {
  expect(embeddings).toHaveLength(values.length)
  for (const embedding of embeddings) {
    expect(embedding).toHaveLength(dimensions)
    expect(embedding.every(value => Number.isFinite(value))).toBe(true)
  }
}

test('toEmbeddings uses the self-hosted model and preserves vector count and dimensions', async () => {
  const data = values.map(() => Array.from({ length: 384 }, (_, i) => i / 384))
  run.mockResolvedValue({ shape: [values.length, 384], data })
  const agent = Object.create(RagAgent.prototype) as RagAgent
  Object.defineProperty(agent, 'env', { value: { AI: { run } } })

  const embeddings = await agent.toEmbeddings(values)

  expect(run).toHaveBeenCalledWith(
    'multilingual-e5-small-fp32',
    expect.objectContaining({ text: values.map(value => `passage: ${value}`) }),
    expect.anything(),
  )
  expect(embeddings).toEqual(data)
  checkDimensions(embeddings)
})

// Opt in explicitly: this test uses Cloudflare authentication and billable AI usage.
test.skipIf(process.env.RUN_REMOTE_EMBEDDINGS !== '1')(
  'Cloudflare benchmark model returns 768-dimensional vectors from the real binding',
  async () => {
    vi.unstubAllGlobals()
    const { getPlatformProxy } = await import('wrangler')
    const platform = await getPlatformProxy({
      configPath: 'tests/wrangler.embeddings.jsonc',
      persist: false,
      remoteBindings: true,
    })

    try {
      const agent = Object.create(RagAgent.prototype) as RagAgent
      Object.defineProperty(agent, 'env', { value: platform.env })
      const {embeddings} = await embedMany({model: agent.embedModel, values})

      console.log('Actual embedding dimensions:', embeddings.map(vector => vector.length))
      checkDimensions(embeddings, 768)
    } finally {
      await platform.dispose()
    }
  },
  120_000,
)
