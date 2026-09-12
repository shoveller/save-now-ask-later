import {AIChatAgent} from "@cloudflare/ai-chat";
import {callable} from "agents";
import {createWorkersAI} from "workers-ai-provider";
import {
    convertToModelMessages,
    createUIMessageStreamResponse, embed,
    embedMany,
    streamText,
    tool,
    toUIMessageStream
} from "ai";
import {drizzle, DrizzleSqliteDODatabase} from "drizzle-orm/durable-sqlite";
import {migrate} from "drizzle-orm/durable-sqlite/migrator";
import migrations from "../drizzle/migrations.js";
import {chunksTable} from "./schema.ts";
import {z} from "zod";
import {eq} from "drizzle-orm";
import {createAI} from "./createAI.ts";
import type {BenchmarkSample} from "../shared/benchmark.ts";

export class RagAgent extends AIChatAgent {
    db: DrizzleSqliteDODatabase | undefined

    override onStart() {
        this.db = drizzle(this.ctx.storage)
        migrate(this.db, migrations)
    }

    @callable()
    async toMarkdown(url: string) {
        return await fetch(`https://api.cloudflare.com/client/v4/accounts/${this.env.ACCOUNT_ID}/browser-rendering/markdown`, {
            method: 'post',
            headers: {
                Authorization: `Bearer ${this.env.CF_API_KEY}`,
                "Content-Type": "text/markdown; charset=utf-8"
            },
            body: JSON.stringify({url})
        }).then(async (res) => await res.json<{ success: boolean, result: string, meta: unknown }>())
    }

    toChunks(value: string) {
        const chunkSize = 1000
        const overlap = 200
        const characters = Array.from(value)
        const chunks: string[] = []

        for (let start = 0; start < characters.length; start += chunkSize - overlap) {
            const chunk = characters.slice(start, start + chunkSize).join('')
            if (chunk.trim()) chunks.push(chunk)
            if (start + chunkSize >= characters.length) break
        }

        return chunks
    }

    get embedModel() {
        const workersAi = createWorkersAI({binding: this.env.AI})
        return workersAi.textEmbeddingModel("@cf/google/embeddinggemma-300m")
    }

    get embedModel2() {
        const ai = createAI()

        return ai.embeddingModel('embeddinggemma:300m')
    }

    get llm() {
        const ai = createAI()

        return ai('gemini-3.8-flash-high')
    }

    @callable()
    async benchmarkEmbedding(input: unknown): Promise<BenchmarkSample> {
        const {model: key, text} = z.object({
            model: z.enum(['embedModel', 'embedModel2']),
            text: z.string().min(1).max(8000).refine(value => value.trim().length > 0)
        }).parse(input)
        const model = this[key]
        const started = performance.now()
        try {
            const {embedding} = await embed({
                model,
                value: text,
                maxRetries: 0,
                abortSignal: AbortSignal.timeout(60_000)
            })
            const durationMs = performance.now() - started
            if (!embedding.length || !embedding.every(Number.isFinite)) {
                throw new Error('Invalid embedding response')
            }
            return {model: key, durationMs, dimensions: embedding.length, error: null}
        } catch (error) {
            // Provider errors may contain request URLs or credentials; keep them server-side.
            console.error('Embedding benchmark failed', key, error)
            return {
                model: key,
                durationMs: performance.now() - started,
                dimensions: null,
                error: '임베딩 호출 실패 또는 60초 제한 초과. 서버 로그를 확인하세요.'
            }
        }
    }

    async toEmbeddings(values: string[]) {
        const {embeddings} = await embedMany({
            model: this.embedModel2,
            values
        })

        return embeddings
    }

    saveChunk({ id, text, source }: {id: string, source: string, text: string}) {
        const db = this.db
        if (!db) {
            throw new Error('Database is not initialized')
        }

        db.insert(chunksTable).values({
            id,
            source,
            text
        }).run()
    }

    async toVectors({value, source, onChunk}:{value: string, source: string, onChunk: (param: {id: string, chunk: string}) => void}) {
        const chunks = this.toChunks(value)
        const embeddings = await this.toEmbeddings(chunks)

        return chunks.map((chunk, index) => {
            const id = crypto.randomUUID()
            onChunk({ id, chunk })

            return {
                id,
                values: embeddings[index],
                metadata: {source}
            }
        })
    }

    async ingest() {
        const url = 'https://en.wikipedia.org/wiki/Korea'
        const {result} = await this.toMarkdown(url)

        const vectors = await this.toVectors({
            source: url,
            value: result,
            onChunk: ({ id, chunk }) => this.saveChunk({ id, source: url, text: chunk })
        })
        await this.env.VECTORIZE.upsert(vectors)
    }

    async onChatMessage() {
        const result = streamText({
            model: this.llm,
            messages: await convertToModelMessages(this.messages),
            tools: {
                recall: tool({
                    description:
                        `Search ingested documents for chunks relevant to a query. 
                        Call this before answering questions about previously-saved content.`,
                    inputSchema: z.object({
                        query: z.string().meta({description: "What to look up. "})
                    }),
                    execute: async ({query}) => {
                        const {embedding} = await embed({
                            model: this.embedModel,
                            value: query
                        })

                        const {matches} = await this.env.VECTORIZE.query(embedding, {
                            topK: 5
                        })

                        return matches.map(match => {
                            // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                            // @ts-expect-error
                            const [row] = this.db.select().from(chunksTable).where((record) => eq(record.id, match.id)).run()
                            return row
                        })
                    }
                })
            }
        })

        return createUIMessageStreamResponse({
            stream: toUIMessageStream({stream: result.stream})
        })
    }
}
