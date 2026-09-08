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
import {createOpenAICompatible} from "@ai-sdk/openai-compatible";
import {z} from "zod";
import {eq} from "drizzle-orm";

export class RagAgent extends AIChatAgent {
    db: DrizzleSqliteDODatabase | undefined

    onStart() {
        void this.ctx.blockConcurrencyWhile(async () => {
            this.db = drizzle(this.ctx.storage)
            migrate(this.db, migrations)
        })
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

    async toEmbeddings(values: string[]) {
        const {embeddings} = await embedMany({
            model: this.embedModel,
            values
        })

        return embeddings
    }

    async ingest() {
        const db = this.db
        if (!db) {
            throw new Error('Database is not initialized')
        }

        const url = 'https://en.wikipedia.org/wiki/Korea'
        const {success, result} = await this.toMarkdown(url)
        if (!success || typeof result !== 'string') {
            throw new Error('Failed to retrieve Markdown for chunking')
        }
        const chunks = this.toChunks(result)
        const embeddings = await this.toEmbeddings(chunks)
        const vectors = chunks.map((chunk, index) => {
            const id = crypto.randomUUID()
            db.insert(chunksTable).values({
                id,
                source: url,
                text: chunk
            }).run()

            return {
                id,
                values: embeddings[index],
                metadata: {source: url}
            }
        })
        await this.env.VECTORIZE.upsert(vectors)
    }

    get llmModel() {
        const proxy = createOpenAICompatible({
            name: 'proxy',
            baseURL: 'https://cli-proxy.illuwa.click/v1',
            apiKey: this.env.API_SERVER_KEY
        })

        return proxy('gemini-3.8-flash-high')
    }

    async onChatMessage() {
        const result = streamText({
            model: this.llmModel,
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
