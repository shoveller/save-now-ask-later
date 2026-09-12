import {createOpenAICompatible} from "@ai-sdk/openai-compatible";
import { env } from "cloudflare:workers";

export const createAI = () => {
    return createOpenAICompatible({
        name: 'proxy',
        baseURL: 'https://models.illuwa.click/v1',
        apiKey: env.API_SERVER_KEY
    })
}