import {routeAgentRequest} from "agents";

export { RagAgent } from './RagAgent.ts'

export default {
  async fetch(request, env) {
	return await routeAgentRequest(request, env) || new Response(null, { status: 404 });
  },
} satisfies ExportedHandler<Env>;
