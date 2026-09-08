import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

import { cloudflare } from "@cloudflare/vite-plugin";
import agents from "agents/vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [agents(), react(), cloudflare()],
})