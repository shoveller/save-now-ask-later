import {defineConfig} from "drizzle-kit";

export default defineConfig({
    schema: './worker/schema.ts',
    out: './drizzle',
    dialect: 'sqlite',
    driver: 'durable-sqlite'
})