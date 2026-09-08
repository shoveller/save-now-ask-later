import { sqliteTable, text } from 'drizzle-orm/sqlite-core'
import {createSelectSchema, createUpdateSchema} from "drizzle-orm/zod";

export const chunksTable = sqliteTable('chunks', {
  id: text('id').primaryKey(),
  source: text('source').notNull(),
  text: text('text').notNull(),
})

export const chunkSelectSchema = createSelectSchema(chunksTable)
export const chunkUpdateSchema = createUpdateSchema(chunksTable)