import {and, eq} from 'drizzle-orm'
import type {DrizzleSqliteDODatabase} from 'drizzle-orm/durable-sqlite'
import {migrate} from 'drizzle-orm/durable-sqlite/migrator'
import {integer, sqliteTable, text} from 'drizzle-orm/sqlite-core'
import migrations from '../drizzle/migrations.js'

// Read-only SQLite catalog and the installed Drizzle rc.4 migration journal.
// Kept outside schema.ts so drizzle-kit does not generate these internal tables.
const catalog = sqliteTable('sqlite_master', {name: text(), type: text()})
const journal = sqliteTable('__drizzle_migrations', {
    id: integer().primaryKey(),
    hash: text().notNull(),
    createdAt: integer('created_at'),
    name: text(),
    appliedAt: text('applied_at'),
})
const sourcesMigration = '20260908163212_handy_blonde_phantom'

export function migrateStorage(db: DrizzleSqliteDODatabase) {
    migrate(db, {migrations: Object.fromEntries(
        Object.entries(migrations.migrations).filter(([name]) => name < sourcesMigration),
    )})

    // Older app versions created sources outside Drizzle. Adopt it without
    // recreating the table or losing its recorded titles and save times.
    if (db.select().from(catalog).where(and(eq(catalog.type, 'table'), eq(catalog.name, 'sources'))).get()
        && !db.select().from(journal).where(eq(journal.name, sourcesMigration)).get()) {
        db.insert(journal).values({
            hash: '', name: sourcesMigration,
            createdAt: Date.UTC(2026, 8, 8, 16, 32, 12),
            appliedAt: new Date().toISOString(),
        }).run()
    }
    migrate(db, migrations)
}
