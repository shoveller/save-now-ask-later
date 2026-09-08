import { defineConfig } from 'vitest/config'
import ts from 'typescript'

export default defineConfig({
  plugins: [{
    name: 'test-agent-decorators',
    enforce: 'pre',
    transform(code, id) {
      if (!id.endsWith('/worker/RagAgent.ts')) return
      return ts.transpileModule(code, {
        compilerOptions: {
          target: ts.ScriptTarget.ES2022,
          module: ts.ModuleKind.ESNext,
        },
      }).outputText
    },
  }],
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
})
