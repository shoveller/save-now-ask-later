# Save Now, Ask Later

## 현재 임베딩 구성

- 모델: `multilingual-e5-small-fp32` (384차원)
- 경로: 앱 → `https://models.illuwa.click/v1` → Cloudflare Mesh → Infinity
- 문서에는 `passage: `, 검색 질문에는 `query: ` 접두어 적용
- Vectorize: `save-now-ask-later-e5-fp32` (cosine)
- 기존 `hello-vectorize` 768차원 인덱스, `save-now-ask-later-e5-384` INT8 인덱스와 SQLite 원문은 보존됩니다. 앱의 기존 문서는 **다시 저장/재임베딩해야 FP32 인덱스에서 검색**됩니다. Hindsight 두 곳은 원래 FP32를 사용했으므로 이 앱과 달리 기존 벡터를 그대로 재사용합니다.

E5의 입력 잘림을 피하기 위해 청크는 문자별 NFKC 정규화 후 UTF-8 크기를 합산해 최대 480 bytes, 겹침은 최대 96 bytes로 제한합니다. 원문 자체는 보존합니다. 이는 정확한 토큰 수 계산이 아닌 보수적인 입력 예산이며, 대형 토크나이저를 Worker 메모리에 올리지 않기 위한 선택입니다. 문서 임베딩은 최대 64개씩 순차 요청하고, Vectorize 쓰기는 최대 100개씩 나눕니다.

`/bench`는 동일한 짧은 본문을 Cloudflare Gemma와 E5로 비교합니다. E5에는 모델 권장 접두어를 붙이고 BFF 캐시를 우회합니다. `/`의 저장·검색은 BFF 캐시를 정상 사용합니다.

2026-09-13 초기 INT8 전환 검증 기록: 배포된 채팅 API에서 한국 위키 문서를 337개 청크로 저장(임베딩 21,077ms)하고, 인덱싱 완료 후 5개 근거를 검색해 출처와 함께 답변하는 것을 확인했습니다. 현재 FP32 전환과는 별도 측정입니다. Vectorize는 비동기 인덱싱이므로 저장 직후에는 검색 결과가 아직 없을 수 있습니다. 검증은 별도 `migration-check-*` 에이전트에서 수행했습니다.

```sh
pnpm run build
pnpm exec vitest run
pnpm run lint
pnpm run deploy
```

## 원래 Vite 템플릿 안내

### React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])

```

You can also install [eslint-plugin-react-x](https://npmx.dev/package/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://npmx.dev/package/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])

```
