export const e5ModelId = 'multilingual-e5-small-fp32'
export const e5Dimensions = 384
// Conservative UTF-8 budget after compatibility normalization, with room for E5 prefixes/special tokens.
// This avoids loading the large multilingual tokenizer into a memory-limited Worker.
export const e5TextByteLimit = 480
export const e5BatchSize = 64
const encoder = new TextEncoder()

export function embeddingCharacterBytes(character: string) {
  return encoder.encode(character.normalize('NFKC')).length
}

export function embeddingTextBytes(text: string) {
  return Array.from(text).reduce((sum, character) => sum + embeddingCharacterBytes(character), 0)
}

export function embeddingTextPreview(text: string) {
  let bytes = 0
  let result = ''
  for (const character of text) {
    const size = embeddingCharacterBytes(character)
    if (bytes + size > e5TextByteLimit) break
    bytes += size
    result += character
  }
  return result
}
