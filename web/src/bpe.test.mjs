// Проверка JS-токенизатора против канонических ID GPT-2.
// Запуск: npm test (из web/)
import { readFileSync } from 'node:fs';
import { GPT2TokenizerJS } from './bpe.js';

const dir = new URL('../public/', import.meta.url);
const encoder = JSON.parse(readFileSync(new URL('encoder.json', dir), 'utf8'));
const merges = readFileSync(new URL('vocab.bpe', dir), 'utf8')
    .split('\n').map(l => l.trim())
    .filter(l => l && !l.startsWith('#'))
    .map(l => l.split(' '));
const tok = new GPT2TokenizerJS(encoder, merges);

const cases = [
    ['Hello, world!', [15496, 11, 995, 0]],
    ['The capital of France is', [464, 3139, 286, 4881, 318]],
    ['Привет, мир! 🚀 test123  spaces', null],   // только round-trip
];

for (const [text, expected] of cases) {
    const ids = tok.encode(text);
    if (expected && ids.toString() !== expected.toString()) {
        console.error(`FAIL encode(${JSON.stringify(text)}): got ${ids}, want ${expected}`);
        process.exit(1);
    }
    if (tok.decode(ids) !== text) {
        console.error(`FAIL roundtrip(${JSON.stringify(text)}): got ${JSON.stringify(tok.decode(ids))}`);
        process.exit(1);
    }
}
console.log('bpe.js: ok (2 exact + 1 roundtrip)');
