// Порт gpt2rep/tokenizer.py: byte-level BPE GPT-2 (официальные encoder.json + vocab.bpe).
// Совпадает с Python-реализацией: "Hello, world!" -> [15496, 11, 995, 0].

function bytesToUnicode() {
    const bs = [];
    for (let b = 33; b <= 126; b++) bs.push(b);
    for (let b = 161; b <= 172; b++) bs.push(b);
    for (let b = 174; b <= 255; b++) bs.push(b);
    const cs = bs.slice();
    let n = 0;
    for (let b = 0; b < 256; b++) {
        if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
    }
    const map = {};
    bs.forEach((b, i) => { map[b] = String.fromCodePoint(cs[i]); });
    return map;
}

export class GPT2TokenizerJS {
    constructor(encoder, merges) {
        this.encoder = encoder;                       // {токен: id}
        this.decoder = {};
        for (const [t, id] of Object.entries(encoder)) this.decoder[id] = t;
        this.byteEncoder = bytesToUnicode();
        this.byteDecoder = {};
        for (const [b, c] of Object.entries(this.byteEncoder)) this.byteDecoder[c] = Number(b);
        this.bpeRanks = new Map();
        merges.forEach((pair, i) => this.bpeRanks.set(pair.join(" "), i)); // "a b" -> ранг
        this.cache = new Map();
        // Предтокенизация GPT-2 (та же, что в Python-версии с модулем regex)
        this.pat = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
    }

    static async load(dir) {
        const [encoder, mergesText] = await Promise.all([
            fetch(`${dir}/encoder.json`).then(r => r.json()),
            fetch(`${dir}/vocab.bpe`).then(r => { if (!r.ok) throw new Error(`vocab.bpe: HTTP ${r.status}`); return r.text(); }),
        ]);
        const merges = mergesText.split("\n")
            .map(line => line.trim())
            .filter(line => line && !line.startsWith("#"))
            .map(line => line.split(" "));
        return new GPT2TokenizerJS(encoderJson, merges);
    }

    bpe(token) {
        if (this.cache.has(token)) return this.cache.get(token);
        let word = [...token];
        if (word.length >= 2) {
            while (true) {
                let bestRank = Infinity, bestIdx = -1;
                for (let i = 0; i < word.length - 1; i++) {
                    const r = this.bpeRanks.get(word[i] + " " + word[i + 1]);
                    if (r !== undefined && r < bestRank) { bestRank = r; bestIdx = i; }
                }
                if (bestIdx === -1) break;
                word = [
                    ...word.slice(0, bestIdx),
                    word[bestIdx] + word[bestIdx + 1],
                    ...word.slice(bestIdx + 2),
                ];
                if (word.length === 1) break;
            }
        }
        const result = word.join(" ");
        this.cache.set(token, result);
        return result;
    }

    encode(text) {
        const ids = [];
        for (const m of text.matchAll(this.pat)) {
            const bytes = new TextEncoder().encode(m[0]);
            const spaced = Array.from(bytes, b => this.byteEncoder[b]).join("");
            for (const t of this.bpe(spaced).split(" ")) {
                ids.push(this.encoder[t]);
            }
        }
        return ids;
    }

    decode(ids) {
        let s = "";
        for (const id of ids) s += this.decoder[id];
        const bytes = new Uint8Array([...s].map(c => this.byteDecoder[c]));
        return new TextDecoder("utf-8", { fatal: false }).decode(bytes);
    }
}
