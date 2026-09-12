// Инференс GPT-2 в браузере через onnxruntime-web (WebGPU, фолбэк WASM).
// Модель: web/gpt2_124m.onnx (см. scripts/export_onnx.py), токенизатор: bpe.js.

import { GPT2TokenizerJS } from "./bpe.js";

const $ = id => document.getElementById(id);
const EOT = 50256;
let session = null;
let tok = null;
let cfg = null;
let state = null;      // ort.Tensor KV-кэша
let pastLen = 0;
let generating = false;

function zeroState(batch = 1) {
    const dims = [cfg.n_layer, 2, batch, cfg.n_head, 0, cfg.head_dim];
    const data = cfg.dtype === "fp16" ? new Uint16Array(0) : new Float32Array(0);
    return new ort.Tensor(cfg.dtype === "fp16" ? "float16" : "float32", data, dims);
}

function int64Tensor(values) {
    return new ort.Tensor("int64", BigInt64Array.from(values.map(BigInt)), [1, values.length]);
}

function scalarTensor(n) {
    return new ort.Tensor("int64", BigInt64Array.from([BigInt(n)]), []);
}

async function step(ids) {
    const feeds = { ids: int64Tensor(ids), state, past_len: scalarTensor(pastLen) };
    const out = await session.run(feeds);
    state = out.new_state;
    pastLen += ids.length;
    return out.logits.data;   // Float32Array (vocab,)
}

function sample(logits, temperature, topK) {
    // верх-K индексы без сортировки всего массива
    const idx = Array.from({ length: logits.length }, (_, i) => i);
    const k = Math.min(topK, logits.length);
    idx.sort((a, b) => logits[b] - logits[a]);
    let max = -Infinity;
    const top = idx.slice(0, k).map(i => {
        const v = logits[i] / temperature;
        if (v > max) max = v;
        return [v, i];
    });
    let sum = 0;
    const probs = top.map(([v, i]) => { const p = Math.exp(v - max); sum += p; return [p, i]; });
    let r = Math.random() * sum;
    for (const [p, i] of probs) { r -= p; if (r <= 0) return i; }
    return probs[0][1];
}

async function generate() {
    if (generating || !session) return;
    const promptText = $("prompt").value.trim();
    if (!promptText) return;
    const maxTokens = Number($("tokens").value) || 100;
    const temperature = Number($("temperature").value) || 0.8;
    const topK = Number($("topk").value) || 50;

    generating = true;
    $("generate").disabled = true;
    $("stop").disabled = false;
    const outEl = $("output");
    outEl.textContent = promptText;
    state = zeroState();
    pastLen = 0;
    stopFlag = false;

    const t0 = performance.now();
    try {
        const ids = tok.encode(promptText);
        let logits = await step(ids);                    // префилл
        const generated = [...ids];
        for (let i = 0; i < maxTokens; i++) {
            if (stopFlag || pastLen >= cfg.n_positions) break;
            const next = sample(logits, temperature, topK);
            if (next === EOT) break;
            generated.push(next);
            outEl.textContent = tok.decode(generated);
            $("stats").textContent =
                `${i + 1} токенов, ${(pastLen / ((performance.now() - t0) / 1000)).toFixed(0)} ток/с`;
            await new Promise(r => setTimeout(r, 0));    // дать браузеру отрисовать
            logits = await step([next]);
        }
    } catch (e) {
        outEl.textContent += `\n\n[ошибка: ${e.message}]`;
    }
    const dt = (performance.now() - t0) / 1000;
    $("stats").textContent = `готово за ${dt.toFixed(1)} с (${pastLen} токенов)`;
    generating = false;
    $("generate").disabled = false;
    $("stop").disabled = true;
}

let stopFlag = false;

async function init() {
    cfg = await (await fetch("config.json")).json();
    tok = await GPT2TokenizerJS.load(".");
    $("status").textContent = `модель загружается (${cfg.dtype}, шаг ${cfg.ckpt_step})...`;
    const options = { executionProviders: ["webgpu"], graphOptimizationLevel: "all" };
    try {
        session = await ort.InferenceSession.create("gpt2_124m.onnx", options);
        $("status").textContent = "бэкенд: WebGPU";
    } catch (e) {
        console.warn("WebGPU недоступен, фолбэк на WASM:", e);
        session = await ort.InferenceSession.create("gpt2_124m.onnx",
            { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
        $("status").textContent = "бэкенд: WASM (медленнее; включите WebGPU в браузере)";
    }
    $("generate").disabled = false;
}

$("generate").addEventListener("click", generate);
$("stop").addEventListener("click", () => { stopFlag = true; });
init();
