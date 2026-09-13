import { useEffect, useRef, useState } from 'react';
import {
    AppBar, Toolbar, Typography, Container, Paper, TextField, Button,
    Slider, Box, Chip, CircularProgress, Alert, Tooltip,
} from '@mui/material';
import { GPT2TokenizerJS } from './bpe.js';
import * as ort from 'onnxruntime-web';

// wasm-рантаймORT берём с CDN той же версии, что и npm-пакет
const ORT_VERSION = '1.22.0';
const EOT = 50256;

export default function App() {
    const [status, setStatus] = useState('initializing...');
    const [ready, setReady] = useState(false);
    const [backend, setBackend] = useState('');
    const [modelInfo, setModelInfo] = useState('');
    const [output, setOutput] = useState('');
    const [stats, setStats] = useState('');
    const [dl, setDl] = useState(null);    // {got,total} байт при скачивании модели
    const [error, setError] = useState('');
    const [prompt, setPrompt] = useState('The meaning of life is');
    const [isMobile] = useState(() =>
        matchMedia('(pointer: coarse)').matches || innerWidth < 640);
    const [maxTokens, setMaxTokens] = useState(isMobile ? 64 : 150);
    const [temperature, setTemperature] = useState(0.95);
    const [topK, setTopK] = useState(5);

    const sessionRef = useRef(null);
    const tokenizerRef = useRef(null);
    const cfgRef = useRef(null);
    const stateRef = useRef(null);      // ort.Tensor KV-кэша
    const pastLenRef = useRef(0);
    const stopRef = useRef(false);
    const backendRef = useRef('');          // короткое имя: 'WebGPU' | 'WASM'
    const modelBufferRef = useRef(null);   // скачанные байты модели (кэш для WASM-фолбэка)

    useEffect(() => { init(); }, []);

    function zeroState() {
        const dims = [cfgRef.current.n_layer, 2, 1, cfgRef.current.n_head, 0, cfgRef.current.head_dim];
        const data = cfgRef.current.dtype === 'fp16' ? new Uint16Array(0) : new Float32Array(0);
        return new ort.Tensor(cfgRef.current.dtype === 'fp16' ? 'float16' : 'float32', data, dims);
    }

    function int64Tensor(values) {
        return new ort.Tensor('int64', BigInt64Array.from(values.map(BigInt)), [1, values.length]);
    }

    function scalarTensor(n) {
        return new ort.Tensor('int64', BigInt64Array.from([BigInt(n)]), []);
    }

    async function sha256hex(bytes) {
        const d = await crypto.subtle.digest('SHA-256', bytes);
        return [...new Uint8Array(d)].map(b => b.toString(16).padStart(2, '0')).join('');
    }

    // Model source: chunk URLs from config.model_chunks (raw.githubusercontent
    // serves CORS '*'; pin a tag/commit in the URL to bind model to frontend
    // revision) -> local gpt2_124m.onnx (npm run dev).
    async function resolveModelSource() {
        const cfg = cfgRef.current;
        if (cfg.model_chunks && cfg.model_chunks.length)
            return { urls: cfg.model_chunks, label: 'chunks' };
        return { urls: ['gpt2_124m.onnx'], label: 'local file' };
    }

    async function loadModelBuffer() {
        if (modelBufferRef.current) return modelBufferRef.current;
        const source = await resolveModelSource();
        const total = cfgRef.current.model_size || 0;
        const mb = b => (b / 2 ** 20).toFixed(1);
        const parts = [];
        let got = 0, nextMark = 0;
        for (let i = 0; i < source.urls.length; i++) {
            const resp = await fetch(source.urls[i]);
            if (!resp.ok) throw new Error(`part ${i + 1}: HTTP ${resp.status}`);
            const reader = resp.body.getReader();
            const pieces = [];
            let partLen = 0;
            for (;;) {
                const { done: d, value } = await reader.read();
                if (d) break;
                pieces.push(value);
                got += value.length;
                partLen += value.length;
                if (got >= nextMark) {           // троттлинг ре-рендеров: раз в 2 МБ
                    nextMark = got + 2 * 2 ** 20;
                    setDl({ got, total });
                    setStatus(`loading model ${source.label}: ` +
                        (total ? `${mb(got)} / ${mb(total)} MB (${Math.round(got / total * 100)}%)` : `${mb(got)} MB`) +
                        ` · part ${i + 1}/${source.urls.length}`);
                }
            }
            const merged = new Uint8Array(partLen);
            let off = 0;
            for (const p of pieces) { merged.set(p, off); off += p.length; }
            parts.push(merged);
        }
        setDl({ got, total });
        const buf = new Uint8Array(got);
        let off = 0;
        for (const p of parts) { buf.set(p, off); off += p.length; }
        if (cfgRef.current.model_sha256 && await sha256hex(buf) !== cfgRef.current.model_sha256)
            throw new Error('model sha256 mismatch');
        modelBufferRef.current = buf;
        return modelBufferRef.current;
    }

    async function init() {
        try {
            const cfgResp = await fetch('config.json');
            if (!cfgResp.ok) throw new Error(`config.json: HTTP ${cfgResp.status}`);
            cfgRef.current = await cfgResp.json();
            tokenizerRef.current = await GPT2TokenizerJS.load('.');
            setStatus(`loading model (${cfgRef.current.dtype}, step ${cfgRef.current.ckpt_step})...`);
            ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
            const buf = await loadModelBuffer();
            let session, backendName;
            try {
                session = await ort.InferenceSession.create(buf.buffer,
                    { executionProviders: ['webgpu'], graphOptimizationLevel: 'all' });
                setBackend('WebGPU');
                backendName = 'WebGPU';
            } catch (e) {
                console.warn('WebGPU unavailable, falling back to WASM:', e);
                session = await ort.InferenceSession.create(buf.buffer,
                    { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
                setBackend('WASM (slower than WebGPU)');
                backendName = 'WASM';
            }
            sessionRef.current = session;
            backendRef.current = backendName;
            // самопроверка бэкенда: префилл одной строки, argmax должен попасть
            // в эталонную top-5 (посчитано torch-ом на best.pt, fp32)
            const golden = new Set([262, 257, 635, 5140, 287]);
            const probe = [464, 3139, 286, 4881, 318];   // "The capital of France is"
            stateRef.current = zeroState();
            pastLenRef.current = 0;
            const probeLogits = await step(probe);
            let best = -1, bestIdx = -1;
            for (let i = 0; i < probeLogits.length; i++) {
                if (probeLogits[i] > best) { best = probeLogits[i]; bestIdx = i; }
            }
            if (!golden.has(bestIdx)) {
                throw new Error(`self-test failed: ${backendName} predicts token ${bestIdx}, ` +
                    `expected one of [${[...golden].join(', ')}] — the graph/backend is broken`);
            }
            setModelInfo(`step ${cfgRef.current.ckpt_step}, ${cfgRef.current.dtype} · self-test ok`);
            setStatus('ready');
            setReady(true);
        } catch (e) {
            console.error(e);
            setStatus('');
            let msg = e.message;
            if (/SIMD|JIT|initWasm/i.test(msg)) {
                msg += ' — WebAssembly seems to be disabled in this browser ' +
                    '(JIT off?). The model needs WebGPU or WASM; please allow ' +
                    'wasm/JIT for this site or try another browser.';
            }
            setError(`failed to load model: ${msg}`);
        }
    }

    async function step(ids) {
        const feeds = {
            ids: int64Tensor(ids),
            state: stateRef.current,
            past_len: scalarTensor(pastLenRef.current),
        };
        const out = await sessionRef.current.run(feeds);
        stateRef.current = out.new_state;
        pastLenRef.current += ids.length;
        return out.logits.data;
    }

    function sample(logits, temperature, topK) {
        const idx = Array.from({ length: logits.length }, (_, i) => i);
        idx.sort((a, b) => logits[b] - logits[a]);
        let max = -Infinity;
        const top = idx.slice(0, Math.min(topK, logits.length)).map(i => {
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
        if (!sessionRef.current) return;
        const text = prompt.trim();
        if (!text) return;
        setError('');
        setOutput('');
        setStats('');
        stopRef.current = false;
        stateRef.current = zeroState();
        pastLenRef.current = 0;
        const t0 = performance.now();
        try {
            const tok = tokenizerRef.current;
            const ids = tok.encode(text);
            let logits = await step(ids);                    // prefill
            const generated = [...ids];
            for (let i = 0; i < maxTokens; i++) {
                if (stopRef.current || pastLenRef.current >= cfgRef.current.n_positions) break;
                const next = sample(logits, temperature, topK);
                if (next === EOT) break;
                generated.push(next);
                setOutput(tok.decode(generated));
                const tps = pastLenRef.current / ((performance.now() - t0) / 1000);
                setStats(`${i + 1} tokens · ${tps.toFixed(0)} tok/s · ${backendRef.current}`);
                await new Promise(r => setTimeout(r, 0));    // кадр на отрисовку
                logits = await step([next]);
            }
            setStats(s => `${s} · done in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
        } catch (e) {
            console.error(e);
            setError(`generation error: ${e.message}`);
        }
    }

    return (
        <>
            <AppBar position="static">
                <Toolbar>
                    <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                        GPT-2 124M reproduction
                    </Typography>
                    {backend && <Chip label={backend} color="primary" size="small" sx={{ mr: 1 }} />}
                    {modelInfo && <Chip label={modelInfo} variant="outlined" size="small" />}
                </Toolbar>
            </AppBar>
            <Container maxWidth="md" sx={{ py: 3 }}>
                {!ready && error === '' && (
                    <Paper sx={{ p: 3, mb: 2, display: 'flex', alignItems: 'center', gap: 3 }}>
                        <Box sx={{ position: 'relative', display: 'inline-flex', flexShrink: 0 }}>
                            <CircularProgress
                                variant={dl && dl.total ? 'determinate' : 'indeterminate'}
                                value={dl && dl.total ? Math.min(100, dl.got / dl.total * 100) : undefined}
                                size={72} thickness={4} />
                            <Box sx={{ position: 'absolute', inset: 0, display: 'flex',
                                       alignItems: 'center', justifyContent: 'center' }}>
                                <Typography variant="caption" sx={{ fontWeight: 600 }}>
                                    {dl && dl.total ? `${Math.min(100, Math.round(dl.got / dl.total * 100))}%` : ''}
                                </Typography>
                            </Box>
                        </Box>
                        <Box sx={{ minWidth: 0 }}>
                            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>Loading model</Typography>
                            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                                {status}
                            </Typography>
                        </Box>
                    </Paper>
                )}
                {error !== '' && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

                <Paper sx={{ p: 2 }}>
                    <TextField
                        label="Prompt"
                        multiline rows={3} fullWidth
                        value={prompt}
                        onChange={e => setPrompt(e.target.value)}
                        disabled={!ready}
                    />
                    <Box sx={{ display: 'flex', gap: 3, mt: 2, flexWrap: 'wrap' }}>
                        {[
                            { label: 'new tokens', value: maxTokens, set: setMaxTokens, min: 16, max: 512, step: 16, fmt: v => String(v) },
                            { label: 'temperature', value: temperature, set: setTemperature, min: 0.1, max: 1, step: 0.05, fmt: v => v.toFixed(2) },
                            { label: 'top-k', value: topK, set: setTopK, min: 1, max: 20, step: 1, fmt: v => String(v) },
                        ].map(c => (
                            <Box key={c.label} sx={{ flex: '1 1 180px', minWidth: 160 }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                                    <Typography variant="body2" color="text.secondary">{c.label}</Typography>
                                    <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 600 }}>
                                        {c.fmt(c.value)}
                                    </Typography>
                                </Box>
                                <Slider value={c.value} min={c.min} max={c.max} step={c.step}
                                        valueLabelDisplay="auto" aria-label={c.label}
                                        onChange={(_, v) => c.set(v)} disabled={!ready} />
                            </Box>
                        ))}
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, ml: 'auto', flexWrap: 'wrap' }}>
                            <Tooltip title="sampling stops on the <|endoftext|> token">
                                <Button variant="contained" onClick={generate}
                                        disabled={!ready} sx={{ minWidth: 140 }}>
                                    Generate
                                </Button>
                            </Tooltip>
                            <Button variant="outlined" color="error"
                                    onClick={() => { stopRef.current = true; }}>
                                Stop
                            </Button>
                        </Box>
                    </Box>
                </Paper>

                {output !== '' && (
                    <Paper sx={{ p: 2, mt: 2 }}>
                        <Typography sx={{ whiteSpace: 'pre-wrap' }}>{output}</Typography>
                    </Paper>
                )}
                {stats !== '' && (
                    <Typography color="text.secondary" sx={{ mt: 1 }}>{stats}</Typography>
                )}
            </Container>
        </>
    );
}
