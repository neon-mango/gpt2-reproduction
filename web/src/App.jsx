import { useEffect, useRef, useState } from 'react';
import {
    AppBar, Toolbar, Typography, Container, Paper, TextField, Button,
    Slider, Box, Chip, LinearProgress, Alert, Tooltip,
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
    const [error, setError] = useState('');
    const [prompt, setPrompt] = useState('The meaning of life is');
    const [isMobile] = useState(() =>
        matchMedia('(pointer: coarse)').matches || innerWidth < 640);
    const [maxTokens, setMaxTokens] = useState(isMobile ? 64 : 150);
    const [temperature, setTemperature] = useState(0.8);
    const [topK, setTopK] = useState(50);

    const sessionRef = useRef(null);
    const tokenizerRef = useRef(null);
    const cfgRef = useRef(null);
    const stateRef = useRef(null);      // ort.Tensor KV-кэша
    const pastLenRef = useRef(0);
    const stopRef = useRef(false);
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
        const parts = [];
        for (let i = 0; i < source.urls.length; i++) {
            setStatus(`model ${source.label}: part ${i + 1}/${source.urls.length}...`);
            const resp = await fetch(source.urls[i]);
            if (!resp.ok) throw new Error(`part ${i + 1}: HTTP ${resp.status}`);
            parts.push(new Uint8Array(await resp.arrayBuffer()));
            const mb = parts.reduce((s, p) => s + p.length, 0) / 2 ** 20;
            setStatus(`model ${source.label}: ${mb.toFixed(0)} MiB loaded...`);
        }
        const total = parts.reduce((s, p) => s + p.length, 0);
        const buf = new Uint8Array(total);
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
            let session;
            try {
                session = await ort.InferenceSession.create(buf.buffer,
                    { executionProviders: ['webgpu'], graphOptimizationLevel: 'all' });
                setBackend('WebGPU');
            } catch (e) {
                console.warn('WebGPU unavailable, falling back to WASM:', e);
                session = await ort.InferenceSession.create(buf.buffer,
                    { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
                setBackend(isMobile ? 'WASM (phone CPU: ~1-3 tok/s)' : 'WASM (slower)');
            }
            sessionRef.current = session;
            setModelInfo(`step ${cfgRef.current.ckpt_step}, ${cfgRef.current.dtype}`);
            setStatus('ready');
            setReady(true);
        } catch (e) {
            console.error(e);
            setStatus('');
            setError(`failed to load model: ${e.message}`);
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
                setStats(`${i + 1} tokens · ${tps.toFixed(0)} tok/s`);
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
                    <Box>
                        <LinearProgress />
                        <Typography color="text.secondary" sx={{ mt: 1 }}>{status}</Typography>
                    </Box>
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
                        <Box sx={{ minWidth: 150 }}>
                            <Typography gutterBottom color="text.secondary">new tokens</Typography>
                            <Slider value={maxTokens} min={16} max={512} step={16}
                                    valueLabelDisplay="on"
                                    onChange={(_, v) => setMaxTokens(v)} disabled={!ready} />
                        </Box>
                        <Box sx={{ minWidth: 150 }}>
                            <Typography gutterBottom color="text.secondary">temperature</Typography>
                            <Slider value={temperature} min={0.1} max={2} step={0.05}
                                    valueLabelDisplay="on"
                                    valueLabelFormat={v => v.toFixed(2)}
                                    onChange={(_, v) => setTemperature(v)} disabled={!ready} />
                        </Box>
                        <Box sx={{ minWidth: 150 }}>
                            <Typography gutterBottom color="text.secondary">top-k</Typography>
                            <Slider value={topK} min={1} max={200} step={1}
                                    valueLabelDisplay="on"
                                    onChange={(_, v) => setTopK(v)} disabled={!ready} />
                        </Box>
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
