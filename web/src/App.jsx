import React, { useEffect, useRef, useState } from 'react';
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
    const [status, setStatus] = useState('инициализация...');
    const [ready, setReady] = useState(false);
    const [backend, setBackend] = useState('');
    const [modelInfo, setModelInfo] = useState('');
    const [output, setOutput] = useState('');
    const [stats, setStats] = useState('');
    const [error, setError] = useState('');
    const [prompt, setPrompt] = useState('The meaning of life is');
    const [maxTokens, setMaxTokens] = useState(150);
    const [temperature, setTemperature] = useState(0.8);
    const [topK, setTopK] = useState(50);

    const sessionRef = useRef(null);
    const tokenizerRef = useRef(null);
    const cfgRef = useRef(null);
    const stateRef = useRef(null);      // ort.Tensor KV-кэша
    const pastLenRef = useRef(0);
    const stopRef = useRef(false);

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

    // автоопределение USER/REPO на проектных страницах *.github.io
    function detectRepo() {
        const m = location.hostname.match(/^([\w.-]+)\.github\.io$/);
        if (!m) return cfgRef.current.repo || '';
        const seg = location.pathname.split('/').filter(Boolean)[0];
        if (seg && seg !== `${m[1]}.github.io`) return `${m[1]}/${seg}`;
        return `${m[1]}/${m[1]}.github.io`;
    }

    // Порядок источника модели: GitHub Release (latest или тег — тег пинает
    // модель к коммиту фронтенда) -> локальный gpt2_124m.onnx.
    async function resolveModelSource() {
        const cfg = cfgRef.current;
        if (!cfg.release) return { urls: ['gpt2_124m.onnx'], digests: [null], label: 'локальный файл' };
        const repo = cfg.repo || detectRepo();
        if (!repo) throw new Error('укажите repo в config.json (вне *.github.io автоопределение невозможно)');
        const base = `https://api.github.com/repos/${repo}/releases/`;
        const url = cfg.release === 'latest' ? base + 'latest' : base + 'tags/' + cfg.release;
        const rel = await (await fetch(url)).json();
        if (!rel.assets) throw new Error(`GitHub API: ${rel.message || 'релиз не найден'}`);
        const assets = rel.assets
            .filter(a => /gpt2_124m\.part-\d+$/.test(a.name))
            .sort((a, b) => (a.name < b.name ? -1 : 1));
        if (!assets.length) throw new Error(`в релизе ${rel.tag_name} нет частей gpt2_124m.part-*`);
        return {
            urls: assets.map(a => a.browser_download_url),
            digests: assets.map(a => a.digest || null),
            label: `${repo}@${rel.tag_name}`,
        };
    }

    async function loadModelBuffer() {
        const source = await resolveModelSource();
        const parts = [];
        for (let i = 0; i < source.urls.length; i++) {
            setStatus(`модель ${source.label}: часть ${i + 1}/${source.urls.length}...`);
            const resp = await fetch(source.urls[i]);
            if (!resp.ok) throw new Error(`часть ${i + 1}: HTTP ${resp.status}`);
            const bytes = new Uint8Array(await resp.arrayBuffer());
            const want = source.digests && source.digests[i];
            if (want && want.startsWith('sha256:')) {
                const got = await sha256hex(bytes);
                if (got !== want.slice(7)) throw new Error(`часть ${i + 1}: sha256 не совпал`);
            }
            parts.push(bytes);
            const mb = parts.reduce((s, p) => s + p.length, 0) / 2 ** 20;
            setStatus(`модель ${source.label}: ${mb.toFixed(0)} MiB загружено...`);
        }
        const total = parts.reduce((s, p) => s + p.length, 0);
        const buf = new Uint8Array(total);
        let off = 0;
        for (const p of parts) { buf.set(p, off); off += p.length; }
        return buf;
    }

    async function init() {
        try {
            cfgRef.current = await (await fetch('config.json')).json();
            tokenizerRef.current = await GPT2TokenizerJS.load('');
            setStatus(`модель (${cfgRef.current.dtype}, шаг ${cfgRef.current.ckpt_step}) загружается...`);
            ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
            const buf = await loadModelBuffer();
            let session;
            try {
                session = await ort.InferenceSession.create(buf.buffer,
                    { executionProviders: ['webgpu'], graphOptimizationLevel: 'all' });
                setBackend('WebGPU');
            } catch (e) {
                console.warn('WebGPU недоступен, фолбэк на WASM:', e);
                session = await ort.InferenceSession.create(buf.buffer,
                    { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
                setBackend('WASM (медленнее)');
            }
            sessionRef.current = session;
            setModelInfo(`шаг ${cfgRef.current.ckpt_step}, ${cfgRef.current.dtype}`);
            setStatus('готово');
            setReady(true);
        } catch (e) {
            console.error(e);
            setStatus('');
            setError(`не удалось загрузить модель: ${e.message}`);
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
            let logits = await step(ids);                    // префилл
            const generated = [...ids];
            for (let i = 0; i < maxTokens; i++) {
                if (stopRef.current || pastLenRef.current >= cfgRef.current.n_positions) break;
                const next = sample(logits, temperature, topK);
                if (next === EOT) break;
                generated.push(next);
                setOutput(tok.decode(generated));
                const tps = pastLenRef.current / ((performance.now() - t0) / 1000);
                setStats(`${i + 1} токенов · ${tps.toFixed(0)} ток/с`);
                await new Promise(r => setTimeout(r, 0));    // кадр на отрисовку
                logits = await step([next]);
            }
            setStats(s => `${s} · готово за ${((performance.now() - t0) / 1000).toFixed(1)} с`);
        } catch (e) {
            console.error(e);
            setError(`ошибка генерации: ${e.message}`);
        }
    }

    return (
        <>
            <AppBar position="static">
                <Toolbar>
                    <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                        GPT-2 124M — своя репродукция
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
                        label="Промпт"
                        multiline rows={3} fullWidth
                        value={prompt}
                        onChange={e => setPrompt(e.target.value)}
                        disabled={!ready}
                    />
                    <Box sx={{ display: 'flex', gap: 3, mt: 2, flexWrap: 'wrap' }}>
                        <Box sx={{ minWidth: 150 }}>
                            <Typography gutterBottom color="text.secondary">новых токенов</Typography>
                            <Slider value={maxTokens} min={16} max={512} step={16}
                                    valueLabelDisplay="auto"
                                    onChange={(_, v) => setMaxTokens(v)} disabled={!ready} />
                        </Box>
                        <Box sx={{ minWidth: 150 }}>
                            <Typography gutterBottom color="text.secondary">temperature</Typography>
                            <Slider value={temperature} min={0.1} max={2} step={0.05}
                                    valueLabelDisplay="auto"
                                    onChange={(_, v) => setTemperature(v)} disabled={!ready} />
                        </Box>
                        <Box sx={{ minWidth: 150 }}>
                            <Typography gutterBottom color="text.secondary">top-k</Typography>
                            <Slider value={topK} min={1} max={200} step={1}
                                    valueLabelDisplay="auto"
                                    onChange={(_, v) => setTopK(v)} disabled={!ready} />
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, ml: 'auto' }}>
                            <Tooltip title="sampling останавливается по <|endoftext|>">
                                <Button variant="contained" onClick={generate}
                                        disabled={!ready} sx={{ minWidth: 140 }}>
                                    Генерировать
                                </Button>
                            </Tooltip>
                            <Button variant="outlined" color="error"
                                    onClick={() => { stopRef.current = true; }}>
                                Стоп
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
