// Run with node --test image-edit-studio/test_generation.js (no browser/GPU needed).
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(__dirname + '/static/app.js', 'utf8')
    .split('// ========== GENERATE ==========')[1]
    .split('// ========== OUTPUT DISPLAY ==========')[0];

function harness(fetch) {
    const elements = {};
    const values = { prompt: 'Edit this', neg: '', cfg: '1', steps: '4', batch: '1',
        w: '512', h: '512', seed: '-1', maskBlur: '0', resizeMode: 'auto' };
    const messages = [], results = [];
    const ctx = vm.createContext({
        fetch, API: '', generating: false, S: [{ du: 'image' }], maskDU: null,
        $: id => elements[id] ||= { value: values[id] ?? '', style: {}, textContent: '' },
        toast: message => messages.push(message), showR: result => results.push(result),
        rH() {}, hideProgress() {}, setProgress() {},
        AbortController, AbortSignal, TextDecoder,
        setInterval: fn => { ctx.poll = fn; return 1; }, clearInterval() {},
    });
    vm.runInContext(source, ctx);
    return { ctx, messages, results };
}

function response(event, id = 'job') {
    return new Response('data: ' + JSON.stringify(event) + '\n\n', {
        headers: { 'Content-Type': 'text/event-stream', 'X-Job-ID': id },
    });
}

test('Stop during submission cancels server job once its ID arrives', async () => {
    let submit, signal;
    const calls = [];
    const h = harness(async (url, options) => {
        calls.push(url);
        if (url === '/api/generate') {
            signal = options.signal;
            return new Promise(resolve => { submit = resolve; });
        }
        assert.equal(url, '/api/cancel/job');
        assert.equal(signal.aborted, false);
        return Response.json({ ok: true, state: 'cancelling' });
    });
    const running = h.ctx.gen();
    await h.ctx.stopGen();
    assert.equal(signal.aborted, false);
    submit(response({ type: 'cancelled' }));
    await running;
    assert.deepEqual(calls, ['/api/generate', '/api/cancel/job']);
    assert.deepEqual(h.messages, ['Generation cancelled']);
    assert.equal(h.ctx.generating, false);
});

test('polling recovers a completed result when SSE is buffered', async () => {
    const h = harness(async (url, options) => {
        if (url === '/api/generate') {
            return { ok: true, headers: new Headers({ 'X-Job-ID': 'job' }), body: {
                getReader: () => ({ read: () => new Promise((resolve, reject) => {
                    options.signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
                }) }),
            } };
        }
        return Response.json({ type: 'done', results: ['output'], elapsed: 2 });
    });
    const running = h.ctx.gen();
    await new Promise(resolve => setImmediate(resolve));
    await h.ctx.poll();
    await running;
    assert.deepEqual(h.results, [['output']]);
    assert.deepEqual(h.messages, ['Done! in 2s']);
    assert.equal(h.ctx.generating, false);
});
