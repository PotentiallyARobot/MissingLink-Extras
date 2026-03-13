/* ══════════════════════════════════════════════════════════════
   LoRA Browser — CivitAI search & pick for AI Image Edit Studio
   Standalone JS module. Depends on lora_browser.html being in DOM.

   PUBLIC API:
     loraBrowser.open()   — show the browser modal
     loraBrowser.close()  — hide it
     loraBrowser.onPick   — callback(repo) when user selects a LoRA
   ══════════════════════════════════════════════════════════════ */

const loraBrowser = (() => {
    const CIVITAI = 'https://civitai.com/api/v1';
    const BASE_MATCH = ['qwen', 'qwen-image', 'qwen image'];

    // ── State ──
    const st = {
        query: '',
        sort: 'Most Downloaded',
        period: '',
        nsfw: false,
        results: [],
        nextUrl: null,
        total: 0,
        hasMore: true,
        loading: false,
    };
    let abortCtrl = null;
    let debounce = null;
    const cache = new Map();
    let toastTimer = null;

    // ── DOM refs (lazily resolved after DOM is ready) ──
    let _dom = null;
    function dom() {
        if (!_dom) {
            _dom = {
                overlay:    document.getElementById('loraBrowserOverlay'),
                close:      document.getElementById('loraBrowserClose'),
                search:     document.getElementById('lbSearch'),
                sort:       document.getElementById('lbSort'),
                period:     document.getElementById('lbPeriod'),
                nsfw:       document.getElementById('lbNsfwTog'),
                info:       document.getElementById('lbResultInfo'),
                scroll:     document.getElementById('lbScroll'),
                grid:       document.getElementById('lbGrid'),
                error:      document.getElementById('lbError'),
                status:     document.getElementById('lbStatus'),
                sentinel:   document.getElementById('lbSentinel'),
                toast:      document.getElementById('lbToast'),
            };
        }
        return _dom;
    }

    // ── Helpers ──
    const esc = s => s ? s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;') : '';
    function fmtSz(b) { if (!b) return '?'; if (b > 1e9) return (b / 1e9).toFixed(1) + ' GB'; if (b > 1e6) return (b / 1e6).toFixed(0) + ' MB'; return (b / 1e3).toFixed(0) + ' KB'; }
    function fmtN(n) { if (!n) return '0'; return n > 1000 ? (n / 1000).toFixed(1) + 'k' : String(n); }

    function matchBM(model) {
        const vs = model.modelVersions || [];
        return vs.some(v => {
            const bm = (v.baseModel || '').toLowerCase();
            return BASE_MATCH.some(pat => bm.includes(pat));
        });
    }
    function getBM(model) {
        for (const v of model.modelVersions || []) {
            const bm = (v.baseModel || '').toLowerCase();
            if (BASE_MATCH.some(pat => bm.includes(pat))) return v.baseModel;
        }
        return model.modelVersions?.[0]?.baseModel || '?';
    }
    function verImgs(ver) {
        if (!ver || !ver.images) return [];
        return ver.images.filter(i => st.nsfw || (!i.nsfw && (!i.nsfwLevel || i.nsfwLevel <= 2))).slice(0, 6);
    }

    function showToast(msg, isErr) {
        const t = dom().toast;
        t.textContent = msg;
        t.className = 'lb-toast show' + (isErr ? ' err' : '');
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => t.className = 'lb-toast', 3000);
    }

    // ── Fetch from CivitAI ──
    async function doFetch(append) {
        if (abortCtrl) abortCtrl.abort();
        abortCtrl = new AbortController();
        st.loading = true;
        if (!append) renderGrid();
        renderStatus();

        try {
            let url;
            if (append && st.nextUrl) {
                url = st.nextUrl;
            } else {
                const q = st.query || 'qwen image edit';
                const p = [
                    'types=LORA',
                    'sort=' + encodeURIComponent(st.sort),
                    'limit=100',
                ];
                if (q) p.push('query=' + encodeURIComponent(q));
                if (st.period) p.push('period=' + encodeURIComponent(st.period));
                if (st.nsfw) p.push('nsfw=true');
                url = CIVITAI + '/models?' + p.join('&');
            }

            let filtered = [];
            let pages = 0;

            while (url && pages < 3) {
                let data;
                if (cache.has(url)) {
                    data = cache.get(url);
                } else {
                    const r = await fetch(url, { signal: abortCtrl.signal });
                    if (!r.ok) throw new Error('CivitAI API ' + r.status);
                    data = await r.json();
                    cache.set(url, data);
                    if (cache.size > 60) cache.delete(cache.keys().next().value);
                }
                filtered = filtered.concat(
                    (data.items || []).filter(m => matchBM(m))
                );
                st.nextUrl = data.metadata?.nextPage || null;
                st.total = data.metadata?.totalItems || 0;
                pages++;
                if (filtered.length >= 12 || !st.nextUrl) break;
                url = st.nextUrl;
            }

            if (append) st.results = st.results.concat(filtered);
            else st.results = filtered;
            st.hasMore = !!st.nextUrl;
            dom().error.innerHTML = '';
        } catch (e) {
            if (e.name !== 'AbortError')
                dom().error.innerHTML = `<div class="lb-error">${esc(e.message)}</div>`;
        } finally {
            st.loading = false;
            renderGrid();
            renderStatus();
            renderInfo();
            dom().sentinel.style.display = st.hasMore ? 'block' : 'none';
        }
    }

    // ── Render grid ──
    function renderGrid() {
        const g = dom().grid;
        if (st.loading && !st.results.length) { g.innerHTML = ''; return; }

        g.innerHTML = st.results.map((m, i) => {
            const ver = m.modelVersions?.[0];
            const imgs = verImgs(ver);
            const img0 = imgs[0];
            const s = m.stats || {};
            const f = ver?.files?.[0];
            const sz = f?.sizeKB ? f.sizeKB * 1024 : null;
            const bm = getBM(m);

            return `<div class="lb-card"><div class="lb-card-img">
              ${img0 ? `<img data-src="${esc(img0.url)}" alt="" loading="lazy">` : '<div class="lb-no-img">No preview</div>'}
              ${m.nsfw ? '<span class="lb-badge lb-badge-nsfw">NSFW</span>' : ''}
              <span class="lb-badge lb-badge-bm">${esc(bm)}</span>
              <div class="lb-card-ov">
                <button class="lb-btn-use" onclick="event.stopPropagation();loraBrowser._pick(${i})">⚡ Use this LoRA</button>
                <a class="lb-btn-civ" href="https://civitai.com/models/${m.id}" target="_blank" rel="noopener" onclick="event.stopPropagation()">View on CivitAI →</a>
              </div>
            </div><div class="lb-card-body">
              <div class="lb-card-name" title="${esc(m.name)}">${esc(m.name)}</div>
              <div class="lb-card-meta">by ${esc(m.creator?.username || '?')} · ${fmtSz(sz)}</div>
              <div class="lb-card-stats"><span>⬇${fmtN(s.downloadCount)}</span><span>★${s.rating?.toFixed(1) || '—'}</span><span>♥${fmtN(s.favoriteCount)}</span></div>
            </div></div>`;
        }).join('');

        requestAnimationFrame(lazyObserve);
    }

    // ── Lazy image loading ──
    let imgObs;
    function lazyObserve() {
        if (!imgObs) {
            imgObs = new IntersectionObserver(entries => {
                entries.forEach(e => {
                    if (e.isIntersecting) {
                        const img = e.target;
                        if (img.dataset.src) { img.src = img.dataset.src; delete img.dataset.src; }
                        imgObs.unobserve(img);
                    }
                });
            }, { root: dom().scroll, rootMargin: '300px' });
        }
        dom().grid.querySelectorAll('img[data-src]').forEach(i => imgObs.observe(i));
    }

    function renderStatus() {
        const el = dom().status;
        if (st.loading) el.innerHTML = '<div class="lb-loading"><div class="lb-spin"></div><div style="margin-top:8px">Searching CivitAI...</div></div>';
        else if (!st.results.length) el.innerHTML = '<div class="lb-empty">No LoRAs found. Try a different search.</div>';
        else el.innerHTML = '';
    }
    function renderInfo() {
        dom().info.textContent = st.results.length && !st.loading
            ? `${st.results.length} LoRAs found (Qwen Image Edit)` : '';
    }

    // ── Infinite scroll ──
    let sentinelObs;
    function initSentinelObs() {
        if (sentinelObs) return;
        sentinelObs = new IntersectionObserver(entries => {
            if (entries[0].isIntersecting && st.hasMore && !st.loading && st.results.length)
                doFetch(true);
        }, { root: dom().scroll, rootMargin: '600px' });
        sentinelObs.observe(dom().sentinel);
    }

    // ── Pick handler — resolves HF repo or CivitAI download URL ──
    function _pick(idx) {
        const m = st.results[idx];
        if (!m) return;
        const ver = m.modelVersions?.[0];
        const f = ver?.files?.[0];
        if (!f) { showToast('No file found for this LoRA', true); return; }

        // Try to find a HuggingFace-hosted version first
        const hfUrl = f.hashes?.HF || null;

        // Build info object for the callback
        const info = {
            name: m.name,
            civitaiId: m.id,
            versionId: ver.id,
            downloadUrl: f.downloadUrl,
            filename: f.name || m.name.replace(/[^a-zA-Z0-9_.-]/g, '_') + '.safetensors',
            baseModel: getBM(m),
            fileSize: f.sizeKB ? f.sizeKB * 1024 : null,
        };

        showToast('Selected: ' + m.name);

        if (typeof loraBrowser.onPick === 'function') {
            loraBrowser.onPick(info);
        }

        // Close after a small delay so user sees the toast
        setTimeout(() => close(), 400);
    }

    // ── Open / Close ──
    function open() {
        const d = dom();
        d.overlay.classList.add('open');
        if (!st.results.length) doFetch();
        initSentinelObs();
        d.search.focus();
    }
    function close() {
        dom().overlay.classList.remove('open');
    }

    // ── Wire up events (called once after DOM is ready) ──
    let _bound = false;
    function bind() {
        if (_bound) return;
        _bound = true;
        const d = dom();

        d.close.onclick = close;
        d.overlay.addEventListener('click', e => { if (e.target === d.overlay) close(); });

        // Search
        d.search.addEventListener('input', e => {
            clearTimeout(debounce);
            debounce = setTimeout(() => {
                st.query = e.target.value.trim();
                st.results = []; st.nextUrl = null; cache.clear();
                doFetch();
            }, 350);
        });

        // Sort
        d.sort.onchange = e => { st.sort = e.target.value; st.results = []; st.nextUrl = null; cache.clear(); doFetch(); };
        d.period.onchange = e => { st.period = e.target.value; st.results = []; st.nextUrl = null; cache.clear(); doFetch(); };

        // NSFW
        d.nsfw.onclick = () => {
            st.nsfw = !st.nsfw;
            d.nsfw.classList.toggle('on', st.nsfw);
            st.results = []; st.nextUrl = null; cache.clear();
            doFetch();
        };

        // Escape to close
        document.addEventListener('keydown', e => {
            if (e.key === 'Escape' && d.overlay.classList.contains('open')) close();
        });
    }

    // ── Init on DOMContentLoaded ──
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', bind);
    } else {
        // DOM already ready (script loaded at bottom of body)
        setTimeout(bind, 0);
    }

    // ── Public API ──
    return {
        open,
        close,
        onPick: null,  // set by consumer: loraBrowser.onPick = function(info){ ... }
        _pick,         // called from inline onclick in grid cards
    };
})();
