/* ══════════════════════════════════════════════════════════════
   LoRA Browser — CivitAI search & pick
   Perf-focused: single-page fetch, rAF DOM batching, sessionStorage
   ══════════════════════════════════════════════════════════════ */

const loraBrowser = (() => {
    const CIVITAI = 'https://civitai.com/api/v1';
    const BASE_MATCH = ['qwen', 'qwen-image', 'qwen image'];
    const THUMB_W = 100;
    const VIDEO_RE = /\.(mp4|webm|mov|avi|gif)(\?|$)/i;
    const SS_PREFIX = 'lb_';          // sessionStorage key prefix
    const SS_MAX = 30;                // max cached pages

    const st = {
        query: '', sort: 'Most Downloaded', period: '', nsfw: false,
        results: [], nextUrl: null, hasMore: true, loading: false,
    };
    let abortCtrl = null, debounce = null, toastTimer = null;

    // ── sessionStorage cache ──
    // Survives modal close/reopen, cleared on tab close
    function ssKey(url) { return SS_PREFIX + url; }
    function ssGet(url) {
        try { const v = sessionStorage.getItem(ssKey(url)); return v ? JSON.parse(v) : null; }
        catch { return null; }
    }
    function ssPut(url, data) {
        try {
            // Evict oldest if over limit
            const keys = []; for (let i = 0; i < sessionStorage.length; i++) {
                const k = sessionStorage.key(i);
                if (k && k.startsWith(SS_PREFIX)) keys.push(k);
            }
            while (keys.length >= SS_MAX) { sessionStorage.removeItem(keys.shift()); }
            sessionStorage.setItem(ssKey(url), JSON.stringify(data));
        } catch { /* quota exceeded — ignore */ }
    }

    // ── DOM ──
    let _dom = null;
    function dom() {
        return _dom || (_dom = {
            overlay:  document.getElementById('loraBrowserOverlay'),
            close:    document.getElementById('loraBrowserClose'),
            search:   document.getElementById('lbSearch'),
            sort:     document.getElementById('lbSort'),
            period:   document.getElementById('lbPeriod'),
            nsfw:     document.getElementById('lbNsfwTog'),
            info:     document.getElementById('lbResultInfo'),
            scroll:   document.getElementById('lbScroll'),
            grid:     document.getElementById('lbGrid'),
            error:    document.getElementById('lbError'),
            status:   document.getElementById('lbStatus'),
            sentinel: document.getElementById('lbSentinel'),
            toast:    document.getElementById('lbToast'),
        });
    }

    // ── Helpers ──
    const esc = s => s ? s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;') : '';
    const fmtSz = b => !b ? '?' : b>1e9 ? (b/1e9).toFixed(1)+' GB' : b>1e6 ? (b/1e6).toFixed(0)+' MB' : (b/1e3).toFixed(0)+' KB';
    const fmtN = n => !n ? '0' : n>1000 ? (n/1000).toFixed(1)+'k' : String(n);

    function isVideo(url) {
        if (!url) return false;
        return VIDEO_RE.test(url) || url.includes('/transcode=');
    }

    function thumbUrl(url) {
        if (!url) return '';
        try {
            const u = new URL(url), p = u.pathname.split('/');
            for (let i = 0; i < p.length; i++) {
                if (p[i].startsWith('original=') || p[i].startsWith('width=')) {
                    p[i] = 'width=' + THUMB_W; u.pathname = p.join('/'); return u.toString();
                }
            }
            if (p.length > 2) { p.splice(p.length-1, 0, 'width='+THUMB_W); u.pathname = p.join('/'); return u.toString(); }
        } catch {}
        return url;
    }

    function matchBM(m) {
        return (m.modelVersions||[]).some(v => {
            const bm = (v.baseModel||'').toLowerCase();
            return BASE_MATCH.some(p => bm.includes(p));
        });
    }
    function matchLoose(m) {
        if (matchBM(m)) return true;
        const hay = [m.name||'', ...(m.tags||[]), ...(m.modelVersions||[]).map(v=>v.name||'')].join(' ').toLowerCase();
        return BASE_MATCH.some(p => hay.includes(p));
    }
    function getBM(m) {
        for (const v of m.modelVersions||[]) {
            const bm = (v.baseModel||'').toLowerCase();
            if (BASE_MATCH.some(p => bm.includes(p))) return v.baseModel;
        }
        return m.modelVersions?.[0]?.baseModel || '?';
    }
    function pickImage(ver) {
        if (!ver?.images) return null;
        for (const img of ver.images) {
            if (!st.nsfw && (img.nsfw || (img.nsfwLevel && img.nsfwLevel > 2))) continue;
            if (img.type && img.type.toLowerCase() === 'video') continue;
            if (isVideo(img.url)) continue;
            return img;
        }
        return null;
    }

    function showToast(msg, err) {
        const t = dom().toast; t.textContent = msg;
        t.className = 'lb-toast show' + (err ? ' err' : '');
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => t.className = 'lb-toast', 3000);
    }

    // ══════════════════════════════════════════════════════════
    //  FETCH — one page at a time, never blocks scroll
    // ══════════════════════════════════════════════════════════
    async function doFetch(append) {
        if (st.loading) return;              // don't stack fetches
        if (abortCtrl) abortCtrl.abort();
        abortCtrl = new AbortController();
        st.loading = true;
        if (!append) { dom().grid.innerHTML = ''; st.results = []; }
        renderStatus();

        try {
            let url;
            if (append && st.nextUrl) {
                url = st.nextUrl;
            } else {
                const q = st.query.trim() || 'qwen image edit';
                const p = ['types=LORA','sort='+encodeURIComponent(st.sort),'limit=40'];
                p.push('query='+encodeURIComponent(q));
                if (st.period) p.push('period='+encodeURIComponent(st.period));
                if (st.nsfw) p.push('nsfw=true');
                url = CIVITAI + '/models?' + p.join('&');
            }

            // ── Fetch exactly ONE page ──
            let data = ssGet(url);
            if (!data) {
                const r = await fetch(url, { signal: abortCtrl.signal });
                if (!r.ok) throw new Error('CivitAI API ' + r.status);
                data = await r.json();
                ssPut(url, data);
            }

            const matchFn = st.query.trim() ? matchLoose : matchBM;
            const matches = (data.items || []).filter(m => matchFn(m));
            st.nextUrl = data.metadata?.nextPage || null;
            st.hasMore = !!st.nextUrl;

            if (matches.length > 0) {
                const prevLen = st.results.length;
                st.results = st.results.concat(matches);
                // Insert cards off main thread via rAF
                batchAppendCards(matches, prevLen);
            }

            dom().error.innerHTML = '';
            if (!st.results.length && !st.hasMore && st.query.trim()) {
                dom().status.innerHTML = '<div class="lb-empty">No Qwen-compatible LoRAs for "'
                    + esc(st.query) + '".<br>Try broader terms.</div>';
            }

            // If we got 0 matches but there are more pages, auto-fetch next
            // (do it after a frame so we don't block)
            if (matches.length === 0 && st.hasMore) {
                st.loading = false;  // unlock
                requestAnimationFrame(() => doFetch(true));
                return;
            }
        } catch (e) {
            if (e.name !== 'AbortError')
                dom().error.innerHTML = '<div class="lb-error">' + esc(e.message) + '</div>';
        } finally {
            st.loading = false;
            renderStatus(); renderInfo();
            dom().sentinel.style.display = st.hasMore ? 'block' : 'none';
        }
    }

    // ══════════════════════════════════════════════════════════
    //  DOM — batched card insertion via requestAnimationFrame
    // ══════════════════════════════════════════════════════════
    function cardHTML(m, idx) {
        const ver = m.modelVersions?.[0];
        const img0 = pickImage(ver);
        const s = m.stats || {};
        const f = ver?.files?.[0];
        const sz = f?.sizeKB ? f.sizeKB * 1024 : null;
        const bm = getBM(m);
        const src = img0 ? thumbUrl(img0.url) : '';

        return '<div class="lb-card"><div class="lb-card-img">'
          + (src ? '<img data-src="'+esc(src)+'" alt="" decoding="async">' : '<div class="lb-no-img">No preview</div>')
          + (m.nsfw ? '<span class="lb-badge lb-badge-nsfw">NSFW</span>' : '')
          + '<span class="lb-badge lb-badge-bm">'+esc(bm)+'</span>'
          + '<div class="lb-card-ov">'
          + '<button class="lb-btn-use" onclick="event.stopPropagation();loraBrowser._pick('+idx+')">⚡ Use this LoRA</button>'
          + '<a class="lb-btn-civ" href="https://civitai.com/models/'+m.id+'" target="_blank" rel="noopener" onclick="event.stopPropagation()">View on CivitAI →</a>'
          + '</div></div><div class="lb-card-body">'
          + '<div class="lb-card-name" title="'+esc(m.name)+'">'+esc(m.name)+'</div>'
          + '<div class="lb-card-meta">by '+esc(m.creator?.username||'?')+' · '+fmtSz(sz)+'</div>'
          + '<div class="lb-card-stats"><span>⬇'+fmtN(s.downloadCount)+'</span><span>★'+(s.rating?.toFixed(1)||'—')+'</span><span>♥'+fmtN(s.favoriteCount)+'</span></div>'
          + '</div></div>';
    }

    // Insert cards in batches of 6 per animation frame — keeps scroll buttery
    const BATCH_SZ = 6;
    function batchAppendCards(items, startIdx) {
        const g = dom().grid;
        let i = 0;
        function tick() {
            const frag = document.createDocumentFragment();
            const tmp = document.createElement('div');
            const end = Math.min(i + BATCH_SZ, items.length);
            for (; i < end; i++) {
                tmp.innerHTML = cardHTML(items[i], startIdx + i);
                while (tmp.firstChild) frag.appendChild(tmp.firstChild);
            }
            g.appendChild(frag);
            lazyObserve();
            renderInfo();
            if (i < items.length) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    }

    // ── Lazy image loading ──
    let imgObs;
    function lazyObserve() {
        if (!imgObs) {
            imgObs = new IntersectionObserver(entries => {
                for (const e of entries) {
                    if (!e.isIntersecting) continue;
                    const img = e.target;
                    imgObs.unobserve(img);
                    if (!img.dataset.src) continue;
                    const src = img.dataset.src;
                    delete img.dataset.src;
                    img.onload = () => img.classList.add('loaded');
                    img.onerror = () => { img.style.display = 'none'; };
                    img.src = src;
                }
            }, { root: dom().scroll, rootMargin: '500px' });
        }
        // Only observe new unloaded images
        dom().grid.querySelectorAll('img[data-src]:not([src])').forEach(i => imgObs.observe(i));
    }

    function renderStatus() {
        const el = dom().status;
        if (st.loading && !st.results.length)
            el.innerHTML = '<div class="lb-loading"><div class="lb-spin"></div><div style="margin-top:8px">Searching CivitAI...</div></div>';
        else if (!st.loading && !st.results.length)
            el.innerHTML = '<div class="lb-empty">No LoRAs found. Try a different search.</div>';
        else el.innerHTML = '';
    }
    function renderInfo() {
        const n = st.results.length;
        dom().info.textContent = n && !st.loading ? n + ' Qwen-compatible LoRA' + (n!==1?'s':'') + ' found' : '';
    }

    // ── Infinite scroll — fires doFetch(true) for next page ──
    let sentinelObs;
    function initSentinelObs() {
        if (sentinelObs) return;
        sentinelObs = new IntersectionObserver(entries => {
            if (entries[0].isIntersecting && st.hasMore && !st.loading && st.results.length)
                doFetch(true);
        }, { root: dom().scroll, rootMargin: '600px' });
        sentinelObs.observe(dom().sentinel);
    }

    function _pick(idx) {
        const m = st.results[idx]; if (!m) return;
        const ver = m.modelVersions?.[0], f = ver?.files?.[0];
        if (!f) { showToast('No file found', true); return; }
        showToast('Selected: ' + m.name);
        if (typeof loraBrowser.onPick === 'function') loraBrowser.onPick({
            name: m.name, civitaiId: m.id, versionId: ver.id,
            downloadUrl: f.downloadUrl,
            filename: f.name || m.name.replace(/[^a-zA-Z0-9_.-]/g,'_') + '.safetensors',
            baseModel: getBM(m), fileSize: f.sizeKB ? f.sizeKB*1024 : null,
        });
        setTimeout(close, 400);
    }

    function open() {
        dom().overlay.classList.add('open');
        if (!st.results.length) doFetch();
        initSentinelObs();
        requestAnimationFrame(() => dom().search.focus());
    }
    function close() { dom().overlay.classList.remove('open'); }

    function resetAndFetch() {
        st.results = []; st.nextUrl = null; doFetch();
    }

    let _bound = false;
    function bind() {
        if (_bound) return; _bound = true;
        const d = dom();
        d.close.onclick = close;
        d.overlay.addEventListener('click', e => { if (e.target === d.overlay) close(); });
        d.search.addEventListener('input', e => {
            clearTimeout(debounce);
            debounce = setTimeout(() => { st.query = e.target.value.trim(); resetAndFetch(); }, 450);
        });
        d.sort.onchange = e => { st.sort = e.target.value; resetAndFetch(); };
        d.period.onchange = e => { st.period = e.target.value; resetAndFetch(); };
        d.nsfw.onclick = () => { st.nsfw = !st.nsfw; d.nsfw.classList.toggle('on', st.nsfw); resetAndFetch(); };
        document.addEventListener('keydown', e => {
            if (e.key === 'Escape' && d.overlay.classList.contains('open')) close();
        });
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bind);
    else setTimeout(bind, 0);

    return { open, close, onPick: null, _pick };
})();
