/* ══════════════════════════════════════════════════════════════
   LoRA Browser — CivitAI search & pick for AI Image Edit Studio
   ══════════════════════════════════════════════════════════════ */

const loraBrowser = (() => {
    const CIVITAI = 'https://civitai.com/api/v1';
    const BASE_MATCH = ['qwen', 'qwen-image', 'qwen image'];
    const THUMB_W = 100;
    const VIDEO_EXT = /\.(mp4|webm|mov|avi|gif)(\?|$)/i;

    const st = {
        query: '', sort: 'Most Downloaded', period: '', nsfw: false,
        results: [], nextUrl: null, total: 0, hasMore: true, loading: false,
    };
    let abortCtrl = null, debounce = null, toastTimer = null;
    const cache = new Map();

    let _dom = null;
    function dom() {
        if (!_dom) _dom = {
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
        };
        return _dom;
    }

    const esc = s => s ? s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;') : '';
    function fmtSz(b) { if (!b) return '?'; if (b>1e9) return (b/1e9).toFixed(1)+' GB'; if (b>1e6) return (b/1e6).toFixed(0)+' MB'; return (b/1e3).toFixed(0)+' KB'; }
    function fmtN(n) { if (!n) return '0'; return n>1000 ? (n/1000).toFixed(1)+'k' : String(n); }

    /** Is this URL a video? Check extension and known CivitAI video patterns */
    function isVideoUrl(url) {
        if (!url) return false;
        if (VIDEO_EXT.test(url)) return true;
        // CivitAI sometimes serves video at .jpeg URLs with /transcode= path
        if (url.includes('/transcode=')) return true;
        return false;
    }

    function thumbUrl(url) {
        if (!url) return '';
        try {
            const u = new URL(url);
            const parts = u.pathname.split('/');
            for (let i = 0; i < parts.length; i++) {
                if (parts[i].startsWith('original=') || parts[i].startsWith('width=')) {
                    parts[i] = 'width=' + THUMB_W;
                    u.pathname = parts.join('/');
                    return u.toString();
                }
            }
            if (parts.length > 2) {
                parts.splice(parts.length - 1, 0, 'width=' + THUMB_W);
                u.pathname = parts.join('/');
                return u.toString();
            }
        } catch (e) {}
        return url;
    }

    function matchBM(model) {
        return (model.modelVersions || []).some(v => {
            const bm = (v.baseModel || '').toLowerCase();
            return BASE_MATCH.some(pat => bm.includes(pat));
        });
    }
    function matchLoose(model) {
        if (matchBM(model)) return true;
        const hay = [model.name||'', ...(model.tags||[]),
            ...(model.modelVersions||[]).map(v=>v.name||'')].join(' ').toLowerCase();
        return BASE_MATCH.some(pat => hay.includes(pat));
    }
    function getBM(model) {
        for (const v of model.modelVersions||[]) {
            const bm = (v.baseModel||'').toLowerCase();
            if (BASE_MATCH.some(pat => bm.includes(pat))) return v.baseModel;
        }
        return model.modelVersions?.[0]?.baseModel || '?';
    }

    /**
     * Pick the best static image from a version's images array.
     * Skips videos, NSFW (if not enabled), returns first valid image.
     */
    function pickImage(ver) {
        if (!ver || !ver.images) return null;
        for (const img of ver.images) {
            // Skip NSFW unless enabled
            if (!st.nsfw && (img.nsfw || (img.nsfwLevel && img.nsfwLevel > 2))) continue;
            // Skip if the image has a video type field
            if (img.type && img.type.toLowerCase() === 'video') continue;
            // Skip video URLs
            if (isVideoUrl(img.url)) continue;
            return img;
        }
        return null;
    }

    function showToast(msg, isErr) {
        const t = dom().toast;
        t.textContent = msg;
        t.className = 'lb-toast show' + (isErr ? ' err' : '');
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => t.className = 'lb-toast', 3000);
    }

    // ── Fetch ──
    async function doFetch(append) {
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
                const userQ = st.query.trim();
                const q = userQ || 'qwen image edit';
                const p = ['types=LORA','sort='+encodeURIComponent(st.sort),'limit=40'];
                p.push('query='+encodeURIComponent(q));
                if (st.period) p.push('period='+encodeURIComponent(st.period));
                if (st.nsfw) p.push('nsfw=true');
                url = CIVITAI + '/models?' + p.join('&');
            }

            let filtered = [];
            let pages = 0;
            const matchFn = st.query.trim() ? matchLoose : matchBM;

            while (url && pages < 5) {
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

                const allItems = data.items || [];
                const matches = allItems.filter(m => matchFn(m));
                filtered = filtered.concat(matches);
                st.nextUrl = data.metadata?.nextPage || null;
                st.total = data.metadata?.totalItems || 0;
                pages++;

                if (matches.length > 0) {
                    const prevLen = st.results.length;
                    st.results = st.results.concat(matches);
                    appendCards(matches, prevLen);
                    renderInfo();
                }

                if (filtered.length >= 20 || !st.nextUrl || allItems.length === 0) break;
                url = st.nextUrl;
            }

            st.hasMore = !!st.nextUrl;
            dom().error.innerHTML = '';

            if (!st.results.length && st.query.trim()) {
                dom().status.innerHTML = '<div class="lb-empty">No Qwen-compatible LoRAs found for "'
                    + esc(st.query) + '".<br>Try broader terms like "style", "anime", or "realistic".</div>';
            }
        } catch (e) {
            if (e.name !== 'AbortError')
                dom().error.innerHTML = '<div class="lb-error">' + esc(e.message) + '</div>';
        } finally {
            st.loading = false;
            renderStatus();
            renderInfo();
            dom().sentinel.style.display = st.hasMore ? 'block' : 'none';
        }
    }

    function cardHTML(m, idx) {
        const ver = m.modelVersions?.[0];
        const img0 = pickImage(ver);
        const s = m.stats || {};
        const f = ver?.files?.[0];
        const sz = f?.sizeKB ? f.sizeKB * 1024 : null;
        const bm = getBM(m);
        const imgSrc = img0 ? thumbUrl(img0.url) : '';

        // onerror hides the image if it turns out to be a video/broken
        return `<div class="lb-card"><div class="lb-card-img">`
          + (imgSrc
            ? `<img data-src="${esc(imgSrc)}" alt="" loading="lazy" decoding="async">`
            : '<div class="lb-no-img">No preview</div>')
          + (m.nsfw ? '<span class="lb-badge lb-badge-nsfw">NSFW</span>' : '')
          + `<span class="lb-badge lb-badge-bm">${esc(bm)}</span>`
          + `<div class="lb-card-ov">`
          + `<button class="lb-btn-use" onclick="event.stopPropagation();loraBrowser._pick(${idx})">⚡ Use this LoRA</button>`
          + `<a class="lb-btn-civ" href="https://civitai.com/models/${m.id}" target="_blank" rel="noopener" onclick="event.stopPropagation()">View on CivitAI →</a>`
          + `</div></div><div class="lb-card-body">`
          + `<div class="lb-card-name" title="${esc(m.name)}">${esc(m.name)}</div>`
          + `<div class="lb-card-meta">by ${esc(m.creator?.username||'?')} · ${fmtSz(sz)}</div>`
          + `<div class="lb-card-stats"><span>⬇${fmtN(s.downloadCount)}</span><span>★${s.rating?.toFixed(1)||'—'}</span><span>♥${fmtN(s.favoriteCount)}</span></div>`
          + `</div></div>`;
    }

    function appendCards(items, startIdx) {
        const g = dom().grid;
        const frag = document.createDocumentFragment();
        const tmp = document.createElement('div');
        items.forEach((m, i) => {
            tmp.innerHTML = cardHTML(m, startIdx + i);
            while (tmp.firstChild) frag.appendChild(tmp.firstChild);
        });
        g.appendChild(frag);
        lazyObserve();
    }

    let imgObs;
    function lazyObserve() {
        if (!imgObs) {
            imgObs = new IntersectionObserver(entries => {
                for (const e of entries) {
                    if (e.isIntersecting) {
                        const img = e.target;
                        if (img.dataset.src) {
                            img.onload = () => { img.classList.add('loaded'); };
                            img.onerror = () => { img.style.display = 'none'; };
                            img.src = img.dataset.src;
                            delete img.dataset.src;
                        }
                        imgObs.unobserve(img);
                    }
                }
            }, { root: dom().scroll, rootMargin: '400px' });
        }
        dom().grid.querySelectorAll('img[data-src]').forEach(i => imgObs.observe(i));
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
        dom().info.textContent = n && !st.loading
            ? n + ' Qwen-compatible LoRA' + (n !== 1 ? 's' : '') + ' found' : '';
    }

    let sentinelObs;
    function initSentinelObs() {
        if (sentinelObs) return;
        sentinelObs = new IntersectionObserver(entries => {
            if (entries[0].isIntersecting && st.hasMore && !st.loading && st.results.length)
                doFetch(true);
        }, { root: dom().scroll, rootMargin: '400px' });
        sentinelObs.observe(dom().sentinel);
    }

    function _pick(idx) {
        const m = st.results[idx]; if (!m) return;
        const ver = m.modelVersions?.[0];
        const f = ver?.files?.[0];
        if (!f) { showToast('No file found for this LoRA', true); return; }
        const info = {
            name: m.name, civitaiId: m.id, versionId: ver.id,
            downloadUrl: f.downloadUrl,
            filename: f.name || m.name.replace(/[^a-zA-Z0-9_.-]/g,'_') + '.safetensors',
            baseModel: getBM(m), fileSize: f.sizeKB ? f.sizeKB * 1024 : null,
        };
        showToast('Selected: ' + m.name);
        if (typeof loraBrowser.onPick === 'function') loraBrowser.onPick(info);
        setTimeout(() => close(), 400);
    }

    function open() {
        const d = dom();
        d.overlay.classList.add('open');
        if (!st.results.length) doFetch();
        initSentinelObs();
        requestAnimationFrame(() => d.search.focus());
    }
    function close() { dom().overlay.classList.remove('open'); }

    function resetAndFetch() {
        st.results = []; st.nextUrl = null; cache.clear(); doFetch();
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
