/* ForteBank cash-discipline video check.
 * Flow: pick a video -> POST api/v1/analyze -> poll GET api/v1/jobs/{id} every 2 s -> render result.
 * Vanilla JS, relative URLs (works behind any prefix or tunnel), no external dependencies.
 * Every server string is rendered through textContent; links are only created for http(s) URLs.
 */
(function () {
  'use strict';

  const ANALYZE_URL = 'api/v1/analyze';
  const JOB_URL = 'api/v1/jobs/';
  const POLL_INTERVAL_MS = 2000;
  const POLL_LIMIT_MS = 10 * 60 * 1000; // the jury waits 5 min per video; the UI waits twice that
  const POLL_NET_RETRIES = 5;           // consecutive network failures tolerated while polling
  const STORAGE_KEY = 'fortebank_api_key';
  const RU = 'ru-RU';

  const $ = (id) => document.getElementById(id);
  const ui = {
    key: $('api-key'), toggleKey: $('toggle-key'), drop: $('dropzone'), fileInput: $('file-input'),
    fileInfo: $('file-info'), fileName: $('file-name'), fileSize: $('file-size'), fileClear: $('file-clear'),
    submit: $('submit'), cancel: $('cancel'), status: $('status'), result: $('result'),
    modal: $('modal'), modalImg: $('modal-img'), modalCaption: $('modal-caption'), modalClose: $('modal-close'),
  };
  const state = { file: null, busy: false, stopped: false, xhr: null, ticker: null, titleEl: null, noteEl: null, lastFocus: null };
  const fail = (message, code) => Object.assign(new Error(message), { code });
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  // ---- DOM helpers -----------------------------------------------------------------------------
  function h(tag, attrs, ...children) {
    const node = document.createElement(tag);
    Object.keys(attrs || {}).forEach((k) => {
      const v = attrs[k];
      if (v == null) return;
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else if (k.indexOf('on') === 0) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    });
    children.forEach((c) => { if (c != null) node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
    return node;
  }
  const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };

  // ---- Formatting (Russian locale) -------------------------------------------------------------
  const fmtNum = (x, digits) => Number(x).toLocaleString(RU, { maximumFractionDigits: digits });
  const fmtSeconds = (sec) => fmtNum(sec, 1) + ' с';
  const toNum = (x) => (x == null || x === '' ? NaN : Number(x));
  function fmtTimecode(sec) {
    const total = Math.floor(sec);
    return String(Math.floor(total / 60)).padStart(2, '0') + ':' + String(total % 60).padStart(2, '0');
  }
  const fmtTime = (sec) => fmtTimecode(sec) + ' · ' + fmtSeconds(sec);
  function fmtSize(bytes) {
    if (bytes >= 1024 ** 3) return fmtNum(bytes / 1024 ** 3, 2) + ' ГБ';
    if (bytes >= 1024 ** 2) return fmtNum(bytes / 1024 ** 2, 1) + ' МБ';
    if (bytes >= 1024) return fmtNum(bytes / 1024, 0) + ' КБ';
    return bytes + ' Б';
  }
  function plural(n, one, few, many) {
    const n10 = n % 10, n100 = n % 100;
    if (n10 === 1 && n100 !== 11) return one;
    if (n10 >= 2 && n10 <= 4 && (n100 < 12 || n100 > 14)) return few;
    return many;
  }
  const isHttpUrl = (u) => typeof u === 'string' && /^https?:\/\/\S+$/i.test(u);
  const parseJson = (text) => { try { return JSON.parse(text); } catch (e) { return text; } };

  // ---- API key: ?key=... -> localStorage -> input --------------------------------------------
  function loadKey() {
    let key = '';
    try {
      const params = new URLSearchParams(location.search);
      if (params.get('key')) {
        key = params.get('key').trim();
        params.delete('key'); // keep the key out of the address bar and browser history
        const qs = params.toString();
        history.replaceState(null, '', location.pathname + (qs ? '?' + qs : '') + location.hash);
      }
    } catch (e) { /* no URL/history API: ignore */ }
    try {
      if (key) localStorage.setItem(STORAGE_KEY, key);
      else key = localStorage.getItem(STORAGE_KEY) || '';
    } catch (e) { /* storage blocked: the key lives only in the input */ }
    ui.key.value = key;
  }
  const getKey = () => ui.key.value.trim();
  const saveKey = () => { try { localStorage.setItem(STORAGE_KEY, getKey()); } catch (e) { /* ignore */ } };
  const apiHeaders = () => ({ 'X-API-Key': getKey(), 'ngrok-skip-browser-warning': '1' });

  // ---- File selection ----------------------------------------------------------------------------
  function setFile(file) {
    if (!file) return;
    const name = file.name || '';
    const dot = name.lastIndexOf('.');
    const ext = dot >= 0 ? name.slice(dot + 1).toLowerCase() : '';
    const isVideo = ext === 'mov' || ext === 'mp4' || (!ext && /^video\//.test(file.type || ''));
    if (!isVideo) { showError('Поддерживаются только видеофайлы .mov и .mp4. Выбран файл: ' + (name || 'без имени')); return; }
    state.file = file;
    ui.fileName.textContent = name;
    ui.fileSize.textContent = fmtSize(file.size);
    ui.fileInfo.hidden = false;
    ui.drop.classList.add('has-file');
    hideStatus();
    updateSubmit();
  }
  function clearFile() {
    state.file = null;
    ui.fileInput.value = '';
    ui.fileInfo.hidden = true;
    ui.drop.classList.remove('has-file');
    updateSubmit();
  }
  const updateSubmit = () => { ui.submit.disabled = state.busy || !state.file; };

  // ---- Status panel ------------------------------------------------------------------------------
  function setStatus(kind, title, details) {
    clear(ui.status);
    ui.status.className = 'status status-' + kind;
    ui.status.setAttribute('role', kind === 'error' ? 'alert' : 'status');
    const titleEl = h('p', { class: 'status-title', text: title });
    const text = h('div', { class: 'status-text' }, titleEl);
    (details || []).forEach((d) => { if (d) text.appendChild(typeof d === 'string' ? h('p', { class: 'status-detail', text: d }) : d); });
    ui.status.appendChild(h('div', { class: 'status-row' }, kind === 'busy' ? h('span', { class: 'spinner', 'aria-hidden': 'true' }) : null, text));
    ui.status.hidden = false;
    return titleEl;
  }
  const showError = (message) => { setStatus('error', message); };
  const hideStatus = () => { ui.status.hidden = true; clear(ui.status); };
  function startTicker(t0) {
    stopTicker();
    state.ticker = setInterval(() => {
      if (state.titleEl) state.titleEl.textContent = 'Обработка... ' + Math.floor((Date.now() - t0) / 1000) + ' с';
    }, 1000);
  }
  function stopTicker() { if (state.ticker) { clearInterval(state.ticker); state.ticker = null; } }

  // ---- HTTP ----------------------------------------------------------------------------------------
  function extractDetail(body) {
    if (typeof body === 'string') return /^\s*</.test(body) ? '' : body.trim().slice(0, 300); // skip HTML error pages
    if (!body || typeof body !== 'object') return '';
    let d = body.detail != null ? body.detail : (body.error != null ? body.error : (body.message != null ? body.message : body.reason));
    if (Array.isArray(d)) d = d.map((x) => (x && x.msg) || JSON.stringify(x)).join('; ');
    else if (d && typeof d === 'object') d = JSON.stringify(d);
    return d == null ? '' : String(d).slice(0, 300);
  }
  function describeHttpError(status, body) {
    const detail = extractDetail(body);
    const tail = detail ? ': ' + detail : '';
    if (status === 401 || status === 403) return 'Неверный API-ключ';
    if (status === 413) return 'Файл слишком большой: сервер отклонил загрузку' + tail;
    if (status === 404) return 'Задание не найдено на сервере' + tail;
    if (status === 429) return 'Сервер перегружен, попробуйте позже' + tail;
    if (status >= 500) return 'Ошибка сервера (' + status + ')' + tail;
    return 'Сервер отклонил запрос (' + status + ')' + tail;
  }
  // XMLHttpRequest instead of fetch: it reports upload progress for a 30 MB file over a tunnel.
  function upload(file) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      state.xhr = xhr;
      xhr.open('POST', ANALYZE_URL, true);
      const hdrs = apiHeaders();
      Object.keys(hdrs).forEach((k) => xhr.setRequestHeader(k, hdrs[k]));
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && state.titleEl) state.titleEl.textContent = 'Загрузка... ' + Math.round((e.loaded / e.total) * 100) + ' %';
      };
      xhr.upload.onload = () => { if (state.titleEl) state.titleEl.textContent = 'Загрузка... ожидание ответа сервера'; };
      xhr.onload = () => resolve({ status: xhr.status, body: parseJson(xhr.responseText) });
      xhr.onerror = () => reject(fail('Нет соединения с сервером или загрузка прервана. Проверьте сеть, адрес сервиса и размер файла.', 'network'));
      xhr.onabort = () => reject(fail('Проверка отменена', 'abort'));
      const form = new FormData();
      form.append('file', file, file.name);
      xhr.send(form);
    });
  }
  async function fetchJob(jobId) {
    const res = await fetch(JOB_URL + encodeURIComponent(jobId), { headers: apiHeaders(), cache: 'no-store' });
    return { status: res.status, body: parseJson(await res.text()) };
  }
  function progressNote(body) {
    const parts = [];
    if (body.status && body.status !== 'processing') parts.push('статус: ' + body.status);
    if (typeof body.stage === 'string') parts.push('этап: ' + body.stage);
    const p = toNum(body.progress);
    if (isFinite(p)) parts.push(Math.round(p <= 1 ? p * 100 : p) + ' %');
    return parts.join(' · ');
  }
  async function waitForResult(jobId, t0) {
    let netErrors = 0;
    for (;;) {
      if (state.stopped) throw fail('Проверка отменена', 'abort');
      if (Date.now() - t0 > POLL_LIMIT_MS) throw fail('Сервер не вернул результат за 10 минут. Задание ' + jobId + ' можно запросить позже.', 'timeout');
      let r;
      try {
        r = await fetchJob(jobId);
        netErrors = 0;
      } catch (e) {
        if (++netErrors >= POLL_NET_RETRIES) throw fail('Потеряна связь с сервером во время обработки. Задание: ' + jobId, 'network');
        if (state.noteEl) state.noteEl.textContent = 'Нет связи с сервером, повторяю запрос...';
        await sleep(POLL_INTERVAL_MS);
        continue;
      }
      if (r.status >= 400) throw fail(describeHttpError(r.status, r.body), 'http');
      const body = r.body && typeof r.body === 'object' ? r.body : {};
      if (body.status === 'done' || body.status === 'failed') return body;
      if (state.noteEl) state.noteEl.textContent = progressNote(body);
      await sleep(POLL_INTERVAL_MS);
    }
  }

  // ---- Main flow -----------------------------------------------------------------------------------
  function setBusy(busy) {
    state.busy = busy;
    state.stopped = false;
    state.xhr = null;
    ui.submit.hidden = busy;
    ui.cancel.hidden = !busy;
    ui.key.disabled = busy;
    ui.toggleKey.disabled = busy;
    ui.fileInput.disabled = busy;
    ui.fileClear.disabled = busy;
    ui.drop.classList.toggle('is-disabled', busy);
    ui.drop.setAttribute('aria-disabled', String(busy));
    updateSubmit();
  }
  function cancelRun() {
    state.stopped = true;
    if (state.xhr) { try { state.xhr.abort(); } catch (e) { /* already finished */ } }
  }
  async function run() {
    if (state.busy) return;
    if (!getKey()) { showError('Введите API-ключ'); ui.key.focus(); return; }
    if (!state.file) { showError('Выберите видеофайл .mov или .mp4'); return; }
    const file = state.file;
    setBusy(true);
    ui.result.hidden = true;
    clear(ui.result);
    try {
      state.titleEl = setStatus('busy', 'Загрузка...', [file.name + ', ' + fmtSize(file.size)]);
      const up = await upload(file);
      if (up.status >= 400) throw fail(describeHttpError(up.status, up.body), 'http');
      const jobId = up.body && typeof up.body === 'object' ? up.body.job_id : null;
      if (!jobId) throw fail('Сервер не вернул идентификатор задания (job_id)', 'protocol');
      const t0 = Date.now();
      state.noteEl = h('p', { class: 'status-detail' });
      state.titleEl = setStatus('busy', 'Обработка... 0 с', [h('p', { class: 'status-detail' }, 'Задание: ', h('code', { text: String(jobId) })), state.noteEl]);
      startTicker(t0);
      const job = await waitForResult(String(jobId), t0);
      stopTicker();
      hideStatus();
      renderResult(job, { jobId: String(jobId), file, clientSec: (Date.now() - t0) / 1000 });
    } catch (e) {
      stopTicker();
      if (e && e.code === 'abort') hideStatus();
      else if (e && e.code) showError(e.message);
      else { showError('Непредвиденная ошибка в интерфейсе: ' + (e && e.message ? e.message : e)); console.error(e); }
    } finally {
      setBusy(false);
    }
  }

  // ---- Result rendering ---------------------------------------------------------------------------
  function serverProcessingSec(job) {
    const t = job.timings || (job._debug && job._debug.timings) || {};
    const cands = [job.processing_time_sec, job.processing_sec, job.processing_time, job.elapsed_sec, t.total];
    for (const c of cands) if (typeof c === 'number' && isFinite(c)) return c;
    return null;
  }
  function badge(job, count) {
    let kind, title, sub;
    if (job.status === 'failed') {
      kind = 'failed'; title = 'Обработка не удалась';
      sub = 'Ошибка: ' + (extractDetail(job) || 'сервер не сообщил причину');
    } else if (job.verdict === 'violation') {
      kind = 'bad'; title = 'Обнаружены признаки возможных нарушений';
      sub = count ? count + ' ' + plural(count, 'нарушение', 'нарушения', 'нарушений') + '. Детали ниже: таймкод, кадры и нормативное основание.'
        : 'Список нарушений в ответе пуст, детали недоступны.';
    } else if (job.verdict === 'no_violation') {
      kind = 'ok'; title = 'Нарушение не обнаружено';
      sub = 'Признаков нарушений кассовой дисциплины на записи не найдено. Кадры-доказательства прикладываются только к нарушениям.';
    } else {
      kind = 'neutral'; title = 'Вердикт не определён';
      sub = 'Сервер вернул неизвестный вердикт: ' + String(job.verdict);
    }
    return h('div', { class: 'badge badge-' + kind, role: kind === 'ok' ? null : 'alert' },
      h('span', { class: 'badge-icon', 'aria-hidden': 'true' }),
      h('div', { class: 'badge-text' }, h('p', { class: 'badge-title', text: title }), h('p', { class: 'badge-sub', text: sub })));
  }
  function metaLine(job, ctx) {
    const meta = h('div', { class: 'meta' });
    const add = (label, value, mono) => meta.appendChild(h('span', null, label + ': ', h(mono ? 'code' : 'b', { text: value })));
    if (ctx.file) add('Файл', ctx.file.name + ' (' + fmtSize(ctx.file.size) + ')');
    add('Задание', String(job.job_id || ctx.jobId), true);
    const srv = serverProcessingSec(job);
    if (srv != null) add('Время обработки', fmtSeconds(srv));
    else add('Время обработки (по часам браузера)', fmtSeconds(ctx.clientSec));
    return meta;
  }
  const toDataUrl = (b64) => (/^data:/i.test(b64) ? b64 : 'data:image/jpeg;base64,' + b64.replace(/\s+/g, ''));
  function framesBlock(frames) {
    const wrap = h('div', { class: 'frames' });
    const valid = frames.filter((f) => f && typeof f.image_base64 === 'string' && f.image_base64.length > 0);
    if (!valid.length) { wrap.appendChild(h('p', { class: 'frames-empty', text: 'Кадры-доказательства не приложены' })); return wrap; }
    valid.forEach((f) => {
      const src = toDataUrl(f.image_base64);
      const ts = toNum(f.timestamp_sec);
      const label = isFinite(ts) ? fmtTime(ts) : 'время не указано';
      const btn = h('button', { type: 'button', class: 'frame', title: 'Открыть кадр в полном размере' },
        h('img', { src, alt: 'Кадр, ' + label, loading: 'lazy', decoding: 'async' }),
        h('span', { class: 'frame-ts', text: label }));
      btn.addEventListener('click', () => openModal(src, 'Кадр, ' + label, btn));
      wrap.appendChild(btn);
    });
    return wrap;
  }
  function ruleBlock(rule) {
    if (typeof rule === 'string') rule = { name: rule };
    if (!rule || typeof rule !== 'object') return null;
    const src = rule.source && typeof rule.source === 'object' ? rule.source : {};
    const block = h('div', { class: 'rule' }, h('p', { class: 'rule-label', text: 'Нормативное основание' }));
    const name = [rule.id, rule.name].filter((x) => x).join(' · ');
    if (name) block.appendChild(h('p', { class: 'rule-name', text: name }));
    const text = rule.summary || rule.interpretation || rule.description;
    if (text) block.appendChild(h('p', { class: 'rule-text', text: String(text) }));
    if (src.document) block.appendChild(h('p', { class: 'rule-doc', text: String(src.document) }));
    if (src.clause) block.appendChild(h('p', { class: 'rule-clause', text: String(src.clause) }));
    if (isHttpUrl(src.url)) {
      let host = ''; try { host = new URL(src.url).hostname; } catch (e) { /* keep generic label */ }
      block.appendChild(h('a', { class: 'rule-link', href: src.url, target: '_blank', rel: 'noopener noreferrer', text: host ? 'Открыть источник: ' + host : 'Открыть источник' }));
    } else if (src.url) block.appendChild(h('p', { class: 'rule-clause', text: String(src.url) }));
    return block.childNodes.length > 1 ? block : null;
  }
  const compactJson = (obj) => JSON.stringify(obj, (k, v) =>
    (k === 'image_base64' && typeof v === 'string' && v.length > 64 ? v.slice(0, 32) + '… [' + v.length + ' символов]' : v), 2);
  function violationCard(v, index) {
    v = v && typeof v === 'object' ? v : {};
    const tc = toNum(v.timecode_sec);
    const score = toNum(v.score != null ? v.score : v.confidence);
    return h('article', { class: 'card' },
      h('header', { class: 'card-head' },
        h('h3', { class: 'card-type', text: (index + 1) + '. ' + (v.type || 'Иное') }),
        h('span', { class: 'timecode', title: 'Таймкод начала события' }, isFinite(tc) ? fmtTime(tc) : 'таймкод не указан')),
      h('p', { class: 'card-desc', text: v.description || 'Описание отсутствует' }),
      isFinite(score) ? h('p', { class: 'score', text: 'Оценка уверенности: ' + fmtNum(score, 2) }) : null,
      framesBlock(Array.isArray(v.evidence_frames) ? v.evidence_frames : []),
      ruleBlock(v.rule),
      h('details', { class: 'raw' }, h('summary', { text: 'JSON нарушения' }), h('pre', { text: compactJson(v) })));
  }
  function rawJsonBlock(job, jobId) {
    const full = JSON.stringify(job, null, 2);
    const copyBtn = h('button', { type: 'button', class: 'btn btn-ghost btn-sm', text: 'Скопировать' });
    copyBtn.addEventListener('click', () => copyText(full, copyBtn));
    const dlBtn = h('button', { type: 'button', class: 'btn btn-ghost btn-sm', text: 'Скачать .json' });
    dlBtn.addEventListener('click', () => download(full, 'result_' + jobId + '.json'));
    return h('details', { class: 'raw raw-all' },
      h('summary', { text: 'Полный ответ API (JSON)' }),
      h('p', { class: 'hint', text: 'Строки image_base64 сокращены для чтения; копирование и скачивание отдают полный JSON.' }),
      h('div', { class: 'raw-actions' }, copyBtn, dlBtn),
      h('pre', { text: compactJson(job) }));
  }
  function renderResult(job, ctx) {
    const violations = Array.isArray(job.violations) ? job.violations : [];
    clear(ui.result);
    ui.result.appendChild(badge(job, violations.length));
    ui.result.appendChild(metaLine(job, ctx));
    violations.forEach((v, i) => ui.result.appendChild(violationCard(v, i)));
    ui.result.appendChild(rawJsonBlock(job, ctx.jobId));
    ui.result.hidden = false;
    const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    ui.result.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
  }

  // ---- Clipboard, download, modal ------------------------------------------------------------------
  function flash(btn, label) { const old = btn.textContent; btn.textContent = label; setTimeout(() => { btn.textContent = old; }, 1500); }
  function copyText(text, btn) {
    const done = () => flash(btn, 'Скопировано');
    const fallback = () => {
      const ta = h('textarea', { class: 'visually-hidden', 'aria-hidden': 'true' });
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy'); done(); } catch (e) { flash(btn, 'Не удалось скопировать'); }
      document.body.removeChild(ta);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, fallback);
    else fallback();
  }
  function download(text, filename) {
    const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
    const a = h('a', { href: url, download: filename });
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function openModal(src, caption, opener) {
    state.lastFocus = opener || document.activeElement;
    ui.modalImg.src = src;
    ui.modalImg.alt = caption;
    ui.modalCaption.textContent = caption;
    ui.modal.hidden = false;
    document.body.classList.add('modal-open');
    ui.modalClose.focus();
  }
  function closeModal() {
    if (ui.modal.hidden) return;
    ui.modal.hidden = true;
    ui.modalImg.removeAttribute('src');
    document.body.classList.remove('modal-open');
    if (state.lastFocus && state.lastFocus.focus) state.lastFocus.focus();
  }

  // ---- Wiring --------------------------------------------------------------------------------------
  function bind() {
    ui.key.addEventListener('input', saveKey);
    ui.toggleKey.addEventListener('click', () => {
      const show = ui.key.type === 'password';
      ui.key.type = show ? 'text' : 'password';
      ui.toggleKey.textContent = show ? 'Скрыть' : 'Показать';
      ui.toggleKey.setAttribute('aria-pressed', String(show));
    });
    // A programmatic input.click() bubbles back to the dropzone: ignore that echo to avoid recursion.
    ui.drop.addEventListener('click', (e) => { if (!state.busy && e.target !== ui.fileInput) ui.fileInput.click(); });
    ui.drop.addEventListener('keydown', (e) => {
      if ((e.key === 'Enter' || e.key === ' ') && !state.busy) { e.preventDefault(); ui.fileInput.click(); }
    });
    ui.fileInput.addEventListener('change', () => setFile(ui.fileInput.files && ui.fileInput.files[0]));
    ['dragenter', 'dragover'].forEach((t) => ui.drop.addEventListener(t, (e) => { e.preventDefault(); if (!state.busy) ui.drop.classList.add('is-over'); }));
    ['dragleave', 'drop'].forEach((t) => ui.drop.addEventListener(t, () => ui.drop.classList.remove('is-over')));
    ui.drop.addEventListener('drop', (e) => {
      e.preventDefault();
      if (!state.busy) setFile(e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]);
    });
    // Dropping a file outside the zone must not navigate the tab away from the app.
    ['dragover', 'drop'].forEach((t) => document.addEventListener(t, (e) => e.preventDefault()));
    ui.fileClear.addEventListener('click', clearFile);
    ui.submit.addEventListener('click', run);
    ui.cancel.addEventListener('click', cancelRun);
    ui.modal.addEventListener('click', (e) => { if (e.target === ui.modal) closeModal(); });
    ui.modalClose.addEventListener('click', closeModal);
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });
    window.addEventListener('beforeunload', (e) => { if (state.busy) { e.preventDefault(); e.returnValue = ''; } });
  }

  loadKey();
  bind();
  updateSubmit();
})();
