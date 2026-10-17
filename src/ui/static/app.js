/* ForteBank cash-discipline video check: pick a video -> POST api/v1/analyze -> poll GET api/v1/jobs/{id}
 * every 2 s -> render verdict, violation cards, evidence frames and rule sources. Vanilla JS, relative URLs
 * (works behind any prefix or tunnel). Server strings go through textContent; links only for http(s) URLs. */
(function () {
  'use strict';

  const ANALYZE_URL = 'api/v1/analyze', JOB_URL = 'api/v1/jobs/', STORAGE_KEY = 'fortebank_api_key';
  const POLL_INTERVAL_MS = 2000;
  const POLL_LIMIT_MS = 10 * 60 * 1000; // the jury waits 5 min per video; the UI waits twice that
  const POLL_RETRIES = 5;               // consecutive failed polls (network error or 5xx/429) tolerated

  const $ = (id) => document.getElementById(id);
  const ui = {
    key: $('api-key'), toggleKey: $('toggle-key'), drop: $('dropzone'), fileInput: $('file-input'),
    fileInfo: $('file-info'), fileName: $('file-name'), fileSize: $('file-size'), fileClear: $('file-clear'),
    submit: $('submit'), cancel: $('cancel'), lookup: $('lookup'), jobId: $('job-id'), lookupBtn: $('lookup-btn'),
    status: $('status'), result: $('result'),
    modal: $('modal'), modalImg: $('modal-img'), modalCaption: $('modal-caption'), modalClose: $('modal-close'),
  };
  const state = { file: null, busy: false, stopped: false, xhr: null, ticker: null, titleEl: null, noteEl: null, lastFocus: null };
  const fail = (message, code) => Object.assign(new Error(message), { code });
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  // ---- DOM and formatting helpers (Russian locale) ----
  function h(tag, attrs, ...children) {
    const node = document.createElement(tag);
    Object.keys(attrs || {}).forEach((k) => {
      if (attrs[k] == null) return;
      if (k === 'class') node.className = attrs[k]; else if (k === 'text') node.textContent = attrs[k]; else node.setAttribute(k, attrs[k]);
    });
    children.forEach((c) => { if (c != null) node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
    return node;
  }
  const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };
  const line = (cls, text) => (text ? h('p', { class: cls, text: String(text) }) : null);
  // An element with a bold lead-in ("Трактовка: ..."); null when there is no text.
  const labelled = (tag, cls, label, text) => (text ? h(tag, { class: cls }, h('span', { class: 'rule-sub', text: label + ': ' }), String(text)) : null);
  const fmtNum = (x, digits) => Number(x).toLocaleString('ru-RU', { maximumFractionDigits: digits });
  const fmtSeconds = (sec) => fmtNum(sec, 1) + ' с';
  const toNum = (x) => (x == null || x === '' ? NaN : Number(x));
  const pad2 = (n) => String(n).padStart(2, '0');
  // mm:ss and the seconds share one 0.1 s rounding, so 59.96 reads "01:00 · 60 с" rather than "00:59 · 60 с".
  function fmtTime(sec) { const r = Math.round(sec * 10) / 10; return pad2(Math.floor(r / 60)) + ':' + pad2(Math.floor(r) % 60) + ' · ' + fmtSeconds(r); }
  const fmtDate = (d) => { const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(d)); return m ? m[3] + '.' + m[2] + '.' + m[1] : String(d); };
  function fmtSize(bytes) {
    if (bytes >= 1024 ** 3) return fmtNum(bytes / 1024 ** 3, 2) + ' ГБ';
    if (bytes >= 1024 ** 2) return fmtNum(bytes / 1024 ** 2, 1) + ' МБ';
    return bytes >= 1024 ? fmtNum(bytes / 1024, 0) + ' КБ' : bytes + ' Б';
  }
  function plural(n, one, few, many) {
    const n10 = n % 10, n100 = n % 100;
    return n10 === 1 && n100 !== 11 ? one : n10 >= 2 && n10 <= 4 && (n100 < 12 || n100 > 14) ? few : many;
  }
  // Cut long server text at a word boundary and mark the cut instead of chopping mid-word.
  function shorten(text, max) {
    text = String(text).trim();
    if (text.length <= max) return text;
    const cut = text.slice(0, max), sp = cut.lastIndexOf(' ');
    return (sp > max / 2 ? cut.slice(0, sp) : cut).replace(/[\s,;:|.-]+$/, '') + '…';
  }
  const isHttpUrl = (u) => typeof u === 'string' && /^https?:\/\/\S+$/i.test(u);
  const parseJson = (text) => { try { return JSON.parse(text); } catch (e) { return text; } };

  // ---- API key: ?key=... -> localStorage -> input ----
  function loadKey() {
    let key = '';
    try {
      const params = new URLSearchParams(location.search);
      key = (params.get('key') || '').trim();
      params.delete('key'); // keep the key out of the address bar and browser history
      if (key) history.replaceState(null, '', location.pathname + (params.toString() ? '?' + params.toString() : '') + location.hash);
    } catch (e) { /* no URL/history API: ignore */ }
    try { key = key || localStorage.getItem(STORAGE_KEY) || ''; localStorage.setItem(STORAGE_KEY, key); } catch (e) { /* storage blocked */ }
    ui.key.value = key;
  }
  const getKey = () => ui.key.value.trim();
  const saveKey = () => { try { localStorage.setItem(STORAGE_KEY, getKey()); } catch (e) { /* ignore */ } };
  const apiHeaders = () => ({ 'X-API-Key': getKey(), 'ngrok-skip-browser-warning': '1' });

  // ---- File selection ----
  const updateSubmit = () => { ui.submit.disabled = state.busy || !state.file; };
  function setFile(file) {
    if (!file) return;
    const name = file.name || '';
    const ext = name.indexOf('.') >= 0 ? name.slice(name.lastIndexOf('.') + 1).toLowerCase() : '';
    if (ext !== 'mov' && ext !== 'mp4' && !(ext === '' && /^video\//.test(file.type || ''))) {
      showError('Поддерживаются только видеофайлы .mov и .mp4. Выбран файл: ' + (name || 'без имени')); return;
    }
    state.file = file;
    ui.fileName.textContent = name; ui.fileSize.textContent = fmtSize(file.size);
    toggleFileInfo(true); hideStatus();
  }
  function clearFile() { state.file = null; ui.fileInput.value = ''; toggleFileInfo(false); }
  function toggleFileInfo(on) { ui.fileInfo.hidden = !on; ui.drop.classList.toggle('has-file', on); updateSubmit(); }

  // ---- Status panel ----
  function setStatus(kind, title, details) {
    clear(ui.status);
    ui.status.className = 'status status-' + kind;
    ui.status.setAttribute('role', kind === 'error' ? 'alert' : 'status');
    const titleEl = h('p', { class: 'status-title', text: title }), text = h('div', { class: 'status-text' }, titleEl);
    (details || []).forEach((d) => { if (d) text.appendChild(typeof d === 'string' ? h('p', { class: 'status-detail', text: d }) : d); });
    ui.status.appendChild(h('div', { class: 'status-row' }, kind === 'busy' ? h('span', { class: 'spinner', 'aria-hidden': 'true' }) : null, text));
    ui.status.hidden = false;
    return titleEl;
  }
  const showError = (message, details) => { setStatus('error', message, details); };
  const hideStatus = () => { ui.status.hidden = true; clear(ui.status); };
  const setNote = (text) => { if (state.noteEl) state.noteEl.textContent = text; };
  const stopTicker = () => { clearInterval(state.ticker); state.ticker = null; };
  function startTicker(t0) {
    stopTicker(); state.ticker = setInterval(() => { if (state.titleEl) state.titleEl.textContent = 'Обработка... ' + Math.floor((Date.now() - t0) / 1000) + ' с'; }, 1000);
  }

  // ---- HTTP ----
  function extractDetail(body) {
    if (typeof body === 'string') return /^\s*</.test(body) ? '' : shorten(body, 300); // skip HTML error pages
    if (!body || typeof body !== 'object') return '';
    let d = [body.detail, body.error, body.message, body.reason].find((x) => x != null);
    if (Array.isArray(d)) d = d.map((x) => (x && x.msg) || JSON.stringify(x)).join('; ');
    else if (d && typeof d === 'object') d = JSON.stringify(d);
    return d == null ? '' : shorten(d, 300);
  }
  function describeHttpError(status, body) {
    const detail = extractDetail(body), tail = detail ? ': ' + detail : '';
    if (status === 401 || status === 403) return 'Неверный API-ключ';
    if (status === 413) return 'Файл слишком большой: сервер отклонил загрузку' + tail;
    if (status === 404) return 'Задание не найдено на сервере' + tail;
    if (status === 429) return 'Сервер перегружен, попробуйте позже' + tail;
    return (status >= 500 ? 'Ошибка сервера (' : 'Сервер отклонил запрос (') + status + ')' + tail;
  }
  // XMLHttpRequest instead of fetch: it reports upload progress for a 30 MB file over a tunnel.
  function upload(file) {
    return new Promise((resolve, reject) => {
      const xhr = (state.xhr = new XMLHttpRequest()), hdrs = apiHeaders();
      xhr.open('POST', ANALYZE_URL, true);
      Object.keys(hdrs).forEach((k) => xhr.setRequestHeader(k, hdrs[k]));
      const setTitle = (t) => { if (state.titleEl) state.titleEl.textContent = t; };
      xhr.upload.onprogress = (e) => { if (e.lengthComputable) setTitle('Загрузка... ' + Math.round((e.loaded / e.total) * 100) + ' %'); };
      xhr.upload.onload = () => setTitle('Загрузка... ожидание ответа сервера');
      xhr.onload = () => resolve({ status: xhr.status, body: parseJson(xhr.responseText) });
      xhr.onerror = () => reject(fail('Нет соединения с сервером или загрузка прервана. Проверьте сеть, адрес сервиса и размер файла.', 'network'));
      xhr.onabort = () => reject(fail('Проверка отменена', 'abort'));
      const form = new FormData(); form.append('file', file, file.name);
      xhr.send(form);
    });
  }
  async function waitForResult(jobId, t0) {
    let strikes = 0;
    const retry = async (note) => { // transient failure: keep polling, give up after POLL_RETRIES in a row
      if (++strikes >= POLL_RETRIES) throw fail('Потеряна связь с сервером во время обработки. Задание: ' + jobId, 'network');
      setNote(note); await sleep(POLL_INTERVAL_MS);
    };
    for (;;) {
      if (state.stopped) throw fail('Проверка отменена', 'abort');
      if (Date.now() - t0 > POLL_LIMIT_MS) throw fail('Сервер не вернул результат за 10 минут. Задание: ' + jobId, 'timeout');
      let res, body;
      try {
        res = await fetch(JOB_URL + encodeURIComponent(jobId), { headers: apiHeaders(), cache: 'no-store' });
        body = parseJson(await res.text());
      } catch (e) { await retry('Нет связи с сервером, повторяю запрос...'); continue; }
      // A tunnel or proxy may answer 5xx/429 while the job is still running: treat it like a dropped connection.
      if (res.status >= 500 || res.status === 429) { await retry('Сервер временно недоступен (' + res.status + '), повторяю запрос...'); continue; }
      if (res.status >= 400) throw fail(describeHttpError(res.status, body), 'http');
      strikes = 0;
      if (!body || typeof body !== 'object') body = {};
      if (body.status === 'done' || body.status === 'failed') return body;
      setNote(body.status && body.status !== 'processing' ? 'статус: ' + body.status : (typeof body.stage === 'string' ? 'этап: ' + body.stage : ''));
      await sleep(POLL_INTERVAL_MS);
    }
  }

  // ---- Main flow: run() uploads the selected file; run(jobId) looks an existing job up ----
  function setBusy(busy) {
    Object.assign(state, { busy, stopped: false, xhr: null });
    ui.submit.hidden = busy; ui.cancel.hidden = !busy;
    [ui.key, ui.toggleKey, ui.fileInput, ui.fileClear, ui.jobId, ui.lookupBtn].forEach((el) => { el.disabled = busy; });
    ui.drop.classList.toggle('is-disabled', busy); ui.drop.setAttribute('aria-disabled', String(busy));
    updateSubmit();
  }
  function cancelRun() { state.stopped = true; if (state.xhr) state.xhr.abort(); }
  // The job keeps running server-side after a lost connection or a cancel: prefill the lookup form for later.
  function offerLookup(jobId) { ui.jobId.value = jobId; ui.lookup.open = true; }
  const LOOKUP_HINT = 'Задание продолжает выполняться на сервере. Результат можно получить без повторной загрузки видео: '
    + 'форма «Запросить результат по номеру задания» выше уже заполнена.';
  async function run(knownId) {
    if (state.busy) return;
    const key = getKey(), file = knownId == null ? state.file : null;
    if (!key) { showError('Введите API-ключ'); ui.key.focus(); return; }
    if (/[^\x20-\xff]/.test(key)) { showError('В API-ключе недопустимые символы: только латиница, цифры и знаки пунктуации'); ui.key.focus(); return; }
    if (knownId == null && !file) { showError('Выберите видеофайл .mov или .mp4'); return; }
    if (knownId != null && !knownId) { showError('Введите номер задания (job_id)'); ui.jobId.focus(); return; }
    let jobId = knownId || '';
    setBusy(true); ui.result.hidden = true; clear(ui.result);
    try {
      if (!jobId) {
        state.titleEl = setStatus('busy', 'Загрузка...', [file.name + ', ' + fmtSize(file.size)]);
        const up = await upload(file);
        if (up.status >= 400) throw fail(describeHttpError(up.status, up.body), 'http');
        jobId = up.body && typeof up.body === 'object' && up.body.job_id != null ? String(up.body.job_id) : '';
        if (!jobId) throw fail('Сервер не вернул идентификатор задания (job_id)', 'protocol');
      }
      const t0 = Date.now(); state.noteEl = h('p', { class: 'status-detail' });
      state.titleEl = setStatus('busy', 'Обработка... 0 с', [h('p', { class: 'status-detail' }, 'Задание: ', h('code', { text: jobId })), state.noteEl]);
      startTicker(t0);
      const job = await waitForResult(jobId, t0);
      stopTicker(); hideStatus();
      renderResult(job, { jobId, file, clientSec: (Date.now() - t0) / 1000 });
    } catch (e) {
      stopTicker();
      const lost = Boolean(jobId && e && ['network', 'timeout', 'abort'].indexOf(e.code) >= 0);
      if (lost) offerLookup(jobId);
      if (e && e.code === 'abort') setStatus('note', 'Проверка отменена', [lost ? 'Задание ' + jobId + ' продолжит выполняться на сервере; результат можно запросить по его номеру.' : null]);
      else if (e && e.code) showError(e.message, [lost ? LOOKUP_HINT : null]);
      else { console.error(e); showError('Непредвиденная ошибка в интерфейсе: ' + (e && e.message ? e.message : e)); }
    } finally {
      setBusy(false);
    }
  }

  // ---- Result rendering ----
  // `done` + `note` = the pipeline crashed, timed out or was cut short: the verdict is a placeholder, not a finding.
  function noteText(note) {
    if (/^partial/i.test(note)) return 'обработка остановлена по лимиту времени, проанализирована только часть записи';
    if (/error|missing|log/i.test(note)) return 'обработка прервана внутренней ошибкой сервиса, анализ не выполнен';
    return 'отметка сервера: «' + note + '»';
  }
  function failedText(job) {
    const raw = String([job.error, job.detail, job.message].find((x) => x != null) || '').trim();
    const head = shorten(raw.split(' | ')[0], 300) || 'сервер не сообщил причину'; // the API joins ffmpeg's lines with ' | '
    const more = head.length < raw.length ? (/[.!?…]$/.test(head) ? '' : '.') + ' Полный текст ошибки — в блоке «Полный ответ API» ниже.' : '';
    return 'Ошибка: ' + head + more;
  }
  function badge(job, count) {
    const note = typeof job.note === 'string' ? job.note.trim() : '';
    let kind = 'neutral', title = 'Вердикт не определён', sub = 'Сервер вернул неизвестный вердикт: ' + String(job.verdict);
    if (job.status === 'failed') {
      [kind, title, sub] = ['failed', 'Обработка не удалась', failedText(job)];
    } else if (job.verdict === 'violation') {
      [kind, title] = ['bad', 'Обнаружены признаки возможных нарушений'];
      sub = count ? count + ' ' + plural(count, 'нарушение', 'нарушения', 'нарушений') + '. Детали ниже: таймкод, кадры и нормативное основание.'
        : 'Список нарушений в ответе пуст, детали недоступны.';
      if (note) sub += ' Анализ не завершён: ' + noteText(note) + '. Показаны нарушения, найденные до остановки.';
    } else if (note) {
      [kind, title] = ['partial', 'Анализ не завершён, результат неполный'];
      sub = 'Вердикт не вынесен: ' + noteText(note) + '. Нарушения не подтверждены и не исключены — повторите проверку или проверьте журнал сервера.';
    } else if (job.verdict === 'no_violation') {
      [kind, title] = ['ok', 'Нарушение не обнаружено'];
      sub = 'Признаков нарушений кассовой дисциплины на записи не найдено. Кадры-доказательства прикладываются только к нарушениям.';
    }
    return h('div', { class: 'badge badge-' + kind, role: kind === 'ok' ? null : 'alert' }, h('span', { class: 'badge-icon', 'aria-hidden': 'true' }),
      h('div', { class: 'badge-text' }, h('p', { class: 'badge-title', text: title }), h('p', { class: 'badge-sub', text: sub })));
  }
  function metaLine(job, ctx) {
    const meta = h('div', { class: 'meta' });
    const add = (label, value, mono) => meta.appendChild(h('span', null, label + ': ', h(mono ? 'code' : 'b', { text: value })));
    if (ctx.file) add('Файл', ctx.file.name + ' (' + fmtSize(ctx.file.size) + ')');
    add('Задание', String(job.job_id || ctx.jobId), true);
    const t = job.timings || (job._debug && job._debug.timings) || {};
    const srv = [job.processing_time_sec, job.processing_sec, job.elapsed_sec, t.total].find((c) => typeof c === 'number');
    if (srv != null) add('Время обработки', fmtSeconds(srv));
    else if (ctx.file) add('Время обработки (по часам браузера)', fmtSeconds(ctx.clientSec)); // unknown for a looked-up job
    if (typeof job.note === 'string' && job.note) add('Служебная отметка', job.note, true);
    return meta;
  }
  const toDataUrl = (b64) => (/^data:/i.test(b64) ? b64 : 'data:image/jpeg;base64,' + b64.replace(/\s+/g, ''));
  function framesBlock(frames) {
    const valid = frames.filter((f) => f && typeof f.image_base64 === 'string' && f.image_base64.length > 0);
    if (!valid.length) return h('p', { class: 'frames-empty', text: 'Кадры-доказательства не приложены' });
    const wrap = h('div', { class: 'frames' });
    valid.forEach((f) => {
      const src = toDataUrl(f.image_base64), ts = toNum(f.timestamp_sec);
      const time = isFinite(ts) ? fmtTime(ts) : 'время не указано';
      const btn = h('button', { type: 'button', class: 'frame', title: 'Открыть кадр в полном размере' },
        h('img', { src, alt: 'Кадр, ' + time, loading: 'lazy', decoding: 'async' }), h('span', { class: 'frame-ts', text: time }));
      btn.addEventListener('click', () => openModal(src, 'Кадр, ' + time, btn));
      wrap.appendChild(btn);
    });
    return wrap;
  }
  // An entry of docs/legal_sources.json (document, authority/number/date, clause, quote, interpretation, url);
  // older shapes (title/name, summary/note/text) still render.
  function sourceBlock(s) {
    if (typeof s === 'string') s = { document: s };
    if (!s || typeof s !== 'object') return null;
    let link = null;
    if (isHttpUrl(s.url)) {
      let host = '';
      try { host = new URL(s.url).hostname; } catch (e) { /* keep the generic label */ }
      link = h('a', { class: 'rule-link', href: s.url, target: '_blank', rel: 'noopener noreferrer', text: 'Открыть источник' + (host ? ': ' + host : '') });
    }
    const issued = [s.authority, s.number, s.date ? 'от ' + fmtDate(s.date) : ''].filter((x) => x).join(' · ');
    return h('div', { class: 'rule-source' }, line('rule-doc', s.document || s.title || s.name), line('rule-meta', issued),
      line('rule-clause', s.clause), labelled('blockquote', 'rule-quote', 'Текст нормы', s.quote),
      labelled('p', 'rule-text', 'Трактовка', s.interpretation || s.summary || s.note || s.text), link || line('rule-clause', s.url));
  }
  // `rule` may carry one `source` object (task spec) or a `sources` list (pipeline output); both are shown.
  function ruleBlock(rule) {
    if (typeof rule === 'string') rule = { name: rule };
    if (!rule || typeof rule !== 'object') return null;
    const sources = Array.isArray(rule.sources) ? rule.sources : (rule.source ? [rule.source] : []);
    const block = h('div', { class: 'rule' }, h('p', { class: 'rule-label', text: 'Нормативное основание' }),
      line('rule-name', [rule.id, rule.name].filter((x) => x).join(' · ')),
      line('rule-text', rule.summary || rule.interpretation || rule.description), ...sources.map(sourceBlock));
    if (!sources.length) block.appendChild(line('rule-clause', 'Ссылка на нормативный документ не указана'));
    return block;
  }
  const compactJson = (obj) => JSON.stringify(obj, (k, v) =>
    (k === 'image_base64' && typeof v === 'string' && v.length > 64 ? v.slice(0, 32) + '… [' + v.length + ' символов]' : v), 2);
  function violationCard(v, index) {
    v = v && typeof v === 'object' ? v : {};
    const r = v.rule && typeof v.rule === 'object' ? v.rule : {};
    const tc = toNum(v.timecode_sec), tEnd = toNum(r.t_end_sec), score = toNum(r.score != null ? r.score : v.score);
    const extra = [isFinite(tc) && isFinite(tEnd) && tEnd > tc ? 'событие ' + fmtNum(tc, 1) + '–' + fmtSeconds(tEnd) : '',
      isFinite(score) ? 'оценка правила ' + fmtNum(score, 2) : ''].filter((x) => x).join(' · ');
    return h('article', { class: 'card' },
      h('header', { class: 'card-head' }, h('h3', { class: 'card-type', text: (index + 1) + '. ' + (v.type || 'Иное') }),
        h('span', { class: 'timecode', title: 'Таймкод начала события', text: isFinite(tc) ? fmtTime(tc) : 'таймкод не указан' })),
      h('p', { class: 'card-desc', text: v.description || 'Описание отсутствует' }),
      line('score', extra), framesBlock(Array.isArray(v.evidence_frames) ? v.evidence_frames : []), ruleBlock(v.rule),
      h('details', { class: 'raw' }, h('summary', { text: 'JSON нарушения' }), h('pre', { text: compactJson(v) })));
  }
  function rawJsonBlock(job, jobId) {
    const full = JSON.stringify(job, null, 2);
    const copyBtn = h('button', { type: 'button', class: 'btn btn-ghost btn-sm', text: 'Скопировать' });
    const dlBtn = h('button', { type: 'button', class: 'btn btn-ghost btn-sm', text: 'Скачать .json' });
    copyBtn.addEventListener('click', () => copyText(full, copyBtn));
    dlBtn.addEventListener('click', () => download(full, 'result_' + jobId.replace(/[^\w.-]+/g, '_') + '.json'));
    return h('details', { class: 'raw raw-all' }, h('summary', { text: 'Полный ответ API (JSON)' }),
      h('p', { class: 'hint', text: 'Строки image_base64 сокращены для чтения; копирование и скачивание отдают полный JSON.' }),
      h('div', { class: 'raw-actions' }, copyBtn, dlBtn), h('pre', { text: compactJson(job) }));
  }
  function renderResult(job, ctx) {
    const violations = Array.isArray(job.violations) ? job.violations : [];
    clear(ui.result);
    ui.result.appendChild(badge(job, violations.length)); ui.result.appendChild(metaLine(job, ctx));
    violations.forEach((v, i) => ui.result.appendChild(violationCard(v, i)));
    ui.result.appendChild(rawJsonBlock(job, ctx.jobId));
    ui.result.hidden = false;
    const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    ui.result.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
  }

  // ---- Clipboard, download, full-size frame viewer ----
  function flash(btn, label) { const old = btn.textContent; btn.textContent = label; setTimeout(() => { btn.textContent = old; }, 1500); }
  function copyText(text, btn) {
    const done = () => flash(btn, 'Скопировано');
    const fallback = () => { // plain-http origins have no navigator.clipboard
      const ta = document.body.appendChild(h('textarea', { class: 'visually-hidden', 'aria-hidden': 'true' }));
      ta.value = text; ta.select();
      try { document.execCommand('copy'); done(); } catch (e) { flash(btn, 'Не удалось скопировать'); }
      document.body.removeChild(ta);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, fallback); else fallback();
  }
  function download(text, filename) {
    const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
    const a = document.body.appendChild(h('a', { href: url, download: filename }));
    a.click(); document.body.removeChild(a); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function openModal(src, caption, opener) {
    state.lastFocus = opener || document.activeElement;
    ui.modalImg.src = src; ui.modalImg.alt = ui.modalCaption.textContent = caption;
    ui.modal.hidden = false; document.body.classList.add('modal-open'); ui.modalClose.focus();
  }
  function closeModal() {
    if (ui.modal.hidden) return;
    ui.modal.hidden = true; document.body.classList.remove('modal-open');
    ui.modalImg.removeAttribute('src'); // release the decoded full-size image
    if (state.lastFocus) state.lastFocus.focus();
  }

  // ---- Wiring ----
  function bind() {
    ui.key.addEventListener('input', saveKey);
    ui.toggleKey.addEventListener('click', () => {
      const show = ui.key.type === 'password';
      ui.key.type = show ? 'text' : 'password'; ui.toggleKey.textContent = show ? 'Скрыть' : 'Показать'; ui.toggleKey.setAttribute('aria-pressed', String(show));
    });
    // A programmatic input.click() bubbles back to the dropzone: ignore that echo to avoid recursion.
    ui.drop.addEventListener('click', (e) => { if (!state.busy && e.target !== ui.fileInput) ui.fileInput.click(); });
    ui.drop.addEventListener('keydown', (e) => { if ((e.key === 'Enter' || e.key === ' ') && !state.busy) { e.preventDefault(); ui.fileInput.click(); } });
    ui.fileInput.addEventListener('change', () => setFile(ui.fileInput.files && ui.fileInput.files[0]));
    ['dragenter', 'dragover'].forEach((t) => ui.drop.addEventListener(t, (e) => { e.preventDefault(); if (!state.busy) ui.drop.classList.add('is-over'); }));
    ['dragleave', 'drop'].forEach((t) => ui.drop.addEventListener(t, () => ui.drop.classList.remove('is-over')));
    ui.drop.addEventListener('drop', (e) => { e.preventDefault(); if (!state.busy) setFile(e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]); });
    // Dropping a file outside the zone must not navigate the tab away from the app.
    ['dragover', 'drop'].forEach((t) => document.addEventListener(t, (e) => e.preventDefault()));
    ui.fileClear.addEventListener('click', clearFile); ui.cancel.addEventListener('click', cancelRun);
    ui.submit.addEventListener('click', () => run(null));
    ui.lookupBtn.addEventListener('click', () => run(ui.jobId.value.trim()));
    ui.jobId.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); run(ui.jobId.value.trim()); } });
    ui.modal.addEventListener('click', (e) => { if (e.target === ui.modal) closeModal(); });
    ui.modalClose.addEventListener('click', closeModal); document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });
    window.addEventListener('beforeunload', (e) => { if (state.busy) { e.preventDefault(); e.returnValue = ''; } });
  }

  loadKey(); bind(); updateSubmit();
})();
