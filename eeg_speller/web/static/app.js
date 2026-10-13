'use strict';
// Participant window: follows the session started in the experimenter window,
// Shows timed frames and reliably delivers their onset markers to physiology.

const $ = (id) => document.getElementById(id);
const ui = {
  matrix: $('matrix'), state: $('state-pill'), targetChar: $('target-char'), targetProgress: $('target-progress'),
  flashLabel: $('flash-label'), flashHelp: $('flash-help'), flashIcon: $('flash-icon'),
  hits: $('hit-count'), progress: $('progress-fill'), epoch: $('epoch-label'),
  count: $('flash-count'), primary: $('primary-text'), secondary: $('secondary-text'),
  symbol: $('decision-symbol'), decision: $('decision-detail'), note: $('session-note'),
};

let latest = null;        // latest participant snapshot from the server
let runningId = null;     // session whose epochs this window is running
let runToken = 0;
let activeFlash = null;
let activeCells = [];
let epochHits = [];
let epochId = null;
let markerQueue = Promise.resolve();
let markerError = null;
let wake = null;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const show = (s) => (s === ' ' ? '␣' : s || '—');

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

// Ordered delivery does not block rendering. All acknowledgements are checked
// before decoding; failed or missing markers cannot silently produce a symbol.
function report(event, flashId, clientMs) {
  if (!latest || epochId === null) return;
  const body = {session_id: latest.session_id, epoch_id: epochId, event,
    flash_id: flashId, client_ms: clientMs};
  markerQueue = markerQueue.then(async () => {
    if (markerError) return;
    const response = await api('/api/progress', body);
    if (!response.accepted) throw new Error('Метка вспышки отклонена: эпоха изменилась');
  }).catch((error) => { markerError = error; });
}

async function synchronizeClock(sessionId) {
  let best = null;
  for (let i = 0; i < 5; i++) {
    const a = performance.now();
    const reply = await api('/api/clock');
    const b = performance.now();
    const sample = {offset_ms: reply.server_ms - (a + b) / 2, uncertainty_ms: (b - a) / 2};
    if (!best || sample.uncertainty_ms < best.uncertainty_ms) best = sample;
  }
  await api('/api/clock/sync', {session_id: sessionId, ...best});
}

async function waitUntil(atMs) {
  const remaining = atMs - performance.now();
  if (remaining > 0) await sleep(remaining);
}

function frameFlash(flash) {
  return new Promise((resolve) => requestAnimationFrame((timestamp) => {
    showFlash(flash);
    resolve(timestamp);
  }));
}

function setStatus(text, kind = '') {
  ui.state.textContent = text;
  ui.state.className = 'state-pill' + (kind ? ` ${kind}` : '');
}

function setMessage(label, help, icon = '✧') {
  ui.flashLabel.textContent = label;
  ui.flashHelp.textContent = help;
  ui.flashIcon.textContent = icon;
}

function renderMatrix(symbols) {
  ui.matrix.replaceChildren();
  symbols.forEach((symbol, index) => {
    const cell = document.createElement('div');
    cell.className = 'cell' + (symbol === ' ' || symbol === '⌫' ? ' special' : '');
    cell.textContent = symbol === ' ' ? '␣' : symbol;
    cell.setAttribute('role', 'gridcell');
    cell.setAttribute('aria-label', symbol === ' ' ? 'пробел' : symbol === '⌫' ? 'удалить' : symbol);
    cell.dataset.index = String(index);
    ui.matrix.append(cell);
  });
}

function clearFlash() {
  activeFlash = null;
  activeCells.forEach((i) => ui.matrix.children[i]?.classList.remove('flash', 'hit'));
  activeCells = [];
}

function showFlash(flash) {
  clearFlash();
  activeFlash = flash;
  activeCells = flash.indices;
  activeCells.forEach((i) => ui.matrix.children[i]?.classList.add('flash'));
  const name = flash.kind === 'row' ? `Строка ${flash.index + 1}` :
    flash.kind === 'column' ? `Столбец ${flash.index + 1}` : `Символ ${flash.index + 1}`;
  setMessage(`${name} · повтор ${flash.repetition + 1}`,
    latest?.physiology.input_mode === 'keyboard' ? 'Если здесь ваш символ — нажмите пробел сейчас' :
      'Следите за нужным символом. Ответ поступает из ЭЭГ.', '✳');
}

function renderText(state) {
  const keyboard = state.physiology.input_mode === 'keyboard';
  $('participant-instructions').textContent = keyboard ?
    'Следите за нужным символом. Когда подсвечена его строка или столбец, нажмите пробел. Сессию запускает экспериментатор.' :
    (state.physiology.source === 'edf_replay' ?
      'Воспроизводится запись ЭЭГ из EDF. Символ выбирается по сигналу записи; нажатия не нужны.' :
      'Следите за нужным символом и удерживайте на нём внимание. Система принимает ЭЭГ через LSL; нажатия не нужны.');
  $('mode-note').textContent = state.physiology.label;
  ui.note.textContent = 'ЭЭГ, события вспышек и решения сохраняются на этом компьютере.';
  if (ui.matrix.children.length !== state.symbols.length) renderMatrix(state.symbols);
  ui.primary.textContent = state.text || '';
  if (!state.text) {
    const placeholder = document.createElement('span');
    placeholder.className = 'placeholder';
    placeholder.textContent = 'Текст появится здесь';
    ui.primary.append(placeholder);
  }
  ui.secondary.textContent = state.secondary || '—';
  const next = state.target && state.target_index < state.target.length ? state.target[state.target_index] : null;
  ui.targetChar.textContent = next === ' ' ? '␣' : next || '—';
  ui.targetProgress.textContent = state.target ? `${state.target_index} / ${state.target.length} символов` : 'Свободный набор';
  ui.epoch.textContent = state.finished ? 'Эпоха —' :
    `Эпоха ${state.epoch_id + 1}` + (state.retry ? ` · повторная попытка ${state.retry}` : '');
  if (state.last) {
    ui.symbol.textContent = show(state.last.symbol);
    ui.decision.textContent = state.last.symbol ? 'Символ выбран' :
      'Недостаточно надёжного сигнала, повторяем';
  } else {
    ui.symbol.textContent = '—';
    ui.decision.textContent = 'Ожидается первая эпоха';
  }
}

function waitForChange() {
  return new Promise((resolve) => { wake = resolve; });
}

function onLive(data) {
  const state = data.session;
  latest = state;
  if (wake) { const resolve = wake; wake = null; resolve(); }
  if (!state) {
    runToken++; runningId = null; clearFlash();
    setStatus('Ожидание');
    setMessage('Ожидание', 'Экспериментатор настраивает и запускает сессию в своём окне.');
    return;
  }
  if (state.finished) {
    if (runningId === state.session_id || runningId === null) {
      runToken++; runningId = null; clearFlash();
    }
    renderText(state);
    ui.count.textContent = '0 / 0 вспышек';
    ui.progress.style.width = '0%';
    setStatus(state.error_message ? 'Ошибка сигнала' : 'Завершена', state.error_message ? 'error' : '');
    setMessage('Сессия завершена', state.end_reason === 'target_finished' ? 'Все позиции задания обработаны.' :
      state.end_reason === 'source_exhausted' ? 'Запись EDF обработана полностью.' :
      state.error_message || 'Экспериментатор остановил сессию.');
    return;
  }
  if (runningId !== state.session_id) {
    runToken++;
    runningId = state.session_id;
    clearFlash();
    renderText(state);
    runEpochs(runToken, state.session_id);
  }
}

function stillValid(token, sessionId) {
  return token === runToken && latest && latest.session_id === sessionId && !latest.finished;
}

async function runEpochs(token, sessionId) {
  try {
    while (stillValid(token, sessionId)) {
      if (latest.paused) {
        clearFlash();
        ui.count.textContent = '—';
        ui.progress.style.width = '0%';
        setStatus('Пауза', 'paused');
        setMessage('Пауза', 'Экспериментатор поставил паузу. Можно отдохнуть.', '❚❚');
        await waitForChange();
        continue;
      }
      const state = latest;
      const plan = state.plan;
      epochId = state.epoch_id;
      epochHits = Array(plan.length).fill(false);
      ui.hits.textContent = state.physiology.input_mode === 'keyboard' ? '0' : 'ЭЭГ';
      markerQueue = Promise.resolve(); markerError = null;
      ui.count.textContent = `0 / ${plan.length} вспышек`;
      ui.progress.style.width = '0%';
      setStatus('Идёт сессия', 'active');
      setMessage('Приготовьтесь', 'Найдите глазами свой символ — подсветка скоро начнётся');
      await synchronizeClock(sessionId);
      await sleep(state.timing.pause_ms);
      if (!stillValid(token, sessionId) || latest.paused || latest.epoch_id !== epochId) continue;
      report('epoch_begin');
      let firstFrame = null;
      for (let i = 0; i < plan.length; i++) {
        if (!stillValid(token, sessionId)) return;
        if (markerError) throw markerError;
        if (firstFrame !== null) await waitUntil(firstFrame + plan[i].onset_ms);
        const onset = await frameFlash(plan[i]);
        if (firstFrame === null) firstFrame = onset;
        report('flash', i, onset);
        ui.count.textContent = `${i + 1} / ${plan.length} вспышек`;
        ui.progress.style.width = `${100 * (i + 1) / plan.length}%`;
        await waitUntil(onset + state.timing.flash_ms);
        clearFlash();
      }
      if (!stillValid(token, sessionId)) return;
      report('flashes_done');
      await markerQueue;
      if (markerError) throw markerError;
      setMessage('Обрабатываю сигналы…', 'Декодер ожидает реакцию ЭЭГ и объединяет оценки символов');
      setStatus('Декодирование', 'active');
      let next;
      try {
        next = await api('/api/epoch', {session_id: sessionId, epoch_id: epochId,
          ...(state.physiology.input_mode === 'keyboard' ? {hits: epochHits} : {})});
      } catch (error) {
        if (!stillValid(token, sessionId)) return;   // stopped while decoding
        throw error;
      }
      if (token !== runToken) return;
      latest = next;
      renderText(next);
      if (next.finished) { onLive({session: next}); return; }
    }
  } catch (error) {
    clearFlash();
    setStatus('Ошибка', 'error');
    setMessage(error.message, 'Сообщите экспериментатору');
    if (stillValid(token, sessionId)) {
      try { await api('/api/fault', {session_id: sessionId, message: error.message.slice(0, 500)}); }
      catch (_) { /* the original signal/network error remains visible */ }
    }
  } finally {
    epochId = null;
  }
}

async function liveLoop() {
  let since = -1;
  for (;;) {
    try {
      const data = await api(`/api/live?role=participant&since=${since}&timeout=15`);
      since = data.control_version;
      onLive(data);
    } catch (error) {
      setStatus('Нет связи', 'error');
      await sleep(1500);
    }
  }
}

document.addEventListener('keydown', (event) => {
  if (event.code !== 'Space') return;
  event.preventDefault();
  if (!latest || latest.physiology.input_mode !== 'keyboard' || latest.finished ||
      !activeFlash || event.repeat || epochHits[activeFlash.id]) return;
  epochHits[activeFlash.id] = true;
  report('hit', activeFlash.id);
  ui.hits.textContent = String(epochHits.filter(Boolean).length);
  activeCells.forEach((i) => ui.matrix.children[i]?.classList.add('hit'));
  ui.flashIcon.textContent = '✓';
  ui.flashHelp.textContent = 'Сигнал принят';
});

async function init() {
  try {
    const data = await api('/api/configs');
    renderMatrix(data.symbols);
    setStatus('Ожидание');
  } catch (error) {
    setStatus('Нет связи', 'error');
  }
  liveLoop();
}
init();
