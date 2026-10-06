const $ = (id) => document.getElementById(id);
const ui = {
  config: $('config'), detail: $('config-detail'), target: $('target'),
  start: $('start'), stop: $('stop'), matrix: $('matrix'),
  state: $('state-pill'), targetChar: $('target-char'), targetProgress: $('target-progress'),
  flashLabel: $('flash-label'), flashHelp: $('flash-help'), flashIcon: $('flash-icon'),
  hits: $('hit-count'), progress: $('progress-fill'), epoch: $('epoch-label'),
  count: $('flash-count'), primary: $('primary-text'), secondary: $('secondary-text'),
  symbol: $('decision-symbol'), decision: $('decision-detail'), suggestions: $('suggestions'),
  model: $('model-status'), qwen: $('qwen-text'), qwenButton: $('qwen-button'), note: $('session-note')
};

let configs = [];
let session = null;
let activeFlash = null;
let activeCells = [];
let epochHits = [];
let runToken = 0;
let qwenReady = false;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function setStatus(text, kind = '') {
  ui.state.textContent = text;
  ui.state.className = 'state-pill' + (kind ? ` ${kind}` : '');
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
  ui.flashLabel.textContent = `${name} · повтор ${flash.repetition + 1}`;
  ui.flashHelp.textContent = 'Если здесь ваш символ — нажмите пробел сейчас';
  ui.flashIcon.textContent = '✳';
}

function render(state) {
  session = state;
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
  ui.epoch.textContent = `Эпоха ${state.epoch_id + 1}` + (state.retry ? ` · повторная попытка ${state.retry}` : '');
  ui.count.textContent = `0 / ${state.plan.length} вспышек`;
  ui.progress.style.width = '0%';
  if (state.last) {
    ui.symbol.textContent = state.last.symbol === ' ' ? '␣' : state.last.symbol || '—';
    ui.decision.textContent = state.last.symbol ? `Уверенность ${Math.round(state.last.confidence * 100)}% · ${state.last.hits} сигналов` :
      `Недостаточно сигнала · ${state.last.hits} сигналов, повторяем`;
    ui.suggestions.replaceChildren();
    state.last.suggestions.forEach((item) => {
      const badge = document.createElement('span');
      badge.textContent = `${item.symbol === ' ' ? '␣' : item.symbol} ${Math.round(item.probability * 100)}%`;
      ui.suggestions.append(badge);
    });
  }
  if (state.finished) {
    setStatus('Завершена');
    ui.start.disabled = false;
    ui.stop.disabled = true;
    ui.config.disabled = false;
    ui.target.disabled = false;
    ui.flashLabel.textContent = 'Сессия завершена';
    ui.flashHelp.textContent = 'Можно выбрать конфиг и начать новую сессию';
    ui.note.textContent = `Локальная запись: runs/gui/${state.session_id}`;
  } else {
    setStatus('Идёт сессия', 'active');
    ui.start.disabled = true;
    ui.stop.disabled = false;
    ui.config.disabled = true;
    ui.target.disabled = true;
    ui.note.textContent = 'Нажатия и решения записываются только на этом компьютере.';
  }
  ui.qwenButton.disabled = !qwenReady || !state.text;
}

async function runEpochs(token) {
  try {
    while (session && !session.finished && token === runToken) {
      const plan = session.plan;
      const hits = Array(plan.length).fill(false);
      epochHits = hits;
      ui.hits.textContent = '0';
      ui.flashLabel.textContent = 'Приготовьтесь';
      ui.flashHelp.textContent = 'Подсветка начнётся через мгновение';
      await sleep(Math.min(1200, session.timing.pause_ms));
      for (let i = 0; i < plan.length && token === runToken; i++) {
        showFlash(plan[i]);
        ui.count.textContent = `${i + 1} / ${plan.length} вспышек`;
        ui.progress.style.width = `${100 * (i + 1) / plan.length}%`;
        await sleep(session.timing.flash_ms);
        clearFlash();
        if (i + 1 < plan.length) await sleep(session.timing.isi_ms);
      }
      if (token !== runToken) return;
      ui.flashLabel.textContent = 'Обрабатываю сигналы…';
      setStatus('Декодирование', 'active');
      const next = await api('/api/epoch', {session_id: session.session_id, hits});
      if (token !== runToken) return;
      render(next);
    }
  } catch (error) {
    clearFlash();
    setStatus('Ошибка', 'error');
    ui.flashLabel.textContent = error.message;
    ui.flashHelp.textContent = 'Остановите сессию и попробуйте ещё раз';
  }
}

document.addEventListener('keydown', (event) => {
  if (event.code !== 'Space' || !session || session.finished || !activeFlash) return;
  if (event.target instanceof HTMLElement && /INPUT|SELECT|BUTTON|TEXTAREA/.test(event.target.tagName)) return;
  event.preventDefault();
  if (event.repeat || epochHits[activeFlash.id]) return;
  epochHits[activeFlash.id] = true;
  ui.hits.textContent = String(epochHits.filter(Boolean).length);
  activeCells.forEach((i) => ui.matrix.children[i]?.classList.add('hit'));
  ui.flashIcon.textContent = '✓';
  ui.flashHelp.textContent = 'Сигнал принят';
});

ui.config.addEventListener('change', () => {
  const config = configs.find((item) => item.name === ui.config.value);
  if (config) ui.detail.textContent = `${config.paradigm === 'row_column' ? 'Строки и столбцы' : 'Отдельные символы'} · ${config.repetitions} повтора · ${config.flash_ms} мс вспышка + ${config.isi_ms} мс пауза · prior: ${config.model}`;
});

ui.start.addEventListener('click', async () => {
  try {
    const state = await api('/api/start', {config: ui.config.value, target: ui.target.value.toLowerCase().replaceAll('ё', 'е')});
    runToken++;
    ui.start.blur();
    render(state);
    runEpochs(runToken);
  } catch (error) {
    setStatus('Ошибка', 'error');
    ui.flashLabel.textContent = error.message;
  }
});

ui.stop.addEventListener('click', async () => {
  runToken++;
  clearFlash();
  if (!session || session.finished) return;
  try { render(await api('/api/stop', {session_id: session.session_id})); }
  catch (error) { setStatus('Ошибка', 'error'); ui.flashLabel.textContent = error.message; }
});

ui.qwenButton.addEventListener('click', async () => {
  if (!session) return;
  ui.qwenButton.disabled = true;
  ui.qwen.textContent = 'Локальная модель подбирает продолжение…';
  try {
    const result = await api('/api/suggest', {session_id: session.session_id});
    ui.qwen.textContent = result.suggestion || 'Модель не предложила продолжение.';
  } catch (error) { ui.qwen.textContent = error.message; }
  ui.qwenButton.disabled = !qwenReady || !session.text;
});

async function init() {
  try {
    const data = await api('/api/configs');
    configs = data.configs;
    ui.config.replaceChildren();
    configs.forEach((item) => {
      const option = document.createElement('option');
      option.value = item.name;
      option.textContent = item.name === 'gui_slow.yaml' ? 'Медленный старт · gui_slow.yaml' : item.name;
      ui.config.append(option);
    });
    if (configs.some((item) => item.name === 'gui_slow.yaml')) ui.config.value = 'gui_slow.yaml';
    ui.config.dispatchEvent(new Event('change'));
    qwenReady = Boolean(data.qwen.ready);
    ui.model.textContent = qwenReady ? `Символьный prior + ${data.qwen.model} локально` :
      'Символьная n-грамма офлайн; Qwen не загружен';
    ui.qwen.textContent = qwenReady ? 'Нажмите кнопку после набора первого символа.' :
      'Qwen пока не установлен; основной декодер работает офлайн.';
    ui.start.disabled = !configs.length;
    renderMatrix(data.symbols);
  } catch (error) {
    setStatus('Нет связи', 'error');
    ui.detail.textContent = error.message;
    ui.start.disabled = true;
  }
}
init();
