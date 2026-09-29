# EEG Speller v3 — исследовательский прототип

Рабочий прототип программного контура ЭЭГ-спеллера по приложенной [архитектуре v3](docs/ARCHITECTURE.md). Декодирование ЭЭГ заменено симулятором, быстрый языковой prior — символьной n-граммой, исправление — словарной моделью. Эти mock-модели **не являются БЯМ**. Результаты симуляции **не подтверждают требования ТЗ к реальным участникам**.

## Установка

Требуется Python 3.11+. В отдельном виртуальном окружении:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

По умолчанию не нужны интернет и GPU: модель использует `data/corpus_ru_fallback.txt` (оригинальный текст >50 КБ, не содержащий предложений тестовых заданий). Сетевые вызовы языковой модели выключены: `llm.allow_remote: false`. Сырого ЭЭГ в комплекте нет.

## Команды

```bash
.venv/bin/python -m eeg_speller run --config configs/default.yaml
.venv/bin/python -m eeg_speller run --config configs/no_llm.yaml
.venv/bin/python -m eeg_speller bench-llm --config configs/exp01.yaml
.venv/bin/python -m eeg_speller experiment --config configs/compare_llm.yaml
.venv/bin/python -m eeg_speller replay runs/<session_id> --config configs/default.yaml
.venv/bin/python -m eeg_speller export runs/<session_id>
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_markers.py
.venv/bin/python scripts/gen_data_model.py
```

`run` создаёт каталог `runs/<session_id>/` с `events.jsonl`, манифестом, полным конфигом, текстами, таблицами, метриками, контролем целостности и контрольными суммами. `replay` повторно использует записанные физиологические наблюдения и пересчитывает языковую часть. `experiment` запускает парный план с 10 симулированными участниками и записывает `runs/compare_llm/summary.json`; сессии лежат ниже в `paired_compare/`. `export` принимает каталог сессии или серии и создаёт ZIP рядом с ним.

Для быстрого пробного запуска уменьшите `runtime.max_symbols` в копии конфига. `stimulus.paradigm` выбирает `row_column` или `single_symbol`, `fusion.strategy` — `log_pool` или `linear_pool`, `detector.mode` — `label` или `abstain`, `slow_llm.trigger` — момент коррекции. Все параметры и временные допущения описаны в `docs/ASSUMPTIONS.md`.

## Корпус и расширения

`scripts/prepare_corpus.py --offline` создаёт `data/corpus_ru.txt` из fallback. Без `--offline` скрипт загружает только перечисленные в `configs/corpus_sources.yaml` русскоязычные произведения, отрезает служебный текст Gutenberg, нормализует репертуар и удаляет предложения, совпадающие с заданиями. Для запусков по умолчанию скачивание не выполняется. Происхождение и ограничение fallback отмечены в `docs/OPEN_QUESTIONS.md` (Q-34).

`hf_causal` — опциональный локальный бэкенд `transformers` с ленивым импортом и только локальными весами; `openai_compat` доступен только как медленная коррекция при явном `allow_remote: true`. Быстрая модель требует полных логитов следующего токена. Точка подключения команды физиологии — `eeg_speller/physiology/base.py`; `stub_live` намеренно останавливается с `NotImplementedError` (Q-01). Внешние функции TTS и команд над текстом пока только записывают событие.

## Документы

- `00_PROMPT.md`, `input/` и четыре исходных документа в `docs/` — постановка и контракты;
- `docs/TRACEABILITY.md` — трассировка всех NEED-F и REQ;
- `docs/REPORT_NOTES.md` — заготовка отчёта и ограничения;
- `docs/DATA_MODEL.md` — схема записи и экспорт.
