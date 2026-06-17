# Tapeback — критическое ревью архитектуры + roadmap производительности

Дата ревью: 2026-06-17. Ветка: main, версия 0.9.5.
Триггер: «после записи очень долго идёт транскрибация, даже без диаризации».

Ревью основано на чтении всех модулей `src/tapeback/` и проверке фактов грепом.
Это рассуждение по коду, **не по замерам** — поэтому задача №1 в roadmap = тайминги.

---

## Часть 1 — Почему транскрибация долгая (по убыванию вклада)

### 1.1. Стерео-пайплайн транскрибирует аудио ДВА раза подряд
`transcriber.py:142-143` — `transcribe_stereo` зовёт Whisper отдельно на mic и monitor:
```python
mic_segments, mic_info = self.transcribe(mic_16k)
monitor_segments, monitor_info = self.transcribe(monitor_16k)
```
`vad_filter=True` спасает от тишины → не строго ×2, а ×1.3–2 по озвученному времени.
На 4 GB GTX 1650 Ti две модели параллельно не идут → проходы строго последовательны.
Это и есть главная причина «долго даже без диаризации». Архитектурный trade-off
(чистый текст по каналам без crosstalk).

### 1.2. `merge_channels` делает лишний ffmpeg-проход, результат выбрасывается
`pipeline.py:71`: `stereo_path, _mono_16k_path = merge_channels(...)` — второй элемент
никогда не читается (подтверждено грепом). А внутри `merge_channels` (`audio.py:92-111`)
ради него гоняется отдельный ffmpeg с **двумя `loudnorm` + amix + ресемпл по всей длине**.
Чистый баг-перформанса. Чинится удалением второго прохода.
Глубже: pipeline склеивает mic+monitor в стерео (проход A), потом режет обратно
(проход C). Исходные `mic.wav`/`monitor.wav` уже раздельные — 16k можно делать прямо
из них, стерео собирать только для архива/RMS. Сейчас 3 ffmpeg + 4 файла там, где хватит 2.

### 1.3. Модель грузится заново каждый раз + сетевой поход в HuggingFace
`_lazy.py` → новый `WhisperModel` на каждый запуск. Нет `local_files_only` →
faster-whisper лезет в HF за метаданными на каждом старте (видно в логах трея).
+10с холодного старта, риск зависона оффлайн. Для трея модель можно держать тёплой.

### 1.4. `beam_size=5` по умолчанию (`settings.py:49`)
Дефолт «на точность». Greedy (`beam_size=1`) на чистом аудио ×3–4 быстрее.
Настройка уже есть — вопрос дефолта и документации. Менять только после E2E-прогона.

### 1.5. Обычный `transcribe`, а не batched
faster-whisper умеет `BatchedInferencePipeline` (до ×4 на том же чекпойнте/качестве).
Самый «честный» ускоритель без смены модели/качества.

---

## Часть 2 — Архитектура vs прод-стандарты

### P0 — тихий откат на CPU маскирует баги и убивает скорость
`transcriber.py:102-107` ловит **любой** `RuntimeError` на cuda → молча CPU.
В отличие от `diarizer.py:130`, где проверяется текст ошибки (`"CUDA"`/`"out of memory"`).
Любой нерелевантный RuntimeError → CPU → внезапное ×10 и «почему медленно».
Fix: откатываться только на реальную CUDA/OOM-ошибку, иначе пробросить.

### P1
- **Нет единого логирования/наблюдаемости.** 9 `print(file=sys.stderr)` в библиотечном
  коде, трей через `logging`, CLI через `click.echo`. Библиотека не должна `print`.
  Нужен единый `logging.getLogger(__name__)` + **тайминги стадий** (см. roadmap №1).
- **Side-effect при импорте.** `transcriber.py:20-21` на уровне модуля меняет
  `os.environ["LC_MESSAGES"]` + `setlocale` — мутирует локаль всего процесса при импорте.
  Workaround к багу PyAV; место неправильное (локально вокруг вызова / точка входа).
- **Слабая типизация на границах.** Инфо ходит как `dict[str, str | float]`
  (`transcriber.py:112`), достаётся через `float(info.get(...))` (`pipeline.py:88-90`).
  Просится `@dataclass TranscriptionInfo` в models.py. `device`/`compute_type`/
  `whisper_model`/`language` — `str` там, где `Literal`/валидатор ловил бы опечатки.
- **`device: str = "cuda"` жёсткий дефолт.** На CPU-only машине каждый запуск:
  load→RuntimeError→warn→CPU. Нужен `auto` + детект.
- **Daemon-поток обработки гибнет при выходе.** `tray.py:201` — `_do_stop_and_process`
  в `daemon=True`, `_on_quit` гасит луп → поток умирает на полуслове, недописанный md.

### P2
- ffmpeg-ошибки непрозрачны: `capture_output=True, check=True` прячет stderr в
  `CalledProcessError`. Логировать stderr на ошибке.
- Широкие `except Exception` в `is_stereo`/`_get_stereo_source` (`pipeline.py:160,338`) —
  сузить до `wave.Error, OSError`.
- Дублируется CPU-load в `_load_model` и `_fallback_to_cpu` (`transcriber.py:67,81`).

---

## Часть 3 — Что уже хорошо (калибровка)
Разделение models/инфраструктура, lazy-импорт ML (`_lazy.py`), `SecretStr` для токенов,
валидация `session_name` против path-traversal, subprocess без `shell=True`,
fallback-цепочка LLM, покрытие ≥85%, лимит 500 строк/файл с реальной декомпозицией.

---

## Roadmap (по приоритету)

1. ✅ **Тайминги стадий (P1)** — `_timing.py` + обёртки в pipeline (0.9.6).
2. ✅ Убрать мёртвый `mono_16k`-проход в `merge_channels` (§1.2) — `merge_channels` теперь возвращает только stereo (0.9.6).
3. ✅ Сузить CPU-fallback до реальных CUDA/OOM-ошибок (P0) — общий `is_cuda_error` в `_gpu.py`, применён в `transcriber.py` и `diarizer.py` (0.9.6).
4. ✅ `local_files_only` (offline-first, fallback на скачивание) в `transcriber._new_model` (0.9.6). Тёплая модель для трея НЕ делается — конфликт с выгрузкой Whisper перед diarize на 4 GB VRAM; возможна как opt-in позже.
5. `BatchedInferencePipeline` и/или дефолт `beam_size` (§1.4–1.5) — после замеров из №1.
6. Раздельные тайминги mic/monitor + (опц.) пропуск near-silent mic-прохода (§1.1).

Каждый пункт — отдельный коммит, под свой failing-тест (bug-fix workflow проекта).

---

## Детальный план задачи №1 — тайминги стадий

### Цель
Логировать длительность каждой тяжёлой стадии пайплайна, чтобы пользователь видел,
куда уходит время (подтвердить/опровергнуть ранжирование Части 1 цифрами).

### Дизайн (система типов и тестируемость — по правилам проекта)
Новый приватный модуль `src/tapeback/_timing.py` (как `_gpu.py`, `_lazy.py`) —
чистый, тестируется без UI/D-Bus/network:
- `format_stage_duration(stage: str, seconds: float) -> str` — чистая функция формата.
- `stage_timer(stage, report, *, clock=time.monotonic)` — контекстменеджер, мерит
  wall-clock и репортит строку через `report` (= существующий `StatusCallback`).
  Репорт в `finally` → даже упавшая стадия покажет, сколько шла до падения.

Почему через `on_status`, а не через `logging`: это текущий канал прогресса пайплайна.
В CLI on_status = `click.echo(err=True)` (видно), в трее = `logger.info` (видно в логах).
Единое логирование — отдельная P1-задача, не смешиваем здесь два sink.

### Стадии под тайминг (pipeline.py)
- `merge` — `merge_channels` в `stop_and_process`
- `split` — `split_channels_16k` в `process_stereo_file`
- `convert` — `convert_to_mono16k` в `process_mono_file`
- `load model` — `load_transcriber` (холодный старт ~10с — важно показать)
- `transcribe` — `transcribe_stereo` / `transcribe` (раздельный mic/monitor — пункт №6)
- `diarize` — блок Diarizer
- `summarize` — `_maybe_summarize`

### Файлы
| Файл | Что |
|---|---|
| `src/tapeback/_timing.py` | **новый** — `format_stage_duration` + `stage_timer` |
| `src/tapeback/pipeline.py` | обернуть стадии в `with stage_timer(...)` |
| `tests/test_timing.py` | **новый** — юнит-тесты чистых функций |

### Тесты (хардкод ожидаемых значений)
- `test_format_stage_duration` → assert `== "Stage 'transcribe' took 12.3s"`
- `test_stage_timer_reports_elapsed` → fake clock, проверить переданную строку
- `test_stage_timer_reports_on_exception` → исключение внутри `with` репортится и пробрасывается

### Verification
`uv run ruff check --fix && uv run ruff format && uv run ty check && uv run pytest`.
Coverage ≥85%. CHANGELOG: новая patch-секция (проверить `git tag` первым).

### Что НЕ делаем в этой задаче
- Не переводим `print` → `logging` (отдельная P1).
- Не разносим mic/monitor (пункт №6 roadmap).
- Не трогаем дефолты модели/beam_size (пункт №5, нужен E2E).