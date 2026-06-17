## Раздражения (где Claude ошибся)

- **Галлюцинация SHA для GitHub Action** (сессия 2026-05-21, .deb e2e workflow). Я выдумал `actions/download-artifact@d3f86a106a0bac45b6d35f1fd534f73aa55a4fc0` — первые 16 символов совпали с реальным `v4.3.0`, остальные 24 — мусор. CI упал на всей matrix. Фикс задним числом: `scripts/check-workflow-pins.py` + step в `ci.yml`, опрашивает GitHub API. Кандидат в CLAUDE.md: «Перед записью SHA для `uses: owner/repo@<sha>` — обязательно проверить через `curl https://api.github.com/repos/<owner>/<repo>/commits/<sha>` или взять из существующего workflow в репо. Не сочинять».

- **Bundled-python через `uv python install` ломался на CI runner** (та же сессия). Layout `uv python find` оказался разным между моей машиной и ubuntu-24.04 runner — `cp -a` копировал «не то», в .deb попадал битый дереву python без `/opt/tapeback/python/bin/python3.13`. Потратили 3 round-trip с диагностиками. Кандидат в CLAUDE.md: «Для bundled-binaries в дистрибутивах предпочитать download-by-URL (`curl <pinned-tarball-url>`) над download-via-tool. URL детерминирован, layout архива гарантирован, не зависит от версии менеджера».

## Что зашло (паттерны, которые работают)

- **CHANGELOG — кратко, без issue/PR ссылок и attribution** (правка [0.9.5] на 2026-05-21). Я писал «Reported by @doonto on [#3](...)» и «Per the structural feedback on [#3](...): runtime warnings should not assume one distro» — пользователь оба раза удалил. Кандидат в CLAUDE.md → раздел «Versioning & releases»: «CHANGELOG entries: brief, focus on user-visible effect + one-line "why"; cross-references (`#N`, `@user`, "reported by") belong in PR description and GitHub release notes, не в changelog».

- **Вынос pure-функций в отдельные `_`-модули для тестируемости без зависимостей**. В этой сессии: `build_layout` в `_dbusmenu.py` и `detect_tray_env` в `_tray_env.py` тестируются без D-Bus session, без display, без pystray. Тесты гоняются в headless docker без проблем. Кандидат в CLAUDE.md (раздел Testing или Architecture): «Логику, которая не требует UI/D-Bus/network, выносить в отдельный модуль без этих импортов. Тесты тогда не нуждаются в фикстурах с моками всей системы».

- **Крупное ревью → спека в `plans/` → пункты отдельными коммитами под failing-тест** (сессия 2026-06-17, архитектурное ревью + тайминги стадий). Сохранил ревью как `plans/tapeback-architecture-review.md` (findings + roadmap из 6 пунктов), затем взялся за №1 (тайминги) по bug-fix workflow: тест первым (упал `ModuleNotFoundError`), потом реализация. Зашло: ревью не теряется между сессиями, объём каждого коммита под контролем (≤500 строк), приоритеты явные. Кандидат в CLAUDE.md (раздел про планирование/Before finishing): «Результат ревью/анализа на >2 шага — сохранять как спеку в `plans/` с приоритизированным roadmap; реализовывать по одному пункту за коммит».

## Сомнения (не уверен, стоит ли это правило)

- **Hooks для авто-форматирования после Edit/Write**. В отчёте insights советуют `PostToolUse` hook на `ruff format + ruff check --fix`. Соблазнительно — закрывает «забыл прогнать линтер». Минус: лишний шум при каждом edit, может конфликтовать с моими ручными правками формата. Стоит ли пробовать или оставить как сейчас (ручной `uv run ruff check --fix` перед коммитом)?
