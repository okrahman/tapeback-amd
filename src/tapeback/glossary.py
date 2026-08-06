"""Default decoder-bias glossary (faster-whisper `hotwords`).

Whisper mangles English technical vocabulary spoken inside Russian sentences: it
transcribes what the phonemes sound like in the surrounding language, so "tapeback"
came back as "ты пупа ты бэк" and "Whisper" as "виспер". Biasing the decoder towards a
list of expected terms fixes most of it — measured on a 31-minute recording, this list
raised distinct English terms preserved in Latin script from 25 to 33 and cut
low-confidence words from 81.5 to 59.4 per 1000. It also *reduced* hallucinations
(2 → 0 across a benchmark grid), because a biased decoder wanders less.

Kept in its own module rather than in `settings.py` so it can grow without pushing that
file towards the 500-line limit, and so a personal glossary has an obvious place to live
later. Override the whole thing with `TAPEBACK_HOTWORDS`.

Keeping it useful:
- terms only in the language you want them *written* in — listing "Обсидиан" would bias
  towards the Cyrillic spelling, which is the failure being fixed
- prefer terms actually observed coming back mangled; ordinary English words Whisper
  already gets right only dilute the bias and eat the budget
- **there is a hard budget.** faster-whisper truncates hotwords to
  `max_length // 2 - 1` = 223 tokens (transcribe.py) and says nothing about it, so
  everything past that silently stops working. A first draft of this file measured 295
  tokens — a quarter of it was dead weight. Measure, do not estimate: at ~3.3
  characters per token for this content, 223 tokens is roughly 730 characters.
"""

# Grouped by domain purely for editing convenience — the runtime sees one flat string.
# Every entry is either a term observed mangled in a real transcript or a proper noun
# with no Russian spelling for Whisper to fall back on. Ordinary English words are
# deliberately absent: Whisper already gets them right, and each one spends budget.
#
# Harvested from a corpus of the project's own recordings. Terms added because the
# transcripts showed them breaking:
#   "Layout, MV3"   -> LayoutLMv3
#   "Onyx, Runtime" -> ONNX Runtime   (plain "ONNX" was already listed and still split)
#   "RebitMQ"       -> RabbitMQ       (alongside a correct "RabbitMQ" from one speaker)
#   "OpenSWIFO"     -> OpenVINO
#   "QVN, last QVN" -> Qwen
#
# `docTR` was tried and removed: it was measured still coming back as "Dr. OCR" with the
# glossary in place — the hotword lost to the phonetic match with an ordinary English
# word. A term that does not work is not free, it spends budget (see below).
_AI_ML = (
    "LLM, RAG, Whisper, embeddings, vector search, retrieval, agent, MCP, ONNX, "
    "ONNX Runtime, OpenVINO, PyTorch, TorchScript, LayoutLMv3, Tesseract, "
    "Hugging Face, Qwen, OCR, CUDA, GPU, VRAM, diarization"
)

_TOOLS = (
    "tapeback, Obsidian, Jira, Trello, Notion, Slack, Telegram, Excalidraw, GitHub, "
    "Docker, Postgres, RabbitMQ, Sentry, Playwright, Jitsi, Claude, Copilot, "
    "Anthropic, OpenAI, Cloudflare"
)

_PROCESS = (
    "backlog, roadmap, kanban, action items, stakeholder, dogfooding, MVP, deploy, production"
)

_ENGINEERING = (
    "API, REST, backend, frontend, pipeline, migration, CI, markdown, JSON, SQL, "
    "VPN, OAuth, Python, TypeScript, JavaScript, React, TDD"
)

DEFAULT_HOTWORDS = ", ".join((_AI_ML, _TOOLS, _PROCESS, _ENGINEERING))
