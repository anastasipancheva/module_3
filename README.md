**Панчева Анастасия, группа 972403**

> MVP агента-коуча с долгосрочной памятью, двумя персонами, изоляцией тенантов и управляемым забыванием.

Design-doc: [docs/ml_system_design_doc.md](docs/ml_system_design_doc.md)

---

## Быстрый старт

### Требования
- Python 3.12+, [uv](https://github.com/astral-sh/uv)
- OpenRouter API key (бесплатный тир): <https://openrouter.ai>

### 1. Установка
```bash
git clone <repo-url> && cd module_3_NLP_experimental
cp .env.example .env          # вставить OPENROUTER_API_KEY
uv sync                       # установить зависимости
```

### 2. Запустить веб-интерфейс

```bash
uv run streamlit run src/app.py
# → открыть http://localhost:8501
```

### 3. Запустить через Docker (одна команда)

```bash
# Создать .env с ключом, затем:
docker build -t mindly .
docker run -p 8501:8501 --env-file .env -v $(pwd)/data:/app/data mindly
# → http://localhost:8501
```

---

## Как пользоваться

| Действие | Как |
|---|---|
| Сменить пользователя | Поле **User ID** в сайдбаре |
| Сменить персону | Дропдаун **Persona** |
| Удалить всю память | Кнопка **Delete ALL my memory** |
| Удалить конкретную тему | Поле **Forget specific topic…** + кнопка |
| Посмотреть текущие факты | Кнопка **Refresh memory list** |

### Смена персон с сохранением памяти
Память хранится по `user_id`, **не** по персоне. Переключите дропдаун — тот же пользователь, другой тон. Проверьте: скажите агенту свой вес через `wellness-friend`, смените на `tough-love` и спросите про вес — он вспомнит.

### Управляемое забывание
- **Удалить всё**: кнопка "Delete ALL my memory".
- **Удалить конкретную тему**: поле "Forget specific topic…" → введите "вес" → "Forget this topic".

---

## Изоляция тенантов
```bash
uv run python scripts/test_isolation.py
```
Скрипт создаёт двух пользователей (`test_alice_isolation`, `test_bob_isolation`), вводит секрет от Алисы, затем проверяет, что Боб не видит её данные. Выход `0` = OK.

---

## Бенчмарк и эксперимент-трекинг

### Запустить оценку
```bash
# запуск с логированием в W&B
uv run python scripts/eval_benchmark.py
```

### Результаты: Mini LongMemEval

| Параметр | Значение |
|---|---|
| **Benchmark** | Mini LongMemEval (25 QA-пар в стиле Wu et al., 2024) |
| **Метрика** | Soft Exact Match (EM): ответ засчитан, если ключевое слово/число встречается в тексте ответа |
| **Target Score (из ТЗ)** | > 50% EM |
| **Наш результат (прогон 1, 2026-05-04)** | 100% EM (25/25) — [W&B Run](https://wandb.ai/anastasipancheva-tsu/mindly-eval/runs/s0e68pfq) |
| **Наш результат (прогон 2, 2026-06-12)** | **96% EM (24/25)** |
| **Avg Latency** | **2264 ms** |

**Честный анализ результата и ограничений метрики:**

96–100% soft EM — хороший результат для **немедленного recall** (факт введён → тут же спрошен). Но важно понимать три ограничения:

**1. Метрика soft EM слишком мягкая для коротких чисел.**
Единственный провальный кейс: вопрос "How often do I go to the gym?" — ожидалось `"3"`, агент ответил *"three times a week"* (слово вместо цифры). Память сработала корректно, провалилась метрика. Если бы expected было `"three"` — кейс прошёл бы. Аналогично `"1"`, `"5"`, `"6"` как ключевые слова могут случайно совпасть с другим числом в ответе.

**2. Тест не cross-session по-настоящему.**
В eval_benchmark.py все кейсы запущены в одном Python-процессе. Факты сохраняются в ChromaDB и персистентны между реальными перезапусками (что показано в demo-видео), но сам eval не перезапускает процесс между inject и recall.

**3. Расстояние между inject и recall — 1–2 хода.**
Реальный LongMemEval (Wu et al.) тестирует recall спустя сотни ходов и дней. Наш mini-набор — это проверка «ближней памяти» (факт ввели → сразу спросили). Для MVP-демо этого достаточно; для production нужен тест с многодневным gap'ом.

---

## Архитектура памяти
```
Пользователь вводит сообщение
       │
       ▼
┌─────────────────────────────────────────┐
│            MindlyAgent.chat()           │
│                                         │
│  1. mem0.search(query, user_id)         │◄── ChromaDB (векторный индекс)
│     → top-10 релевантных фактов         │    all-MiniLM-L6-v2 embeddings
│                                         │
│  2. Сборка system prompt:               │
│     [Persona prompt] + [Known facts]    │
│                                         │
│  3. OpenAI-compatible API (OpenRouter)  │
│     → streaming ответ                   │
│                                         │
│  4. mem0.add(interaction, user_id)      │──► Fact extraction LLM
│     ← извлечённые факты в ChromaDB      │    (gpt-4o-mini via OpenRouter)
└─────────────────────────────────────────┘
```

---

## Логирование
Все события записываются в `data/log_file.log`:
- `memory_retrieval` — сколько фактов найдено, за сколько мс.
- `stream_done` / `generate_done` — TTFT, общее время, длина ответа.
- `memory_saved` — факт записан в ChromaDB.
- `eval_done` — результаты бенчмарка.

---

## Самооценка по критериям (self-grading)

### Part 1 — Design Document (40 баллов)

| Секция | Оценка | Баллы | Обоснование |
|---|---|---|---|
| 1.1 Зачем идем в разработку | Full | **2/2** | Unit-экономика с числами (12% vs 18–22% Retention), проблема клиента дословно, memory как рычаг |
| 1.2 Бизнес-требования | Full | **3/3** | 10 требований в таблице, каждое tagged Stated/Inferred/Assumed, assumptions расписаны |
| 1.3 Скоуп проекта | Full | **2/2** | MVP vs Phase 2 явно разграничены, investor demo moment описан пошагово |
| 1.4 Предпосылки решения | Full | **2/2** | Данные (нет датасета), модели (LLM + embeddings 384d), вендоры (OpenRouter, Mem0, ChromaDB) |
| 2.1 Постановка задачи | Full | **2/2** | I/O формализованы, Memory и Proactive Recall операционально определены, tenant isolation определён |
| 2.2 Блок-схема решения | Full | **3/3** | ASCII-диаграмма с memory layer в центре, trade-off таблица 5 вариантов, защита выбора Mem0 с 4 причинами |
| 2.3 Этапы решения | Full | **2/2** | 4-недельный план, конкретные даты и артефакты каждого этапа |
| 3.1 Способ оценки пилота | Full | **3/3** | Mini LongMemEval (Wu et al. 2024), Soft EM с формулой, 3 baseline'а (5%/35%/60-68%), честный scope |
| 3.2 Успешный пилот | Full | **2/2** | Два слоя: продуктовый (инвестор видит recall) + инженерный (EM>50%, TTFT<3s, 0 leaks) |
| 3.3 Подготовка пилота | Full | **2/2** | 2 test-пользователя, 6-пунктовый чеклист, пошаговый скрипт 30 мая, backup-план |
| 4.1 Архитектура решения | Full | **3/3** | Все компоненты + обоснование; ChromaDB SQLite = KV/SQL; OpenAI SDK streaming = gateway |
| 4.2 Инфраструктура | Full | **2/2** | Demo sizing (CPU/8GB) + 10k MAU (Pinecone/k8s) + 3 bottleneck'а |
| 4.3 Требования к работе | Full | **2/2** | Таблица p50/p95 для TTFT/retrieval/write, availability 99%, throughput 10 users — числами |
| 4.4 Безопасность системы | Full | **2/2** | Auth + rate limiting + prompt-injection threat model (system/user isolation) + abuse handling |
| 4.5 Безопасность данных | Full | **3/3** | Tenant isolation + GDPR Art.17 + шифрование (MVP plaintext → prod LUKS) + retention 180d |
| 4.6 Издержки | Full | **2/2** | Math-таблица $0.05/user/мес, 10k MAU = $650/мес, hosted vs self-hosted сравнение |
| 4.7 Integration points | Full | **1/1** | Конкретный API (user_id, chat(), forget()), deletion webhook |
| 4.8 Риски | Full | **2/2** | 5 рисков с probability/impact/mitigation, R1 = tenant leakage |
| **ИТОГО Part 1** | | **40/40** | |

### Part 2 — MVP (60 баллов)

| Критерий | Баллы | Статус | Примечание |
|---|---|---|---|
| Cross-session recall (16) | **12–14/16** | ✅ | ChromaDB persistent, demo-видео записано |
| Memory non-trivial (8) | **8/8** | ✅ | Mem0 fact extraction, semantic search, стратегия видна в коде |
| Tenant isolation (6) | **6/6** | ✅ | `test_isolation.py` с exit code, два пользователя, leak check |
| User-controlled forgetting (6) | **6/6** | ✅ | Targeted (search→delete по ID) + full delete_all, оба в UI |
| 2 personas + shared memory (4) | **4/4** | ✅ | Разные system prompts, память по user_id (не по персоне) |
| Streaming (4) | **4/4** | ✅ | `_stream_response()`, TTFT измерен и показан в UI |
| Benchmark + defended (6) | **5/6** | ✅ | 96% soft EM (24/25, прогон 2); честный анализ ограничений метрики |
| Logging + Docker + CLI (4) | **4/4** | ✅ | Loguru полный pipeline, `.env.example`, Dockerfile, `main.py` CLI |
| Experiment tracking W&B (3) | **3/3** | ✅ | [W&B run](https://wandb.ai/anastasipancheva-tsu/mindly-eval/runs/s0e68pfq) с config, metrics, eval_table |
| Git + README (3) | **2–3/3** | ✅ | Две ветки, 8+ коммитов, README с видео и честным анализом |
| **ИТОГО Part 2** | **~54–57/60** | | |

### Итоговая оценка: ~94–97/100
---

## Структура проекта
```
module_3_NLP_experimental/
├── src/
│   ├── agent.py          # MindlyAgent — основной класс
│   └── app.py            # Streamlit UI
├── scripts/
│   ├── eval_benchmark.py # Мини-бенчмарк + W&B
│   └── test_isolation.py # Проверка изоляции тенантов
├── docs/
│   └── ml_system_design_doc.md
├── data/                 # gitignored — логи и ChromaDB
├── Dockerfile
├── pyproject.toml
└── .env.example
```
