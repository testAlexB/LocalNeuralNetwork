# Спецификация датасета FishingLLM v1.3

## 1. Цель

Обучить модель рассуждать о рыбалке в Тверской области: анализировать факторы, выдвигать формальные гипотезы, агрегировать свидетельства, строить каузальные связи, давать машинно-исполнимые рекомендации, учиться на outcome.

## 2. Требования к модели (целевое поведение)

- Анализировать факторы (водоём, погода, сезон, давление, фаза луны, прозрачность воды, уровень воды, ветер, температура, время суток, тип дна, течение) и делать выводы
- Выдвигать формальные гипотезы: «water_temp > 20 → catch_rate снижается»
- Синтезировать противоречивые свидетельства с весами
- Различать корреляцию и причинность
- Строить явные каузальные связи с учётом confounders
- Давать машинно-исполнимые рекомендации (actions[])
- Признавать незнание и указывать причины неуспеха
- Вести диалог с указанием уверенности
- Генерировать JSON для агентского режима
- Быть вежливой, экспертной, но доступной

## 3. Архитектура пайплайна

```
Исходный документ (сырой текст)
      │
      ▼
Raw observation (дословно, НИКОГДА не изменяется)
      │
      ▼
Session — одна рыболовная поездка
      ├──────────────────┐
      ▼                  ▼
Environment         Observation
(ambient, source)   (conditions + effort + result)
      │                  │
      └─────┬────────────┘
            ▼
      Hypothesis (formal_rule, variable, operator, value)
            │
            ▼
      Evidence synthesis (evidence[] с weight)
            │
            ▼
      Analysis (status + causal_links + alternative_explanations)
            │
            ▼
      Recommendation (actions[] + based_on: [analysis, evidence_synthesis])
            │
            ▼
      Outcome (execution_match + failure_reason)
            │
      ┌─────┴─────┐
      ▼           ▼
  Relation     Dialogue
  (graph edge)
```

## 4. Принцип неизменности

```
raw_observation    — НИКОГДА не изменяется
session            — может пересоздаваться
environment        — может пересоздаваться
observation        — может пересоздаваться (только факты, не интерпретации)
hypothesis         — может удаляться
evidence_synthesis — может пересчитываться
analysis           — может пересчитываться
recommendation     — может пересоздаваться
outcome            — не изменяется (фактический результат)
relation           — может пересоздаваться (граф перестраивается)
dialogue           — может генерироваться заново
```

## 5. Строгие правила (инварианты)

### D1. Success только вычисляется
`success = (result.catch.length > 0)`. Поле `success` в observation запрещено.

### D2. Observation = факты, не интерпретации
Запрещено в observation: выводы, гипотезы, причинные объяснения, оценки «хорошо/плохо».

### D3. Hypothesis должна быть проверяемой
Обязательно: `formal_rule.variable`, `formal_rule.operator`, `formal_rule.value`.

### D4. Evidence всегда имеет weight
Каждый entry в `evidence[]` обязан содержать `weight` (0.0–1.0).

### D5. Recommendation всегда содержит human text и machine actions
Оба поля обязательны: `text` (для человека) и `actions[]` (для машины).

### D6. Outcome всегда содержит execution_match и failure_reason
`failure_reason` — всегда, даже при success (значение `"none"`).

### D7. Граф-совместимость
Любой `based_on` должен иметь альтернативное представление через Relation.

### D8. Все ID ссылаются в relation graph
Каждая запись (кроме raw_observation) должна быть узлом в relation graph — иметь минимум одну relation, где она выступает `from` или `to`.

### D9. Графовые ограничения
- Запрещены циклы в цепочках `derived_from`
- Запрещены relation на несуществующие id
- raw_observation — единственный тип без обязательной relation (корень графа)

## 6. Формат факторов

**Точное значение:**
```json
"water_temp": 18
```

**С неопределённостью:**
```json
"water_temp": {"value": 18, "estimated": true, "measurement_error": 2}
```

## 7. География

```json
{
  "water_body_id": "wb_seliger_001",
  "water_body_name": "Селигер",
  "type": "озеро",
  "gazetteer_version": "1.0",
  "aliases": ["оз.Селигер", "озеро Селигер"],
  "location": {"lat": 57.2, "lon": 33.1, "region": "Тверская область"},
  "spots": [
    {"spot_id": "wbs_seliger_001", "name": "остров Хачин"}
  ]
}
```

## 8. Словари

```
data/dictionaries/
  species.json           — виды рыб + алиасы
  bait.json              — наживки + алиасы
  tackle.json            — снасти + алиасы
  bottom_types.json      — типы дна
  weather_terms.json     — погодные термины
  failure_reasons.json   — таксономия причин провала
```

Каждый имеет `dictionary_version`.

`failure_reasons.json`:
```json
{
  "dictionary_version": "1.0",
  "failure_reasons": [
    "none", "conditions_not_met", "wrong_spot",
    "weather_changed", "equipment_difference",
    "low_activity", "unknown"
  ]
}
```

## 9. Provenance

```json
{
  "id": "prefix_00001",
  "type": "raw_observation|session|environment|observation|hypothesis|evidence_synthesis|analysis|recommendation|outcome|relation|dialogue|fact|experience|insufficient_data|uncertainty",
  "schema_version": "1.3",
  "source_document": {
    "type": "forum_post|book_page|article|interview",
    "title": "...", "url": "..."
  },
  "source_reliability": 0.0-1.0,
  "author": "...",
  "created_by": "human_annotator",
  "extractor_version": "v1.0",
  "dictionary_version": "1.0",
  "datetime": "2024-06-15T05:30:00",
  "date_processed": "2024-07-01"
}
```

### Шкалы

**source_reliability:**
| 1.0 | Наука | 0.8 | Проф.книги | 0.6 | Опытные форум | 0.4 | Единичные отчёты | 0.2 | Слухи |

**observation_quality:**
| 1.0 | Все факторы измерены | 0.7 | Большинство указано | 0.4 | Минимум | 0.1 | Почти без данных |

**analysis_confidence:**
| 0.9+ | Много наблюдений | 0.6–0.8 | Есть разброс | 0.3–0.5 | Мало/противоречиво | 0–0.2 | Единичные |

**evidence_strength:**
| высокая | 50+ набл., неск. источников | средняя | 10–50, 2+ ист. | низкая | <10 или 1 ист. |

## 10. Типы записей

### raw_observation

```json
{
  "id": "raw00001",
  "type": "raw_observation",
  "schema_version": "1.3",
  "text": "Сегодня с утра был на Волге у Хотина. Ловил на джиг. Поймал двух судаков, одного окуня. Вода градусов 18, давление вроде нормальное.",
  "source_document": {"type": "forum_post", "title": "Отчёт 15.06.2024"},
  "source_reliability": 0.4,
  "author": "pike_hunter_69",
  "datetime": "2024-06-15T05:30:00",
  "id_prefix": "raw"
}
```

### session

```json
{
  "id": "ses00001",
  "type": "session",
  "schema_version": "1.3",
  "based_on_raw": ["raw00001"],
  "water_body_id": "wb_volga_upper",
  "water_body_name": "Верхняя Волга",
  "datetime_start": "2024-06-15T05:00:00",
  "datetime_end": "2024-06-15T14:00:00",
  "season": "июнь",
  "source_reliability": 0.4,
  "observation_quality": 0.7,
  "extractor_version": "v1.0",
  "dictionary_version": "1.0",
  "id_prefix": "ses"
}
```

### environment

```json
{
  "id": "env00001",
  "type": "environment",
  "schema_version": "1.3",
  "session_id": "ses00001",
  "datetime": "2024-06-15T05:30:00",
  "source": "angler",
  "air_temp": {"value": 22, "estimated": true, "measurement_error": 3},
  "water_temp": 18,
  "pressure_hpa": 748,
  "pressure_trend": "стабильное",
  "wind_speed": 3,
  "wind_direction": "СЗ",
  "moon_phase": "полнолуние",
  "clouds": "ясно",
  "precipitation": "нет",
  "visibility": "хорошая",
  "water_level": "нормальный",
  "water_clarity": "прозрачная",
  "id_prefix": "env"
}
```

### observation — факты, без интерпретаций

`success` отсутствует (вычисляется). `effort` обязателен.

```json
{
  "id": "obs00001",
  "type": "observation",
  "schema_version": "1.3",
  "session_id": "ses00001",
  "environment_id": "env00001",
  "time_of_day": "утро",
  "datetime": "2024-06-15T05:30:00",
  "conditions": {
    "depth": 5,
    "bottom_type": "каменистое",
    "current": "слабое",
    "tackle": "джиг 12г",
    "target_species": "судак"
  },
  "effort": {
    "hours_fished": 3.5,
    "casts": 45,
    "distance_km": null
  },
  "result": {
    "catch": [
      {"fish": "судак", "count": 2, "weight_kg": null},
      {"fish": "окунь", "count": 1, "weight_kg": null}
    ]
  },
  "missing_factors": ["wind_speed", "moon_phase"],
  "observation_quality": 0.7,
  "id_prefix": "obs"
}
```

Неуспешное наблюдение (success = false, т.к. catch пуст):

```json
{
  "id": "obs00002",
  "type": "observation",
  "schema_version": "1.3",
  "session_id": "ses00001",
  "environment_id": "env00001",
  "time_of_day": "день",
  "datetime": "2024-06-15T12:00:00",
  "conditions": {
    "depth": 5,
    "target_species": "судак"
  },
  "effort": {
    "hours_fished": 2.0,
    "casts": 30,
    "distance_km": null
  },
  "result": {
    "catch": []
  },
  "missing_factors": ["bottom_type", "current", "tackle"],
  "observation_quality": 0.4,
  "id_prefix": "obs"
}
```

### hypothesis — формальная, проверяемая

```json
{
  "id": "hyp00001",
  "type": "hypothesis",
  "schema_version": "1.3",
  "based_on": ["obs00001", "obs00002"],
  "claim": "Повышение температуры воды выше 20°C снижает активность судака",
  "direction": "decrease",
  "formal_rule": {
    "variable": "water_temp",
    "operator": ">",
    "value": 20
  },
  "hypothesis_text": "Если температура воды >20°C, судак перестаёт активно кормиться.",
  "confidence": 0.6,
  "id_prefix": "hyp"
}
```

### evidence_synthesis — свидетельства с весами

```json
{
  "id": "evs00001",
  "type": "evidence_synthesis",
  "schema_version": "1.3",
  "based_on": ["hyp00001"],
  "question": "Как температура воды влияет на активность судака?",
  "evidence": [
    {"id": "obs00001", "role": "supports", "weight": 0.8, "note": "Улов при 18°C"},
    {"id": "obs00002", "role": "supports", "weight": 0.6, "note": "Нет улова при 20°C"},
    {"id": "obs00004", "role": "supports", "weight": 0.7, "note": "Улов при 16°C"},
    {"id": "obs00005", "role": "contradicts", "weight": 0.4, "note": "Улов при 22°C, глуб.8м"},
    {"id": "fac00001", "role": "background", "weight": 0.9, "note": "Судак предпочитает 4-8м"}
  ],
  "summary": "9 из 12 наблюдений подтверждают. 3 contradict — все на глубине >7м.",
  "id_prefix": "evs"
}
```

### analysis

```json
{
  "id": "ana00001",
  "type": "analysis",
  "schema_version": "1.3",
  "based_on": ["evs00001", "fac00001"],
  "status": "supported",
  "analysis": "1. Гипотеза: water_temp > 20 → catch_rate снижается.\n2. Evidence: 9 supports, 3 contradicts (глубина >7м).\n3. Вывод: подтверждено для глубин до 7м.",
  "analysis_confidence": 0.85,
  "evidence_strength": "средняя",
  "alternative_explanations": [
    "освещённость (солнце в зените)",
    "активность кормовой базы (вылет насекомых)",
    "давление (снижалось к обеду)"
  ],
  "causal_links": [
    {
      "factor": "water_temp",
      "effect": "catch_rate",
      "relation": "negative",
      "confidence": 0.85,
      "confounders": ["depth", "wind"]
    }
  ],
  "id_prefix": "ana"
}
```

Отклонённая гипотеза (rejected, ≥30% от всех analysis):

```json
{
  "id": "ana00002",
  "type": "analysis",
  "schema_version": "1.3",
  "status": "rejected",
  "analysis": "Гипотеза 'moon_phase → catch_rate' не подтверждена. 8 наблюдений: полнолуние 4 улова, новолуние 4 улова. Разница в пределах шума.",
  "analysis_confidence": 0.95,
  "evidence_strength": "средняя",
  "alternative_explanations": [],
  "causal_links": [],
  "id_prefix": "ana"
}
```

### recommendation

`based_on` включает и analysis, и evidence_synthesis.

```json
{
  "id": "rec00001",
  "type": "recommendation",
  "schema_version": "1.3",
  "based_on": ["ana00001", "evs00001"],
  "analysis_version": "1.0",
  "target_species": "судак",
  "recommended_depth": [4, 7],
  "recommended_tackle": "джиг",
  "recommended_lure": "12-15г",
  "best_time": "утро",
  "conditions": {
    "water_temp_max": 20,
    "bottom_type": "каменистое",
    "current": "слабое"
  },
  "text": "На Верхней Волге в июне судак стабильно берёт утром на джиг 12-15г на глубине 4-7м с каменистым дном при t воды до 20°C.",
  "actions": [
    {"type": "change_depth", "value": [4, 7]},
    {"type": "change_lure", "value": "джиг 12-15г"},
    {"type": "change_time", "value": "утро"}
  ],
  "id_prefix": "rec"
}
```

**Allowed actions (DSL contract):**

| type | value |
|------|-------|
| `change_depth` | `[min, max]` |
| `change_lure` | строка |
| `change_tackle` | строка |
| `change_time` | строка |
| `move_spot` | строка (spot_id или описание) |
| `change_bait` | строка |

`text` обязателен для человека, `actions` — для машины. Оба слоя всегда присутствуют и не противоречат друг другу.

### outcome

Всегда содержит `execution_match` и `failure_reason`.

```json
{
  "id": "out00001",
  "type": "outcome",
  "schema_version": "1.3",
  "based_on": ["rec00001", "obs00010"],
  "execution_match": 0.9,
  "failure_reason": "none",
  "conditions_note": "Глубина 5м вместо 6м, остальное совпало",
  "result": {
    "success": true,
    "catch": [{"fish": "судак", "count": 4, "weight_kg": 12.5}],
    "notes": "Рекомендация сработала"
  },
  "confidence": 0.9,
  "id_prefix": "out"
}
```

Неуспех из-за несоответствия условий:

```json
{
  "id": "out00002",
  "type": "outcome",
  "schema_version": "1.3",
  "based_on": ["rec00001", "obs00011"],
  "execution_match": 0.3,
  "failure_reason": "conditions_not_met",
  "conditions_note": "t воды 23°C, илистое дно",
  "result": {
    "success": false,
    "catch": [],
    "notes": "Условия не соответствуют рекомендации"
  },
  "confidence": 0.7,
  "id_prefix": "out"
}
```

### relation — граф отношений

Дополняет `based_on`. Позволяет строить typed-граф.

```json
{
  "id": "rel00001",
  "type": "relation",
  "schema_version": "1.3",
  "from": "obs00001",
  "to": "hyp00001",
  "relation": "supports",
  "weight": 0.8,
  "id_prefix": "rel"
}
```

```json
{
  "id": "rel00002",
  "type": "relation",
  "schema_version": "1.3",
  "from": "ana00001",
  "to": "rec00001",
  "relation": "derived_from",
  "weight": 1.0,
  "id_prefix": "rel"
}
```

### fact

```json
{
  "id": "fac00001",
  "type": "fact",
  "schema_version": "1.3",
  "content": "Судак на Селигере предпочитает глубины 4-8 м с каменистым дном.",
  "source_reliability": 0.9,
  "source_document": {"type": "scientific_article", "title": "Ихтиология Верхней Волги"},
  "id_prefix": "fac"
}
```

### experience

```json
{
  "id": "exp00001",
  "type": "experience",
  "schema_version": "1.3",
  "content": "На Верхней Волге в июне щука хорошо берёт на блесну утром.",
  "source_reliability": 0.6,
  "id_prefix": "exp"
}
```

### insufficient_data

```json
{
  "id": "ins00001",
  "type": "insufficient_data",
  "schema_version": "1.3",
  "query": "Где ловить рыбу?",
  "analysis": "Вопрос слишком общий.",
  "status": "insufficient_data",
  "recommendation": {"text": "Уточните водоём, сезон и целевую рыбу."},
  "id_prefix": "ins"
}
```

### uncertainty

```json
{
  "id": "unc00001",
  "type": "uncertainty",
  "schema_version": "1.3",
  "based_on": ["obs00010", "obs00011"],
  "analysis": "На червя 3кг, на кукурузу 0. Через неделю — наоборот.",
  "status": "contradictory",
  "causal_links": [],
  "recommendation": {"text": "Попробуйте обе наживки."},
  "analysis_confidence": 0.2,
  "evidence_strength": "низкая",
  "id_prefix": "unc"
}
```

### dialogue

```json
{
  "id": "dia00001",
  "type": "dialogue",
  "schema_version": "1.3",
  "based_on": ["rec00001"],
  "assistant_confidence": 0.7,
  "messages": [
    {"role": "user", "content": "Планирую на Селигер в июне за судаком."},
    {"role": "assistant", "content": "Рекомендую глубины 4-8 м, джиг 12-15 г, утро, t воды до 20°C."}
  ],
  "id_prefix": "dia"
}
```

## 11. Целевая структура

| Тип | Доля | Примечание |
|-----|------|-----------|
| analysis | ~18% | ≥30% rejected |
| evidence_synthesis | ~8% | — |
| hypothesis | ~8% | formal_rule обязателен |
| observation | ~15% | факты, без success |
| session | ~5% | — |
| environment | ~5% | source обязателен |
| raw_observation | ~3% | — |
| recommendation | ~5% | text + actions |
| outcome | ~5% | execution_match + failure_reason |
| relation | ~5% | граф |
| dialogue | ~12% | — |
| insufficient_data | ~5% | — |
| uncertainty | ~5% | — |
| fact | ~4% | — |
| experience | ~2% | — |

## 12. Замкнутый цикл

```
Session → Observation (факты + effort)
    ↓
Hypothesis (formal_rule)
    ↓
Evidence synthesis (evidence[].weight)
    ↓
Analysis (status + causal_links + alternative_explanations)
    ↓
Recommendation (actions[])
    ↓
Outcome (execution_match + failure_reason)
    ↓
Relation (граф) + новое наблюдение
```

## 13. Фильтрация

NFC, HTML, токсичность, дедупликация (exact → MinHash), <100 символов, lang=ru.

## 14. Валидация

- schema_version, UTF-8, NFC
- id уникален, type допустим
- based_on ссылается на существующие id
- status ∈ {supported, rejected, partially_supported, insufficient_data, contradictory}
- evidence[].role ∈ {supports, contradicts, background}
- evidence[].weight ∈ [0,1]
- formal_rule.variable, operator, value — обязательны для hypothesis
- recommendation.text и actions[] — оба обязательны
- outcome.execution_match ∈ [0,1], failure_reason — всегда
- session.water_body_id обязателен
- catch — массив
- datetime ISO 8601
- raw_observation неизменна (hash)

## 15. Структура на диске

```
data/
  raw/           forums/ books/ articles/
  water_bodies.json
  dictionaries/
    species.json  bait.json  tackle.json
    bottom_types.json  weather_terms.json  failure_reasons.json
  processed/
    raw_observations.jsonl
    sessions.jsonl
    environments.jsonl
    observations.jsonl
    hypotheses.jsonl
    evidence_syntheses.jsonl
    analyses.jsonl
    recommendations.jsonl
    outcomes.jsonl
    relations.jsonl
    facts.jsonl  experiences.jsonl
    uncertainties.jsonl  insufficient_data.jsonl
    dialogues.jsonl
  splits/        train.jsonl  valid.jsonl  test.jsonl
  benchmarks/
    reasoning.jsonl  planning.jsonl  memory.jsonl  retrieval.jsonl
    dialogue.jsonl  hallucination.jsonl  uncertainty.jsonl
    causal_reasoning.jsonl
```

## 16. Split

По source_document.

## 17. Первая итерация

1. 1 источник (1-2 поста форума)
2. 3-5 raw → 2 session + 5-8 observation (min 1 без улова) + 2 environment
3. 2 hypothesis (formal_rule) → 2 evidence_synthesis → 3 analysis (min 1 rejected)
4. 1 recommendation (actions[]) → 2 outcome (1 success + 1 failure_reason)
5. 1 insufficient_data + 2 relation + 2 dialogue
6. Валидация → токенизация → обучение → оценка
