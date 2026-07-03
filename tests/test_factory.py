"""Shared test factories for validate_corpus tests.

Single source of truth for record generation — used by both
deterministic pytest tests and Hypothesis property-based tests.
"""
from pathlib import Path

from fishingllm.validate_corpus import (
    check_record, ACCEPTED_SCHEMA_VERSION,
    VALID_TYPES, VALID_STATUSES, VALID_EVIDENCE_ROLES,
    VALID_CAUSAL_RELATIONS, VALID_RELATION_TYPES, VALID_ACTION_TYPES,
    VALID_FAILURE_REASONS, VALID_OPERATORS,
)


# ── Helpers ──

def codes(errors: list[str]) -> set[str]:
    return {e.split("]")[0].lstrip("[") if "]" in e else e for e in errors}


def run_check(rec: dict, ctx: dict | None = None) -> list[str]:
    return check_record(rec, ctx or {})


# ── Record factory ──

def make(rtype: str, rid: str, *, mode: str = "valid", **overrides) -> dict:
    """Build a record with given type and id.

    Modes:
      "minimal" — only id/type/schema_version/id_prefix + overrides.
      "valid"   — full valid record with rich defaults + overrides.
    """
    base = {
        "id": rid,
        "type": rtype,
        "schema_version": ACCEPTED_SCHEMA_VERSION,
        "id_prefix": rid.split("_")[0] if "_" in rid else rid[:4],
    }
    if mode == "minimal":
        return {**base, **overrides}

    defaults = {
        "raw_observation": {"text": "Был на рыбалке. Поймал судака."},
        "session": {"water_body_id": "wb_volga_upper",
                    "datetime_start": "2024-06-15T05:00:00",
                    "datetime_end": "2024-06-15T12:00:00"},
        "observation": {
            "conditions": {"depth": 5, "target_species": "судак"},
            "effort": {"hours_fished": 3.0, "casts": 40},
            "result": {"catch": [{"fish": "судак", "count": 1}]},
            "missing_factors": [],
            "observation_quality": 0.7,
        },
        "environment": {"session_id": "ses00001", "datetime": "2024-06-15T05:30:00",
                        "source": "angler", "water_temp": 18, "pressure_hpa": 748},
        "hypothesis": {
            "claim": "t > 20 снижает активность", "confidence": 0.6,
            "formal_rule": {"variable": "water_temp", "operator": ">", "value": 20},
        },
        "evidence_synthesis": {
            "evidence": [{"id": "obs00001", "role": "supports", "weight": 0.8}],
            "summary": "Тест",
        },
        "analysis": {
            "status": "supported", "analysis": "Тест",
            "analysis_confidence": 0.8, "evidence_strength": "низкая",
            "causal_links": [], "alternative_explanations": [],
        },
        "recommendation": {
            "text": "Лови утром", "target_species": "судак",
            "actions": [{"type": "change_time", "value": "утро"}],
        },
        "outcome": {
            "execution_match": 0.8, "failure_reason": "none",
            "result": {"catch": [], "success": True},
        },
        "relation": {
            "from": "obs00001", "to": "hyp00001",
            "relation": "supports", "weight": 0.7,
        },
        "dialogue": {
            "assistant_confidence": 0.7,
            "messages": [{"role": "user", "content": "Привет"},
                         {"role": "assistant", "content": "Здравствуйте"}],
        },
        "fact": {"content": "Факт", "source_reliability": 0.8},
        "experience": {"content": "Опыт", "source_reliability": 0.6},
        "insufficient_data": {
            "query": "Где ловить?", "analysis": "Слишком общий вопрос",
            "status": "insufficient_data", "recommendation": {"text": "Уточните"},
        },
        "uncertainty": {
            "analysis": "Противоречие", "status": "contradictory",
            "causal_links": [], "recommendation": {"text": "Нет данных"},
            "analysis_confidence": 0.2, "evidence_strength": "низкая",
        },
    }
    return {**base, **defaults.get(rtype, {}), **overrides}


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
