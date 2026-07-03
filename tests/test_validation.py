import json
import unicodedata
from pathlib import Path
import pytest
from fishingllm.validate_corpus import validate_corpus, read_jsonl, CorpusError


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _make_valid_record(rtype: str, rid: str, **overrides) -> dict:
    base = {
        "id": rid,
        "type": rtype,
        "schema_version": "1.3",
        "id_prefix": rid.split("_")[0] if "_" in rid else rid[:4],
    }
    type_defaults = {
        "raw_observation": {"text": "Был на рыбалке. Поймал судака."},
        "session": {"water_body_id": "wb_volga_upper", "datetime_start": "2024-06-15T05:00:00", "datetime_end": "2024-06-15T12:00:00"},
        "observation": {
            "conditions": {"depth": 5, "target_species": "судак"},
            "effort": {"hours_fished": 3.0, "casts": 40},
            "result": {"catch": [{"fish": "судак", "count": 1}]},
            "missing_factors": [],
        },
        "environment": {"session_id": "ses00001", "datetime": "2024-06-15T05:30:00", "source": "angler", "water_temp": 18, "pressure_hpa": 748},
        "hypothesis": {
            "claim": "t > 20 снижает активность",
            "formal_rule": {"variable": "water_temp", "operator": ">", "value": 20},
            "confidence": 0.6,
        },
        "evidence_synthesis": {
            "evidence": [
                {"id": "obs00001", "role": "supports", "weight": 0.8},
            ],
            "summary": "Тест",
        },
        "analysis": {
            "status": "supported",
            "analysis": "Тест",
            "analysis_confidence": 0.8,
            "evidence_strength": "низкая",
            "causal_links": [],
            "alternative_explanations": [],
        },
        "recommendation": {
            "text": "Лови утром",
            "actions": [{"type": "change_time", "value": "утро"}],
            "target_species": "судак",
        },
        "outcome": {
            "execution_match": 0.8,
            "failure_reason": "none",
            "result": {"catch": [], "success": True},
        },
        "relation": {
            "from": "obs00001",
            "to": "hyp00001",
            "relation": "supports",
            "weight": 0.7,
        },
        "dialogue": {
            "assistant_confidence": 0.7,
            "messages": [
                {"role": "user", "content": "Привет"},
                {"role": "assistant", "content": "Здравствуйте"},
            ],
        },
        "fact": {"content": "Факт", "source_reliability": 0.8},
        "experience": {"content": "Опыт", "source_reliability": 0.6},
        "insufficient_data": {
            "query": "Где ловить?",
            "analysis": "Слишком общий вопрос",
            "status": "insufficient_data",
            "recommendation": {"text": "Уточните"},
        },
        "uncertainty": {
            "analysis": "Противоречие",
            "status": "contradictory",
            "causal_links": [],
            "recommendation": {"text": "Нет данных"},
            "analysis_confidence": 0.2,
            "evidence_strength": "низкая",
        },
    }
    result = {**base, **type_defaults.get(rtype, {}), **overrides}
    return result


def test_validate_real_corpus():
    errors = validate_corpus(DATA_DIR)
    assert len(errors) == 0, f"Corpus validation failed:\n" + "\n".join(errors)


def test_validate_observation_no_success():
    from fishingllm.validate_corpus import _validate_record
    rec = _make_valid_record("observation", "obs_test01", success=True)
    try:
        _validate_record(rec, "test.jsonl", "obs_test01")
        assert False, "Should reject observation with success field"
    except CorpusError:
        pass


def test_validate_hypothesis_no_formal_rule():
    from fishingllm.validate_corpus import _validate_record
    rec = _make_valid_record("hypothesis", "hyp_test01")
    rec.pop("formal_rule", None)
    try:
        _validate_record(rec, "test.jsonl", "hyp_test01")
        assert False, "Should reject hypothesis without formal_rule"
    except CorpusError:
        pass


def test_validate_recommendation_no_actions():
    from fishingllm.validate_corpus import _validate_record
    rec = _make_valid_record("recommendation", "rec_test01")
    rec["actions"] = []
    try:
        _validate_record(rec, "test.jsonl", "rec_test01")
        assert False, "Should reject recommendation without actions"
    except CorpusError:
        pass


def test_validate_outcome_no_execution_match():
    from fishingllm.validate_corpus import _validate_record
    rec = _make_valid_record("outcome", "out_test01")
    rec.pop("execution_match", None)
    try:
        _validate_record(rec, "test.jsonl", "out_test01")
        assert False, "Should reject outcome without execution_match"
    except CorpusError:
        pass


def test_validate_outcome_no_failure_reason():
    from fishingllm.validate_corpus import _validate_record
    rec = _make_valid_record("outcome", "out_test01")
    rec.pop("failure_reason", None)
    try:
        _validate_record(rec, "test.jsonl", "out_test01")
        assert False, "Should reject outcome without failure_reason"
    except CorpusError:
        pass


def test_validate_outcome_bad_failure_reason():
    from fishingllm.validate_corpus import _validate_record
    rec = _make_valid_record("outcome", "out_test01", failure_reason="invalid_reason")
    try:
        _validate_record(rec, "test.jsonl", "out_test01")
        assert False, "Should reject outcome with invalid failure_reason"
    except CorpusError:
        pass


def test_validate_relation_no_weight():
    from fishingllm.validate_corpus import _validate_record
    rec = _make_valid_record("relation", "rel_test01")
    rec.pop("weight", None)
    try:
        _validate_record(rec, "test.jsonl", "rel_test01")
        assert False, "Should reject relation without weight"
    except CorpusError:
        pass


def test_validate_duplicate_ids():
    import tempfile, os
    from fishingllm.validate_corpus import validate_corpus, read_jsonl
    with tempfile.TemporaryDirectory() as tmp:
        proc = Path(tmp) / "processed"
        proc.mkdir(parents=True)
        f = proc / "test.jsonl"
        with open(f, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(_make_valid_record("fact", "fac_dup01", content="A"), ensure_ascii=False) + "\n")
            fh.write(json.dumps(_make_valid_record("fact", "fac_dup01", content="B"), ensure_ascii=False) + "\n")
        errors = validate_corpus(tmp)
        dup_errors = [e for e in errors if "duplicate" in e]
        assert len(dup_errors) >= 1, "Should detect duplicate IDs"


def test_validate_invalid_relation_type():
    from fishingllm.validate_corpus import _validate_record
    rec = _make_valid_record("relation", "rel_test01", relation="invalid_type")
    try:
        _validate_record(rec, "test.jsonl", "rel_test01")
        assert False, "Should reject invalid relation type"
    except CorpusError:
        pass


def test_validate_session_no_water_body():
    from fishingllm.validate_corpus import _validate_record
    rec = _make_valid_record("session", "ses_test01")
    rec.pop("water_body_id", None)
    try:
        _validate_record(rec, "test.jsonl", "ses_test01")
        assert False, "Should reject session without water_body_id"
    except CorpusError:
        pass


def test_validate_nfc():
    from fishingllm.validate_corpus import check_nfc, CorpusError
    check_nfc("нормально", "test", 1)
    # "ё" as Cyrillic 'е' (0x0435) + combining diaeresis (0x0308) is non-NFC
    non_nfc = "е\u0308рш"  # decomposed ёрш
    if unicodedata.normalize("NFC", non_nfc) == non_nfc:
        pytest.skip("System already NFC-normalized this character")
    try:
        check_nfc(non_nfc, "test", 1)
        assert False, "Should reject non-NFC text"
    except CorpusError:
        pass
