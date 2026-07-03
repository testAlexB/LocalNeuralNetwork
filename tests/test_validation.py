import json
import tempfile
from pathlib import Path
import pytest
from fishingllm.validate_corpus import (
    validate_corpus, _validate_record, check_nfc,
    CorpusError, ACCEPTED_SCHEMA_VERSION,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _make_valid_record(rtype: str, rid: str, **overrides) -> dict:
    base = {
        "id": rid,
        "type": rtype,
        "schema_version": ACCEPTED_SCHEMA_VERSION,
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
            "observation_quality": 0.7,
        },
        "environment": {"session_id": "ses00001", "datetime": "2024-06-15T05:30:00", "source": "angler", "water_temp": 18, "pressure_hpa": 748},
        "hypothesis": {
            "claim": "t > 20 снижает активность",
            "formal_rule": {"variable": "water_temp", "operator": ">", "value": 20},
            "confidence": 0.6,
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
            "text": "Лови утром",
            "actions": [{"type": "change_time", "value": "утро"}],
            "target_species": "судак",
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
            "messages": [{"role": "user", "content": "Привет"}, {"role": "assistant", "content": "Здравствуйте"}],
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
    result = {**base, **type_defaults.get(rtype, {}), **overrides}
    return result


def validate_corpus_from_records(records: list[dict]) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        proc = Path(tmp) / "processed"
        proc.mkdir(parents=True)
        f = proc / "test.jsonl"
        with open(f, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return validate_corpus(tmp)


# ======================= Real corpus =======================

def test_validate_real_corpus():
    errors = validate_corpus(DATA_DIR)
    assert len(errors) == 0, "\n".join(errors)


# ======================= D1: success forbidden =======================

@pytest.mark.parametrize("val", [True, False])
def test_observation_success_forbidden(val):
    rec = _make_valid_record("observation", "obs_test01", success=val)
    with pytest.raises(CorpusError, match="success is forbidden"):
        _validate_record(rec, "test.jsonl", "obs_test01")


# ======================= Observation schema =======================

def test_observation_missing_effort():
    rec = _make_valid_record("observation", "obs_test01")
    rec.pop("effort", None)
    with pytest.raises(CorpusError, match="effort is required"):
        _validate_record(rec, "test.jsonl", "obs_test01")


def test_observation_missing_result():
    rec = _make_valid_record("observation", "obs_test01")
    rec.pop("result", None)
    with pytest.raises(CorpusError, match="result is required"):
        _validate_record(rec, "test.jsonl", "obs_test01")


def test_observation_catch_null():
    rec = _make_valid_record("observation", "obs_test01")
    rec["result"]["catch"] = None
    with pytest.raises(CorpusError, match="catch must be an array"):
        _validate_record(rec, "test.jsonl", "obs_test01")


def test_observation_catch_dict():
    rec = _make_valid_record("observation", "obs_test01")
    rec["result"]["catch"] = {"fish": "судак", "count": 1}
    with pytest.raises(CorpusError, match="catch must be an array"):
        _validate_record(rec, "test.jsonl", "obs_test01")


def test_observation_quality_low():
    rec = _make_valid_record("observation", "obs_test01", observation_quality=-0.1)
    with pytest.raises(CorpusError, match="observation_quality must be 0..1"):
        _validate_record(rec, "test.jsonl", "obs_test01")


def test_observation_quality_high():
    rec = _make_valid_record("observation", "obs_test01", observation_quality=1.5)
    with pytest.raises(CorpusError, match="observation_quality must be 0..1"):
        _validate_record(rec, "test.jsonl", "obs_test01")


# ======================= D3: formal_rule =======================

def test_hypothesis_no_formal_rule():
    rec = _make_valid_record("hypothesis", "hyp_test01")
    rec.pop("formal_rule", None)
    with pytest.raises(CorpusError, match="formal_rule is required"):
        _validate_record(rec, "test.jsonl", "hyp_test01")


def test_hypothesis_invalid_operator():
    rec = _make_valid_record("hypothesis", "hyp_test01")
    rec["formal_rule"]["operator"] = "foobar"
    with pytest.raises(CorpusError, match="operator.*foobar"):
        _validate_record(rec, "test.jsonl", "hyp_test01")


# ======================= D4: evidence =======================

def test_evidence_weight_out_of_range():
    rec = _make_valid_record("evidence_synthesis", "evs_test01")
    rec["evidence"] = [{"id": "obs00001", "role": "supports", "weight": 1.5}]
    with pytest.raises(CorpusError, match="weight must be 0..1"):
        _validate_record(rec, "test.jsonl", "evs_test01")


def test_evidence_weight_missing():
    rec = _make_valid_record("evidence_synthesis", "evs_test01")
    rec["evidence"] = [{"id": "obs00001", "role": "supports"}]
    with pytest.raises(CorpusError, match="weight must be 0..1"):
        _validate_record(rec, "test.jsonl", "evs_test01")


def test_evidence_invalid_role():
    rec = _make_valid_record("evidence_synthesis", "evs_test01")
    rec["evidence"] = [{"id": "obs00001", "role": "banana", "weight": 0.8}]
    with pytest.raises(CorpusError, match="role.*banana"):
        _validate_record(rec, "test.jsonl", "evs_test01")


# ======================= D5: recommendation =======================

def test_recommendation_no_actions():
    rec = _make_valid_record("recommendation", "rec_test01")
    rec["actions"] = []
    with pytest.raises(CorpusError, match="actions.*required"):
        _validate_record(rec, "test.jsonl", "rec_test01")


def test_recommendation_invalid_action_type():
    rec = _make_valid_record("recommendation", "rec_test01")
    rec["actions"] = [{"type": "fly_to_mars", "value": 123}]
    with pytest.raises(CorpusError, match="fly_to_mars"):
        _validate_record(rec, "test.jsonl", "rec_test01")


def test_recommendation_action_missing_value():
    rec = _make_valid_record("recommendation", "rec_test01")
    rec["actions"] = [{"type": "change_time"}]
    with pytest.raises(CorpusError, match="missing.*value"):
        _validate_record(rec, "test.jsonl", "rec_test01")


# ======================= D6: outcome =======================

def test_outcome_no_execution_match():
    rec = _make_valid_record("outcome", "out_test01")
    rec.pop("execution_match", None)
    with pytest.raises(CorpusError, match="execution_match must be 0..1"):
        _validate_record(rec, "test.jsonl", "out_test01")


def test_outcome_negative_execution_match():
    rec = _make_valid_record("outcome", "out_test01", execution_match=-0.1)
    with pytest.raises(CorpusError, match="execution_match must be 0..1"):
        _validate_record(rec, "test.jsonl", "out_test01")


def test_outcome_high_execution_match():
    rec = _make_valid_record("outcome", "out_test01", execution_match=1.4)
    with pytest.raises(CorpusError, match="execution_match must be 0..1"):
        _validate_record(rec, "test.jsonl", "out_test01")


def test_outcome_no_failure_reason():
    rec = _make_valid_record("outcome", "out_test01")
    rec.pop("failure_reason", None)
    with pytest.raises(CorpusError, match="failure_reason is required"):
        _validate_record(rec, "test.jsonl", "out_test01")


def test_outcome_bad_failure_reason():
    rec = _make_valid_record("outcome", "out_test01", failure_reason="invalid_reason")
    with pytest.raises(CorpusError, match="failure_reason.*not in dictionary"):
        _validate_record(rec, "test.jsonl", "out_test01")


# ======================= Relation =======================

def test_relation_no_weight():
    rec = _make_valid_record("relation", "rel_test01")
    rec.pop("weight", None)
    with pytest.raises(CorpusError, match="weight must be 0..1"):
        _validate_record(rec, "test.jsonl", "rel_test01")


def test_relation_invalid_type():
    rec = _make_valid_record("relation", "rel_test01", relation="invalid_type")
    with pytest.raises(CorpusError, match="relation type.*invalid"):
        _validate_record(rec, "test.jsonl", "rel_test01")


def test_relation_no_from():
    rec = _make_valid_record("relation", "rel_test01")
    rec.pop("from", None)
    with pytest.raises(CorpusError, match="relation.from"):
        _validate_record(rec, "test.jsonl", "rel_test01")


# ======================= Analysis =======================

def test_analysis_invalid_status():
    rec = _make_valid_record("analysis", "ana_test01", status="foobar")
    with pytest.raises(CorpusError, match="status.*foobar"):
        _validate_record(rec, "test.jsonl", "ana_test01")


def test_analysis_confidence_low():
    rec = _make_valid_record("analysis", "ana_test01", analysis_confidence=-0.1)
    with pytest.raises(CorpusError, match="analysis_confidence must be 0..1"):
        _validate_record(rec, "test.jsonl", "ana_test01")


def test_analysis_confidence_high():
    rec = _make_valid_record("analysis", "ana_test01", analysis_confidence=1.4)
    with pytest.raises(CorpusError, match="analysis_confidence must be 0..1"):
        _validate_record(rec, "test.jsonl", "ana_test01")


def test_causal_links_invalid_relation():
    rec = _make_valid_record("analysis", "ana_test01")
    rec["causal_links"] = [
        {"factor": "water_temp", "effect": "catch_rate", "relation": "banana", "confidence": 0.8}
    ]
    with pytest.raises(CorpusError, match="relation.*banana"):
        _validate_record(rec, "test.jsonl", "ana_test01")


def test_causal_links_missing_factor():
    rec = _make_valid_record("analysis", "ana_test01")
    rec["causal_links"] = [{"factor": "", "effect": "catch_rate"}]
    with pytest.raises(CorpusError, match="causal_link requires factor"):
        _validate_record(rec, "test.jsonl", "ana_test01")


def test_causal_links_confidence_out_of_range():
    rec = _make_valid_record("analysis", "ana_test01")
    rec["causal_links"] = [
        {"factor": "water_temp", "effect": "catch_rate", "confidence": 1.5}
    ]
    with pytest.raises(CorpusError, match="causal_link.confidence must be 0..1"):
        _validate_record(rec, "test.jsonl", "ana_test01")


# ======================= based_on type consistency =======================

def test_based_on_nonexistent():
    rec = _make_valid_record("analysis", "ana_test01", based_on=["ana99999"])
    errors = validate_corpus_from_records([rec])
    assert any("ana99999" in e for e in errors)


def test_based_on_wrong_type_no_match():
    obs = _make_valid_record("observation", "obs_parent01")
    dia = _make_valid_record("dialogue", "dia_test01", based_on=["obs_parent01"])
    errors = validate_corpus_from_records([obs, dia])
    assert any("based_on" in e and "obs_parent01" in e for e in errors), \
        "Dialogue should not allow based_on referencing observation directly"


# ======================= schema_version =======================

def test_schema_version_wrong():
    rec = _make_valid_record("fact", "fac_test01")
    rec["schema_version"] = "1.2"
    errors = validate_corpus_from_records([rec])
    assert any("schema_version '1.2' != '1.3'" in e for e in errors)


def test_schema_version_missing():
    rec = _make_valid_record("fact", "fac_test01")
    rec.pop("schema_version", None)
    errors = validate_corpus_from_records([rec])
    assert any("missing schema_version" in e for e in errors)


# ======================= Duplicate IDs =======================

def test_duplicate_id_across_files():
    rec1 = _make_valid_record("fact", "fac_dup01")
    rec2 = _make_valid_record("experience", "fac_dup01")
    errors = validate_corpus_from_records([rec1, rec2])
    assert any("duplicate" in e for e in errors)


def test_duplicate_id_single_file():
    rec = _make_valid_record("fact", "fac_dup01")
    errors = validate_corpus_from_records([rec, rec])
    assert any("duplicate" in e for e in errors)


# ======================= Datetime =======================

@pytest.mark.parametrize("bad_dt,pattern", [
    ("2024-99-99", "not valid ISO8601"),
    ("not-a-date", "not valid ISO8601"),
    ("", "non-empty ISO8601"),  # empty → caught by presence check
    ("2024/06/15", "not valid ISO8601"),
])
def test_invalid_datetime(bad_dt, pattern):
    rec = _make_valid_record("environment", "env_test01", datetime=bad_dt)
    with pytest.raises(CorpusError, match=pattern):
        _validate_record(rec, "test.jsonl", "env_test01")


# ======================= NFC =======================

def test_nfc():
    check_nfc("нормально", "test", 1)
    non_nfc = "е\u0308рш"
    import unicodedata
    if unicodedata.normalize("NFC", non_nfc) == non_nfc:
        pytest.skip("System already NFC-normalized")
    with pytest.raises(CorpusError, match="not NFC-normalized"):
        check_nfc(non_nfc, "test", 1)
