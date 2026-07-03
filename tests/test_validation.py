import json
import tempfile
from pathlib import Path
import pytest
from fishingllm.validate_corpus import (
    validate_corpus, check_record,
    ACCEPTED_SCHEMA_VERSION,
    E_FIELD_REQUIRED, E_FIELD_FORBIDDEN, E_RANGE, E_FIELD_ENUM,
    E_SCHEMA_VERSION, E_DATETIME, E_CATCH_SCHEMA, E_ACTION_SCHEMA,
    E_EVIDENCE, E_CAUSAL_LINK, E_ANALYSIS_STATUS, E_BASED_ON,
    E_MISSING_FACTORS, E_NFC, E_RAW_CHANGED, E_DUPLICATE,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ── Factories ──

def _make(rtype: str, rid: str, *, mode: str = "valid", **overrides) -> dict:
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
    result = {**base, **defaults.get(rtype, {}), **overrides}
    return result


def run_check(rec: dict) -> list[str]:
    return check_record(rec, {})


def codes(errors: list[str]) -> set[str]:
    return {e.split("]")[0].lstrip("[") if "]" in e else e for e in errors}


# ── Real corpus ──

def test_validate_real_corpus():
    errors = validate_corpus(DATA_DIR)
    assert len(errors) == 0, "\n".join(errors)


# ── D1: success forbidden ──

@pytest.mark.parametrize("val", [True, False])
def test_observation_success_forbidden(val):
    errs = run_check(_make("observation", "obs_forbidden", success=val))
    assert E_FIELD_FORBIDDEN in codes(errs)


# ── Observation schema ──

def test_observation_missing_effort():
    rec = _make("observation", "obs_me")
    rec.pop("effort")
    assert E_FIELD_REQUIRED in codes(run_check(rec))


def test_observation_missing_result():
    rec = _make("observation", "obs_mr")
    rec.pop("result")
    assert E_FIELD_REQUIRED in codes(run_check(rec))


def test_observation_catch_null():
    rec = _make("observation", "obs_cn")
    rec["result"]["catch"] = None
    assert E_CATCH_SCHEMA in codes(run_check(rec))


def test_observation_catch_dict():
    rec = _make("observation", "obs_cd")
    rec["result"]["catch"] = {"fish": "судак", "count": 1}
    assert E_CATCH_SCHEMA in codes(run_check(rec))


@pytest.mark.parametrize("oq", [-0.1, 1.5])
def test_observation_quality_bounds(oq):
    assert E_RANGE in codes(run_check(_make("observation", "obs_qb", observation_quality=oq)))


# ── Catch deep schema ──

def test_catch_missing_fish():
    rec = _make("observation", "obs_cmf")
    rec["result"]["catch"] = [{"count": 2}]
    assert E_CATCH_SCHEMA in codes(run_check(rec))


def test_catch_negative_count():
    rec = _make("observation", "obs_nc")
    rec["result"]["catch"] = [{"fish": "судак", "count": -1}]
    assert E_CATCH_SCHEMA in codes(run_check(rec))


def test_catch_negative_weight():
    rec = _make("observation", "obs_nw")
    rec["result"]["catch"] = [{"fish": "судак", "count": 1, "weight_kg": -1.5}]
    assert E_CATCH_SCHEMA in codes(run_check(rec))


# ── D3: formal_rule ──

def test_hypothesis_no_formal_rule():
    rec = _make("hypothesis", "hyp_nfr")
    rec.pop("formal_rule")
    assert E_FIELD_REQUIRED in codes(run_check(rec))


def test_hypothesis_invalid_operator():
    rec = _make("hypothesis", "hyp_io")
    rec["formal_rule"]["operator"] = "foobar"
    assert E_FIELD_ENUM in codes(run_check(rec))


# ── Evidence schema ──

def test_evidence_weight_out_of_range():
    rec = _make("evidence_synthesis", "evs_wor")
    rec["evidence"] = [{"id": "obs00001", "role": "supports", "weight": 1.5}]
    assert E_EVIDENCE in codes(run_check(rec))


def test_evidence_weight_missing():
    rec = _make("evidence_synthesis", "evs_wm")
    rec["evidence"] = [{"id": "obs00001", "role": "supports"}]
    assert E_EVIDENCE in codes(run_check(rec))


def test_evidence_invalid_role():
    rec = _make("evidence_synthesis", "evs_ir")
    rec["evidence"] = [{"id": "obs00001", "role": "banana", "weight": 0.8}]
    assert E_EVIDENCE in codes(run_check(rec))


def test_evidence_ref_nonexistent():
    rec = _make("evidence_synthesis", "evs_rn")
    rec["evidence"] = [{"id": "nonexistent", "role": "supports", "weight": 0.8}]
    assert E_EVIDENCE in codes(run_check(rec))


# ── Action schema ──

def test_recommendation_no_actions():
    rec = _make("recommendation", "rec_na")
    rec["actions"] = []
    assert E_ACTION_SCHEMA in codes(run_check(rec))


def test_recommendation_missing_value():
    rec = _make("recommendation", "rec_mv")
    rec["actions"] = [{"type": "change_time"}]
    assert E_ACTION_SCHEMA in codes(run_check(rec))


def test_action_change_time_enum():
    rec = _make("recommendation", "rec_cte")
    rec["actions"] = [{"type": "change_time", "value": "полночь"}]
    assert E_ACTION_SCHEMA in codes(run_check(rec))


def test_action_change_depth_schema():
    rec = _make("recommendation", "rec_cds")
    rec["actions"] = [{"type": "change_depth", "value": "глубоко"}]
    assert E_ACTION_SCHEMA in codes(run_check(rec))


def test_action_change_depth_range():
    rec = _make("recommendation", "rec_cdr")
    rec["actions"] = [{"type": "change_depth", "value": [4]}]
    assert E_ACTION_SCHEMA in codes(run_check(rec))


# ── Causal links ──

def test_causal_links_invalid_relation():
    rec = _make("analysis", "ana_clir")
    rec["causal_links"] = [{"factor": "water_temp", "effect": "catch_rate", "relation": "banana", "confidence": 0.8}]
    assert E_CAUSAL_LINK in codes(run_check(rec))


def test_causal_links_missing_factor():
    rec = _make("analysis", "ana_clmf")
    rec["causal_links"] = [{"factor": "", "effect": "catch_rate"}]
    assert E_CAUSAL_LINK in codes(run_check(rec))


def test_causal_links_missing_effect():
    rec = _make("analysis", "ana_clme")
    rec["causal_links"] = [{"factor": "water_temp", "effect": ""}]
    assert E_CAUSAL_LINK in codes(run_check(rec))


def test_causal_links_confidence_out_of_range():
    rec = _make("analysis", "ana_clco")
    rec["causal_links"] = [{"factor": "water_temp", "effect": "catch_rate", "confidence": 1.5}]
    assert E_CAUSAL_LINK in codes(run_check(rec))


def test_causal_links_confounders_not_list():
    rec = _make("analysis", "ana_clcn")
    rec["causal_links"] = [{"factor": "water_temp", "effect": "catch_rate", "confounders": "not_a_list"}]
    assert E_CAUSAL_LINK in codes(run_check(rec))


# ── Outcome ──

def test_outcome_no_execution_match():
    rec = _make("outcome", "out_nem")
    rec.pop("execution_match")
    assert E_FIELD_REQUIRED in codes(run_check(rec))


@pytest.mark.parametrize("em", [-0.1, 1.4])
def test_outcome_execution_match_bounds(em):
    assert E_RANGE in codes(run_check(_make("outcome", "out_emb", execution_match=em)))


def test_outcome_bad_failure_reason():
    rec = _make("outcome", "out_bfr", failure_reason="invalid")
    assert E_FIELD_ENUM in codes(run_check(rec))


# ── Relation ──

def test_relation_no_weight():
    rec = _make("relation", "rel_nw")
    rec.pop("weight")
    assert E_FIELD_REQUIRED in codes(run_check(rec))


def test_relation_invalid_type():
    rec = _make("relation", "rel_it", relation="invalid_type")
    assert E_FIELD_ENUM in codes(run_check(rec))


def test_relation_no_from():
    rec = _make("relation", "rel_nf")
    rec.pop("from")
    assert E_FIELD_REQUIRED in codes(run_check(rec))


# ── Analysis status consistency ──

def test_analysis_status_supported_no_evidence():
    rec = _make("analysis", "ana_ssne", status="supported")
    rec["causal_links"] = []
    rec["based_on"] = []
    assert len(run_check(rec)) == 0


def test_analysis_status_rejected_without_contradicts():
    evs = _make("evidence_synthesis", "evs_swc_ref")
    evs["evidence"] = [{"id": "obs00001", "role": "supports", "weight": 0.8}]
    ana = _make("analysis", "ana_rwc", status="rejected", based_on=["evs_swc_ref"])
    errs = check_record(ana, {"evs_swc_ref": evs})
    assert E_ANALYSIS_STATUS in codes(errs)


def test_analysis_status_contradictory_balanced():
    evs = _make("evidence_synthesis", "evs_cb_ref")
    evs["evidence"] = [
        {"id": "obs00001", "role": "supports", "weight": 0.8},
        {"id": "obs00002", "role": "contradicts", "weight": 0.6},
    ]
    ana = _make("analysis", "ana_cb", status="supported", based_on=["evs_cb_ref"])
    errs = check_record(ana, {"evs_cb_ref": evs})
    assert len(errs) == 0, str(errs)


# ── BasedOn type consistency ──

def test_based_on_nonexistent():
    ana = _make("analysis", "ana_bne", based_on=["nonexistent"])
    assert E_BASED_ON in codes(run_check(ana))


def test_based_on_wrong_type():
    obs = _make("observation", "obs_bwt_parent")
    dia = _make("dialogue", "dia_bwt", based_on=["obs_bwt_parent"])
    errs = check_record(dia, {"obs_bwt_parent": obs})
    assert E_BASED_ON in codes(errs)


# ── Schema version ──

def test_schema_version_wrong():
    assert E_SCHEMA_VERSION in codes(run_check(_make("fact", "fac_svw", schema_version="1.2")))


def test_schema_version_missing():
    rec = _make("fact", "fac_svm")
    rec.pop("schema_version")
    assert E_SCHEMA_VERSION in codes(run_check(rec))


# ── Duplicate IDs ──

def test_duplicate_id_single_file():
    rec = _make("fact", "fac_dup")
    errs = _validate([rec, rec])
    assert E_DUPLICATE in codes(errs) or "duplicate" in str(errs)


def test_duplicate_id_across_files():
    r1 = _make("fact", "fac_dup2")
    r2 = _make("experience", "fac_dup2")
    errs = _validate([r1, r2])
    assert E_DUPLICATE in codes(errs) or "duplicate" in str(errs)


# ── Datetime ──

@pytest.mark.parametrize("dt", [
    "2024-99-99",
    "not-a-date",
    "",
    "2024/06/15",
])
def test_invalid_datetime(dt):
    assert E_DATETIME in codes(run_check(_make("environment", "env_idt", datetime=dt)))


# ── Empty catch (valid) ──

def test_observation_empty_catch():
    rec = _make("observation", "obs_ec")
    rec["result"]["catch"] = []
    assert len(run_check(rec)) == 0


# ── Missing fields per type ──

@pytest.mark.parametrize("rtype,rid,field", [
    ("raw_observation",   "r_raw_missing",   "text"),
    ("evidence_synthesis", "r_evs_missing",  "evidence"),
    ("dialogue",          "r_dia_missing",   "messages"),
    ("fact",              "r_fac_missing",   "content"),
    ("experience",        "r_exp_missing",   "content"),
    ("insufficient_data", "r_ins_missing",   "query"),
    ("uncertainty",       "r_unc_missing",   "analysis"),
])
def test_missing_required_field(rtype, rid, field):
    rec = _make(rtype, rid)
    rec.pop(field)
    assert E_FIELD_REQUIRED in codes(run_check(rec))


@pytest.mark.parametrize("rtype,rid", [
    ("raw_observation",   "r_raw_valid"),
    ("evidence_synthesis", "r_evs_valid"),
    ("dialogue",          "r_dia_valid"),
    ("fact",              "r_fac_valid"),
    ("experience",        "r_exp_valid"),
    ("insufficient_data", "r_ins_valid"),
    ("uncertainty",       "r_unc_valid"),
])
def test_valid_rare_type(rtype, rid):
    ctx = {}
    if rtype == "evidence_synthesis":
        ctx["obs00001"] = {"id": "obs00001", "type": "observation"}
    rec = _make(rtype, rid)
    errs = check_record(rec, ctx)
    assert len(errs) == 0, f"{rtype}: {errs}"


# ── MissingFactors ──

def test_missing_factors_valid():
    rec = _make("observation", "obs_mfv")
    rec["missing_factors"] = ["ветер", "течение"]
    assert len(run_check(rec)) == 0


def test_missing_factors_non_string():
    rec = _make("observation", "obs_mfns")
    rec["missing_factors"] = [42]
    assert E_MISSING_FACTORS in codes(run_check(rec))


# ── NFC ──

def test_nfc_non_normalized_content():
    rec = _make("fact", "fac_nfc",
                content="\u0065\u0301\u006b\u0073\u0070\u0065\u0072\u0074")
    assert E_NFC in codes(run_check(rec))


# ── RawImmutable stateless ──

def test_raw_immutable_stateless():
    r1 = _make("raw_observation", "raw_a", text="version one")
    r2 = _make("raw_observation", "raw_b", text="version two")
    assert len(run_check(r1)) == 0
    assert len(run_check(r2)) == 0


# ── _make minimal mode ──

def test_make_minimal_valid():
    rec = _make("observation", "obs_min", mode="minimal")
    assert rec["id"] == "obs_min"
    assert rec["type"] == "observation"
    assert "conditions" not in rec


def test_make_minimal_overrides():
    rec = _make("observation", "obs_min2", mode="minimal",
                conditions={"depth": 3})
    assert rec["conditions"] == {"depth": 3}


# ── Snapshot: real corpus must validate ──

def test_snapshot_corpus_clean():
    errs = validate_corpus(DATA_DIR)
    assert errs == [], f"Snapshot mismatch — corpus has {len(errs)} errors:\n" + "\n".join(errs)


# ── Helper ──

def _validate(records: list[dict]) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        proc = Path(tmp) / "processed"
        proc.mkdir(parents=True)
        with open(proc / "test.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return validate_corpus(tmp)
