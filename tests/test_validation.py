import json
import tempfile
from pathlib import Path
import pytest
from fishingllm.validate_corpus import (
    validate_corpus, check_record,
    E_FIELD_REQUIRED, E_FIELD_FORBIDDEN, E_RANGE, E_FIELD_ENUM,
    E_SCHEMA_VERSION, E_DATETIME, E_CATCH_SCHEMA, E_ACTION_SCHEMA,
    E_EVIDENCE, E_CAUSAL_LINK, E_ANALYSIS_STATUS, E_BASED_ON,
    E_MISSING_FACTORS, E_NFC, E_DUPLICATE, E_HYPOTHESIS,
    E_ENVIRONMENT, E_GRAPH_BASED_ON, E_GRAPH_CONNECTIVITY,
    E_OUTCOME, E_CYCLE,
)
from test_factory import make, codes, run_check, DATA_DIR


# ── Snapshot: real corpus must validate ──

def test_snapshot_corpus_clean():
    assert validate_corpus(DATA_DIR) == []


# ── D1: success forbidden ──

@pytest.mark.parametrize("val", [True, False])
def test_observation_success_forbidden(val):
    assert E_FIELD_FORBIDDEN in codes(run_check(make("observation", "obs_fb", success=val)))


# ── Observation schema ──

def test_observation_missing_effort():
    rec = make("observation", "obs_me")
    rec.pop("effort")
    assert E_FIELD_REQUIRED in codes(run_check(rec))


def test_observation_missing_result():
    rec = make("observation", "obs_mr")
    rec.pop("result")
    assert E_FIELD_REQUIRED in codes(run_check(rec))


def test_observation_catch_null():
    rec = make("observation", "obs_cn")
    rec["result"]["catch"] = None
    assert E_CATCH_SCHEMA in codes(run_check(rec))


def test_observation_catch_dict():
    rec = make("observation", "obs_cd")
    rec["result"]["catch"] = {"fish": "судак", "count": 1}
    assert E_CATCH_SCHEMA in codes(run_check(rec))


@pytest.mark.parametrize("oq", [-0.1, 1.5])
def test_observation_quality_bounds(oq):
    assert E_RANGE in codes(run_check(make("observation", "obs_qb", observation_quality=oq)))


# ── Catch deep schema ──

def test_catch_missing_fish():
    rec = make("observation", "obs_cmf")
    rec["result"]["catch"] = [{"count": 2}]
    assert E_CATCH_SCHEMA in codes(run_check(rec))


def test_catch_negative_count():
    rec = make("observation", "obs_nc")
    rec["result"]["catch"] = [{"fish": "судак", "count": -1}]
    errs = run_check(rec)
    assert E_CATCH_SCHEMA in codes(errs)
    assert any("count" in e for e in errs)


def test_catch_negative_weight():
    rec = make("observation", "obs_nw")
    rec["result"]["catch"] = [{"fish": "судак", "count": 1, "weight_kg": -1.5}]
    errs = run_check(rec)
    assert E_CATCH_SCHEMA in codes(errs)
    assert any("weight_kg" in e for e in errs)


# ── D3: formal_rule ──

def test_hypothesis_no_formal_rule():
    rec = make("hypothesis", "hyp_nfr")
    rec.pop("formal_rule")
    assert E_FIELD_REQUIRED in codes(run_check(rec))


def test_hypothesis_invalid_operator():
    rec = make("hypothesis", "hyp_io")
    rec["formal_rule"]["operator"] = "foobar"
    assert E_FIELD_ENUM in codes(run_check(rec))


# ── Evidence schema ──

def test_evidence_weight_out_of_range():
    rec = make("evidence_synthesis", "evs_wor")
    rec["evidence"] = [{"id": "obs00001", "role": "supports", "weight": 1.5}]
    assert E_EVIDENCE in codes(run_check(rec))


def test_evidence_weight_missing():
    rec = make("evidence_synthesis", "evs_wm")
    rec["evidence"] = [{"id": "obs00001", "role": "supports"}]
    assert E_EVIDENCE in codes(run_check(rec))


def test_evidence_invalid_role():
    rec = make("evidence_synthesis", "evs_ir")
    rec["evidence"] = [{"id": "obs00001", "role": "banana", "weight": 0.8}]
    errs = run_check(rec)
    assert E_EVIDENCE in codes(errs)
    assert any("banana" in e for e in errs)


def test_evidence_ref_nonexistent():
    rec = make("evidence_synthesis", "evs_rn")
    rec["evidence"] = [{"id": "nonexistent", "role": "supports", "weight": 0.8}]
    errs = run_check(rec)
    assert E_EVIDENCE in codes(errs)
    assert any("nonexistent" in e for e in errs)


# ── Action schema ──

def test_recommendation_no_actions():
    rec = make("recommendation", "rec_na")
    rec["actions"] = []
    assert E_ACTION_SCHEMA in codes(run_check(rec))


def test_recommendation_missing_value():
    rec = make("recommendation", "rec_mv")
    rec["actions"] = [{"type": "change_time"}]
    assert E_ACTION_SCHEMA in codes(run_check(rec))


def test_action_change_time_enum():
    rec = make("recommendation", "rec_cte")
    rec["actions"] = [{"type": "change_time", "value": "полночь"}]
    errs = run_check(rec)
    assert E_ACTION_SCHEMA in codes(errs)
    assert any("полночь" in e for e in errs)


def test_action_change_depth_schema():
    rec = make("recommendation", "rec_cds")
    rec["actions"] = [{"type": "change_depth", "value": "глубоко"}]
    assert E_ACTION_SCHEMA in codes(run_check(rec))


def test_action_change_depth_range():
    rec = make("recommendation", "rec_cdr")
    rec["actions"] = [{"type": "change_depth", "value": [4]}]
    assert E_ACTION_SCHEMA in codes(run_check(rec))


# ── Causal links ──

def test_causal_links_invalid_relation():
    rec = make("analysis", "ana_clir")
    rec["causal_links"] = [{"factor": "water_temp", "effect": "catch_rate",
                            "relation": "banana", "confidence": 0.8}]
    errs = run_check(rec)
    assert E_CAUSAL_LINK in codes(errs)
    assert any("banana" in e for e in errs)


def test_causal_links_missing_factor():
    rec = make("analysis", "ana_clmf")
    rec["causal_links"] = [{"factor": "", "effect": "catch_rate"}]
    assert E_CAUSAL_LINK in codes(run_check(rec))


def test_causal_links_missing_effect():
    rec = make("analysis", "ana_clme")
    rec["causal_links"] = [{"factor": "water_temp", "effect": ""}]
    assert E_CAUSAL_LINK in codes(run_check(rec))


def test_causal_links_confidence_out_of_range():
    rec = make("analysis", "ana_clco")
    rec["causal_links"] = [{"factor": "water_temp", "effect": "catch_rate",
                            "confidence": 1.5}]
    assert E_CAUSAL_LINK in codes(run_check(rec))


def test_causal_links_duplicate():
    rec = make("analysis", "ana_cldup")
    link = {"factor": "water_temp", "effect": "catch_rate"}
    rec["causal_links"] = [link, link]
    assert E_CAUSAL_LINK in codes(run_check(rec))


def test_causal_links_factor_equals_effect():
    rec = make("analysis", "ana_clfe")
    rec["causal_links"] = [{"factor": "water_temp", "effect": "water_temp"}]
    assert E_CAUSAL_LINK in codes(run_check(rec))


def test_causal_links_confounders_not_list():
    rec = make("analysis", "ana_clcn")
    rec["causal_links"] = [{"factor": "water_temp", "effect": "catch_rate",
                            "confounders": "not_a_list"}]
    assert E_CAUSAL_LINK in codes(run_check(rec))


# ── Outcome ──

def test_outcome_no_execution_match():
    rec = make("outcome", "out_nem")
    rec.pop("execution_match")
    assert E_FIELD_REQUIRED in codes(run_check(rec))


@pytest.mark.parametrize("em", [-0.1, 1.4])
def test_outcome_execution_match_bounds(em):
    assert E_RANGE in codes(run_check(make("outcome", "out_emb", execution_match=em)))


def test_outcome_bad_failure_reason():
    rec = make("outcome", "out_bfr", failure_reason="invalid")
    errs = run_check(rec)
    assert E_FIELD_ENUM in codes(errs)
    assert any("invalid" in e for e in errs)


# ── Relation ──

def test_relation_no_weight():
    rec = make("relation", "rel_nw")
    rec.pop("weight")
    assert E_FIELD_REQUIRED in codes(run_check(rec))


def test_relation_invalid_type():
    rec = make("relation", "rel_it", relation="invalid_type")
    errs = run_check(rec)
    assert E_FIELD_ENUM in codes(errs)
    assert any("invalid_type" in e for e in errs)


def test_relation_no_from():
    rec = make("relation", "rel_nf")
    rec.pop("from")
    assert E_FIELD_REQUIRED in codes(run_check(rec))


# ── Analysis status consistency ──

def test_analysis_status_supported_no_evidence():
    rec = make("analysis", "ana_ssne", status="supported")
    rec["causal_links"] = []
    rec["based_on"] = []
    assert run_check(rec) == []


def test_analysis_status_rejected_without_contradicts():
    evs = make("evidence_synthesis", "evs_swc_ref")
    evs["evidence"] = [{"id": "obs00001", "role": "supports", "weight": 0.8}]
    ana = make("analysis", "ana_rwc", status="rejected", based_on=["evs_swc_ref"])
    assert E_ANALYSIS_STATUS in codes(check_record(ana, {"evs_swc_ref": evs}))


def test_analysis_status_contradictory_balanced():
    evs = make("evidence_synthesis", "evs_cb_ref")
    evs["evidence"] = [
        {"id": "obs00001", "role": "supports", "weight": 0.8},
        {"id": "obs00002", "role": "contradicts", "weight": 0.6},
    ]
    ana = make("analysis", "ana_cb", status="supported", based_on=["evs_cb_ref"])
    assert check_record(ana, {"evs_cb_ref": evs}) == []


# ── BasedOn type consistency ──

def test_based_on_nonexistent():
    ana = make("analysis", "ana_bne", based_on=["nonexistent"])
    assert E_BASED_ON in codes(run_check(ana))


def test_based_on_wrong_type():
    obs = make("observation", "obs_bwt_parent")
    dia = make("dialogue", "dia_bwt", based_on=["obs_bwt_parent"])
    assert E_BASED_ON in codes(check_record(dia, {"obs_bwt_parent": obs}))


# ── Schema version ──

def test_schema_version_wrong():
    assert E_SCHEMA_VERSION in codes(run_check(make("fact", "fac_svw", schema_version="1.2")))


def test_schema_version_missing():
    rec = make("fact", "fac_svm")
    rec.pop("schema_version")
    assert E_SCHEMA_VERSION in codes(run_check(rec))


# ── Duplicate IDs ──

def test_duplicate_id_single_file():
    rec = make("fact", "fac_dup")
    assert E_DUPLICATE in codes(_validate([rec, rec]))


def test_duplicate_id_across_files():
    r1 = make("fact", "fac_dup2")
    r2 = make("experience", "fac_dup2")
    assert E_DUPLICATE in codes(_validate([r1, r2]))


# ── Datetime ──

@pytest.mark.parametrize("dt", ["2024-99-99", "not-a-date", "", "2024/06/15"])
def test_invalid_datetime(dt):
    assert E_DATETIME in codes(run_check(make("environment", "env_idt", datetime=dt)))


# ── Empty catch (valid) ──

def test_observation_empty_catch():
    rec = make("observation", "obs_ec")
    rec["result"]["catch"] = []
    assert run_check(rec) == []


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
    rec = make(rtype, rid)
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
    ctx = {"obs00001": {"id": "obs00001", "type": "observation"}} if rtype == "evidence_synthesis" else {}
    assert check_record(make(rtype, rid), ctx) == []


# ── MissingFactors ──

def test_missing_factors_valid():
    rec = make("observation", "obs_mfv")
    rec["missing_factors"] = ["ветер", "течение"]
    assert run_check(rec) == []


def test_missing_factors_non_string():
    rec = make("observation", "obs_mfns")
    rec["missing_factors"] = [42]
    assert E_MISSING_FACTORS in codes(run_check(rec))


# ── NFC ──

def test_nfc_non_normalized_content():
    rec = make("fact", "fac_nfc",
               content="\u0065\u0301\u006b\u0073\u0070\u0065\u0072\u0074")
    assert E_NFC in codes(run_check(rec))


# ── RawImmutable stateless ──

def test_raw_immutable_stateless():
    assert run_check(make("raw_observation", "raw_a", text="version one")) == []
    assert run_check(make("raw_observation", "raw_b", text="version two")) == []


# ── _make minimal mode ──

def test_make_minimal_valid():
    rec = make("observation", "obs_min", mode="minimal")
    assert rec["id"] == "obs_min"
    assert rec["type"] == "observation"
    assert "conditions" not in rec


def test_make_minimal_overrides():
    rec = make("observation", "obs_min2", mode="minimal", conditions={"depth": 3})
    assert rec["conditions"] == {"depth": 3}


# ── Environment reference (graph-level) ──

def test_environment_ref_exists():
    obs = make("observation", "obs_ere")
    obs["environment_id"] = "env_ref"
    env = make("environment", "env_ref", mode="minimal",
               session_id="ses_x", datetime="2024-06-15T06:00:00",
               water_temp=18, pressure_hpa=750, wind_speed=2,
               moon_phase="полнолуние", precipitation="нет",
               water_clarity="прозрачная", water_level="нормальный")
    rel_env = make("relation", "rel_ere1")
    rel_env["from"] = "env_ref"
    rel_env["to"] = "obs_ere"
    rel_obs = make("relation", "rel_ere2")
    rel_obs["from"] = "obs_ere"
    rel_obs["to"] = "env_ref"
    errs = _validate([env, obs, rel_env, rel_obs])
    assert E_ENVIRONMENT not in codes(errs)


def test_environment_ref_nonexistent():
    obs = make("observation", "obs_ern")
    obs["environment_id"] = "env_nonexistent"
    env = make("environment", "env_real", mode="minimal",
               session_id="ses_x", datetime="2024-06-15T06:00:00",
               water_temp=18, pressure_hpa=750, wind_speed=2,
               moon_phase="полнолуние", precipitation="нет",
               water_clarity="прозрачная", water_level="нормальный")
    assert E_ENVIRONMENT in codes(_validate([env, obs]))


def test_environment_ref_wrong_type():
    obs = make("observation", "obs_erwt")
    obs["environment_id"] = "ses_wrong"
    ses = make("session", "ses_wrong", mode="minimal",
               water_body_id="wb_test", datetime_start="2024-06-15T05:00:00",
               datetime_end="2024-06-15T12:00:00")
    assert E_ENVIRONMENT in codes(_validate([ses, obs]))


# ── Evidence reference type (schema-level) ──

def test_evidence_ref_type_valid():
    rec = make("evidence_synthesis", "evs_rtv")
    hyp = make("hypothesis", "hyp_rtv_ref")
    rec["evidence"] = [{"id": "hyp_rtv_ref", "role": "supports", "weight": 0.8}]
    assert E_EVIDENCE not in codes(run_check(rec, {"hyp_rtv_ref": hyp}))


def test_evidence_ref_type_invalid():
    rec = make("evidence_synthesis", "evs_rti")
    rel = make("relation", "rel_rti_ref")
    rec["evidence"] = [{"id": "rel_rti_ref", "role": "supports", "weight": 0.8}]
    assert E_EVIDENCE in codes(run_check(rec, {"rel_rti_ref": rel}))


# ── Graph-based_on consistency (graph-level) ──

def test_graph_based_on_has_edge():
    ana = make("analysis", "ana_gbo")
    evs = make("evidence_synthesis", "evs_gbo_ref")
    ana["based_on"] = ["evs_gbo_ref"]
    ana["causal_links"] = []
    rel = make("relation", "rel_gbo")
    rel["from"] = "ana_gbo"
    rel["to"] = "evs_gbo_ref"
    assert E_GRAPH_BASED_ON not in codes(_validate([ana, evs, rel]))


def test_graph_based_on_missing_edge():
    ana = make("analysis", "ana_gbom")
    evs = make("evidence_synthesis", "evs_gbom_ref")
    ana["based_on"] = ["evs_gbom_ref"]
    ana["causal_links"] = []
    rel = make("relation", "rel_gbom_other")
    rel["from"] = "ana_gbom"
    rel["to"] = "some_other"
    assert E_GRAPH_BASED_ON in codes(_validate([ana, evs, rel]))


# ── Graph connectivity (graph-level) ──

def test_graph_connectivity_no_edges():
    obs = make("observation", "obs_gc")
    assert E_GRAPH_CONNECTIVITY in codes(_validate([obs]))


def test_graph_connectivity_with_edge():
    obs = make("observation", "obs_gce")
    obs_self = make("observation", "obs_gce_self")
    rel = make("relation", "rel_gce")
    rel["from"] = "obs_gce"
    rel["to"] = "obs_gce_self"
    assert E_GRAPH_CONNECTIVITY not in codes(_validate([obs, obs_self, rel]))


# ── Hypothesis formal_rule completeness ──

def test_hypothesis_missing_variable():
    rec = make("hypothesis", "hyp_mv")
    rec["formal_rule"] = {"operator": ">", "value": 20}
    assert E_HYPOTHESIS in codes(run_check(rec))


def test_hypothesis_missing_operator():
    rec = make("hypothesis", "hyp_mo")
    rec["formal_rule"] = {"variable": "water_temp", "value": 20}
    rec["claim"] = "test"
    rec["confidence"] = 0.5
    assert E_HYPOTHESIS in codes(run_check(rec))


def test_hypothesis_type_mismatch_numeric():
    rec = make("hypothesis", "hyp_tmn")
    rec["formal_rule"] = {"variable": "water_temp", "operator": ">", "value": "warm"}
    assert E_HYPOTHESIS in codes(run_check(rec))


def test_hypothesis_type_mismatch_list():
    rec = make("hypothesis", "hyp_tml")
    rec["formal_rule"] = {"variable": "moon_phase", "operator": "in", "value": "полнолуние"}
    assert E_HYPOTHESIS in codes(run_check(rec))


def test_hypothesis_between_not_list():
    rec = make("hypothesis", "hyp_btn")
    rec["formal_rule"] = {"variable": "water_temp", "operator": "between", "value": 20}
    assert E_HYPOTHESIS in codes(run_check(rec))


# ── NestedFieldEnum message (regression: f-string contained literal ".join") ──

def test_nested_field_enum_error_message():
    rec = make("hypothesis", "hyp_nfe")
    rec["formal_rule"] = {"variable": "water_temp", "operator": "bad_op", "value": 20}
    errs = run_check(rec)
    for e in errs:
        assert "'.join" not in e, f"NestedFieldEnum message contains literal '.join': {e}"
        assert "self._path" not in e, f"NestedFieldEnum message contains literal '_path': {e}"


# ── BasedOnRule: unsupported type gives single error, not per-ref ──

def test_based_on_unsupported_type_single_error():
    rec = make("observation", "obs_bon")
    rec["based_on"] = ["obs_ref1", "obs_ref2"]
    obs_ref1 = make("observation", "obs_ref1")
    obs_ref2 = make("observation", "obs_ref2")
    errs = run_check(rec, {"obs_ref1": obs_ref1, "obs_ref2": obs_ref2})
    e_based_on = [e for e in errs if e.startswith("[E_BASED_ON]")]
    assert len(e_based_on) == 1, f"expected 1 E_BASED_ON error, got {len(e_based_on)}: {e_based_on}"


# ── Causal reasoning ──

def test_causal_links_missing_factor_and_effect():
    rec = make("analysis", "ana_clm")
    rec["causal_links"] = [{"relation": "positive", "confidence": 0.5}]
    assert E_CAUSAL_LINK in codes(run_check(rec))


def test_causal_links_empty_factor():
    rec = make("analysis", "ana_clef")
    rec["causal_links"] = [{"factor": "", "effect": "catch_rate", "relation": "positive"}]
    assert E_CAUSAL_LINK in codes(run_check(rec))


# ── Missing environment fields (schema-level) ──

def test_environment_missing_wind_speed():
    rec = make("environment", "env_mws")
    rec.pop("wind_speed", None)
    assert E_FIELD_REQUIRED in codes(run_check(rec))


def test_environment_missing_moon_phase():
    rec = make("environment", "env_mmp")
    rec.pop("moon_phase", None)
    assert E_FIELD_REQUIRED in codes(run_check(rec))


def test_environment_missing_precipitation():
    rec = make("environment", "env_mp")
    rec.pop("precipitation", None)
    assert E_FIELD_REQUIRED in codes(run_check(rec))


def test_environment_missing_water_clarity():
    rec = make("environment", "env_mwc")
    rec.pop("water_clarity", None)
    assert E_FIELD_REQUIRED in codes(run_check(rec))


def test_environment_missing_session_id():
    rec = make("environment", "env_msid")
    rec.pop("session_id", None)
    assert E_FIELD_REQUIRED in codes(run_check(rec))


# ── Outcome success invariant (cross-record) ──

def test_outcome_success_empty_catch_true():
    rec = make("outcome", "out_sect")
    rec["result"] = {"catch": [], "success": True}
    assert E_OUTCOME in codes(run_check(rec))


def test_outcome_success_nonempty_catch_false():
    rec = make("outcome", "out_scf")
    rec["result"] = {"catch": [{"fish": "судак", "count": 2}], "success": False}
    assert E_OUTCOME in codes(run_check(rec))


def test_outcome_success_missing_result():
    rec = make("outcome", "out_msr")
    rec.pop("result", None)
    assert E_FIELD_REQUIRED in codes(run_check(rec))


# ── Observation invariant: environment_id and time_of_day ──

def test_observation_missing_environment_id():
    rec = make("observation", "obs_meid")
    rec.pop("environment_id", None)
    assert E_FIELD_REQUIRED in codes(run_check(rec))


def test_observation_missing_time_of_day():
    rec = make("observation", "obs_mtod")
    rec.pop("time_of_day", None)
    assert E_FIELD_REQUIRED in codes(run_check(rec))


# ── D9: derived_from multi-edge (regression: old code overwrote multiple edges) ──

def test_d9_no_false_cycle_on_dag():
    a = make("analysis", "ana_d9_dag")
    a["based_on"] = ["evs_dag"]
    a["causal_links"] = []
    evs = make("evidence_synthesis", "evs_dag")
    evs["evidence"] = [{"id": "obs_dag", "role": "supports", "weight": 0.8}]
    obs = make("observation", "obs_dag")
    rel1 = make("relation", "rel_d9_1")
    rel1["from"] = "evs_dag"
    rel1["to"] = "ana_d9_dag"
    rel1["relation"] = "derived_from"
    rel2 = make("relation", "rel_d9_2")
    rel2["from"] = "evs_dag"
    rel2["to"] = "obs_dag"
    rel2["relation"] = "derived_from"
    rel3 = make("relation", "rel_d9_3")
    rel3["from"] = "ana_d9_dag"
    rel3["to"] = "obs_dag"
    rel3["relation"] = "supports"
    errs = _validate([a, evs, obs, rel1, rel2, rel3])
    assert E_CYCLE not in codes(errs)


# ── Helper ──

def _validate(records: list[dict]) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        proc = Path(tmp) / "processed"
        proc.mkdir(parents=True)
        sorted_records = sorted(records, key=lambda r: (r.get("type", ""), r.get("id", "")))
        with open(proc / "test.jsonl", "w", encoding="utf-8") as f:
            for r in sorted_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return validate_corpus(tmp)
