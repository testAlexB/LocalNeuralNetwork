"""Property-based tests for validate_corpus using Hypothesis.

Tests invariants: valid records pass, specific mutations cause specific error codes.
"""
import json
import tempfile
from pathlib import Path
from hypothesis import given, strategies as st, assume
from fishingllm.validate_corpus import (
    validate_corpus, check_record,
    VALID_TYPES, VALID_STATUSES, VALID_EVIDENCE_ROLES,
    VALID_CAUSAL_RELATIONS, VALID_RELATION_TYPES, VALID_ACTION_TYPES,
    VALID_FAILURE_REASONS, VALID_OPERATORS, ACCEPTED_SCHEMA_VERSION,
    E_FIELD_REQUIRED, E_FIELD_ENUM, E_RANGE, E_SCHEMA_VERSION,
    E_DATETIME, E_EVIDENCE, E_BASED_ON, E_ANALYSIS_STATUS,
    E_FIELD_FORBIDDEN, E_CATCH_SCHEMA, E_ACTION_SCHEMA, E_CAUSAL_LINK,
    E_MISSING_FACTORS,
)

# ── Building blocks ──

def _make(rid: str, rtype: str, **overrides) -> dict:
    base = {
        "id": rid,
        "type": rtype,
        "schema_version": ACCEPTED_SCHEMA_VERSION,
        "id_prefix": rid.split("_")[0] if "_" in rid else rid[:4],
    }
    return {**base, **overrides}


def codes(errors: list[str]) -> set[str]:
    return {e.split("]")[0].lstrip("[") if "]" in e else e for e in errors}


# ── Strategies ──

ids = st.text(min_size=3, max_size=20,
              alphabet=st.characters(min_codepoint=97, max_codepoint=122) |
                        st.characters(min_codepoint=48, max_codepoint=57)).map(
    lambda s: "x_" + s)

water_ids = st.sampled_from(["wb_volga_upper", "wb_volga_lower", "wb_seliger", "wb_tvertsa"])
species = st.sampled_from(["судак", "окунь", "щука", "лещ", "плотва", "карась"])
bait = st.sampled_from(["червь", "блесна", "силикон", "мормыш", "опарыш"])
weather = st.sampled_from(["ясно", "облачно", "дождь", "ветер"])
datetimes = st.datetimes().map(lambda d: d.strftime("%Y-%m-%dT%H:%M:%S"))
actions = st.sampled_from(list(VALID_ACTION_TYPES))
action_value = st.sampled_from(["утро", "день", "вечер", "ночь", [2, 5], "блесна"])
operators = st.sampled_from(list(VALID_OPERATORS))
statuses = st.sampled_from(list(VALID_STATUSES))
ev_roles = st.sampled_from(list(VALID_EVIDENCE_ROLES))
rel_types = st.sampled_from(list(VALID_RELATION_TYPES))
fail_reasons = st.sampled_from(list(VALID_FAILURE_REASONS))
causal_rels = st.sampled_from(list(VALID_CAUSAL_RELATIONS))
confidences = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
weights = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
depths = st.floats(min_value=0.5, max_value=30.0, allow_nan=False, allow_infinity=False)
pressures = st.floats(min_value=720.0, max_value=780.0, allow_nan=False, allow_infinity=False)
temperatures = st.floats(min_value=-5.0, max_value=35.0, allow_nan=False, allow_infinity=False)
counts = st.integers(min_value=0, max_value=100)
factors = st.lists(st.text(min_size=1, max_size=20), max_size=5)
texts = st.text(min_size=1, max_size=200)
source_reliabilities = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# ── Hypothesis: valid records produce no errors ──

@st.composite
def valid_record(draw, rtype: str, rid: str):
    """Build a valid record for given type."""
    base = _make(rid, rtype)
    if rtype == "raw_observation":
        return {**base, "text": draw(texts)}
    if rtype == "session":
        return {**base,
                "water_body_id": draw(water_ids),
                "datetime_start": draw(datetimes),
                "datetime_end": draw(datetimes)}
    if rtype == "observation":
        nfish = draw(st.integers(min_value=0, max_value=3))
        catch = []
        for _ in range(nfish):
            catch.append({"fish": draw(species), "count": draw(counts)})
        return {**base,
                "conditions": {"depth": draw(depths), "target_species": draw(species)},
                "effort": {"hours_fished": draw(st.floats(0.1, 24.0)),
                           "casts": draw(st.integers(1, 200))},
                "result": {"catch": catch},
                "missing_factors": draw(factors),
                "observation_quality": draw(st.floats(0.0, 1.0))}
    if rtype == "environment":
        return {**base,
                "session_id": draw(ids),
                "datetime": draw(datetimes),
                "source": "angler",
                "water_temp": draw(temperatures),
                "pressure_hpa": draw(pressures)}
    if rtype == "hypothesis":
        return {**base,
                "claim": draw(texts),
                "confidence": draw(confidences),
                "formal_rule": {"variable": "water_temp",
                                "operator": draw(operators),
                                "value": 20}}
    if rtype == "evidence_synthesis":
        return {**base,
                "evidence": [{"id": draw(ids), "role": draw(ev_roles),
                              "weight": draw(weights)}],
                "summary": draw(texts)}
    if rtype == "analysis":
        return {**base,
                "status": draw(statuses),
                "analysis": draw(texts),
                "analysis_confidence": draw(confidences),
                "evidence_strength": "низкая",
                "causal_links": [],
                "alternative_explanations": []}
    if rtype == "recommendation":
        return {**base,
                "text": draw(texts),
                "target_species": draw(species),
                "actions": [{"type": draw(actions), "value": draw(action_value)}]}
    if rtype == "outcome":
        return {**base,
                "execution_match": draw(confidences),
                "failure_reason": draw(fail_reasons),
                "result": {"catch": [], "success": True}}
    if rtype == "relation":
        return {**base,
                "from": draw(ids),
                "to": draw(ids),
                "relation": draw(rel_types),
                "weight": draw(weights)}
    if rtype == "dialogue":
        return {**base,
                "assistant_confidence": draw(confidences),
                "messages": [{"role": "user", "content": draw(texts)},
                             {"role": "assistant", "content": draw(texts)}]}
    if rtype == "fact":
        return {**base, "content": draw(texts),
                "source_reliability": draw(source_reliabilities)}
    if rtype == "experience":
        return {**base, "content": draw(texts),
                "source_reliability": draw(source_reliabilities)}
    if rtype == "insufficient_data":
        return {**base,
                "query": draw(texts),
                "analysis": draw(texts),
                "status": "insufficient_data",
                "recommendation": {"text": draw(texts)}}
    if rtype == "uncertainty":
        return {**base,
                "analysis": draw(texts),
                "status": "contradictory",
                "causal_links": [],
                "recommendation": {"text": draw(texts)},
                "analysis_confidence": draw(confidences),
                "evidence_strength": "низкая"}
    return base


@given(rid=ids)
def test_fact_minimal_valid(rid):
    rec = _make(rid, "fact", content="факт", source_reliability=0.8)
    assert len(check_record(rec, {})) == 0


@given(rid=ids)
def test_experience_minimal_valid(rid):
    rec = _make(rid, "experience", content="опыт", source_reliability=0.6)
    assert len(check_record(rec, {})) == 0


@given(rid=ids)
def test_raw_observation_valid(rid):
    rec = _make(rid, "raw_observation", text="just some text")
    assert len(check_record(rec, {})) == 0


@given(rid=ids)
def test_fact_valid(rid):
    rec = _make(rid, "fact", content="факт", source_reliability=0.8)
    assert len(check_record(rec, {})) == 0


@given(rid=ids)
def test_experience_valid(rid):
    rec = _make(rid, "experience", content="опыт", source_reliability=0.6)
    assert len(check_record(rec, {})) == 0


# ── Mutation tests: required field deletion → E_FIELD_REQUIRED ──

@given(rid=ids)
def test_raw_missing_text(rid):
    rec = _make(rid, "raw_observation")
    rec.pop("text", None)
    assert E_FIELD_REQUIRED in codes(check_record(rec, {}))


@given(rid=ids)
def test_fact_missing_content(rid):
    rec = _make(rid, "fact", source_reliability=0.5)
    rec.pop("content", None)
    assert E_FIELD_REQUIRED in codes(check_record(rec, {}))


@given(rid=ids)
def test_fact_missing_reliability(rid):
    rec = _make(rid, "fact", content="что-то")
    rec.pop("source_reliability", None)
    assert E_FIELD_REQUIRED in codes(check_record(rec, {}))


@given(rid=ids)
def test_session_missing_water_body(rid):
    rec = _make(rid, "session", datetime_start="2024-01-01T00:00:00",
                datetime_end="2024-01-01T01:00:00")
    rec.pop("water_body_id", None)
    assert E_FIELD_REQUIRED in codes(check_record(rec, {}))


@given(rid=ids)
def test_observation_missing_missing_factors(rid):
    rec = _make(rid, "observation")
    rec.pop("missing_factors", None)
    assert E_FIELD_REQUIRED in codes(check_record(rec, {}))


# ── Mutation: out-of-range → E_RANGE ──

@given(rid=ids, oq=st.floats(min_value=-10, max_value=-0.01, allow_nan=False, allow_infinity=False))
def test_observation_quality_too_low(rid, oq):
    rec = _make(rid, "observation", observation_quality=oq)
    assert E_RANGE in codes(check_record(rec, {}))


@given(rid=ids, ac=st.floats(min_value=1.01, max_value=10, allow_nan=False, allow_infinity=False))
def test_analysis_confidence_too_high(rid, ac):
    rec = _make(rid, "analysis", analysis_confidence=ac,
                status="supported", analysis="x",
                evidence_strength="низкая", causal_links=[])
    assert E_RANGE in codes(check_record(rec, {}))


# ── Mutation: invalid enum → E_FIELD_ENUM ──

@given(rid=ids, bad_status=st.text(min_size=1, max_size=20))
def test_relation_invalid_type_fuzzy(rid, bad_status):
    assume(bad_status not in VALID_RELATION_TYPES)
    rec = _make(rid, "relation", from_id="a", to="b",
                relation=bad_status, weight=0.5)
    assert E_FIELD_ENUM in codes(check_record(rec, {}))


@given(rid=ids, bad_reason=st.text(min_size=1, max_size=20))
def test_outcome_invalid_failure_reason(rid, bad_reason):
    assume(bad_reason not in VALID_FAILURE_REASONS)
    rec = _make(rid, "outcome", execution_match=0.5,
                failure_reason=bad_reason, result={"catch": []})
    assert E_FIELD_ENUM in codes(check_record(rec, {}))


# ── Schema version ──

@given(rid=ids, bad_ver=st.text(min_size=1, max_size=10))
def test_bad_schema_version(rid, bad_ver):
    assume(bad_ver != ACCEPTED_SCHEMA_VERSION)
    rec = _make(rid, "fact", schema_version=bad_ver, content="x",
                source_reliability=0.5)
    assert E_SCHEMA_VERSION in codes(check_record(rec, {}))


# ── Evidence cross-ref ──

def test_evidence_missing_ref():
    rec = _make("evs_prop", "evidence_synthesis",
                evidence=[{"id": "ghost", "role": "supports", "weight": 0.8}],
                summary="x")
    assert E_EVIDENCE in codes(check_record(rec, {}))


# ── Whole-corpus roundtrip: valid corpus stays valid ──

def test_corpus_roundtrip_clean():
    errs = validate_corpus(Path(__file__).resolve().parent.parent / "data")
    assert errs == [], f"Corpus snapshot mismatch:\n" + "\n".join(errs)
