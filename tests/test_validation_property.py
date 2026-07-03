"""Property-based tests for validate_corpus using Hypothesis.

Tests invariants: valid records pass, mutations cause specific error codes.
"""
from pathlib import Path
from hypothesis import given, strategies as st, assume, settings
from fishingllm.validate_corpus import (
    validate_corpus,
    ACCEPTED_SCHEMA_VERSION,
    VALID_RELATION_TYPES, VALID_FAILURE_REASONS,
    E_FIELD_REQUIRED, E_RANGE, E_FIELD_ENUM, E_SCHEMA_VERSION,
    E_EVIDENCE, E_MISSING_FACTORS,
)
from test_factory import make, codes, run_check, DATA_DIR


PROFILE = settings(max_examples=20)


# ── Strategies ──

safe_chars = st.characters(min_codepoint=97, max_codepoint=122) | st.characters(min_codepoint=48, max_codepoint=57)
ids = st.text(min_size=3, max_size=20, alphabet=safe_chars).map(lambda s: "x_" + s)


# ── Minimal valid records: must pass ──

@PROFILE
@given(rid=ids)
def test_raw_observation_valid(rid):
    assert run_check(make("raw_observation", rid, text="test")) == []


@PROFILE
@given(rid=ids)
def test_fact_valid(rid):
    assert run_check(make("fact", rid, content="факт", source_reliability=0.8)) == []


@PROFILE
@given(rid=ids)
def test_experience_valid(rid):
    assert run_check(make("experience", rid, content="опыт", source_reliability=0.6)) == []


# ── Mutation: required field deletion → E_FIELD_REQUIRED ──

@PROFILE
@given(rid=ids)
def test_raw_missing_text(rid):
    rec = make("raw_observation", rid)
    rec.pop("text", None)
    assert E_FIELD_REQUIRED in codes(run_check(rec))


@PROFILE
@given(rid=ids)
def test_fact_missing_content(rid):
    rec = make("fact", rid, source_reliability=0.5)
    rec.pop("content", None)
    assert E_FIELD_REQUIRED in codes(run_check(rec))


@PROFILE
@given(rid=ids)
def test_fact_missing_reliability(rid):
    rec = make("fact", rid, content="what")
    rec.pop("source_reliability", None)
    assert E_FIELD_REQUIRED in codes(run_check(rec))


@PROFILE
@given(rid=ids)
def test_session_missing_water_body(rid):
    rec = make("session", rid,
               datetime_start="2024-01-01T00:00:00",
               datetime_end="2024-01-01T01:00:00")
    rec.pop("water_body_id", None)
    assert E_FIELD_REQUIRED in codes(run_check(rec))


@PROFILE
@given(rid=ids)
def test_observation_missing_missing_factors(rid):
    rec = make("observation", rid)
    rec.pop("missing_factors", None)
    assert E_FIELD_REQUIRED in codes(run_check(rec))


# ── Mutation: out-of-range → E_RANGE ──

@PROFILE
@given(rid=ids, oq=st.floats(min_value=-10, max_value=-0.01, allow_nan=False, allow_infinity=False))
def test_observation_quality_too_low(rid, oq):
    assert E_RANGE in codes(run_check(make("observation", rid, observation_quality=oq)))


@PROFILE
@given(rid=ids, ac=st.floats(min_value=1.01, max_value=10, allow_nan=False, allow_infinity=False))
def test_analysis_confidence_too_high(rid, ac):
    rec = make("analysis", rid, analysis_confidence=ac,
               status="supported", analysis="x",
               evidence_strength="низкая", causal_links=[])
    assert E_RANGE in codes(run_check(rec))


# ── Mutation: invalid enum → E_FIELD_ENUM ──

BAD_REL = "not_a_valid_relation"
BAD_REASON = "not_a_valid_reason"


@PROFILE
@given(rid=ids)
def test_relation_invalid_type_fuzzy(rid):
    assume(BAD_REL not in VALID_RELATION_TYPES)
    rec = make("relation", rid, to="b", relation=BAD_REL, weight=0.5,
               **{"from": "a"})
    assert E_FIELD_ENUM in codes(run_check(rec))


@PROFILE
@given(rid=ids)
def test_outcome_invalid_failure_reason(rid):
    assume(BAD_REASON not in VALID_FAILURE_REASONS)
    rec = make("outcome", rid, execution_match=0.5, failure_reason=BAD_REASON,
               result={"catch": []})
    assert E_FIELD_ENUM in codes(run_check(rec))


# ── Schema version ──

@PROFILE
@given(rid=ids)
def test_bad_schema_version(rid):
    rec = make("fact", rid, schema_version="9.9", content="x", source_reliability=0.5)
    assert E_SCHEMA_VERSION in codes(run_check(rec))


# ── Evidence cross-ref ──

def test_evidence_missing_ref():
    rec = make("evidence_synthesis", "evs_prop",
               evidence=[{"id": "ghost", "role": "supports", "weight": 0.8}],
               summary="x")
    assert E_EVIDENCE in codes(run_check(rec))


# ── Whole-corpus roundtrip: valid corpus stays valid ──

def test_corpus_roundtrip_clean():
    assert validate_corpus(DATA_DIR) == []
