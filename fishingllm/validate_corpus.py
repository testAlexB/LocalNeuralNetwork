import json
import unicodedata
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

VALID_TYPES = frozenset({
    "raw_observation", "session", "environment", "observation",
    "hypothesis", "evidence_synthesis", "analysis", "recommendation",
    "outcome", "relation", "dialogue", "fact", "experience",
    "insufficient_data", "uncertainty",
})

VALID_STATUSES = frozenset({
    "supported", "rejected", "partially_supported",
    "insufficient_data", "contradictory",
})

VALID_EVIDENCE_ROLES = frozenset({"supports", "contradicts", "background"})

VALID_RELATION_TYPES = frozenset({
    "supports", "contradicts", "derived_from",
    "generated_from", "validated_by", "background",
})

VALID_ACTION_TYPES = frozenset({
    "change_depth", "change_lure", "change_tackle",
    "change_time", "move_spot", "change_bait",
})

VALID_FAILURE_REASONS = frozenset({
    "none", "conditions_not_met", "wrong_spot",
    "weather_changed", "equipment_difference",
    "low_activity", "unknown",
})

VALID_OPERATORS = frozenset({">", "<", ">=", "<=", "==", "!=", "in", "between"})

VALID_CAUSAL_RELATIONS = frozenset({"positive", "negative", "nonlinear"})

ACCEPTED_SCHEMA_VERSION = "1.3"

ISO8601_REQUIRED_KEYS = frozenset({"datetime", "datetime_start", "datetime_end"})


class CorpusError(Exception):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise CorpusError(f"{path.name}:{i}: invalid JSON — {e}")
    return records


def check_nfc(text: str, path: str, line: int):
    normalized = unicodedata.normalize("NFC", text)
    if text != normalized:
        raise CorpusError(f"{path}:{line}: text is not NFC-normalized")


def check_datetime(value: str, rid: str, field: str):
    if not isinstance(value, str) or not value.strip():
        raise CorpusError(f"{rid}: {field} must be a non-empty ISO8601 string")
    try:
        datetime.fromisoformat(value)
    except ValueError:
        raise CorpusError(f"{rid}: {field} '{value}' is not valid ISO8601")


ALLOWED_BASED_ON: dict[str, frozenset] = {
    "analysis": frozenset({"evidence_synthesis", "fact", "experience", "observation", "hypothesis"}),
    "evidence_synthesis": frozenset({"hypothesis", "observation", "fact", "experience"}),
    "hypothesis": frozenset({"observation", "fact", "experience"}),
    "recommendation": frozenset({"analysis", "evidence_synthesis", "fact", "experience"}),
    "outcome": frozenset({"recommendation", "observation", "session"}),
    "dialogue": frozenset({"recommendation", "analysis", "evidence_synthesis"}),
    "uncertainty": frozenset({"observation", "analysis", "evidence_synthesis"}),
}


def validate_corpus(data_dir: str | Path) -> list[str]:
    data_dir = Path(data_dir)
    errors: list[str] = []

    all_records: dict[str, dict[str, Any]] = {}
    raw_hashes: dict[str, str] = {}

    processed_dir = data_dir / "processed"

    for jsonl_path in sorted(processed_dir.glob("*.jsonl")):
        fname = jsonl_path.name
        records = read_jsonl(jsonl_path)
        for rec in records:
            rid = rec.get("id")
            if not rid:
                errors.append(f"{fname}: record missing 'id'")
                continue

            if rid in all_records:
                errors.append(f"duplicate id '{rid}' in {fname}")
            all_records[rid] = rec

            rtype = rec.get("type", "unknown")

            sv = rec.get("schema_version")
            if sv is None:
                errors.append(f"{rid}: missing schema_version")
            elif sv != ACCEPTED_SCHEMA_VERSION:
                errors.append(f"{rid}: schema_version '{sv}' != '{ACCEPTED_SCHEMA_VERSION}'")

            prefix = rec.get("id_prefix", "")
            if prefix and not rid.startswith(prefix):
                errors.append(f"{rid}: id does not start with id_prefix '{prefix}'")

            if rtype not in VALID_TYPES:
                errors.append(f"{rid}: invalid type '{rtype}'")

            if rtype == "raw_observation":
                text = rec.get("text", "")
                h = sha256(text.encode("utf-8")).hexdigest()
                if rid in raw_hashes:
                    if raw_hashes[rid] != h:
                        errors.append(f"{rid}: raw_observation text has changed")
                else:
                    raw_hashes[rid] = h

            try:
                _validate_record(rec, fname, rid, all_records)
            except CorpusError as e:
                errors.append(str(e))

    # based_on references and type consistency
    for rid, rec in all_records.items():
        based_on = rec.get("based_on", [])
        if isinstance(based_on, list):
            for ref in based_on:
                if ref not in all_records:
                    errors.append(f"{rid}: based_on references non-existent '{ref}'")

    rtype = rec.get("type")
    allowed_types = ALLOWED_BASED_ON.get(rtype)
    if allowed_types is not None:
        for ref in based_on:
            ref_rec = all_records.get(ref)
            if ref_rec:
                ref_type = ref_rec.get("type")
                if ref_type not in allowed_types:
                    errors.append(
                        f"{rid}: based_on '{ref}' has type '{ref_type}' "
                        f"not allowed for {rtype} (allowed: {sorted(allowed_types)})"
                    )

    # D8: all IDs (except raw and relation) must be in relation graph
    relation_targets = set()
    for rec in all_records.values():
        if rec.get("type") == "relation":
            if rec.get("from"):
                relation_targets.add(rec["from"])
            if rec.get("to"):
                relation_targets.add(rec["to"])

    for rid, rec in all_records.items():
        rtype = rec.get("type")
        if rtype in ("raw_observation", "relation"):
            continue
        if rid not in relation_targets:
            errors.append(f"{rid}: not referenced in any relation (D8)")

    # D9: no cycles in derived_from chain
    derived = {}
    for rec in all_records.values():
        if rec.get("type") == "relation" and rec.get("relation") == "derived_from":
            derived[rec["from"]] = rec["to"]
    visited_global = set()
    for start in derived:
        if start in visited_global:
            continue
        visited = set()
        current = start
        while current in derived:
            if current in visited:
                errors.append(f"cycle in derived_from graph involving '{current}'")
                break
            visited.add(current)
            visited_global.add(current)
            current = derived[current]

    return errors


def _validate_record(rec: dict, fname: str = "", rid: str = "", all_records: dict | None = None):
    rtype = rec.get("type")

    # ISO8601 validation for datetime fields
    for key in ISO8601_REQUIRED_KEYS:
        if key in rec:
            check_datetime(rec[key], rid, key)

    if rtype == "raw_observation":
        if not isinstance(rec.get("text"), str) or not rec["text"].strip():
            raise CorpusError(f"{rid}: raw_observation.text is required")
        check_nfc(rec["text"], fname, 0)

    elif rtype == "session":
        if not rec.get("water_body_id"):
            raise CorpusError(f"{rid}: session.water_body_id is required")
        for key in ("datetime_start", "datetime_end"):
            val = rec.get(key)
            if val:
                check_datetime(val, rid, key)

    elif rtype == "observation":
        if "success" in rec:
            raise CorpusError(f"{rid}: observation.success is forbidden (D1)")
        if "conditions" not in rec:
            raise CorpusError(f"{rid}: observation.conditions is required")
        if "effort" not in rec:
            raise CorpusError(f"{rid}: observation.effort is required")
        if "result" not in rec:
            raise CorpusError(f"{rid}: observation.result is required")
        catch = rec["result"].get("catch", [])
        if not isinstance(catch, list):
            raise CorpusError(f"{rid}: result.catch must be an array")
        mf = rec.get("missing_factors", [])
        if not isinstance(mf, list):
            raise CorpusError(f"{rid}: missing_factors must be an array")
        oq = rec.get("observation_quality")
        if oq is not None and (not isinstance(oq, (int, float)) or not (0 <= oq <= 1)):
            raise CorpusError(f"{rid}: observation_quality must be 0..1")

    elif rtype == "hypothesis":
        fr = rec.get("formal_rule")
        if not isinstance(fr, dict):
            raise CorpusError(f"{rid}: hypothesis.formal_rule is required (D3)")
        if not fr.get("variable"):
            raise CorpusError(f"{rid}: formal_rule.variable is required")
        op = fr.get("operator")
        if not op:
            raise CorpusError(f"{rid}: formal_rule.operator is required")
        if op not in VALID_OPERATORS:
            raise CorpusError(f"{rid}: formal_rule.operator '{op}' not in {sorted(VALID_OPERATORS)}")
        if "value" not in fr:
            raise CorpusError(f"{rid}: formal_rule.value is required")

    elif rtype == "evidence_synthesis":
        ev = rec.get("evidence", [])
        if not isinstance(ev, list):
            raise CorpusError(f"{rid}: evidence must be an array")
        for i, e in enumerate(ev):
            if e.get("role") not in VALID_EVIDENCE_ROLES:
                raise CorpusError(f"{rid}: evidence[{i}].role '{e.get('role')}' invalid")
            w = e.get("weight")
            if not isinstance(w, (int, float)) or not (0 <= w <= 1):
                raise CorpusError(f"{rid}: evidence[{i}].weight must be 0..1 (D4)")

    elif rtype == "analysis":
        status = rec.get("status")
        if status not in VALID_STATUSES:
            raise CorpusError(f"{rid}: analysis.status '{status}' invalid")
        ac = rec.get("analysis_confidence")
        if ac is not None and (not isinstance(ac, (int, float)) or not (0 <= ac <= 1)):
            raise CorpusError(f"{rid}: analysis_confidence must be 0..1")
        for cl in rec.get("causal_links", []):
            if not cl.get("factor") or not cl.get("effect"):
                raise CorpusError(f"{rid}: causal_link requires factor and effect")
            rel = cl.get("relation")
            if rel and rel not in VALID_CAUSAL_RELATIONS:
                raise CorpusError(f"{rid}: causal_link.relation '{rel}' not in {sorted(VALID_CAUSAL_RELATIONS)}")
            conf = cl.get("confidence")
            if conf is not None and (not isinstance(conf, (int, float)) or not (0 <= conf <= 1)):
                raise CorpusError(f"{rid}: causal_link.confidence must be 0..1")
            confounders = cl.get("confounders")
            if confounders is not None and not isinstance(confounders, list):
                raise CorpusError(f"{rid}: causal_link.confounders must be an array")

    elif rtype == "recommendation":
        if not rec.get("text"):
            raise CorpusError(f"{rid}: recommendation.text is required (D5)")
        actions = rec.get("actions", [])
        if not isinstance(actions, list) or len(actions) == 0:
            raise CorpusError(f"{rid}: recommendation.actions[] is required (D5)")
        for i, a in enumerate(actions):
            if a.get("type") not in VALID_ACTION_TYPES:
                raise CorpusError(f"{rid}: actions[{i}].type '{a.get('type')}' not in allowed DSL")
            if "value" not in a:
                raise CorpusError(f"{rid}: actions[{i}] is missing 'value'")

    elif rtype == "outcome":
        em = rec.get("execution_match")
        if not isinstance(em, (int, float)) or not (0 <= em <= 1):
            raise CorpusError(f"{rid}: outcome.execution_match must be 0..1 (D6)")
        fr = rec.get("failure_reason")
        if not fr:
            raise CorpusError(f"{rid}: outcome.failure_reason is required (D6)")
        if fr not in VALID_FAILURE_REASONS:
            raise CorpusError(f"{rid}: failure_reason '{fr}' not in dictionary")

    elif rtype == "relation":
        if rec.get("relation") not in VALID_RELATION_TYPES:
            raise CorpusError(f"{rid}: relation type '{rec.get('relation')}' invalid")
        if not rec.get("from") or not rec.get("to"):
            raise CorpusError(f"{rid}: relation.from and .to are required")
        w = rec.get("weight")
        if not isinstance(w, (int, float)) or not (0 <= w <= 1):
            raise CorpusError(f"{rid}: relation.weight must be 0..1")
