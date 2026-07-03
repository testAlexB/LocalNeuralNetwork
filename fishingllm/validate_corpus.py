import json
import unicodedata
from abc import ABC, abstractmethod
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────
# Error codes
# ──────────────────────────────────────────────

E_SCHEMA_VERSION = "E_SCHEMA_VERSION"
E_ID_PREFIX = "E_ID_PREFIX"
E_TYPE_INVALID = "E_TYPE_INVALID"
E_RAW_CHANGED = "E_RAW_CHANGED"
E_DATETIME = "E_DATETIME"
E_NFC = "E_NFC"
E_FIELD_REQUIRED = "E_FIELD_REQUIRED"
E_FIELD_FORBIDDEN = "E_FIELD_FORBIDDEN"
E_FIELD_ENUM = "E_FIELD_ENUM"
E_RANGE = "E_RANGE"
E_TYPE = "E_TYPE"
E_CATCH_SCHEMA = "E_CATCH_SCHEMA"
E_ACTION_SCHEMA = "E_ACTION_SCHEMA"
E_EVIDENCE = "E_EVIDENCE"
E_CAUSAL_LINK = "E_CAUSAL_LINK"
E_ANALYSIS_STATUS = "E_ANALYSIS_STATUS"
E_BASED_ON = "E_BASED_ON"
E_MISSING_FACTORS = "E_MISSING_FACTORS"
E_DUPLICATE = "E_DUPLICATE"
E_GRAPH = "E_GRAPH"
E_CYCLE = "E_CYCLE"


class CorpusError(Exception):
    def __init__(self, code: str, message: str, rid: str = "",
                 field: str = "", meta: dict | None = None):
        self.code = code
        self.field = field
        self.rid = rid
        self.meta = meta or {}
        full = f"[{code}] {rid}: {message}".strip() if rid else f"[{code}] {message}"
        super().__init__(full)


def err(code: str, rid: str, msg: str, **kw) -> str:
    return str(CorpusError(code, msg, rid=rid, **kw))


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

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
VALID_CAUSAL_RELATIONS = frozenset({"positive", "negative", "nonlinear"})
VALID_RELATION_TYPES = frozenset({
    "supports", "contradicts", "derived_from",
    "generated_from", "validated_by", "background",
})
VALID_ACTION_TYPES = frozenset({
    "change_depth", "change_lure", "change_tackle",
    "change_time", "move_spot", "change_bait",
})
VALID_FAILURE_REASONS = frozenset({
    "none", "conditions_not_met", "wrong_spot", "weather_changed",
    "equipment_difference", "low_activity", "unknown",
})
VALID_OPERATORS = frozenset({">", "<", ">=", "<=", "==", "!=", "in", "between"})
ACCEPTED_SCHEMA_VERSION = "1.3"
ISO8601_KEYS = frozenset({"datetime", "datetime_start", "datetime_end"})

ALLOWED_BASED_ON: dict[str, frozenset] = {
    "analysis": frozenset({"evidence_synthesis", "fact", "experience", "observation", "hypothesis"}),
    "evidence_synthesis": frozenset({"hypothesis", "observation", "fact", "experience"}),
    "hypothesis": frozenset({"observation", "fact", "experience"}),
    "recommendation": frozenset({"analysis", "evidence_synthesis", "fact", "experience"}),
    "outcome": frozenset({"recommendation", "observation", "session"}),
    "dialogue": frozenset({"recommendation", "analysis", "evidence_synthesis"}),
    "uncertainty": frozenset({"observation", "analysis", "evidence_synthesis"}),
}

ACTION_SCHEMA: dict[str, dict] = {
    "change_time": {"value_type": str, "allowed_values": ["утро", "день", "вечер", "ночь"]},
    "change_depth": {"value_type": list, "item_type": (int, float), "min_items": 2, "max_items": 2},
    "change_lure":  {"value_type": str, "min_length": 1},
    "change_tackle": {"value_type": str, "min_length": 1},
    "move_spot":    {"value_type": str, "min_length": 1},
    "change_bait":  {"value_type": str, "min_length": 1},
}

CATCH_ITEM_SCHEMA = {
    "fish": {"required": True, "type": str, "min_length": 1},
    "count": {"required": True, "type": int, "min": 0},
    "weight_kg": {"required": False, "type": (int, float, type(None)), "min": 0},
}

REQUIRED_FIELDS: dict[str, list[str]] = {
    "raw_observation": ["text"],
    "session": ["water_body_id", "datetime_start", "datetime_end"],
    "observation": ["conditions", "effort", "result"],
    "hypothesis": ["formal_rule", "claim", "confidence"],
    "evidence_synthesis": ["evidence", "summary"],
    "analysis": ["status", "analysis", "analysis_confidence", "evidence_strength", "causal_links"],
    "recommendation": ["text", "actions", "target_species"],
    "outcome": ["execution_match", "failure_reason", "result"],
    "relation": ["from", "to", "relation", "weight"],
    "dialogue": ["messages", "assistant_confidence"],
    "fact": ["content", "source_reliability"],
    "experience": ["content", "source_reliability"],
    "insufficient_data": ["query", "analysis", "status", "recommendation"],
    "uncertainty": ["analysis", "status", "causal_links", "recommendation", "analysis_confidence", "evidence_strength"],
}


# ──────────────────────────────────────────────
# Rule Engine
# ──────────────────────────────────────────────

class Rule(ABC):
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        ...

    def reset(self) -> None:
        ...


class ApplyToTypes(Rule):
    def __init__(self, name: str, types: set[str], description: str = "", **kwargs):
        super().__init__(name, description)
        self._types = types
        self._kwargs = kwargs

    def _applies(self, rec: dict) -> bool:
        return rec.get("type") in self._types

    def check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        if not self._applies(rec):
            return []
        return self._check(rec, all_records)

    def _check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        return []


class FieldPresence(ApplyToTypes):
    def __init__(self, types: set[str], required: list[str]):
        super().__init__("field_presence", types, f"Required fields: {required}")
        self._required = required

    def _check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        rid = rec.get("id", "?")
        codes = []
        for f in self._required:
            if f not in rec:
                codes.append(err(E_FIELD_REQUIRED, rid, f"'{f}' is required", field=f))
        return codes


class FieldType(ApplyToTypes):
    def __init__(self, types: set[str], field: str, expected_type: type | tuple):
        super().__init__("field_type", types, f"{field} must be {expected_type}")
        self._field = field
        self._expected = expected_type

    def _check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        rid = rec.get("id", "?")
        val = rec.get(self._field)
        if val is not None and not isinstance(val, self._expected):
            return [err(E_TYPE, rid, f"{self._field} must be {self._expected}",
                        field=self._field)]
        return []


class NestedFieldEnum(Rule):
    def __init__(self, types: set[str], path: str, allowed: frozenset):
        super().__init__("nested_field_enum", f"{path} in {sorted(allowed)}")
        self._types = types
        self._path = path.split(".")
        self._allowed = allowed

    def check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        rid = rec.get("id", "?")
        if rec.get("type") not in self._types:
            return []
        val = rec
        for key in self._path:
            if not isinstance(val, dict):
                return []
            val = val.get(key)
            if val is None:
                return []
        if isinstance(val, list):
            for v in val:
                if v not in self._allowed:
                    return [err(E_FIELD_ENUM, rid,
                                f"'.'.join(self._path) '{v}' not in {sorted(self._allowed)}",
                                field='.'.join(self._path))]
        elif val not in self._allowed:
            return [err(E_FIELD_ENUM, rid,
                        f"'.'.join(self._path) '{val}' not in {sorted(self._allowed)}",
                        field='.'.join(self._path))]
        return []


class FieldEnum(ApplyToTypes):
    def __init__(self, types: set[str], field: str, allowed: frozenset):
        super().__init__("field_enum", types, f"{field} in {sorted(allowed)}")
        self._field = field
        self._allowed = allowed

    def _check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        rid = rec.get("id", "?")
        val = rec.get(self._field)
        if val is not None and val not in self._allowed:
            return [err(E_FIELD_ENUM, rid,
                        f"{self._field} '{val}' not in {sorted(self._allowed)}",
                        field=self._field)]
        return []


class FieldRange(ApplyToTypes):
    def __init__(self, types: set[str], field: str, lo: float, hi: float):
        super().__init__("field_range", types, f"{field} in [{lo}, {hi}]")
        self._field = field
        self._lo = lo
        self._hi = hi

    def _check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        rid = rec.get("id", "?")
        val = rec.get(self._field)
        if val is not None and isinstance(val, (int, float)):
            if not (self._lo <= val <= self._hi):
                return [err(E_RANGE, rid,
                            f"{self._field} must be [{self._lo}, {self._hi}] (got {val})",
                            field=self._field)]
        return []


class ForbiddenField(ApplyToTypes):
    def __init__(self, types: set[str], field: str):
        super().__init__("forbidden_field", types, f"{field} forbidden")
        self._field = field

    def _check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        rid = rec.get("id", "?")
        if self._field in rec:
            return [err(E_FIELD_FORBIDDEN, rid, f"{self._field} is forbidden",
                        field=self._field)]
        return []


class SchemaVersion(Rule):
    def __init__(self):
        super().__init__("schema_version", "schema_version must be 1.3")

    def check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        rid = rec.get("id", "?")
        sv = rec.get("schema_version")
        if sv is None:
            return [err(E_SCHEMA_VERSION, rid, "missing schema_version")]
        if sv != ACCEPTED_SCHEMA_VERSION:
            return [err(E_SCHEMA_VERSION, rid,
                        f"schema_version '{sv}' != '{ACCEPTED_SCHEMA_VERSION}'")]
        return []


class IdPrefix(Rule):
    def __init__(self):
        super().__init__("id_prefix", "id must start with id_prefix")

    def check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        rid = rec.get("id", "?")
        prefix = rec.get("id_prefix", "")
        if prefix and not rid.startswith(prefix):
            return [err(E_ID_PREFIX, rid,
                        f"id does not start with id_prefix '{prefix}'")]
        return []


class TypeValidity(Rule):
    def __init__(self):
        super().__init__("type_validity", "type must be in VALID_TYPES")

    def check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        rid = rec.get("id", "?")
        rtype = rec.get("type", "unknown")
        if rtype not in VALID_TYPES:
            return [err(E_TYPE_INVALID, rid, f"invalid type '{rtype}'")]
        return []


class RawImmutable(Rule):
    def __init__(self):
        super().__init__("raw_immutable")
        self.reset()

    def reset(self) -> None:
        self._hashes: dict[str, str] = {}

    def check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        if rec.get("type") != "raw_observation":
            return []
        rid = rec.get("id", "?")
        text = rec.get("text", "")
        h = sha256(text.encode("utf-8")).hexdigest()
        if rid in self._hashes:
            if self._hashes[rid] != h:
                return [err(E_RAW_CHANGED, rid, "raw_observation text has changed")]
        else:
            self._hashes[rid] = h
        return []


class ObservationCatchRule(ApplyToTypes):
    def __init__(self):
        super().__init__("observation_catch_schema", {"observation"})

    def _check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        rid = rec.get("id", "?")
        errors: list[str] = []
        result = rec.get("result", {})
        catch = result.get("catch", [])
        if not isinstance(catch, list):
            return [err(E_CATCH_SCHEMA, rid, "result.catch must be an array",
                        field="result.catch")]
        if not catch:
            return errors
        for i, item in enumerate(catch):
            if not isinstance(item, dict):
                errors.append(err(E_CATCH_SCHEMA, rid,
                                  f"result.catch[{i}] must be an object"))
                continue
            for field, rules in CATCH_ITEM_SCHEMA.items():
                val = item.get(field)
                if rules["required"] and val is None:
                    errors.append(err(E_CATCH_SCHEMA, rid,
                                      f"result.catch[{i}].{field} is required",
                                      field=f"result.catch[{i}].{field}"))
                elif val is not None:
                    if not isinstance(val, rules["type"]):
                        errors.append(err(E_CATCH_SCHEMA, rid,
                                          f"result.catch[{i}].{field} must be {rules['type']}"))
                    elif isinstance(val, (int, float)) and val < rules.get("min", 0):
                        errors.append(err(E_CATCH_SCHEMA, rid,
                                          f"result.catch[{i}].{field} must be >= {rules['min']}"))
                    elif isinstance(val, str) and len(val) < rules.get("min_length", 0):
                        errors.append(err(E_CATCH_SCHEMA, rid,
                                          f"result.catch[{i}].{field} too short"))
        return errors


class ActionSchemaRule(ApplyToTypes):
    def __init__(self):
        super().__init__("action_schema", {"recommendation"})

    def _check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        rid = rec.get("id", "?")
        errors: list[str] = []
        actions = rec.get("actions", [])
        if not actions:
            return [err(E_ACTION_SCHEMA, rid, "actions list is empty (need at least 1)")]
        for i, a in enumerate(actions):
            atype = a.get("type")
            schema = ACTION_SCHEMA.get(atype)
            if schema is None:
                continue
            val = a.get("value")
            if val is None:
                errors.append(err(E_ACTION_SCHEMA, rid,
                                  f"actions[{i}] is missing 'value'"))
                continue
            if not isinstance(val, schema["value_type"]):
                errors.append(err(E_ACTION_SCHEMA, rid,
                                  f"actions[{i}].value must be {schema['value_type']}"))
                continue
            if isinstance(val, str) and schema.get("allowed_values"):
                if val not in schema["allowed_values"]:
                    errors.append(err(E_ACTION_SCHEMA, rid,
                                      f"actions[{i}].value '{val}' not in {schema['allowed_values']}"))
            if isinstance(val, str) and schema.get("min_length"):
                if len(val) < schema["min_length"]:
                    errors.append(err(E_ACTION_SCHEMA, rid,
                                      f"actions[{i}].value too short"))
            if isinstance(val, list):
                if len(val) < schema.get("min_items", 0) or len(val) > schema.get("max_items", 999):
                    errors.append(err(E_ACTION_SCHEMA, rid,
                                      f"actions[{i}].value length {len(val)} out of range"))
                for j, v in enumerate(val):
                    if not isinstance(v, schema.get("item_type", object)):
                        errors.append(err(E_ACTION_SCHEMA, rid,
                                          f"actions[{i}].value[{j}] type mismatch"))
        return errors


class EvidenceSchemaRule(ApplyToTypes):
    def __init__(self):
        super().__init__("evidence_schema", {"evidence_synthesis"})

    def _check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        rid = rec.get("id", "?")
        errors: list[str] = []
        ev = rec.get("evidence", [])
        for i, e in enumerate(ev):
            ref_id = e.get("id")
            if ref_id and ref_id not in all_records:
                errors.append(err(E_EVIDENCE, rid,
                                  f"evidence[{i}].id '{ref_id}' does not exist"))
            if e.get("role") not in VALID_EVIDENCE_ROLES:
                errors.append(err(E_EVIDENCE, rid,
                                  f"evidence[{i}].role '{e.get('role')}' invalid"))
            w = e.get("weight")
            if not isinstance(w, (int, float)) or not (0 <= w <= 1):
                errors.append(err(E_EVIDENCE, rid,
                                  f"evidence[{i}].weight must be 0..1"))
        return errors


class CausalLinkRule(ApplyToTypes):
    def __init__(self):
        super().__init__("causal_links", {"analysis", "uncertainty"})

    def _check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        rid = rec.get("id", "?")
        errors: list[str] = []
        for cl in rec.get("causal_links", []):
            if not isinstance(cl, dict):
                errors.append(err(E_CAUSAL_LINK, rid, "causal_link must be an object"))
                continue
            factor = cl.get("factor")
            effect = cl.get("effect")
            if not factor or not isinstance(factor, str):
                errors.append(err(E_CAUSAL_LINK, rid,
                                  "causal_link.factor is required non-empty string"))
            if not effect or not isinstance(effect, str):
                errors.append(err(E_CAUSAL_LINK, rid,
                                  "causal_link.effect is required non-empty string"))
            rel = cl.get("relation")
            if rel is not None and rel not in VALID_CAUSAL_RELATIONS:
                errors.append(err(E_CAUSAL_LINK, rid,
                                  f"causal_link.relation '{rel}' not in {sorted(VALID_CAUSAL_RELATIONS)}"))
            conf = cl.get("confidence")
            if conf is not None and (not isinstance(conf, (int, float)) or not (0 <= conf <= 1)):
                errors.append(err(E_CAUSAL_LINK, rid,
                                  "causal_link.confidence must be 0..1"))
            cf = cl.get("confounders")
            if cf is not None and not isinstance(cf, list):
                errors.append(err(E_CAUSAL_LINK, rid,
                                  "causal_link.confounders must be an array"))
            elif isinstance(cf, list):
                for j, c in enumerate(cf):
                    if not isinstance(c, str):
                        errors.append(err(E_CAUSAL_LINK, rid,
                                          f"causal_link.confounders[{j}] must be string"))
        return errors


class AnalysisStatusRule(ApplyToTypes):
    def __init__(self):
        super().__init__("analysis_status_consistency", {"analysis"})

    def _check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        rid = rec.get("id", "?")
        errors: list[str] = []
        status = rec.get("status")
        if status not in VALID_STATUSES:
            return [err(E_ANALYSIS_STATUS, rid, f"status '{status}' invalid")]

        evs_ids = [ref for ref in rec.get("based_on", []) if ref in all_records
                   and all_records[ref].get("type") == "evidence_synthesis"]
        if not evs_ids:
            return errors

        supports = 0
        contradicts = 0
        background = 0
        for eid in evs_ids:
            evs = all_records[eid]
            for e in evs.get("evidence", []):
                role = e.get("role")
                if role == "supports":
                    supports += 1
                elif role == "contradicts":
                    contradicts += 1
                elif role == "background":
                    background += 1

        if status == "supported" and supports == 0 and contradicts > 0:
            errors.append(err(E_ANALYSIS_STATUS, rid,
                              f"status='supported' but no supporting evidence ({contradicts} contradict)"))
        if status == "rejected" and contradicts == 0:
            errors.append(err(E_ANALYSIS_STATUS, rid,
                              "status='rejected' but no contradicting evidence"))
        if status == "contradictory" and (supports == 0 or contradicts == 0):
            errors.append(err(E_ANALYSIS_STATUS, rid,
                              "status='contradictory' requires both supports and contradicts evidence"))

        return errors


class BasedOnRule(Rule):
    def __init__(self):
        super().__init__("based_on_consistency", "based_on must reference existing records of allowed types")

    def check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        rid = rec.get("id", "?")
        rtype = rec.get("type")
        errors: list[str] = []
        based_on = rec.get("based_on", [])
        if not isinstance(based_on, list):
            return [err(E_BASED_ON, rid, "based_on must be an array")]
        allowed_types = ALLOWED_BASED_ON.get(rtype)
        for ref in based_on:
            if ref not in all_records:
                errors.append(err(E_BASED_ON, rid,
                                  f"based_on references non-existent '{ref}'"))
                continue
            if allowed_types is not None:
                ref_type = all_records[ref].get("type")
                if ref_type not in allowed_types:
                    errors.append(err(E_BASED_ON, rid,
                                      f"based_on '{ref}' has type '{ref_type}' not allowed for {rtype}"))
        return errors


class D8GraphRule(Rule):
    def __init__(self):
        super().__init__("D8_graph", "All IDs (except raw/relation) must be in relation graph")

    def check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        return []

    def check_global(self, all_records: dict[str, dict]) -> list[str]:
        errors: list[str] = []
        targets: set[str] = set()
        for rec in all_records.values():
            if rec.get("type") == "relation":
                if rec.get("from"):
                    targets.add(rec["from"])
                if rec.get("to"):
                    targets.add(rec["to"])
        for rid, rec in all_records.items():
            rtype = rec.get("type")
            if rtype in ("raw_observation", "relation"):
                continue
            if rid not in targets:
                errors.append(err(E_GRAPH, rid, "not referenced in any relation (D8)"))
        return errors


class D9CycleRule(Rule):
    def __init__(self):
        super().__init__("D9_no_cycles", "No cycles in derived_from chain")

    def check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        return []

    def check_global(self, all_records: dict[str, dict]) -> list[str]:
        errors: list[str] = []
        derived: dict[str, str] = {}
        for rec in all_records.values():
            if rec.get("type") == "relation" and rec.get("relation") == "derived_from":
                derived[rec["from"]] = rec["to"]
        visited_global: set[str] = set()
        for start in derived:
            if start in visited_global:
                continue
            visited: set[str] = set()
            current = start
            while current in derived:
                if current in visited:
                    errors.append(err(E_CYCLE, current,
                                      f"cycle in derived_from graph involving '{current}'"))
                    break
                visited.add(current)
                visited_global.add(current)
                current = derived[current]
        return errors


class DatetimeRule(Rule):
    def __init__(self):
        super().__init__("datetime", "ISO8601 datetime validation")

    def check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        rid = rec.get("id", "?")
        errors: list[str] = []
        for key in ISO8601_KEYS:
            val = rec.get(key)
            if val is not None:
                if not isinstance(val, str) or not val.strip():
                    errors.append(err(E_DATETIME, rid,
                                      f"{key} must be a non-empty ISO8601 string"))
                else:
                    try:
                        datetime.fromisoformat(val)
                    except ValueError:
                        errors.append(err(E_DATETIME, rid,
                                          f"{key} '{val}' is not valid ISO8601"))
        return errors


class NfcRule(Rule):
    def __init__(self):
        super().__init__("nfc", "NFC normalization check")

    def check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        errors: list[str] = []
        for key, val in rec.items():
            if isinstance(val, str):
                norm = unicodedata.normalize("NFC", val)
                if val != norm:
                    errors.append(err(E_NFC, rec.get("id", "?"),
                                      f"field '{key}' is not NFC-normalized"))
        return errors


class MissingFactorsRule(ApplyToTypes):
    def __init__(self):
        super().__init__("missing_factors", {"observation"})

    def _check(self, rec: dict, all_records: dict[str, dict]) -> list[str]:
        rid = rec.get("id", "?")
        mf = rec.get("missing_factors", [])
        if not isinstance(mf, list):
            return [err(E_MISSING_FACTORS, rid, "missing_factors must be an array")]
        for i, f in enumerate(mf):
            if not isinstance(f, str):
                return [err(E_MISSING_FACTORS, rid,
                            f"missing_factors[{i}] must be string")]
        return []


# ──────────────────────────────────────────────
# Registry — grouped by level
# ──────────────────────────────────────────────

SCHEMA_RULES: list[Rule] = [
    SchemaVersion(),
    IdPrefix(),
    TypeValidity(),
    DatetimeRule(),
    NfcRule(),
    ForbiddenField({"observation"}, "success"),
    # field presence (all 15 types)
    FieldPresence({"raw_observation"}, ["text"]),
    FieldPresence({"session"}, ["water_body_id", "datetime_start", "datetime_end"]),
    FieldPresence({"observation"}, ["conditions", "effort", "result", "missing_factors"]),
    FieldPresence({"hypothesis"}, ["formal_rule", "claim", "confidence"]),
    FieldPresence({"evidence_synthesis"}, ["evidence", "summary"]),
    FieldPresence({"analysis"}, ["status", "analysis", "causal_links"]),
    FieldPresence({"recommendation"}, ["text", "actions"]),
    FieldPresence({"outcome"}, ["execution_match", "failure_reason"]),
    FieldPresence({"relation"}, ["from", "to", "relation", "weight"]),
    FieldPresence({"dialogue"}, ["messages", "assistant_confidence"]),
    FieldPresence({"fact"}, ["content", "source_reliability"]),
    FieldPresence({"experience"}, ["content", "source_reliability"]),
    FieldPresence({"insufficient_data"}, ["query", "analysis", "status", "recommendation"]),
    FieldPresence({"uncertainty"}, ["analysis", "status", "causal_links", "recommendation", "analysis_confidence", "evidence_strength"]),
    # field enums
    FieldEnum({"analysis"}, "status", VALID_STATUSES),
    FieldEnum({"relation"}, "relation", VALID_RELATION_TYPES),
    FieldEnum({"outcome"}, "failure_reason", VALID_FAILURE_REASONS),
    NestedFieldEnum({"hypothesis"}, "formal_rule.operator", VALID_OPERATORS),
    # field ranges
    FieldRange({"observation"}, "observation_quality", 0.0, 1.0),
    FieldRange({"analysis"}, "analysis_confidence", 0.0, 1.0),
    FieldRange({"outcome"}, "execution_match", 0.0, 1.0),
    FieldRange({"relation"}, "weight", 0.0, 1.0),
    # deep schema
    ObservationCatchRule(),
    ActionSchemaRule(),
    EvidenceSchemaRule(),
    CausalLinkRule(),
    MissingFactorsRule(),
]

SEMANTICS_RULES: list[Rule] = [
    RawImmutable(),
    AnalysisStatusRule(),
    BasedOnRule(),
]

GRAPH_RULES: list[Rule] = [
    D8GraphRule(),
    D9CycleRule(),
]

PER_RECORD_RULES: list[Rule] = [*SCHEMA_RULES, *SEMANTICS_RULES]


# ──────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise CorpusError("E_JSON", f"{path.name}:{i}: invalid JSON — {e}") from e
    return records


def check_record(rec: dict, all_records: dict[str, dict] | None = None) -> list[str]:
    errors: list[str] = []
    for rule in PER_RECORD_RULES:
        errors.extend(rule.check(rec, all_records or {}))
    return errors


def validate_corpus(data_dir: str | Path) -> list[str]:
    data_dir = Path(data_dir)
    errors: list[str] = []
    all_records: dict[str, dict] = {}

    for rule in PER_RECORD_RULES:
        rule.reset()

    processed_dir = data_dir / "processed"

    # Phase 1: load all records, check duplicates
    for jsonl_path in sorted(processed_dir.glob("*.jsonl")):
        for rec in read_jsonl(jsonl_path):
            rid = rec.get("id")
            if not rid:
                errors.append(f"{jsonl_path.name}: record missing 'id'")
                continue
            if rid in all_records:
                errors.append(err(E_DUPLICATE, rid, f"duplicate in {jsonl_path.name}"))
            all_records[rid] = rec

    # Phase 2: validate each record with full cross-reference context
    for rid, rec in all_records.items():
        for rule in PER_RECORD_RULES:
            errors.extend(rule.check(rec, all_records))

    # Phase 3: global rules (graph, cycles)
    for rule in GRAPH_RULES:
        if hasattr(rule, "check_global"):
            errors.extend(rule.check_global(all_records))

    return errors
