#!/usr/bin/env python3
"""Bounded deterministic evidence fusion for Role 4.

The HGB model remains trained on the 37 shared SkillCorner features. This
module adds only explicitly configured competition-Parquet evidence in H/A
log-odds space. It never fits weights, calibrates, or tunes on the target match.

Supported evidence kinds:
  * team_code: categorical team code (+1 Home, -1 Away, 0 neutral)
  * signed: signed numeric signal already in [-1, 1]
  * signed_scaled: tanh(value / scale), positive means Home
  * home_probability: probability in [0, 1] mapped to [-1, +1]
  * paired_count_difference: tanh((home_count-away_count)/scale)

Supported multiplicative quality gates:
  * binary: only a configured on_value has reliability 1
  * category_map: categorical value -> reliability in [0, 1]
  * exp_decay: exp(-max(value, 0)/scale)
  * min_ratio: min(value/scale, 1)
  * pair_min_ratio: min(home/scale, away/scale, 1)

Missing evidence or gate values are neutral/no-vote (gate reliability 0).
Correlated signals are clipped by group caps and then by a global cap.
"""
from __future__ import annotations

from typing import Any
import math

import numpy as np
import pandas as pd
from scipy.special import expit, logit


def _category_key(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return None
        if float(value).is_integer():
            return str(int(value))
        return str(float(value)).upper()
    text = str(value).strip().upper()
    return text if text else None


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise KeyError(f"Required evidence/quality column {column!r} is missing.")
    raw = df[column]
    out = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & out.isna()
    if invalid.any():
        examples = raw[invalid].astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(f"Column {column!r} must be numeric; invalid examples: {examples}")
    vals = out.dropna().to_numpy(dtype=float)
    if vals.size and not np.isfinite(vals).all():
        raise ValueError(f"Column {column!r} contains +/-Inf.")
    return out.astype(float)


def _quality_gate(df: pd.DataFrame, gate: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    kind = gate.get("kind")
    col = gate.get("column")
    if kind in {"binary", "category_map", "exp_decay", "min_ratio"}:
        if not isinstance(col, str) or not col:
            raise ValueError(f"Quality gate {kind!r} must define column.")
    if kind == "binary":
        values = _numeric(df, col)
        on_value = float(gate.get("on_value", 1))
        rel = (values == on_value).astype(float).fillna(0.0).to_numpy()
    elif kind == "category_map":
        if col not in df.columns:
            raise KeyError(f"Quality gate column {col!r} is missing.")
        mapping = gate.get("mapping", {})
        keys = df[col].map(_category_key)
        norm_map = {_category_key(k): float(v) for k, v in mapping.items()}
        rel = keys.map(norm_map).fillna(float(gate.get("unknown_reliability", 0.0))).to_numpy(dtype=float)
    elif kind == "exp_decay":
        values = _numeric(df, col)
        scale = float(gate.get("scale", 0))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Quality gate {col!r}: scale must be > 0.")
        if (values.dropna() < 0).any():
            raise ValueError(f"Quality gate {col!r}: values cannot be negative.")
        rel = np.exp(-values.fillna(np.inf).to_numpy(dtype=float) / scale)
        rel[~np.isfinite(rel)] = 0.0
    elif kind == "min_ratio":
        values = _numeric(df, col)
        scale = float(gate.get("scale", 0))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Quality gate {col!r}: scale must be > 0.")
        rel = (values / scale).clip(lower=0.0, upper=1.0).fillna(0.0).to_numpy(dtype=float)
    elif kind == "pair_min_ratio":
        home_col = gate.get("home_column")
        away_col = gate.get("away_column")
        if not home_col or not away_col:
            raise ValueError("pair_min_ratio requires home_column and away_column.")
        home = _numeric(df, home_col)
        away = _numeric(df, away_col)
        scale = float(gate.get("scale", 0))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("pair_min_ratio scale must be > 0.")
        rel = pd.concat([home / scale, away / scale], axis=1).min(axis=1).clip(0.0, 1.0).fillna(0.0).to_numpy(dtype=float)
        col = f"{home_col}+{away_col}"
    else:
        raise ValueError(f"Unsupported quality gate kind: {kind!r}")

    if not np.isfinite(rel).all() or ((rel < 0.0) | (rel > 1.0)).any():
        raise ValueError(f"Quality gate {col!r} must resolve to finite reliability in [0, 1].")
    return rel, {"kind": kind, "column": col, "mean_reliability": float(np.mean(rel)) if len(rel) else 0.0}


def _rule_signal(df: pd.DataFrame, rule: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    name = str(rule.get("name") or rule.get("column") or "unnamed_rule")
    kind = rule.get("kind")
    n = len(df)
    known_mask = np.zeros(n, dtype=bool)
    unknown_count = 0

    if kind == "paired_count_difference":
        home_col = rule.get("home_column")
        away_col = rule.get("away_column")
        if not home_col or not away_col:
            raise ValueError(f"Rule {name!r} requires home_column and away_column.")
        home = _numeric(df, home_col)
        away = _numeric(df, away_col)
        scale = float(rule.get("scale", 0))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Rule {name!r}: scale must be > 0.")
        known_mask = (home.notna() & away.notna()).to_numpy()
        diff = (home.fillna(0.0) - away.fillna(0.0)).to_numpy(dtype=float)
        signal = np.tanh(diff / scale)
        signal[~known_mask] = 0.0
        source_columns = [home_col, away_col]
        unknown_count = int((~known_mask).sum())
    else:
        column = rule.get("column")
        if not isinstance(column, str) or not column:
            raise ValueError(f"Evidence rule {name!r} must define a non-empty column.")
        if column not in df.columns:
            raise KeyError(f"Evidence rule {name!r}: required column {column!r} is missing.")
        raw = df[column]
        known_mask = raw.notna().to_numpy()
        source_columns = [column]

        if kind == "team_code":
            home_values = {_category_key(v) for v in rule.get("home_values", [1, "HOME", "H"])}
            away_values = {_category_key(v) for v in rule.get("away_values", [-1, "AWAY", "A"])}
            neutral_values = {_category_key(v) for v in rule.get("neutral_values", [0, "UNKNOWN", "NONE"])}
            if home_values & away_values or (home_values | away_values) & neutral_values:
                raise ValueError(f"Rule {name!r}: team value mappings overlap.")
            keys = raw.map(_category_key)
            home_mask = keys.isin(home_values).to_numpy()
            away_mask = keys.isin(away_values).to_numpy()
            neutral_mask = keys.isin(neutral_values).to_numpy() | ~known_mask
            unknown_mask = ~(home_mask | away_mask | neutral_mask)
            if unknown_mask.any() and rule.get("unknown_value_policy", "error") == "error":
                examples = raw.iloc[np.flatnonzero(unknown_mask)].astype(str).drop_duplicates().head(8).tolist()
                raise ValueError(f"Rule {name!r}: unrecognized values in {column!r}: {examples}")
            signal = np.zeros(n, dtype=float)
            signal[home_mask] = 1.0
            signal[away_mask] = -1.0
            known_mask = home_mask | away_mask
            unknown_count = int(unknown_mask.sum())
        elif kind == "signed":
            numeric = _numeric(df, column)
            vals = numeric.fillna(0.0).to_numpy(dtype=float)
            if ((vals < -1.0) | (vals > 1.0)).any():
                raise ValueError(f"Rule {name!r}: signed values must be in [-1, 1].")
            signal = vals
        elif kind == "signed_scaled":
            numeric = _numeric(df, column)
            scale = float(rule.get("scale", 0))
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"Rule {name!r}: scale must be > 0.")
            vals = numeric.fillna(0.0).to_numpy(dtype=float)
            signal = np.tanh(vals / scale)
            known_mask = numeric.notna().to_numpy()
            signal[~known_mask] = 0.0
        elif kind == "home_probability":
            numeric = _numeric(df, column)
            vals = numeric.dropna().to_numpy(dtype=float)
            if vals.size and ((vals < 0.0) | (vals > 1.0)).any():
                raise ValueError(f"Rule {name!r}: home_probability must be in [0, 1].")
            signal = 2.0 * numeric.fillna(0.5).to_numpy(dtype=float) - 1.0
            known_mask = numeric.notna().to_numpy()
        else:
            raise ValueError(f"Rule {name!r}: unsupported kind {kind!r}.")

    quality_gates = rule.get("quality_gates", [])
    gate_reports: list[dict[str, Any]] = []
    for gate in quality_gates:
        reliability, gate_info = _quality_gate(df, gate)
        signal *= reliability
        gate_reports.append(gate_info)

    signal = np.clip(signal, -1.0, 1.0)
    signal[~np.isfinite(signal)] = 0.0
    info = {
        "name": name,
        "kind": kind,
        "source_columns": source_columns,
        "group": str(rule.get("group", "default")),
        "rows_with_directional_value": int(np.count_nonzero(signal)),
        "rows_with_recognized_value": int(np.count_nonzero(known_mask)),
        "unknown_category_rows": int(unknown_count),
        "weight_log_odds": float(rule.get("weight_log_odds", 0.0)),
        "quality_gates": gate_reports,
        "sign_semantics": rule.get("sign_semantics"),
    }
    return signal, info


def apply_directional_evidence(
    df: pd.DataFrame,
    p_home_model: np.ndarray | pd.Series,
    evidence_config: dict[str, Any],
    model_feature_names: list[str] | tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Adjust HGB conditional H/A probabilities with fixed bounded evidence."""
    p_home = np.asarray(p_home_model, dtype=float).reshape(-1)
    if len(p_home) != len(df):
        raise ValueError("p_home_model length does not match input dataframe.")
    if not np.isfinite(p_home).all() or ((p_home < 0.0) | (p_home > 1.0)).any():
        raise ValueError("Model p_home values must be finite and in [0, 1].")

    enabled = bool(evidence_config.get("enabled", False))
    rules = evidence_config.get("rules", []) if enabled else []
    if not isinstance(rules, (list, tuple)):
        raise TypeError("role4_evidence_fusion.rules must be a list or tuple.")

    feature_set = set(model_feature_names)
    group_contributions: dict[str, np.ndarray] = {}
    group_caps = evidence_config.get("group_caps_log_odds", {})
    rule_reports: list[dict[str, Any]] = []
    diagnostics = pd.DataFrame(index=df.index)
    all_used_columns: set[str] = set()

    for rule in rules:
        if not isinstance(rule, dict):
            raise TypeError("Each evidence rule must be a dictionary.")
        if not rule.get("active", True):
            rule_reports.append({"name": rule.get("name", "unnamed"), "active": False,
                                 "reason": rule.get("disabled_reason", "disabled_in_config")})
            continue
        source_cols = ([rule.get("column")] if rule.get("column") else [])
        source_cols += [rule.get("home_column"), rule.get("away_column")]
        gate_cols: list[str] = []
        for gate in rule.get("quality_gates", []):
            gate_cols.extend([gate.get("column"), gate.get("home_column"), gate.get("away_column")])
        cols = [c for c in source_cols + gate_cols if c]
        duplicated = sorted(set(cols) & feature_set)
        if duplicated:
            raise ValueError(
                f"Evidence rule {rule.get('name')!r} would double-count HGB model features: {duplicated}."
            )
        signal, info = _rule_signal(df, rule)
        weight = float(rule.get("weight_log_odds", 0.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Evidence rule {info['name']!r}: weight_log_odds must be finite and >= 0.")
        info["active"] = True
        info["weight_log_odds"] = weight
        group = info["group"]
        group_contributions.setdefault(group, np.zeros(len(df), dtype=float))
        contribution = signal * weight
        group_contributions[group] += contribution
        diagnostics[f"role4_evidence_{info['name']}_signal"] = signal
        info["source_columns"].extend([c for c in gate_cols if c not in info["source_columns"]])
        all_used_columns.update(cols)
        rule_reports.append(info)

    group_delta = np.zeros(len(df), dtype=float)
    group_diag = pd.DataFrame(index=df.index)
    for group, values in group_contributions.items():
        cap = group_caps.get(group)
        if cap is not None:
            cap = float(cap)
            if not math.isfinite(cap) or cap < 0.0:
                raise ValueError(f"Invalid log-odds cap for evidence group {group!r}.")
            values = np.clip(values, -cap, cap)
        group_diag[f"role4_evidence_group_{group}_delta"] = values
        group_delta += values

    global_cap = float(evidence_config.get("max_total_adjustment_log_odds", 0.75))
    if not math.isfinite(global_cap) or global_cap < 0.0:
        raise ValueError("max_total_adjustment_log_odds must be finite and >= 0.")
    if bool(evidence_config.get("uncertainty_gating", True)):
        uncertainty_gate = 4.0 * p_home * (1.0 - p_home)
    else:
        uncertainty_gate = np.ones(len(df), dtype=float)
    delta = np.clip(group_delta * uncertainty_gate, -global_cap, global_cap)

    epsilon = float(evidence_config.get("probability_epsilon", 1e-6))
    if not 0.0 < epsilon < 0.5:
        raise ValueError("probability_epsilon must be in (0, 0.5).")
    adjusted = expit(logit(np.clip(p_home, epsilon, 1.0 - epsilon)) + delta)
    away = 1.0 - adjusted

    diagnostics["role4_evidence_log_odds_delta"] = delta
    diagnostics["role4_evidence_abs_strength"] = np.abs(delta)
    diagnostics["role4_evidence_rule_count"] = np.full(
        len(df), sum(1 for r in rule_reports if r.get("active")), dtype=np.int16
    )
    diagnostics["role4_evidence_uncertainty_gate"] = uncertainty_gate
    for col in group_diag.columns:
        diagnostics[col] = group_diag[col].to_numpy()

    context_only = list(evidence_config.get("context_only_columns", []))
    report = {
        "enabled": enabled,
        "method": "bounded_log_odds_evidence",
        "rows": int(len(df)),
        "rules_configured": len(rules),
        "rules_applied_count": sum(1 for r in rule_reports if r.get("active")),
        "rules_applied": rule_reports,
        "evidence_columns_used": sorted(all_used_columns),
        "context_only_columns_not_used_as_directional_votes": context_only,
        "group_caps_log_odds": group_caps,
        "max_total_adjustment_log_odds": global_cap,
        "uncertainty_gating": bool(evidence_config.get("uncertainty_gating", True)),
        "observed_adjustment_abs_max": float(np.max(np.abs(delta))) if len(delta) else 0.0,
        "observed_adjustment_abs_mean": float(np.mean(np.abs(delta))) if len(delta) else 0.0,
        "weight_policy": "fixed_domain_heuristics_not_fitted_or_calibrated_on_target_match",
        "warning": (
            "Evidence is heuristic and uncalibrated. Evaluate using external-match OOF data only. "
            "Signs for lead-lag columns are assumed from their base feature definitions and should be verified in feature code."
        ),
    }
    return adjusted, away, diagnostics, report


def fuse_three_state_probabilities(
    p_home_live: np.ndarray | pd.Series,
    p_away_live: np.ndarray | pd.Series,
    p_dead_ball: np.ndarray | pd.Series,
    fusion_config: dict[str, Any],
    role3_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert conditional live H/A plus Role 3 P(D) into a simplex H/A/D."""
    if not fusion_config.get("enabled", False):
        raise RuntimeError("Three-state probability fusion is disabled in config.py.")
    if role3_config.get("confirmed_semantics") != role3_config.get("required_semantics_for_fusion"):
        raise RuntimeError("Role 3 must confirm P(D) means out-of-play probability only.")
    if fusion_config.get("require_role4_semantics_confirmation", True) and not fusion_config.get(
        "role4_conditional_semantics_confirmed", False
    ):
        raise RuntimeError("Role 4 conditional live-play semantics have not been confirmed.")

    ph = np.asarray(p_home_live, dtype=float).reshape(-1)
    pa = np.asarray(p_away_live, dtype=float).reshape(-1)
    pd_ = np.asarray(p_dead_ball, dtype=float).reshape(-1)
    if not (len(ph) == len(pa) == len(pd_)):
        raise ValueError("H/A/D probability arrays must have equal lengths.")
    for name, vals in (("p_home_live", ph), ("p_away_live", pa), ("p_dead_ball", pd_)):
        if not np.isfinite(vals).all() or ((vals < 0) | (vals > 1)).any():
            raise ValueError(f"{name} must contain finite probabilities in [0, 1].")
    tolerance = float(fusion_config.get("sum_to_one_tolerance", 1e-9))
    if not np.allclose(ph + pa, 1.0, atol=tolerance, rtol=0.0):
        raise ValueError("Conditional H/A probabilities must sum to one.")
    live_mass = 1.0 - pd_
    p_home = live_mass * ph
    p_away = live_mass * pa
    p_dead = pd_.copy()
    if not np.allclose(p_home + p_away + p_dead, 1.0, atol=tolerance, rtol=0.0):
        raise RuntimeError("Final H/A/D probability simplex check failed.")
    return p_home, p_away, p_dead
