#!/usr/bin/env python3
"""Run Role 4 HGB inference while preserving the full competition Parquet.

The model consumes only its stored 37 SkillCorner feature names. All original
competition columns are retained. Explicitly configured extra columns can
adjust H/A probabilities through role4_evidence_fusion.py. Role 3 P(D) can be
read from the same Parquet or safely joined from a separate Role 3 Parquet.

By default, final H/A/D fusion is fail-closed until config.py confirms the
semantics of Role 3 P(D) and the conditional interpretation of Role 4 H/A.
Use --conditional-only for a clearly labelled intermediate export, not as the
final Role 5 contract.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

from role4_evidence_fusion import apply_directional_evidence, fuse_three_state_probabilities

LOG = logging.getLogger("role4_inference")


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono config.py: {config_path}")
    spec = importlib.util.spec_from_file_location("role4_runtime_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nie można załadować konfiguracji: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_config"):
        raise AttributeError("config.py musi udostępniać get_config().")
    return module.get_config()


def _default_config_path() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "config.py",
        here.parent / "config.py",
        Path.cwd() / "config.py",
        Path.cwd() / "src" / "possession" / "config.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0]


def _normalise_key(series: pd.Series) -> pd.Series:
    """Normalise join keys so integer 12 and string '12' match safely."""
    def one(value: Any) -> str | None:
        if pd.isna(value):
            return None
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return None
            if float(value).is_integer():
                return str(int(value))
        return str(value).strip()
    return series.map(one)


def _resolve_identifier_column(
    df: pd.DataFrame,
    preferred: str | None,
    aliases: tuple[str, ...] | list[str] = (),
    override: str | None = None,
    kind: str = "identifier",
    required: bool = False,
) -> str | None:
    """Resolve an identifier using an explicit CLI override, config name, then aliases."""
    if override:
        if override not in df.columns:
            raise KeyError(
                f"Wskazana kolumna {kind} {override!r} nie istnieje. "
                f"Dostępne kolumny: {list(df.columns)}"
            )
        return override
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(str(x) for x in aliases if x)
    # Preserve order and avoid duplicate checks.
    for candidate in dict.fromkeys(candidates):
        if candidate in df.columns:
            return candidate
    if required:
        raise KeyError(
            f"Nie znaleziono kolumny {kind}. Sprawdzono: {candidates}. "
            f"Dostępne kolumny: {list(df.columns)}"
        )
    return None


def _load_role3_column(
    df: pd.DataFrame,
    role3_path: Path | None,
    match_col: str | None,
    frame_col: str | None,
    p_dead_col: str,
    identifier_cfg: dict[str, Any] | None = None,
    role3_frame_column_override: str | None = None,
    role3_match_column_override: str | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Attach P(D), joining on available identifiers without inventing match IDs.

    If the competition input has no match_id, a frame-only join is allowed only
    when both files unambiguously represent one match. A separate Role 3 file
    always needs a frame identifier.
    """
    if role3_path is None:
        return df, p_dead_col in df.columns
    if frame_col is None:
        raise ValueError(
            "Nie można dołączyć osobnego Parqueta Roli 3 bez kolumny klatki. "
            "Podaj --frame-column albo zachowaj wspólny identyfikator klatki w obu plikach."
        )
    if p_dead_col in df.columns:
        raise ValueError(
            f"Kolumna {p_dead_col!r} jest już w Parquecie wejściowym; nie podawaj jednocześnie --role3-parquet."
        )

    role3 = pd.read_parquet(role3_path)
    id_cfg = identifier_cfg or {}
    role3_match = _resolve_identifier_column(
        role3,
        match_col,
        id_cfg.get("match_id_aliases", ()),
        override=role3_match_column_override,
        kind="match_id w Parquecie Roli 3",
        required=False,
    )
    role3_frame = _resolve_identifier_column(
        role3,
        frame_col,
        id_cfg.get("frame_id_aliases", ()),
        override=role3_frame_column_override,
        kind="frame_id w Parquecie Roli 3",
        required=True,
    )
    if p_dead_col not in role3.columns:
        raise KeyError(f"Role 3 Parquet nie zawiera kolumny prawdopodobieństwa {p_dead_col!r}.")

    left = df.copy()
    right = role3.copy()
    if left[frame_col].isna().any() or right[role3_frame].isna().any():
        raise ValueError("Parquet wejściowy lub Role 3 ma brakujące identyfikatory klatek.")

    join_spec: list[tuple[str, str, str]] = []
    # Match ID is used when available in both files. If missing in one or both,
    # only allow a frame-only join when the available side represents one match.
    if match_col is not None and role3_match is not None:
        if left[match_col].isna().any() or right[role3_match].isna().any():
            raise ValueError("Parquet wejściowy lub Role 3 ma brakujące match_id.")
        left_match_count = left[match_col].nunique(dropna=True)
        right_match_count = right[role3_match].nunique(dropna=True)
        join_spec.append((match_col, role3_match, "__role4_match_key"))
    else:
        left_match_count = left[match_col].nunique(dropna=True) if match_col else 1
        right_match_count = right[role3_match].nunique(dropna=True) if role3_match else 1
        if left_match_count != 1 or right_match_count != 1:
            raise ValueError(
                "Role 3 nie ma wspólnego klucza match_id z wejściem. Join tylko po klatce "
                "jest bezpieczny wyłącznie dla pojedynczego meczu po obu stronach."
            )

    join_spec.append((frame_col, role3_frame, "__role4_frame_key"))
    for left_col, right_col, key in join_spec:
        left[key] = _normalise_key(left[left_col])
        right[key] = _normalise_key(right[right_col])

    join_keys = [item[2] for item in join_spec]
    if left[join_keys].isna().any().any() or right[join_keys].isna().any().any():
        raise ValueError("Nie można normalizować części kluczy do joinu Roli 3.")
    if left[join_keys].duplicated().any():
        raise ValueError(f"Parquet wejściowy ma zduplikowane klucze joinu: {join_keys}.")
    if right[join_keys].duplicated().any():
        raise ValueError(f"Role 3 Parquet ma zduplikowane klucze joinu: {join_keys}.")

    right = right[join_keys + [p_dead_col]]
    merged = left.merge(right, on=join_keys, how="left", validate="one_to_one", sort=False, indicator=True)
    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        examples = merged.loc[merged["_merge"] != "both", join_keys].head(5).to_dict("records")
        raise ValueError(f"Role 3 nie pokrywa {unmatched} klatek wejściowych; przykłady: {examples}")
    merged = merged.drop(columns=["_merge", *join_keys])
    return merged, True


def _predict_hgb(artifact_path: Path, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono modelu: {artifact_path}")
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict) or "estimator" not in artifact or "feature_names" not in artifact:
        raise ValueError(
            "Nieznany format artefaktu. Oczekuję słownika z kluczami 'estimator' i 'feature_names', "
            "takiego jak zapisuje train_role4_models.py."
        )
    estimator = artifact["estimator"]
    features = list(artifact["feature_names"])
    if len(features) != 37 or len(set(features)) != 37:
        raise ValueError(f"Artefakt musi zawierać dokładnie 37 unikalnych cech; otrzymano {len(features)}.")
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise KeyError(f"Competition Parquet nie zawiera cech modelu: {missing}")
    X = df.loc[:, features].copy()
    for col in features:
        converted = pd.to_numeric(X[col], errors="coerce")
        invalid = X[col].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"Cecha modelowa {col!r} zawiera nienumeryczne wartości.")
        X[col] = converted.astype(float)
    numeric = X.to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError("Cechy modelowe zawierają +/-Inf.")
    classes = np.asarray(estimator.classes_)
    home_idx = np.flatnonzero(classes == 0)
    away_idx = np.flatnonzero(classes == 1)
    if len(home_idx) != 1 or len(away_idx) != 1:
        raise ValueError(f"Model nie zawiera oczekiwanych klas H=0 i A=1: {classes.tolist()}")
    probs = estimator.predict_proba(X)
    p_home = probs[:, int(home_idx[0])]
    p_away = probs[:, int(away_idx[0])]
    if not np.isfinite(probs).all() or ((probs < 0.0) | (probs > 1.0)).any():
        raise ValueError("HGB zwrócił nieprawidłowe prawdopodobieństwa.")
    if not np.allclose(p_home + p_away, 1.0, atol=1e-9, rtol=0.0):
        raise ValueError("HGB H/A probabilities do not sum to one.")
    return p_home, p_away, features, str(artifact.get("model_name", "hist_gradient_boosting"))


def run_inference(
    input_path: Path,
    model_path: Path,
    output_path: Path,
    config_path: Path,
    role3_path: Path | None = None,
    conditional_only: bool = False,
    allow_column_count_mismatch: bool = False,
    match_column_override: str | None = None,
    frame_column_override: str | None = None,
    allow_row_order_without_frame: bool = False,
    role3_probability_column_override: str | None = None,
    role3_frame_column_override: str | None = None,
    role3_match_column_override: str | None = None,
    confirm_role3_out_of_play_only: bool = False,
    confirm_role4_conditional_live_semantics: bool = False,
    enable_three_state_fusion: bool = False,
) -> dict[str, Any]:
    cfg = _load_config(config_path)
    full_cfg = cfg["full_match_parquet"]
    # Copy these small sections before applying explicit run-time confirmations.
    # The on-disk config remains unchanged and no semantic assumption is hidden.
    role3_cfg = dict(cfg["role3_output"])
    fusion_cfg = dict(cfg["probability_fusion"])
    if role3_probability_column_override:
        role3_cfg["probability_column"] = role3_probability_column_override
    if confirm_role3_out_of_play_only:
        role3_cfg["confirmed_semantics"] = role3_cfg["required_semantics_for_fusion"]
    if confirm_role4_conditional_live_semantics:
        fusion_cfg["role4_conditional_semantics_confirmed"] = True
    if enable_three_state_fusion:
        fusion_cfg["enabled"] = True
    evidence_cfg = cfg.get("role4_evidence_fusion", {"enabled": False, "rules": []})

    df = pd.read_parquet(input_path)
    original_columns = list(df.columns)
    expected_cols = int(full_cfg.get("expected_columns", 111))
    if not allow_column_count_mismatch and len(original_columns) != expected_cols:
        raise ValueError(
            f"Pełny Parquet powinien mieć {expected_cols} kolumn, ma {len(original_columns)}. "
            "Zweryfikuj schemat albo użyj --allow-column-count-mismatch po świadomej kontroli."
        )

    match_col = _resolve_identifier_column(
        df,
        full_cfg.get("match_id_column", "match_id"),
        full_cfg.get("match_id_aliases", ()),
        override=match_column_override,
        kind="match_id",
        required=False,
    )
    frame_col = _resolve_identifier_column(
        df,
        full_cfg.get("frame_id_column", "frame_id"),
        full_cfg.get("frame_id_aliases", ()),
        override=frame_column_override,
        kind="frame_id",
        required=False,
    )

    if match_col is None:
        if not full_cfg.get("allow_single_match_without_match_id", True):
            raise KeyError(
                "Parquet nie zawiera match_id. Dla pojedynczego meczu można to jawnie dopuścić "
                "w full_match_parquet.allow_single_match_without_match_id."
            )
        LOG.warning(
            "Parquet nie zawiera kolumny match_id. Przyjmuję, że plik reprezentuje dokładnie jeden mecz; "
            "nie dodaję syntetycznego match_id do danych wyjściowych."
        )
    if frame_col is None:
        if not (allow_row_order_without_frame or full_cfg.get("allow_row_order_without_frame_id", False)):
            raise KeyError(
                "Nie znaleziono identyfikatora klatki. Sprawdzono frame_id oraz aliasy "
                f"{list(full_cfg.get('frame_id_aliases', ())) }. "
                f"Dostępne kolumny: {original_columns}. "
                "Jeśli Parquet jest już w chronologicznej kolejności, możesz jawnie użyć "
                "--allow-row-order-without-frame; bez tego nie zakładam kolejności wierszy."
            )
        LOG.warning("Brak kolumny frame_id/frame; używam kolejności wierszy jako identyfikatora klatki.")

    # Validate keys where they exist. Single-match files need only a frame key;
    # multi-match files require both match and frame identifiers.
    if match_col is not None and df[match_col].isna().any():
        raise ValueError(f"Kolumna {match_col!r} zawiera brakujące match_id.")
    if match_col is not None and df[match_col].nunique(dropna=True) > 1 and frame_col is None:
        raise ValueError("Wielomeczowy Parquet wymaga identyfikatora klatki.")
    if frame_col is not None and df[frame_col].isna().any():
        raise ValueError(f"Kolumna {frame_col!r} zawiera brakujące identyfikatory klatek.")

    duplicate_key_cols = ([match_col] if match_col else []) + ([frame_col] if frame_col else [])
    if duplicate_key_cols and df.duplicated(duplicate_key_cols).any():
        raise ValueError(f"Parquet wejściowy ma zduplikowane kombinacje {duplicate_key_cols}.")

    df["__role4_original_order"] = np.arange(len(df), dtype=np.int64)
    id_cfg = {
        "match_id_aliases": full_cfg.get("match_id_aliases", ()),
        "frame_id_aliases": full_cfg.get("frame_id_aliases", ()),
    }
    df, has_p_dead = _load_role3_column(
        df,
        role3_path,
        match_col,
        frame_col,
        role3_cfg["probability_column"],
        id_cfg,
        role3_frame_column_override=role3_frame_column_override,
        role3_match_column_override=role3_match_column_override,
    )
    df = df.sort_values("__role4_original_order", kind="stable").reset_index(drop=True)

    p_home_model, p_away_model, model_features, model_name = _predict_hgb(model_path, df)
    p_home_live, p_away_live, evidence_diag, evidence_report = apply_directional_evidence(
        df=df,
        p_home_model=p_home_model,
        evidence_config=evidence_cfg,
        model_feature_names=model_features,
    )

    if conditional_only:
        for col in ("p_home_ha", "p_away_ha"):
            if col in df.columns:
                raise ValueError(f"Input Parquet already contains {col!r}; refusing to overwrite.")
        df["p_home_ha"] = p_home_live
        df["p_away_ha"] = p_away_live
        if has_p_dead:
            p_dead = pd.to_numeric(df[role3_cfg["probability_column"]], errors="coerce")
            if p_dead.isna().any() or not np.isfinite(p_dead.to_numpy(dtype=float)).all():
                raise ValueError("P(D) zawiera NaN lub wartości nieskończone.")
            if ((p_dead < 0.0) | (p_dead > 1.0)).any():
                raise ValueError("P(D) musi mieścić się w [0, 1].")
    else:
        if not has_p_dead:
            raise ValueError(
                f"Brak {role3_cfg['probability_column']!r}. Podaj --role3-parquet albo dodaj wynik Roli 3 do wejściowego Parquetu."
            )
        p_dead = pd.to_numeric(df[role3_cfg["probability_column"]], errors="coerce").to_numpy(dtype=float)
        ph, pa, pd_ = fuse_three_state_probabilities(p_home_live, p_away_live, p_dead, fusion_cfg, role3_cfg)
        # p_dead is the Role 3 source column by default. It is expected to
        # already be present after the Role 3 join, so only H/A output columns
        # are unconditional collision errors. If a custom Role 3 source name is
        # configured, an unrelated pre-existing p_dead is ambiguous.
        for col in ("p_home", "p_away"):
            if col in df.columns:
                raise ValueError(f"Input Parquet already contains {col!r}; refusing to overwrite.")
        if (
            "p_dead" in df.columns
            and role3_cfg["probability_column"] != "p_dead"
        ):
            raise ValueError(
                "Input already contains 'p_dead' but Role 3 uses a different source column; "
                "refusing to overwrite an ambiguous probability column."
            )
        df["p_home"] = ph
        df["p_away"] = pa
        df["p_dead"] = pd_
        sums = df[["p_home", "p_away", "p_dead"]].sum(axis=1).to_numpy(dtype=float)
        if not np.allclose(sums, 1.0, atol=float(fusion_cfg.get("sum_to_one_tolerance", 1e-9)), rtol=0.0):
            raise RuntimeError("Final H/A/D probability vector does not sum to one.")

    df = df.drop(columns=["__role4_original_order"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    # Diagnostics must use the *resolved physical column name* (e.g. `frame`),
    # never the canonical config label (`frame_id`) blindly. This prevents the
    # KeyError seen when the input schema stores the frame key as `frame`.
    diagnostic_candidates: list[str] = []
    if match_col is not None:
        diagnostic_candidates.append(match_col)
    if frame_col is not None:
        diagnostic_candidates.append(frame_col)
    if full_cfg.get("period_column"):
        diagnostic_candidates.append(str(full_cfg["period_column"]))
    diagnostic_columns = list(dict.fromkeys(c for c in diagnostic_candidates if c in df.columns))
    diagnostics_df = df.loc[:, diagnostic_columns].copy()
    if frame_col is None:
        # This is only reachable after explicit row-order opt-in above.
        diagnostics_df["__role4_row_order"] = np.arange(len(df), dtype=np.int64)
    diagnostics_df["p_home_model"] = p_home_model
    diagnostics_df["p_away_model"] = p_away_model
    diagnostics_df["p_home_ha"] = p_home_live
    diagnostics_df["p_away_ha"] = p_away_live
    for col in evidence_diag.columns:
        diagnostics_df[col] = evidence_diag[col].to_numpy()
    diagnostics_path = output_path.with_name(output_path.stem + "_frame_diagnostics.parquet")
    diagnostics_df.to_parquet(diagnostics_path, index=False)

    evidence_columns: set[str] = set()
    quality_gate_columns: set[str] = set()
    active_rules = evidence_cfg.get("rules", []) if evidence_cfg.get("enabled", False) else []
    for rule in active_rules:
        if not rule.get("active", True):
            continue
        for key in ("column", "home_column", "away_column"):
            if rule.get(key):
                evidence_columns.add(rule[key])
        for gate in rule.get("quality_gates", []):
            for key in ("column", "home_column", "away_column"):
                if gate.get(key):
                    quality_gate_columns.add(gate[key])
    id_cols = {c for c in (match_col, frame_col) if c is not None}
    period_col = full_cfg.get("period_column")
    if period_col in df.columns:
        id_cols.add(period_col)
    if has_p_dead:
        evidence_columns.add(role3_cfg["probability_column"])
    context_only_columns = set(evidence_cfg.get("context_only_columns", [])) & set(original_columns)
    routed = set(model_features) | evidence_columns | quality_gate_columns | id_cols | context_only_columns
    original_routed = [c for c in original_columns if c in routed]
    report = {
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "diagnostics_path": str(diagnostics_path.resolve()),
        "model_artifact": str(model_path.resolve()),
        "model_name": model_name,
        "rows": int(len(df)),
        "matches": int(df[match_col].nunique()) if match_col else 1,
        "match_id_column_used": match_col,
        "frame_id_column_used": frame_col,
        "sequence_key_source": frame_col if frame_col else "parquet_row_order",
        "single_match_assumption_used": match_col is None,
        "input_columns_preserved": len(original_columns),
        "expected_input_columns": expected_cols,
        "output_columns": len(df.columns),
        "model_feature_count": len(model_features),
        "model_features": model_features,
        "evidence_fusion": evidence_report,
        "evidence_columns_used": sorted(evidence_columns),
        "quality_gate_columns_used": sorted(quality_gate_columns),
        "context_only_columns_preserved": sorted(context_only_columns),
        "role3_p_dead_present": bool(has_p_dead),
        "output_mode": "conditional_HA_only" if conditional_only else "final_HAD_simplex",
        "original_columns_missing_after_inference": [c for c in original_columns if c not in df.columns],
        "preserved_but_not_used_by_role4": [c for c in original_columns if c not in routed],
        "probability_contract": (
            "conditional H/A only; p_home_ha + p_away_ha = 1; not final Role 5 simplex"
            if conditional_only else "p_home + p_away + p_dead = 1"
        ),
    }
    report_path = output_path.with_name(output_path.stem + "_routing_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Model: %s", model_name)
    LOG.info("Rows: %d | matches: %d | model features: %d", len(df), report["matches"], len(model_features))
    LOG.info(
        "Resolved identifier columns: match=%s | frame=%s",
        match_col or "single-match assumption",
        frame_col or "explicit row order",
    )
    LOG.info("Evidence rules applied: %d", len(evidence_report["rules_applied"]))
    LOG.info("Output Parquet: %s", output_path)
    LOG.info("Frame diagnostics: %s", diagnostics_path)
    LOG.info("Routing report: %s", report_path)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Role 4 HGB inference + bounded evidence fusion.")
    parser.add_argument("--parquet", required=True, help="Competition Parquet with the configured 111-column schema.")
    parser.add_argument("--model", default="role4_artifacts/role4_hist_gradient_boosting.joblib")
    parser.add_argument("--config", default=str(_default_config_path()))
    parser.add_argument("--role3-parquet", default=None, help="Optional Role 3 Parquet; joined by match_id+frame_id when both exist, otherwise frame-only for one match.")
    parser.add_argument("--output", default="role4_predictions.parquet")
    parser.add_argument("--conditional-only", action="store_true", help="Export conditional H/A only; not final H/A/D contract.")
    parser.add_argument("--allow-column-count-mismatch", action="store_true")
    parser.add_argument("--match-column", default=None, help="Explicit match identifier column; optional for a single-match Parquet.")
    parser.add_argument("--frame-column", default=None, help="Explicit frame identifier column (for example: frame or frame_id).")
    parser.add_argument("--allow-row-order-without-frame", action="store_true", help="Explicitly trust input row order if no frame identifier exists.")
    parser.add_argument("--role3-probability-column", default=None, help="Override P(D) column name in the Role 3 Parquet.")
    parser.add_argument("--role3-frame-column", default=None, help="Explicit Role 3 frame-key column override.")
    parser.add_argument("--role3-match-column", default=None, help="Explicit Role 3 match-key column override.")
    parser.add_argument("--confirm-role3-out-of-play-only", action="store_true", help="Explicitly confirm Role 3 P(D) means out-of-play only, not contested in-play.")
    parser.add_argument("--confirm-role4-conditional-live", action="store_true", help="Explicitly confirm Role 4 H/A is interpreted conditional on live play.")
    parser.add_argument("--enable-three-state-fusion", action="store_true", help="Enable H/A/D fusion for this run after semantic checks.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    run_inference(
        input_path=Path(args.parquet),
        model_path=Path(args.model),
        output_path=Path(args.output),
        config_path=Path(args.config),
        role3_path=Path(args.role3_parquet) if args.role3_parquet else None,
        conditional_only=bool(args.conditional_only),
        allow_column_count_mismatch=bool(args.allow_column_count_mismatch),
        match_column_override=args.match_column,
        frame_column_override=args.frame_column,
        allow_row_order_without_frame=bool(args.allow_row_order_without_frame),
        role3_probability_column_override=args.role3_probability_column,
        role3_frame_column_override=args.role3_frame_column,
        role3_match_column_override=args.role3_match_column,
        confirm_role3_out_of_play_only=bool(args.confirm_role3_out_of_play_only),
        confirm_role4_conditional_live_semantics=bool(args.confirm_role4_conditional_live),
        enable_three_state_fusion=bool(args.enable_three_state_fusion),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOG.exception("Role 4 inference failed: %s", exc)
        raise SystemExit(1)
