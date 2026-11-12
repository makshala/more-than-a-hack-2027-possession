#!/usr/bin/env python3
"""One-command Role 4 pipeline: competition Parquet + Role 3 P(D) -> 4 columns.

Inputs:
  1) Full competition Parquet (expected 111 columns, one match).
  2) Role 3 Parquet containing P(D) and a frame identifier.

Output:
  frame_id, p_home, p_away, p_dead

This wrapper does not retrain HGB. It calls the existing Role 4 inference and
Evidence Fusion code, performs the Role 3 join and final H/A/D probability
fusion, then exports only the compact frame-probability contract.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import role4_inference as inference  # noqa: E402

LOG = logging.getLogger("run_role4_pipeline")


def _default_config_path() -> Path:
    return HERE.parent / "config.py"


def _semantic_preflight(
    cfg: dict[str, Any],
    confirm_role3_out_of_play_only: bool,
    confirm_role4_conditional_live: bool,
) -> tuple[bool, bool]:
    role3_cfg = cfg["role3_output"]
    fusion_cfg = cfg["probability_fusion"]
    role3_required = role3_cfg.get("required_semantics_for_fusion", "out_of_play_only")
    role3_ok = role3_cfg.get("confirmed_semantics") == role3_required or confirm_role3_out_of_play_only
    role4_ok = bool(fusion_cfg.get("role4_conditional_semantics_confirmed", False)) or confirm_role4_conditional_live

    missing: list[str] = []
    if not role3_ok:
        missing.append(
            "Role 3: potwierdź, że P(D) oznacza wyłącznie piłkę poza grą (out-of-play), "
            "a nie contested in-play"
        )
    if not role4_ok:
        missing.append(
            "Role 4: potwierdź interpretację H/A jako prawdopodobieństw warunkowych dla live play"
        )
    if missing:
        raise ValueError(
            "Blokada semantyczna fuzji H/A/D. Przed uruchomieniem finalnego wektora potwierdź: "
            + "; ".join(missing)
            + ". Po potwierdzeniu ustaw odpowiednie pola w config.py lub użyj jawnych flag CLI."
        )
    return role3_ok, role4_ok


def run_role4_pipeline(
    competition_parquet: Path,
    role3_parquet: Path,
    model_path: Path,
    config_path: Path,
    output_path: Path,
    role3_probability_column: str | None = None,
    competition_frame_column: str | None = None,
    role3_frame_column: str | None = None,
    competition_match_column: str | None = None,
    role3_match_column: str | None = None,
    output_frame_column: str = "frame_id",
    full_enriched_output: Path | None = None,
    confirm_role3_out_of_play_only: bool = False,
    confirm_role4_conditional_live: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Execute one-match Role 4 and emit a compact 4-column probability Parquet."""
    for label, path in (
        ("competition Parquet", competition_parquet),
        ("Role 3 Parquet", role3_parquet),
        ("HGB model", model_path),
        ("config.py", config_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Nie znaleziono {label}: {path}")

    output_path = output_path.resolve()
    if output_path in {competition_parquet.resolve(), role3_parquet.resolve(), model_path.resolve()}:
        raise ValueError("Plik wyjściowy nie może nadpisywać pliku wejściowego ani modelu.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Plik wyjściowy już istnieje: {output_path}. Użyj --overwrite.")
    if full_enriched_output is not None:
        full_enriched_output = full_enriched_output.resolve()
        if full_enriched_output in {output_path, competition_parquet.resolve(), role3_parquet.resolve(), model_path.resolve()}:
            raise ValueError("Pełny plik wzbogacony nie może nadpisywać innego wejścia ani małego eksportu.")
        if full_enriched_output.exists() and not overwrite:
            raise FileExistsError(f"Pełny plik wzbogacony już istnieje: {full_enriched_output}. Użyj --overwrite.")
        full_enriched_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = inference._load_config(config_path)
    role3_ok, role4_ok = _semantic_preflight(
        cfg,
        confirm_role3_out_of_play_only=confirm_role3_out_of_play_only,
        confirm_role4_conditional_live=confirm_role4_conditional_live,
    )
    configured_p_dead = cfg["role3_output"].get("probability_column", "p_dead")
    p_dead_column = role3_probability_column or configured_p_dead

    # Full enriched output and diagnostics are temporary. The wrapper's public
    # contract remains exactly frame_id + H/A/D probabilities.
    with tempfile.TemporaryDirectory(prefix="role4_pipeline_", dir=output_path.parent) as temp_dir:
        temp_full = Path(temp_dir) / "role4_full_intermediate.parquet"
        full_report = inference.run_inference(
            input_path=competition_parquet,
            model_path=model_path,
            output_path=temp_full,
            config_path=config_path,
            role3_path=role3_parquet,
            conditional_only=False,
            allow_column_count_mismatch=False,
            match_column_override=competition_match_column,
            frame_column_override=competition_frame_column,
            allow_row_order_without_frame=False,
            role3_probability_column_override=p_dead_column,
            role3_frame_column_override=role3_frame_column,
            role3_match_column_override=role3_match_column,
            confirm_role3_out_of_play_only=confirm_role3_out_of_play_only,
            confirm_role4_conditional_live_semantics=confirm_role4_conditional_live,
            enable_three_state_fusion=True,
        )

        frame_source = full_report.get("frame_id_column_used")
        if not frame_source:
            raise ValueError(
                "Inferencja nie rozpoznała kolumny frame. Wrapper wymaga jawnego klucza klatki; "
                "nie używa kolejności wierszy jako zamiennika."
            )
        if int(full_report.get("matches", 1)) != 1:
            raise ValueError(
                "Kompaktowy eksport frame + P(H)/P(A)/P(D) jest przeznaczony dla jednego meczu. "
                "Wejście zawiera więcej niż jeden match_id; bez match_id klatki mogłyby być niejednoznaczne."
            )

        full_output = pd.read_parquet(temp_full)
        required_output = [frame_source, "p_home", "p_away", "p_dead"]
        missing = [col for col in required_output if col not in full_output.columns]
        if missing:
            raise KeyError(f"Wynik Role 4 nie zawiera wymaganych kolumn: {missing}")
        compact = full_output.loc[:, required_output].copy()
        compact = compact.rename(columns={frame_source: output_frame_column})
        expected_columns = [output_frame_column, "p_home", "p_away", "p_dead"]
        compact = compact.loc[:, expected_columns]

        if compact[output_frame_column].isna().any():
            raise ValueError("W wyjściowym Parquecie występują brakujące identyfikatory klatek.")
        if compact[output_frame_column].duplicated().any():
            examples = compact.loc[
                compact[output_frame_column].duplicated(keep=False), output_frame_column
            ].head(8).tolist()
            raise ValueError(f"Wyjściowy frame key nie jest unikalny dla jednego meczu: {examples}")

        prob_cols = ["p_home", "p_away", "p_dead"]
        for col in prob_cols:
            compact[col] = pd.to_numeric(compact[col], errors="coerce")
        probs = compact[prob_cols].to_numpy(dtype=float)
        if not np.isfinite(probs).all():
            raise ValueError("Finalny wektor H/A/D zawiera NaN lub wartości nieskończone.")
        if ((probs < 0.0) | (probs > 1.0)).any():
            raise ValueError("Finalne prawdopodobieństwa H/A/D muszą należeć do [0, 1].")
        sums = probs.sum(axis=1)
        tolerance = float(cfg["probability_fusion"].get("sum_to_one_tolerance", 1e-9))
        max_sum_error = float(np.max(np.abs(sums - 1.0))) if len(sums) else 0.0
        if not np.allclose(sums, 1.0, atol=tolerance, rtol=0.0):
            raise RuntimeError(
                f"Finalny wektor H/A/D nie sumuje się do 1; max abs error={max_sum_error:.3g}."
            )

        if full_enriched_output is not None:
            temp_full_output = full_enriched_output.with_name(full_enriched_output.stem + ".tmp.parquet")
            try:
                full_output.to_parquet(temp_full_output, index=False)
                temp_full_output.replace(full_enriched_output)
            finally:
                if temp_full_output.exists():
                    temp_full_output.unlink()

        temp_output = output_path.with_name(output_path.stem + ".tmp.parquet")
        try:
            compact.to_parquet(temp_output, index=False)
            temp_output.replace(output_path)
        finally:
            if temp_output.exists():
                temp_output.unlink()

    result = {
        "status": "success",
        "role": "Role 4 only",
        "competition_parquet": str(competition_parquet.resolve()),
        "role3_parquet": str(role3_parquet.resolve()),
        "model": str(model_path.resolve()),
        "config": str(config_path.resolve()),
        "output_parquet": str(output_path),
        "full_enriched_output_parquet": str(full_enriched_output) if full_enriched_output else None,
        "rows": int(len(compact)),
        "input_competition_columns": int(full_report["input_columns_preserved"]),
        "output_columns": len(expected_columns),
        "output_column_order": expected_columns,
        "output_frame_column": output_frame_column,
        "source_frame_column": frame_source,
        "matches": 1,
        "role3_probability_source_column": p_dead_column,
        "role3_semantics_confirmed": role3_ok,
        "role4_conditional_live_semantics_confirmed": role4_ok,
        "model_name": full_report.get("model_name"),
        "model_feature_count": full_report.get("model_feature_count"),
        "evidence_rules_applied": full_report.get("evidence_fusion", {}).get("rules_applied_count"),
        "evidence_report": full_report.get("evidence_fusion", {}),
        "probability_sum_max_abs_error": max_sum_error,
        "probability_sum_tolerance": tolerance,
        "output_contract": [output_frame_column, "p_home", "p_away", "p_dead"],
        "note": (
            "Compact probability-only Role 4 output. Original 111-column competition context is not copied here; "
            "Role 5 can join it back by frame_id if its existing contract requires those context columns."
        ),
    }
    report_path = output_path.with_name(output_path.stem + "_report.json")
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Role 4 pipeline complete: %d frames", len(compact))
    LOG.info("Output columns: %s", ", ".join(expected_columns))
    LOG.info("Max probability sum error: %.3g", max_sum_error)
    LOG.info("Output Parquet: %s", output_path)
    LOG.info("Report JSON: %s", report_path)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command Role 4: 111-column competition Parquet + Role 3 P(D) -> frame,H,A,D Parquet."
    )
    parser.add_argument("--competition-parquet", required=True, help="Pełny 111-kolumnowy Parquet meczu.")
    parser.add_argument("--role3-parquet", required=True, help="Parquet Roli 3 z P(D) i identyfikatorem klatki.")
    parser.add_argument("--model", default="role4_artifacts/role4_hist_gradient_boosting.joblib")
    parser.add_argument("--config", default=str(_default_config_path()))
    parser.add_argument("--output", default="role4_artifacts/role5_probabilities.parquet")
    parser.add_argument(
        "--full-enriched-output",
        default=None,
        help="Opcjonalny Parquet 111 kolumn + P(H)/P(A)/P(D), jeśli Rola 5 potrzebuje jednego pełnego DataFrame.",
    )
    parser.add_argument("--role3-probability-column", default=None, help="Nazwa kolumny P(D), domyślnie z config.py.")
    parser.add_argument("--competition-frame-column", default=None, help="Override klucza klatki w Parquecie 111-kolumnowym.")
    parser.add_argument("--role3-frame-column", default=None, help="Override klucza klatki w Parquecie Roli 3.")
    parser.add_argument("--competition-match-column", default=None, help="Opcjonalny match_id w Parquecie konkursowym.")
    parser.add_argument("--role3-match-column", default=None, help="Opcjonalny match_id w Parquecie Roli 3.")
    parser.add_argument("--output-frame-column", default="frame_id", help="Nazwa klucza klatki w małym wyjściowym Parquecie.")
    parser.add_argument(
        "--confirm-role3-out-of-play-only",
        action="store_true",
        help="Potwierdź, że P(D) oznacza wyłącznie piłkę poza grą, nie contested in-play.",
    )
    parser.add_argument(
        "--confirm-role4-conditional-live",
        action="store_true",
        help="Potwierdź interpretację HGB H/A jako P(H/A | live play).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Pozwól zastąpić istniejący plik wyjściowy.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_role4_pipeline(
        competition_parquet=Path(args.competition_parquet),
        role3_parquet=Path(args.role3_parquet),
        model_path=Path(args.model),
        config_path=Path(args.config),
        output_path=Path(args.output),
        role3_probability_column=args.role3_probability_column,
        competition_frame_column=args.competition_frame_column,
        role3_frame_column=args.role3_frame_column,
        competition_match_column=args.competition_match_column,
        role3_match_column=args.role3_match_column,
        output_frame_column=args.output_frame_column,
        full_enriched_output=Path(args.full_enriched_output) if args.full_enriched_output else None,
        confirm_role3_out_of_play_only=args.confirm_role3_out_of_play_only,
        confirm_role4_conditional_live=args.confirm_role4_conditional_live,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOG.exception("Role 4 pipeline failed: %s", exc)
        raise SystemExit(1)
