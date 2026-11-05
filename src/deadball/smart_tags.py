# src/deadball/smart_tags.py
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from kloppy import sportscode
    HAS_KLOPPY = True
except ImportError:
    HAS_KLOPPY = False

N_FRAMES: int = 199_319
FPS: float = 25.0

# Mapowanie nazw drużyn z XML na standard H / A / None
TEAM_MAP = {
    "barcelona": "H",
    "fc barcelona": "H",
    "home": "H",
    "atletic club": "A",
    "athletic club": "A",
    "away": "A",
    "n/a": None,
    "na": None,
    "": None,
}

# Typy stałych fragmentów gry z grupy tagów 'Type'
SET_PIECE_TYPES = {
    "throw-in": "throw_in",
    "corner kick": "corner_kick",
    "goal kick": "goal_kick",
    "free kick": "free_kick",
    "kick off (start)": "kick_off",
    "kick off (after goal)": "kick_off",
}


def _parse_xml_stdlib(xml_path: Union[str, Path]) -> pd.DataFrame:
    """Fallbackowy parser SportsCode XML w oparciu o bibliotekę standardową."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    records = []

    for inst in root.findall(".//instance"):
        start_el = inst.find("start")
        end_el = inst.find("end")
        code_el = inst.find("code")
        if start_el is None or end_el is None or code_el is None:
            continue

        row = {
            "start_s": float(start_el.text.strip()),
            "end_s": float(end_el.text.strip()),
            "code": code_el.text.strip().upper(),
            "Team": None,
            "Type": None,
            "Half": None,
            "Side": None,
        }

        for label in inst.findall("label"):
            group_el = label.find("group")
            text_el = label.find("text")
            if group_el is not None and text_el is not None and group_el.text and text_el.text:
                g_name = group_el.text.strip()
                t_val = text_el.text.strip()
                if g_name.lower() == "team":
                    row["Team"] = t_val
                elif g_name.lower() == "type":
                    row["Type"] = t_val
                elif g_name.lower() == "half":
                    row["Half"] = t_val
                elif g_name.lower() == "side":
                    row["Side"] = t_val

        records.append(row)

    return pd.DataFrame(records)


def load_smart_tags_df(xml_path: Union[str, Path]) -> pd.DataFrame:
    """
    Wczytuje plik SportsCode XML (przez kloppy lub stdlib) i konwertuje czasy
    w sekundach na klatki wideo za pomocą round(seconds * 25).
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"Brak pliku Smart Tagging XML: {xml_path}")

    if HAS_KLOPPY:
        try:
            sc_dataset = sportscode.load(str(xml_path))
            raw_df = sc_dataset.to_df()
            df = pd.DataFrame()
            df["start_s"] = raw_df["timestamp"].dt.total_seconds()
            df["end_s"] = raw_df["end_timestamp"].dt.total_seconds()
            df["code"] = raw_df["code"].astype(str).str.strip().str.upper()

            # Bezpieczne pobieranie kolumn niezależnie od wielkości liter (Title Case w kloppy)
            col_map = {c.lower(): c for c in raw_df.columns}
            for target in ["Team", "Type", "Half", "Side"]:
                src_col = col_map.get(target.lower())
                df[target] = raw_df[src_col].astype(object).where(raw_df[src_col].notna(), None) if src_col else None
        except Exception:
            df = _parse_xml_stdlib(xml_path)
    else:
        df = _parse_xml_stdlib(xml_path)

    if df.empty:
        return df

    # Konwersja czasu na klatki: round(seconds * 25) przycięta do [0, 199318]
    df["start_frame"] = np.clip(np.rint(df["start_s"] * FPS).astype(np.int32), 0, N_FRAMES - 1)
    df["end_frame"] = np.clip(np.rint(df["end_s"] * FPS).astype(np.int32), 0, N_FRAMES - 1)

    # Normalizacja nazwy drużyny
    df["team_clean"] = (
        df["Team"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .map(TEAM_MAP)
    )
    df["type_clean"] = (
        df["Type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .map(SET_PIECE_TYPES)
    )

    return df.sort_values("start_frame").reset_index(drop=True)


def resolve_na_restart_team(
    events_df: pd.DataFrame,
    stage1_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Rozwiązuje problem brakujących drużyn (Team == N/A, szczególnie dla Throw-in).
    Wykorzystuje:
    1. Dane śledzenia z stage1_df (odległość graczy H vs A do piłki przy linii bocznej
       w oknie [-25, +35] klatek wokół końca tagu wznowienia).
    2. Kontekst sąsiednich tagów taktycznych (z uwzględnieniem odwróconej polaryzacji
       dla tagów obronnych DEFEND*).
    """
    df = events_df.copy()
    if df.empty:
        return df

    df["resolved_team"] = df["team_clean"]

    # Wyodrębnienie pomocniczych wektorów głosowania z innych tagów na osi klatek
    tag_team_votes = np.zeros(N_FRAMES, dtype=np.float32)
    for row in df.itertuples():
        if row.team_clean in ("H", "A"):
            sign = 1.0 if row.team_clean == "H" else -1.0
            # Tagi obronne (DEFENDING IN ...) oznaczają posiadanie drużyny przeciwnej
            if str(row.code).startswith("DEFEND"):
                sign = -sign
            s_f = max(0, int(row.start_frame))
            e_f = min(N_FRAMES - 1, int(row.end_frame))
            tag_team_votes[s_f : e_f + 1] += sign

    # Indeksy zdarzeń wymagających estymacji drużyny (szczególnie SET PIECES / Throw-in)
    na_mask = df["resolved_team"].isna() & (
        (df["code"] == "SET PIECES") | df["type_clean"].notna()
    )

    for idx in df[na_mask].index:
        start_f = int(df.loc[idx, "start_frame"])
        end_f = int(df.loc[idx, "end_frame"])
        # Okno wokół momentu wznowienia gry (koniec tagu z uwzględnieniem jittera ±2s)
        w_start = max(0, end_f - 50)
        w_end = min(N_FRAMES - 1, end_f + 50)

        inferred_team = None

        # Krok 1: Jeżeli w stage1_df istnieje już wyliczony st_restart_team lub cechy odległości
        if stage1_df is not None:
            if "st_restart_team" in stage1_df.columns:
                st_rest = stage1_df["st_restart_team"].to_numpy()[w_start : w_end + 1]
                nonzero = st_rest[st_rest != 0]
                if nonzero.size > 0:
                    inferred_team = "H" if np.sum(nonzero) >= 0 else "A"

            if inferred_team is None and {
                "home_near1_dist_m",
                "away_near1_dist_m",
                "ball_near_boundary",
            }.issubset(stage1_df.columns):
                sub = stage1_df.iloc[max(0, start_f - 25) : w_end + 1]
                # Przy wrzucie z autu zawodnik wznawiający stoi najbliżej piłki przy linii bocznej
                h_dist = sub["home_near1_dist_m"].to_numpy()
                a_dist = sub["away_near1_dist_m"].to_numpy()
                near_b = sub["ball_near_boundary"].to_numpy() > 0

                weights = np.where(near_b, 2.0, 1.0)
                # Dodatnia różnica (a_dist - h_dist) oznacza, że gracz Home jest bliżej piłki
                diff_score = np.sum((a_dist - h_dist) * weights)
                if abs(diff_score) > 1e-3:
                    inferred_team = "H" if diff_score > 0 else "A"

        # Krok 2: Fallback do bilansu głosów z tagów bezpośrednio po wznowieniu [end_f, end_f + 125]
        if inferred_team is None:
            post_w_end = min(N_FRAMES - 1, end_f + 125)
            vote_sum = float(np.sum(tag_team_votes[end_f : post_w_end + 1]))
            if abs(vote_sum) > 0:
                inferred_team = "H" if vote_sum > 0 else "A"
            else:
                # Domyślne rozstrzygnięcie z szerszego okna
                broad_sum = float(np.sum(tag_team_votes[max(0, start_f - 75) : post_w_end + 1]))
                inferred_team = "H" if broad_sum >= 0 else "A"

        df.loc[idx, "resolved_team"] = inferred_team

    return df


def build_smart_tag_priors(
    xml_path: Optional[Union[str, Path]] = None,
    stage1_df: Optional[pd.DataFrame] = None,
    jitter_before_s: float = 3.0,
    jitter_after_s: float = 4.0,
) -> pd.DataFrame:
    """
    Generuje wektory priorytetów czasowych (kotwic) na osi 0..199318.
    Uwzględnia jitter ±2-5 sekund poprzez rozszerzenie okien stałych fragmentów gry
    oraz buduje miękki rozkład prawdopodobieństwa a priori (prior_p_dead).
    """
    out = pd.DataFrame({"frame": np.arange(N_FRAMES, dtype=np.int32)})

    pad_before = int(round(jitter_before_s * FPS))  # np. 75 klatek (3s)
    pad_after = int(round(jitter_after_s * FPS))    # np. 100 klatek (4s)

    is_sp_window = np.zeros(N_FRAMES, dtype=bool)
    sp_prior_prob = np.zeros(N_FRAMES, dtype=np.float32)
    restart_hint = np.full(N_FRAMES, None, dtype=object)
    sp_type_arr = np.full(N_FRAMES, None, dtype=object)

    if xml_path is not None and Path(xml_path).exists():
        events_df = load_smart_tags_df(xml_path)
        events_df = resolve_na_restart_team(events_df, stage1_df=stage1_df)

        # Filtrowanie stałych fragmentów gry oraz bramek
        sp_df = events_df[
            (events_df["code"] == "SET PIECES") | events_df["type_clean"].notna()
        ].copy()
        goals_df = events_df[events_df["code"] == "GOALS"].copy()

        for row in sp_df.itertuples():
            s_f = int(row.start_frame)
            e_f = int(row.end_frame)
            win_s = max(0, s_f - pad_before)
            win_e = min(N_FRAMES - 1, e_f + pad_after)

            is_sp_window[win_s : win_e + 1] = True

            # Rdzeń tagu ma wysokie p_dead, a brzegi (jitter ±2-5s) opadają liniowo
            core_s = max(0, s_f)
            core_e = min(N_FRAMES - 1, e_f)
            sp_prior_prob[core_s : core_e + 1] = np.maximum(
                sp_prior_prob[core_s : core_e + 1], 0.85
            )

            if win_s < core_s:
                ramp_up = np.linspace(0.15, 0.85, core_s - win_s, endpoint=False, dtype=np.float32)
                sp_prior_prob[win_s:core_s] = np.maximum(sp_prior_prob[win_s:core_s], ramp_up)

            if win_e > core_e:
                ramp_down = np.linspace(0.85, 0.10, win_e - core_e, endpoint=True, dtype=np.float32)
                sp_prior_prob[core_e + 1 : win_e + 1] = np.maximum(
                    sp_prior_prob[core_e + 1 : win_e + 1], ramp_down
                )

            if row.resolved_team in ("H", "A"):
                restart_hint[win_s : win_e + 1] = row.resolved_team
            if row.type_clean is not None:
                sp_type_arr[win_s : win_e + 1] = row.type_clean

        # Obsługa martwego czasu po golu aż do wznowienia ze środka (Kick Off)
        kickoffs = sp_df[sp_df["type_clean"] == "kick_off"]["start_frame"].to_numpy()
        for grow in goals_df.itertuples():
            g_start = int(grow.start_frame)
            next_ko = kickoffs[kickoffs > g_start]
            g_end = int(next_ko[0]) if next_ko.size > 0 else min(N_FRAMES - 1, int(grow.end_frame) + 1500)
            is_sp_window[g_start : g_end + 1] = True
            sp_prior_prob[g_start : g_end + 1] = 0.98
            # Jeśli gola zdobył zespół X, ze środka wznawia zespół przeciwny
            if grow.resolved_team in ("H", "A"):
                ko_team = "A" if grow.resolved_team == "H" else "H"
                restart_hint[g_start : g_end + 1] = ko_team

    # Integracja/wzmocnienie sygnału z gotowych kolumn st_* z pliku stage1_tracking.parquet
    if stage1_df is not None:
        st_cols = [
            c for c in [
                "st_set_piece", "st_throw_in", "st_corner_kick",
                "st_goal_kick", "st_free_kick", "st_kick_off"
            ] if c in stage1_df.columns
        ]
        if st_cols:
            raw_st_active = stage1_df[st_cols].max(axis=1).to_numpy() > 0
            # Poszerzenie okna o bufor jittera za pomocą splotu (dylatacja binarna 1D)
            kernel = np.ones(pad_before + pad_after + 1, dtype=np.int32)
            dilated = np.convolve(raw_st_active.astype(np.int32), kernel, mode="same") > 0
            is_sp_window |= dilated

            # Wygładzony prior z flag stage1
            smooth_kernel = np.hanning(2 * pad_after + 1)
            smooth_kernel /= smooth_kernel.sum()
            st_prob = np.convolve(raw_st_active.astype(np.float32), smooth_kernel, mode="same")
            sp_prior_prob = np.maximum(sp_prior_prob, np.clip(st_prob * 1.4, 0.0, 0.88))

        if "st_post_goal_dead" in stage1_df.columns:
            pg_dead = stage1_df["st_post_goal_dead"].to_numpy() > 0
            is_sp_window |= pg_dead
            sp_prior_prob = np.where(pg_dead, 0.98, sp_prior_prob)

        if "st_restart_team" in stage1_df.columns:
            st_rt = stage1_df["st_restart_team"].to_numpy()
            # Rozciągnięcie wskazówki st_restart_team na całe okno kotwicy
            for val, label in [(1, "H"), (-1, "A")]:
                mask_val = st_rt == val
                if np.any(mask_val):
                    dil_rt = np.convolve(mask_val.astype(np.int32), np.ones(151, dtype=np.int32), mode="same") > 0
                    fill_mask = (restart_hint == None) & dil_rt & is_sp_window  # noqa: E711
                    restart_hint[fill_mask] = label

        if "st_team_prior" in stage1_df.columns:
            st_tp = stage1_df["st_team_prior"].to_numpy()
            missing_hint = (restart_hint == None) & is_sp_window  # noqa: E711
            restart_hint[missing_hint & (st_tp == 1)] = "H"
            restart_hint[missing_hint & (st_tp == -1)] = "A"

    out["is_set_piece_window"] = is_sp_window
    out["sp_prior_prob"] = sp_prior_prob.astype(np.float32)
    out["restart_team_hint"] = pd.Categorical(restart_hint, categories=["H", "A"])
    out["sp_type"] = sp_type_arr

    return out