from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    from kloppy import sportscode
    HAS_KLOPPY = True
except ImportError:
    HAS_KLOPPY = False

N_FRAMES: int = 199_319
FPS: float = 25.0

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

SET_PIECE_TYPES = {
    "throw-in": "throw_in",
    "corner kick": "corner_kick",
    "goal kick": "goal_kick",
    "free kick": "free_kick",
    "kick off (start)": "kick_off",
    "kick off (after goal)": "kick_off",
}

# Tagi oznaczające żywą grę (używane jako ujemny prior dla D oraz do detekcji startu akcji po wznowieniu)
LIVE_OFFENSIVE_CODES = {
    "BUILD UP",
    "PROGRESSION",
    "CREATING CHANCES",
    "ATTACKING TRANSITION",
    "LONG BALL",
    "BALL IN FINAL THIRD",
    "BALL IN THE BOX",
    "PLAYERS IN THE BOX",
}

LIVE_DEFENSIVE_CODES = {
    "DEFENDING IN ATTACKING THIRD",
    "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN DEFENSIVE THIRD",
    "DEFENSIVE TRANSITION",
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
    Wczytuje wszystkie 638 instancji z pliku SportsCode XML i mapuje czas
    w sekundach na klatki wideo [0..199318] za pomocą round(seconds * 25).
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

            col_map = {c.lower(): c for c in raw_df.columns}
            for target in ["Team", "Type", "Half", "Side"]:
                src_col = col_map.get(target.lower())
                df[target] = (
                    raw_df[src_col].astype(object).where(raw_df[src_col].notna(), None)
                    if src_col
                    else None
                )
        except Exception:
            df = _parse_xml_stdlib(xml_path)
    else:
        df = _parse_xml_stdlib(xml_path)

    if df.empty:
        return df

    df["start_frame"] = np.clip(np.rint(df["start_s"] * FPS).astype(np.int32), 0, N_FRAMES - 1)
    df["end_frame"] = np.clip(np.rint(df["end_s"] * FPS).astype(np.int32), 0, N_FRAMES - 1)

    df["team_clean"] = (
        df["Team"].fillna("").astype(str).str.strip().str.lower().map(TEAM_MAP)
    )
    df["type_clean"] = (
        df["Type"].fillna("").astype(str).str.strip().str.lower().map(SET_PIECE_TYPES)
    )

    # Drużyna efektywnie posiadająca piłkę w danym tagu (odwrócenie polaryzacji dla tagów obronnych)
    eff_team = []
    for code_val, t_val in zip(df["code"], df["team_clean"]):
        if t_val not in ("H", "A"):
            eff_team.append(None)
        elif str(code_val) in LIVE_DEFENSIVE_CODES or str(code_val).startswith("DEFEND"):
            eff_team.append("A" if t_val == "H" else "H")
        else:
            eff_team.append(t_val)
    df["effective_poss_team"] = eff_team

    return df.sort_values("start_frame").reset_index(drop=True)


def resolve_na_restart_team(
    events_df: pd.DataFrame,
    stage1_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Estymuje drużynę wznawiającą dla tagów z Team == N/A (w tym wszystkich 26 Throw-in).
    Łączy 4 źródła informacji:
    1. Kolumnę st_restart_team z stage1_tracking.parquet.
    2. Start nowego tagu żywej fazy (BUILD UP, PROGRESSION, LONG BALL, DEFEND*) tuż po wznowieniu.
    3. Geometrię zawodników przy piłce i linii bocznej (near1_dist_m, in_3m, ball_near_boundary).
    4. Asymetryczny skok kontroli boiska po wznowieniu (pitch_ctrl_leadlag_25f, near1_diff_leadlag_25f).
    """
    df = events_df.copy()
    if df.empty:
        return df

    df["resolved_team"] = df["effective_poss_team"]

    # Budowa osi głosów posiadania ze wszystkich rozwiązanych tagów
    tag_team_votes = np.zeros(N_FRAMES, dtype=np.float32)
    onset_team_votes = np.zeros(N_FRAMES, dtype=np.float32)

    for row in df.itertuples():
        if row.effective_poss_team in ("H", "A"):
            sign = 1.0 if row.effective_poss_team == "H" else -1.0
            s_f = max(0, int(row.start_frame))
            e_f = min(N_FRAMES - 1, int(row.end_frame))
            tag_team_votes[s_f : e_f + 1] += sign
            # Silny impuls na początku tagu żywej fazy (np. BUILD UP lub PROGRESSION zaczyna się od wrzutu z autu)
            if str(row.code) in (LIVE_OFFENSIVE_CODES | LIVE_DEFENSIVE_CODES):
                onset_end = min(N_FRAMES - 1, s_f + 50)
                onset_team_votes[s_f : onset_end + 1] += 2.5 * sign

    na_mask = df["resolved_team"].isna() & (
        (df["code"] == "SET PIECES") | df["type_clean"].notna()
    )

    for idx in df[na_mask].index:
        start_f = int(df.loc[idx, "start_frame"])
        end_f = int(df.loc[idx, "end_frame"])
        # W SportsCode wznowienie (rzut z autu) następuje zazwyczaj w drugiej połowie tagu lub tuż przed end_frame
        w_start = max(0, start_f + int(0.35 * (end_f - start_f)))
        w_end = min(N_FRAMES - 1, end_f + 45)

        score_h_minus_a = 0.0

        # Źródło 1: Tagi żywej gry rozpoczynające się w oknie wznowienia [w_start, end_f + 75]
        post_tag_slice = slice(w_start, min(N_FRAMES, end_f + 75))
        onset_sum = float(np.sum(onset_team_votes[post_tag_slice]))
        vote_sum = float(np.sum(tag_team_votes[post_tag_slice]))
        score_h_minus_a += 0.35 * np.sign(onset_sum) * min(abs(onset_sum), 5.0)
        score_h_minus_a += 0.20 * np.sign(vote_sum) * min(abs(vote_sum) / 25.0, 3.0)

        # Źródło 2: Cechy geometryczne i taktyczne ze stage1_tracking.parquet
        if stage1_df is not None:
            if "st_restart_team" in stage1_df.columns:
                st_rest = stage1_df["st_restart_team"].to_numpy()[ max(0, end_f - 35) : w_end + 1 ]
                nonzero = st_rest[st_rest != 0]
                if nonzero.size > 0:
                    score_h_minus_a += 2.5 * float(np.sign(np.sum(nonzero)))

            if {
                "home_near1_dist_m",
                "away_near1_dist_m",
                "ball_near_boundary",
            }.issubset(stage1_df.columns):
                sub = stage1_df.iloc[max(0, start_f) : w_end + 1]
                h_dist = sub["home_near1_dist_m"].to_numpy(dtype=np.float32)
                a_dist = sub["away_near1_dist_m"].to_numpy(dtype=np.float32)
                near_b = sub["ball_near_boundary"].to_numpy(dtype=np.int8) > 0

                # Przy wrzucie z autu wykonawca trzyma piłkę (< 2.0m), a rywal stoi > 2.0m dalej
                taker_mask = (np.minimum(h_dist, a_dist) < 2.2) & (np.abs(a_dist - h_dist) > 1.2)
                weights = np.where(near_b & taker_mask, 3.5, np.where(near_b, 1.5, 0.8))
                geom_diff = float(np.sum((a_dist - h_dist) * weights)) / max(len(sub), 1)
                score_h_minus_a += np.clip(geom_diff, -3.0, 3.0)

            if {"home_in_3m", "away_in_3m"}.issubset(stage1_df.columns):
                sub_3m = stage1_df.iloc[w_start : w_end + 1]
                h3 = sub_3m["home_in_3m"].to_numpy(dtype=np.float32)
                a3 = sub_3m["away_in_3m"].to_numpy(dtype=np.float32)
                # Klatki, w których dokładnie jedna drużyna ma gracza w promieniu 3m od piłki
                exclusive_h = np.sum((h3 >= 1) & (a3 == 0))
                exclusive_a = np.sum((a3 >= 1) & (h3 == 0))
                if exclusive_h + exclusive_a > 0:
                    score_h_minus_a += 1.8 * float(exclusive_h - exclusive_a) / float(exclusive_h + exclusive_a)

            if "pitch_ctrl_leadlag_25f" in stage1_df.columns:
                sub_ll = stage1_df.iloc[max(0, end_f - 25) : min(N_FRAMES, end_f + 35)]
                pc_ll = float(np.mean(sub_ll["pitch_ctrl_leadlag_25f"].to_numpy(dtype=np.float32)))
                score_h_minus_a += np.clip(pc_ll * 6.0, -2.0, 2.0)

        df.loc[idx, "resolved_team"] = "H" if score_h_minus_a >= 0.0 else "A"

    return df


def build_smart_tag_priors(
    xml_path: Optional[Union[str, Path]] = None,
    stage1_df: Optional[pd.DataFrame] = None,
    jitter_before_s: float = 3.5,
    jitter_after_s: float = 3.5,
) -> pd.DataFrame:
    """
    Buduje wektory priorytetów czasowych na osi 0..199318:
    - is_set_piece_window: okno kotwicy wokół stałych fragmentów gry, bramek i powtórek TV.
    - sp_prior_prob: prawdopodobieństwo a priori stanu D.
    - live_play_suppression: ujemny prior z tagów żywej gry (PROGRESSION, ATTACKING TRANSITION itp.).
    - live_tag_onset_signal: impuls początku nowej fazy taktycznej (BUILD UP, LONG BALL) ułatwiający detekcję klatki 0.
    - restart_team_hint: kategoryczna wskazówka drużyny wznawiającej (H, A, None).
    - sp_type: dokładny typ wznowienia (throw_in, corner_kick, goal_kick, free_kick, kick_off, post_goal).
    """
    out = pd.DataFrame({"frame": np.arange(N_FRAMES, dtype=np.int32)})

    pad_before = int(round(jitter_before_s * FPS))  # 88 klatek (3.5s przed startem tagu)
    pad_after = int(round(jitter_after_s * FPS))    # 88 klatek (3.5s po końcu tagu)

    is_sp_window = np.zeros(N_FRAMES, dtype=bool)
    sp_prior_prob = np.zeros(N_FRAMES, dtype=np.float32)
    live_play_suppression = np.zeros(N_FRAMES, dtype=np.float32)
    live_tag_onset_signal = np.zeros(N_FRAMES, dtype=np.float32)
    restart_hint = np.full(N_FRAMES, None, dtype=object)
    sp_type_arr = np.full(N_FRAMES, None, dtype=object)

    if xml_path is not None and Path(xml_path).exists():
        events_df = load_smart_tags_df(xml_path)
        events_df = resolve_na_restart_team(events_df, stage1_df=stage1_df)

        sp_df = events_df[
            (events_df["code"] == "SET PIECES") | events_df["type_clean"].notna()
        ].copy()
        goals_df = events_df[events_df["code"] == "GOALS"].copy()
        live_df = events_df[
            events_df["code"].isin(LIVE_OFFENSIVE_CODES | LIVE_DEFENSIVE_CODES)
        ].copy()

        # 1. Przetwarzanie tagów żywej gry (ujemny prior dla D + impulsy startowe faz BUILD UP / LONG BALL)
        for lrow in live_df.itertuples():
            ls = max(0, int(lrow.start_frame))
            le = min(N_FRAMES - 1, int(lrow.end_frame))
            code_str = str(lrow.code)

            if code_str in ("ATTACKING TRANSITION", "CREATING CHANCES", "PROGRESSION"):
                live_play_suppression[ls : le + 1] = np.maximum(
                    live_play_suppression[ls : le + 1], 0.35
                )
            elif code_str in ("BUILD UP", "BALL IN FINAL THIRD", "BALL IN THE BOX"):
                live_play_suppression[ls : le + 1] = np.maximum(
                    live_play_suppression[ls : le + 1], 0.20
                )

            # Początek tagu BUILD UP / LONG BALL / PROGRESSION często pokrywa się z wznowieniem gry
            if code_str in ("BUILD UP", "LONG BALL", "PROGRESSION", "BALL IN THE BOX"):
                onset_w = slice(max(0, ls - 12), min(N_FRAMES, ls + 13))
                win_len = onset_w.stop - onset_w.start
                if win_len > 0:
                    tri = np.bartlett(win_len).astype(np.float32)
                    live_tag_onset_signal[onset_w] = np.maximum(live_tag_onset_signal[onset_w], tri * 0.65)

        # 2. Przetwarzanie stałych fragmentów gry (SET PIECES)
        for row in sp_df.itertuples():
            s_f = int(row.start_frame)
            e_f = int(row.end_frame)
            win_s = max(0, s_f - pad_before)
            win_e = min(N_FRAMES - 1, e_f + pad_after)

            is_sp_window[win_s : win_e + 1] = True

            # W SportsCode tag SET PIECES obejmuje przygotowanie do wznowienia oraz sam moment
            # wykonania (często kończąc się 1-2s po kopnięciu piłki). Dlatego szczyt priora
            # przypada na pierwszą część i środek tagu, a pod koniec tagu łagodnie opada.
            core_s = max(0, s_f - 25)
            core_peak_end = max(core_s + 1, e_f - 25)
            sp_prior_prob[core_s : core_peak_end + 1] = np.maximum(
                sp_prior_prob[core_s : core_peak_end + 1], 0.88
            )

            if win_s < core_s:
                ramp_up = np.linspace(0.15, 0.88, core_s - win_s, endpoint=False, dtype=np.float32)
                sp_prior_prob[win_s:core_s] = np.maximum(sp_prior_prob[win_s:core_s], ramp_up)

            if win_e > core_peak_end:
                ramp_down = np.linspace(
                    0.88, 0.08, win_e - core_peak_end, endpoint=True, dtype=np.float32
                )
                sp_prior_prob[core_peak_end + 1 : win_e + 1] = np.maximum(
                    sp_prior_prob[core_peak_end + 1 : win_e + 1], ramp_down
                )

            if row.resolved_team in ("H", "A"):
                restart_hint[win_s : win_e + 1] = row.resolved_team
            if row.type_clean is not None:
                sp_type_arr[win_s : win_e + 1] = row.type_clean

        # 3. Martwy czas po bramkach (GOALS -> Kick Off)
        kickoffs = sp_df[sp_df["type_clean"] == "kick_off"]["start_frame"].to_numpy()
        for grow in goals_df.itertuples():
            g_start = int(grow.start_frame)
            next_ko = kickoffs[kickoffs > g_start]
            g_end = (
                int(next_ko[0])
                if next_ko.size > 0
                else min(N_FRAMES - 1, int(grow.end_frame) + 1500)
            )
            is_sp_window[g_start : g_end + 1] = True
            sp_prior_prob[g_start : g_end + 1] = 0.99
            sp_type_arr[g_start : g_end + 1] = "post_goal"
            if grow.resolved_team in ("H", "A"):
                ko_team = "A" if grow.resolved_team == "H" else "H"
                restart_hint[g_start : g_end + 1] = ko_team

    # 4. Integracja cech st_* oraz brd_is_replay ze stage1_tracking.parquet
    if stage1_df is not None:
        type_col_map = {
            "st_throw_in": "throw_in",
            "st_corner_kick": "corner_kick",
            "st_goal_kick": "goal_kick",
            "st_free_kick": "free_kick",
            "st_kick_off": "kick_off",
        }
        st_cols = [c for c in ["st_set_piece", *type_col_map.keys()] if c in stage1_df.columns]
        if st_cols:
            raw_st_active = stage1_df[st_cols].max(axis=1).to_numpy() > 0
            kernel = np.ones(pad_before + pad_after + 1, dtype=np.int32)
            dilated = np.convolve(raw_st_active.astype(np.int32), kernel, mode="same") > 0
            is_sp_window |= dilated

            smooth_kernel = np.hanning(2 * pad_after + 1)
            smooth_kernel /= smooth_kernel.sum()
            st_prob = np.convolve(raw_st_active.astype(np.float32), smooth_kernel, mode="same")
            sp_prior_prob = np.maximum(sp_prior_prob, np.clip(st_prob * 1.45, 0.0, 0.88))

            # Uzupełnienie sp_type z kolumn binarnych stage1 (jeśli nie podano pliku XML)
            for col_name, t_label in type_col_map.items():
                if col_name in stage1_df.columns:
                    col_mask = stage1_df[col_name].to_numpy(dtype=np.int8) > 0
                    if np.any(col_mask):
                        dil_type = (
                            np.convolve(col_mask.astype(np.int32), np.ones(121, dtype=np.int32), mode="same")
                            > 0
                        )
                        fill_t = (sp_type_arr == None) & dil_type  # noqa: E711
                        sp_type_arr[fill_t] = t_label

        if "st_post_goal_dead" in stage1_df.columns:
            pg_dead = stage1_df["st_post_goal_dead"].to_numpy(dtype=np.int8) > 0
            is_sp_window |= pg_dead
            sp_prior_prob = np.where(pg_dead, 0.99, sp_prior_prob)
            sp_type_arr = np.where(pg_dead, "post_goal", sp_type_arr)

        # Powtórki TV (brd_is_replay == 1) jako kotwice przerw pominiętych przez XML (np. faule, VAR, auty)
        if "brd_is_replay" in stage1_df.columns:
            brd_replay = stage1_df["brd_is_replay"].to_numpy(dtype=np.int8) == 1
            replay_lead = np.roll(brd_replay, -40)
            replay_lead[-40:] = False
            replay_window = brd_replay | replay_lead

            is_sp_window |= replay_window
            replay_tail_cut = np.roll(~brd_replay, -30)
            replay_tail_cut[-30:] = True
            safe_replay_core = brd_replay & ~replay_tail_cut

            sp_prior_prob = np.where(
                safe_replay_core,
                np.maximum(sp_prior_prob, 0.90),
                np.where(replay_window, np.maximum(sp_prior_prob, 0.65), sp_prior_prob),
            )

        if "st_restart_team" in stage1_df.columns:
            st_rt = stage1_df["st_restart_team"].to_numpy(dtype=np.int8)
            for val, label in [(1, "H"), (-1, "A")]:
                mask_val = st_rt == val
                if np.any(mask_val):
                    dil_rt = (
                        np.convolve(mask_val.astype(np.int32), np.ones(161, dtype=np.int32), mode="same")
                        > 0
                    )
                    fill_mask = (restart_hint == None) & dil_rt & is_sp_window  # noqa: E711
                    restart_hint[fill_mask] = label

        if "st_team_prior" in stage1_df.columns:
            st_tp = stage1_df["st_team_prior"].to_numpy(dtype=np.int8)
            missing_hint = (restart_hint == None) & is_sp_window  # noqa: E711
            restart_hint[missing_hint & (st_tp == 1)] = "H"
            restart_hint[missing_hint & (st_tp == -1)] = "A"

    # Wewnątrz potwierdzonego okna stałego fragmentu zerujemy tłumienie z tagów żywej gry
    live_play_suppression = np.where(is_sp_window, 0.0, live_play_suppression).astype(np.float32)

    out["is_set_piece_window"] = is_sp_window
    out["sp_prior_prob"] = sp_prior_prob.astype(np.float32)
    out["live_play_suppression"] = live_play_suppression
    out["live_tag_onset_signal"] = live_tag_onset_signal.astype(np.float32)
    out["restart_team_hint"] = pd.Categorical(restart_hint, categories=["H", "A"])
    out["sp_type"] = sp_type_arr

    return out