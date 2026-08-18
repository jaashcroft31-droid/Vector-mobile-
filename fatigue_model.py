"""
Longitudinal fatigue model for the Climbing Fatigue Monitor.

Modelled on TrainingPeaks' Fitness/Fatigue/Form (PMC) approach, adapted to
session-level pull-up velocity:

    Session load  One number per session (a day with both Pre-session and
                  Post-session reps): how much peak + average velocity
                  dropped from pre- to post-session.

                      pct_diff_peak = (post_peak - pre_peak) / pre_peak
                      pct_diff_avg  = (post_avg  - pre_avg)  / pre_avg
                      session_load  = (pct_diff_peak + pct_diff_avg) * -100

                  A session where velocity dropped gives a positive load
                  (harder/more fatiguing session -> bigger number). A
                  session that finished faster than it started (rare)
                  gives a negative load.

    Fatigue       An exponentially weighted average of session load:
                      - on a day with a session: fatigue moves toward
                        that day's load, weighted by FATIGUE_EXPONENT
                      - on a rest day: fatigue simply decays by
                        FATIGUE_DECAY
                  so it reacts quickly to a hard session and fades during
                  rest - the "ATL" side of a PMC-style model.

    Capacity      A plain 35-day moving average of session load (rest
                  days count as 0 load) - the slower "how much load
                  you've been handling lately" baseline, the "CTL" side.

    Freshness     Capacity minus Fatigue. Positive = carrying more base
                  capacity than current fatigue (fresh). Negative =
                  recent load is outstripping that base (run down).

Every day in the model needs a Pre-session AND a Post-session rep set
logged the same day to register as a "session" with a real load - days
missing either side count as rest days (load 0) for fatigue/capacity
purposes, which is why the model looks flat until both sides of a
session are consistently logged.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

FATIGUE_EXPONENT = 0.2   # weight given to each new session's load
FATIGUE_DECAY = 0.1      # fraction fatigue fades by, per rest day
CAPACITY_WINDOW_DAYS = 35

# Relative session load (for the calendar heat map): each session is
# classified against that athlete's own recent *typical session* load -
# deliberately NOT `capacity`, since capacity is diluted by rest days
# (by design, for the Fitness/Fatigue chart) and would make almost every
# normal session look "heavy" simply because most calendar days aren't
# session days. "Typical" here means the mean session_load over the prior
# LOAD_TYPICAL_WINDOW_DAYS days' worth of *sessions only*, looking
# backward from (but not including) the day being classified - so a
# single huge session can't inflate its own comparison baseline.
LOAD_TYPICAL_WINDOW_DAYS = 35
LOAD_LIGHT_THRESHOLD = 0.7   # below 70% of recent typical -> light
LOAD_HEAVY_THRESHOLD = 1.3   # above 130% of recent typical -> heavy
LOAD_TYPICAL_MIN_FLOOR = 1.0  # typical load below this is too small/noisy
                               # to classify meaningfully -> "moderate"

SESSION_LOAD_COLUMNS = ["rep_date", "session_load"]
MODEL_COLUMNS = ["date", "session_load", "fatigue", "capacity", "freshness"]
RELATIVE_LOAD_COLUMNS = ["rep_date", "relative_load_ratio", "load_band"]


def compute_session_loads(log_df: pd.DataFrame) -> pd.DataFrame:
    """One row per day that has both Pre-session and Post-session reps
    logged: that day's session load. A day with only one side (or
    neither) simply doesn't appear here - compute_fatigue_model treats
    any day missing from this table as a rest day (session_load = 0)."""
    if log_df.empty:
        return pd.DataFrame(columns=SESSION_LOAD_COLUMNS)

    df = log_df.copy()
    df["rep_date_parsed"] = pd.to_datetime(df["rep_date"]).dt.date

    rows = []
    for d, group in df.groupby("rep_date_parsed"):
        pre = group[group["category"] == "Pre-session"]
        post = group[group["category"] == "Post-session"]
        if pre.empty or post.empty:
            continue

        pre_avg = float(pre["avg_velocity_ms"].mean())
        pre_peak = float(pre["peak_velocity_ms"].mean())
        post_avg = float(post["avg_velocity_ms"].mean())
        post_peak = float(post["peak_velocity_ms"].mean())

        if pre_avg == 0 or pre_peak == 0:
            continue

        pct_diff_avg = (post_avg - pre_avg) / pre_avg
        pct_diff_peak = (post_peak - pre_peak) / pre_peak
        session_load = (pct_diff_peak + pct_diff_avg) * -100

        rows.append({"rep_date": d, "session_load": session_load})

    return (
        pd.DataFrame(rows, columns=SESSION_LOAD_COLUMNS)
        .sort_values("rep_date")
        .reset_index(drop=True)
    )


def compute_fatigue_model(log_df: pd.DataFrame, through_date: date | None = None) -> pd.DataFrame:
    """Day-by-day Fatigue / Capacity / Freshness from the first logged rep
    through `through_date` (defaults to today). Every calendar day in
    that range gets a row, including rest days - that's what lets Fatigue
    actually decay and Capacity's 35-day window be a true calendar
    window rather than just "the last 35 sessions"."""
    if log_df.empty:
        return pd.DataFrame(columns=MODEL_COLUMNS)

    session_loads = compute_session_loads(log_df)

    all_dates = pd.to_datetime(log_df["rep_date"]).dt.date
    start_date = all_dates.min()
    end_date = through_date or date.today()
    if end_date < start_date:
        end_date = start_date

    load_by_date = dict(zip(session_loads["rep_date"], session_loads["session_load"]))

    rows = []
    fatigue = 0.0
    loads_window: list[float] = []
    d = start_date
    while d <= end_date:
        has_session = d in load_by_date
        load = load_by_date.get(d, 0.0)

        if has_session:
            fatigue = fatigue * (1 - FATIGUE_EXPONENT) + load * FATIGUE_EXPONENT
        else:
            fatigue = fatigue * (1 - FATIGUE_DECAY)

        loads_window.append(load)
        if len(loads_window) > CAPACITY_WINDOW_DAYS:
            loads_window.pop(0)
        capacity = sum(loads_window) / len(loads_window)

        freshness = capacity - fatigue

        rows.append({
            "date": d,
            "session_load": load if has_session else None,
            "fatigue": fatigue,
            "capacity": capacity,
            "freshness": freshness,
        })
        d += timedelta(days=1)

    return pd.DataFrame(rows, columns=MODEL_COLUMNS)


def compute_relative_session_load(log_df: pd.DataFrame) -> pd.DataFrame:
    """One row per day with a computable session_load: how that session
    compares to this athlete's own recent typical session (see the
    LOAD_* constants above for the exact definition), classified into
    "light" / "moderate" / "heavy". Powers the calendar's colour-coded
    heat map.

    This is a relative, adaptive read of a single signal (velocity loss)
    - useful for spotting a session that stands out from someone's own
    recent pattern, but it is NOT a validated overtraining/injury-risk
    score on its own. Load:capacity-style ratios are genuinely contested
    in the sports-science literature even with much richer data than a
    single metric provides, so treat "heavy" as "worth a look", not as a
    diagnosis.

    The very first session (and any day whose recent window has no prior
    sessions to compare against yet) has no baseline to judge against and
    is classified "moderate" by default, with relative_load_ratio = None.
    """
    session_loads = compute_session_loads(log_df)
    if session_loads.empty:
        return pd.DataFrame(columns=RELATIVE_LOAD_COLUMNS)

    session_loads = session_loads.sort_values("rep_date").reset_index(drop=True)
    dates = session_loads["rep_date"].tolist()
    loads = session_loads["session_load"].tolist()

    rows = []
    for i, d in enumerate(dates):
        window_start = d - timedelta(days=LOAD_TYPICAL_WINDOW_DAYS)
        prior_loads = [loads[j] for j in range(i) if window_start <= dates[j] < d]

        if not prior_loads:
            ratio, band = None, "moderate"
        else:
            typical = sum(prior_loads) / len(prior_loads)
            if typical < LOAD_TYPICAL_MIN_FLOOR:
                ratio, band = None, "moderate"
            else:
                ratio = loads[i] / typical
                if ratio < LOAD_LIGHT_THRESHOLD:
                    band = "light"
                elif ratio > LOAD_HEAVY_THRESHOLD:
                    band = "heavy"
                else:
                    band = "moderate"

        rows.append({"rep_date": d, "relative_load_ratio": ratio, "load_band": band})

    return pd.DataFrame(rows, columns=RELATIVE_LOAD_COLUMNS)
