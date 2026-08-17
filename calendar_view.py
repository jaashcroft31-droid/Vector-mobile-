"""
Calendar / session-history view for the Climbing Fatigue Monitor.

A "session" is inferred as: all reps logged under the same rep_date,
regardless of category (Pre/Mid/Post-session reps on the same day all
count as one session).

Session duration defaults to the span between the earliest and latest
logged_at timestamp that day, but a manual `duration_override_min` (set
via app.py's session editor) wins if one exists for that day. Session
title and notes are optional, set the same way, and also live redundantly
on every rep row for that day (last non-empty value wins if they were
ever set more than once) - see app.py's Calendar section for where all
three get written.

Session load comes from fatigue_model.compute_session_loads() (only
defined for days with both a Pre-session and Post-session set) and is
merged in here so the calendar can show it alongside session time.
"""

from __future__ import annotations

import calendar as _calendar
from datetime import date, timedelta

import pandas as pd

import fatigue_model as fm

WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

SESSION_COLUMNS = [
    "rep_date", "reps", "pre_reps", "mid_reps", "post_reps",
    "duration_min", "duration_is_override", "session_load",
    "session_title", "session_notes", "avg_velocity_ms", "peak_velocity_ms",
]

WEEKLY_COLUMNS = [
    "week_start", "week_end", "sessions", "total_duration_min", "total_session_load",
    "avg_velocity_ms", "best_velocity_ms", "end_fatigue", "end_capacity", "end_freshness",
]


def _last_non_empty(series: pd.Series) -> str:
    non_empty = series[series.astype(str).str.strip() != ""]
    return str(non_empty.iloc[-1]) if not non_empty.empty else ""


def compute_sessions(log_df: pd.DataFrame) -> pd.DataFrame:
    """One row per rep_date that has at least one logged rep: total reps,
    per-category rep counts, session duration (manual override if set,
    otherwise estimated from timestamps), session load (if that day has
    both Pre- and Post-session data), title/notes (if set), and the day's
    average/peak velocity across all reps (any category)."""
    if log_df.empty:
        return pd.DataFrame(columns=SESSION_COLUMNS)

    df = log_df.copy()
    df["rep_date_parsed"] = pd.to_datetime(df["rep_date"]).dt.date
    df["logged_at_parsed"] = pd.to_datetime(df["logged_at"])
    df["duration_override_parsed"] = (
        pd.to_numeric(df["duration_override_min"], errors="coerce")
        if "duration_override_min" in df.columns else float("nan")
    )
    df["session_title_parsed"] = df["session_title"].fillna("").astype(str) if "session_title" in df.columns else ""
    df["session_notes_parsed"] = df["session_notes"].fillna("").astype(str) if "session_notes" in df.columns else ""

    rows = []
    for d, group in df.groupby("rep_date_parsed"):
        span = group["logged_at_parsed"].max() - group["logged_at_parsed"].min()
        auto_duration = span.total_seconds() / 60.0

        overrides = group["duration_override_parsed"].dropna()
        if not overrides.empty:
            duration_min = float(overrides.iloc[-1])
            duration_is_override = True
        else:
            duration_min = auto_duration
            duration_is_override = False

        rows.append({
            "rep_date": d,
            "reps": int(len(group)),
            "pre_reps": int((group["category"] == "Pre-session").sum()),
            "mid_reps": int((group["category"] == "Mid-session").sum()),
            "post_reps": int((group["category"] == "Post-session").sum()),
            "duration_min": duration_min,
            "duration_is_override": duration_is_override,
            "session_title": _last_non_empty(group["session_title_parsed"]),
            "session_notes": _last_non_empty(group["session_notes_parsed"]),
            "avg_velocity_ms": float(group["avg_velocity_ms"].mean()),
            "peak_velocity_ms": float(group["peak_velocity_ms"].max()),
        })
    sessions_df = pd.DataFrame(rows).sort_values("rep_date").reset_index(drop=True)

    session_loads = fm.compute_session_loads(log_df)  # columns: rep_date, session_load
    sessions_df = sessions_df.merge(session_loads, on="rep_date", how="left")

    return sessions_df[SESSION_COLUMNS]


def _end_of_week_fatigue(fatigue_df: pd.DataFrame | None, week_start: date, week_end: date) -> dict:
    """The Fatigue/Capacity/Freshness values as of the latest day within
    [week_start, week_end] that fatigue_df actually covers - the real
    Sunday for a fully-elapsed week, or today for the week still in
    progress."""
    if fatigue_df is None or fatigue_df.empty:
        return {"end_fatigue": None, "end_capacity": None, "end_freshness": None}
    in_week = fatigue_df[(fatigue_df["date"] >= week_start) & (fatigue_df["date"] <= week_end)]
    if in_week.empty:
        return {"end_fatigue": None, "end_capacity": None, "end_freshness": None}
    last_row = in_week.sort_values("date").iloc[-1]
    return {
        "end_fatigue": float(last_row["fatigue"]),
        "end_capacity": float(last_row["capacity"]),
        "end_freshness": float(last_row["freshness"]),
    }


def compute_weekly_summary(sessions_df: pd.DataFrame, fatigue_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per Mon-Sun week that has at least one session: number of
    sessions, total session time and total session load for the week, the
    week's average/best velocity, and (if fatigue_df is supplied, from
    fatigue_model.compute_fatigue_model) the Fatigue/Capacity/Freshness
    values as of the end of that week."""
    if sessions_df.empty:
        return pd.DataFrame(columns=WEEKLY_COLUMNS)

    df = sessions_df.copy()
    df["week_start"] = df["rep_date"].apply(lambda d: d - timedelta(days=d.weekday()))

    rows = []
    for week_start, group in df.groupby("week_start"):
        week_end = week_start + timedelta(days=6)
        row = {
            "week_start": week_start,
            "week_end": week_end,
            "sessions": int(len(group)),
            "total_duration_min": float(group["duration_min"].sum()),
            "total_session_load": float(group["session_load"].fillna(0).sum()),
            "avg_velocity_ms": float(group["avg_velocity_ms"].mean()),
            "best_velocity_ms": float(group["peak_velocity_ms"].max()),
        }
        row.update(_end_of_week_fatigue(fatigue_df, week_start, week_end))
        rows.append(row)
    return pd.DataFrame(rows, columns=WEEKLY_COLUMNS).sort_values("week_start").reset_index(drop=True)


def month_grid(year: int, month: int) -> list[list[int]]:
    """Weeks (Monday-first) of day-of-month numbers for the given month;
    0 means that grid cell falls outside the month."""
    return _calendar.Calendar(firstweekday=0).monthdayscalendar(year, month)


def week_start_for_row(year: int, month: int, week_row: list[int]) -> date | None:
    """The Monday date for a month_grid() row, found via whichever cell
    in the row is a real day-of-month (works even for weeks that overlap
    the previous/next month, since those cells are 0)."""
    for col_index, day_num in enumerate(week_row):
        if day_num != 0:
            return date(year, month, day_num) - timedelta(days=col_index)
    return None


def build_calendar_html(
    year: int, month: int, sessions_df: pd.DataFrame, weekly_df: pd.DataFrame
) -> str:
    """A scrollable, month-view HTML table: one row per week, one column
    per weekday plus a trailing "week total" column (mirroring a
    TrainingPeaks-style calendar). Read-only - Streamlit can't wire click
    events onto custom HTML without a fragile JS bridge, so day drill-down
    and editing happens via the day-picker + edit form in the app instead."""
    sessions_by_date = {row["rep_date"]: row for _, row in sessions_df.iterrows()}
    weekly_by_start = {row["week_start"]: row for _, row in weekly_df.iterrows()}
    weeks = month_grid(year, month)

    header_cells = "".join(f"<th>{d}</th>" for d in WEEKDAY_LABELS) + "<th>Week total</th>"

    body_rows = []
    for week in weeks:
        day_cells = []
        for day_num in week:
            if day_num == 0:
                day_cells.append("<td class='cfm-empty'></td>")
                continue
            d = date(year, month, day_num)
            sess = sessions_by_date.get(d)
            if sess is not None:
                load_line = (
                    f"<div class='cfm-load'>load {sess['session_load']:.0f}</div>"
                    if pd.notna(sess["session_load"]) else ""
                )
                day_cells.append(
                    "<td class='cfm-session'>"
                    f"<div class='cfm-daynum'>{day_num}</div>"
                    f"<div class='cfm-badge'>{int(sess['reps'])} reps</div>"
                    f"<div class='cfm-mins'>{sess['duration_min']:.0f} min</div>"
                    f"{load_line}"
                    "</td>"
                )
            else:
                day_cells.append(f"<td><div class='cfm-daynum'>{day_num}</div></td>")

        week_start = week_start_for_row(year, month, week)
        week_row = weekly_by_start.get(week_start) if week_start is not None else None
        if week_row is not None:
            total_cell = (
                "<td class='cfm-weektotal'>"
                f"<div class='cfm-wt-duration'>{week_row['total_duration_min']:.0f} min</div>"
                f"<div class='cfm-wt-load'>load {week_row['total_session_load']:.0f}</div>"
                "</td>"
            )
        else:
            total_cell = "<td class='cfm-weektotal cfm-empty'></td>"

        body_rows.append(f"<tr>{''.join(day_cells)}{total_cell}</tr>")

    return f"""
<style>
.cfm-cal-wrap {{ overflow-x: auto; }}
.cfm-cal {{ border-collapse: collapse; width: 100%; min-width: 620px; font-size: 0.82rem; }}
.cfm-cal th {{ text-align: center; padding: 4px 6px; font-weight: 600; color: #6b7280; }}
.cfm-cal td {{ border: 1px solid #e5e7eb; vertical-align: top; padding: 4px 6px; min-height: 56px; width: 12%; }}
.cfm-cal td.cfm-empty {{ background: #fafafa; border-color: #f0f0f0; }}
.cfm-daynum {{ font-weight: 600; color: #374151; }}
.cfm-session {{ background: #ecfdf5; }}
.cfm-badge {{ color: #059669; font-size: 0.78em; }}
.cfm-mins {{ color: #6b7280; font-size: 0.78em; }}
.cfm-load {{ color: #7c3aed; font-size: 0.78em; }}
.cfm-weektotal {{ background: #f3f4f6; width: 16%; }}
.cfm-wt-duration {{ font-weight: 600; color: #374151; }}
.cfm-wt-load {{ color: #7c3aed; font-size: 0.78em; }}
</style>
<div class="cfm-cal-wrap">
<table class="cfm-cal">
<thead><tr>{header_cells}</tr></thead>
<tbody>{''.join(body_rows)}</tbody>
</table>
</div>
"""
