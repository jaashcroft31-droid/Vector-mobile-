"""
Calendar / session-history view for the Climbing Fatigue Monitor.

A "session" is inferred as: all reps logged under the same rep_date,
regardless of category (Pre/Mid/Post-session reps on the same day all
count as one session). Session duration is approximated as the span
between the earliest and latest logged_at timestamp that day - the best
available proxy given the app doesn't ask for an explicit session
start/end, and it assumes reps are logged in real time during the
session. If reps are ever backfilled long after the fact, duration for
that day will be inaccurate.
"""

from __future__ import annotations

import calendar as _calendar
from datetime import date, timedelta

import pandas as pd

WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

SESSION_COLUMNS = [
    "rep_date", "reps", "pre_reps", "mid_reps", "post_reps",
    "duration_min", "avg_velocity_ms", "peak_velocity_ms",
]

WEEKLY_COLUMNS = [
    "week_start", "week_end", "sessions", "total_duration_min",
    "avg_velocity_ms", "best_velocity_ms",
]


def compute_sessions(log_df: pd.DataFrame) -> pd.DataFrame:
    """One row per rep_date that has at least one logged rep: total reps,
    per-category rep counts, session duration (minutes), and the day's
    average/peak velocity across all reps (any category)."""
    if log_df.empty:
        return pd.DataFrame(columns=SESSION_COLUMNS)

    df = log_df.copy()
    df["rep_date_parsed"] = pd.to_datetime(df["rep_date"]).dt.date
    df["logged_at_parsed"] = pd.to_datetime(df["logged_at"])

    rows = []
    for d, group in df.groupby("rep_date_parsed"):
        span = group["logged_at_parsed"].max() - group["logged_at_parsed"].min()
        rows.append({
            "rep_date": d,
            "reps": int(len(group)),
            "pre_reps": int((group["category"] == "Pre-session").sum()),
            "mid_reps": int((group["category"] == "Mid-session").sum()),
            "post_reps": int((group["category"] == "Post-session").sum()),
            "duration_min": span.total_seconds() / 60.0,
            "avg_velocity_ms": float(group["avg_velocity_ms"].mean()),
            "peak_velocity_ms": float(group["peak_velocity_ms"].max()),
        })
    return pd.DataFrame(rows, columns=SESSION_COLUMNS).sort_values("rep_date").reset_index(drop=True)


def compute_weekly_summary(sessions_df: pd.DataFrame) -> pd.DataFrame:
    """One row per Mon-Sun week that has at least one session: number of
    sessions, total session time (minutes), the week's average velocity,
    and the week's best (peak) velocity - across all reps of any category
    logged that week."""
    if sessions_df.empty:
        return pd.DataFrame(columns=WEEKLY_COLUMNS)

    df = sessions_df.copy()
    df["week_start"] = df["rep_date"].apply(lambda d: d - timedelta(days=d.weekday()))

    rows = []
    for week_start, group in df.groupby("week_start"):
        rows.append({
            "week_start": week_start,
            "week_end": week_start + timedelta(days=6),
            "sessions": int(len(group)),
            "total_duration_min": float(group["duration_min"].sum()),
            "avg_velocity_ms": float(group["avg_velocity_ms"].mean()),
            "best_velocity_ms": float(group["peak_velocity_ms"].max()),
        })
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
    TrainingPeaks-style calendar). Read-only - day drill-down is a
    separate selector in the app, since clickable cells would need JS
    Streamlit doesn't give us for free."""
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
                day_cells.append(
                    "<td class='cfm-session'>"
                    f"<div class='cfm-daynum'>{day_num}</div>"
                    f"<div class='cfm-badge'>{int(sess['reps'])} reps</div>"
                    f"<div class='cfm-mins'>{sess['duration_min']:.0f} min</div>"
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
                f"<div class='cfm-wt-stats'>avg {week_row['avg_velocity_ms']:.2f} &middot; "
                f"best {week_row['best_velocity_ms']:.2f} m/s</div>"
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
.cfm-cal td {{ border: 1px solid #e5e7eb; vertical-align: top; padding: 4px 6px; height: 56px; width: 12%; }}
.cfm-cal td.cfm-empty {{ background: #fafafa; border-color: #f0f0f0; }}
.cfm-daynum {{ font-weight: 600; color: #374151; }}
.cfm-session {{ background: #ecfdf5; }}
.cfm-badge {{ color: #059669; font-size: 0.78em; }}
.cfm-mins {{ color: #6b7280; font-size: 0.78em; }}
.cfm-weektotal {{ background: #f3f4f6; width: 16%; }}
.cfm-wt-duration {{ font-weight: 600; color: #374151; }}
.cfm-wt-stats {{ color: #6b7280; font-size: 0.78em; }}
</style>
<div class="cfm-cal-wrap">
<table class="cfm-cal">
<thead><tr>{header_cells}</tr></thead>
<tbody>{''.join(body_rows)}</tbody>
</table>
</div>
"""
