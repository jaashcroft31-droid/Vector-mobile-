"""
Calendar / session-history view for the Climbing Fatigue Monitor.
 
A "session" is inferred as: all reps logged under the same rep_date,
regardless of category (Pre/Mid/Post-session reps on the same day all
count as one session). A session can also be added by hand from the
calendar: that's a single row with category fatigue_model.MANUAL_CATEGORY
and no velocity data, carrying the date, start time, title, notes,
duration and an estimated load.
 
Session duration defaults to the span between the earliest and latest
logged_at timestamp of that day's reps (manual-session rows are left out
of that span, since their logged_at is just when they were typed in), but
a manual `duration_override_min` (set via app.py's session editor, or when
adding a session by hand) wins if one exists for that day. Session title,
notes, start time and load override are optional, set the same way, and
also live redundantly on every row for that day (last non-empty value
wins if they were ever set more than once) - see app.py's Calendar
section for where they get written.
 
Session load comes from fatigue_model.compute_session_loads() (defined for
days with both a Pre-session and Post-session set, or with a load entered
by hand) and is merged in here so the calendar can show it alongside
session time.
"""
 
from __future__ import annotations
 
import calendar as _calendar
from datetime import date, timedelta
from html import escape as _escape
 
import pandas as pd
 
import fatigue_model as fm
 
WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
 
# Maps compute_relative_session_load's load_band to the day cell's colour
# class - a session with no computable load (no Pre/Post pairing that day
# and no load entered by hand) falls back to "cfm-session", a neutral grey
# rather than any of the three heat-map colours, since there's genuinely
# nothing to classify.
LOAD_BAND_CLASS = {
    "light": "cfm-load-light",
    "moderate": "cfm-load-moderate",
    "heavy": "cfm-load-heavy",
}
 
SESSION_COLUMNS = [
    "rep_date", "reps", "pre_reps", "mid_reps", "post_reps",
    "is_manual", "session_time",
    "duration_min", "duration_is_override", "session_load",
    "load_is_override", "calculated_load",
    "relative_load_ratio", "load_band",
    "session_title", "session_notes", "avg_velocity_ms", "peak_velocity_ms",
]
 
WEEKLY_COLUMNS = [
    "week_start", "week_end", "sessions", "total_duration_min", "total_session_load",
    "avg_velocity_ms", "best_velocity_ms", "end_fatigue", "end_capacity", "end_freshness",
]
 
 
def _last_non_empty(series: pd.Series) -> str:
    non_empty = series[series.astype(str).str.strip() != ""]
    return str(non_empty.iloc[-1]) if not non_empty.empty else ""
 
 
def _format_time(raw: str) -> str:
    """Normalises a stored start time to HH:MM. Google Sheets sometimes
    hands "18:30" back as "18:30:00", so parse rather than trust the
    string; anything unparseable is shown as-is."""
    raw = str(raw).strip()
    if not raw:
        return ""
    try:
        return pd.to_datetime(raw).strftime("%H:%M")
    except (ValueError, TypeError):
        return raw
 
 
def compute_sessions(log_df: pd.DataFrame) -> pd.DataFrame:
    """One row per rep_date that has at least one logged rep or manual
    session: total reps, per-category rep counts, whether it was added by
    hand, start time (manual sessions only), session duration (manual
    override if set, otherwise estimated from timestamps), session load
    (calculated from Pre/Post data, or entered by hand), title/notes (if
    set), and the day's average/peak velocity across all reps (any
    category)."""
    if log_df.empty:
        return pd.DataFrame(columns=SESSION_COLUMNS)
 
    df = log_df.copy()
    df["rep_date_parsed"] = pd.to_datetime(df["rep_date"]).dt.date
    df["logged_at_parsed"] = pd.to_datetime(df["logged_at"])
    df["is_manual_row"] = df["category"] == fm.MANUAL_CATEGORY
    df["duration_override_parsed"] = (
        pd.to_numeric(df["duration_override_min"], errors="coerce")
        if "duration_override_min" in df.columns else float("nan")
    )
    df["session_title_parsed"] = df["session_title"].fillna("").astype(str) if "session_title" in df.columns else ""
    df["session_notes_parsed"] = df["session_notes"].fillna("").astype(str) if "session_notes" in df.columns else ""
    df["session_time_parsed"] = df["session_time"].fillna("").astype(str) if "session_time" in df.columns else ""
 
    rows = []
    for d, group in df.groupby("rep_date_parsed"):
        reps_group = group[~group["is_manual_row"]]
 
        if not reps_group.empty:
            span = reps_group["logged_at_parsed"].max() - reps_group["logged_at_parsed"].min()
            auto_duration = span.total_seconds() / 60.0
        else:
            auto_duration = 0.0
 
        overrides = group["duration_override_parsed"].dropna()
        if not overrides.empty:
            duration_min = float(overrides.iloc[-1])
            duration_is_override = True
        else:
            duration_min = auto_duration
            duration_is_override = False
 
        avg_vel = pd.to_numeric(reps_group["avg_velocity_ms"], errors="coerce")
        peak_vel = pd.to_numeric(reps_group["peak_velocity_ms"], errors="coerce")
 
        rows.append({
            "rep_date": d,
            "reps": int(len(reps_group)),
            "pre_reps": int((group["category"] == "Pre-session").sum()),
            "mid_reps": int((group["category"] == "Mid-session").sum()),
            "post_reps": int((group["category"] == "Post-session").sum()),
            "is_manual": bool(group["is_manual_row"].any()),
            "session_time": _format_time(_last_non_empty(group["session_time_parsed"])),
            "duration_min": duration_min,
            "duration_is_override": duration_is_override,
            "session_title": _last_non_empty(group["session_title_parsed"]),
            "session_notes": _last_non_empty(group["session_notes_parsed"]),
            "avg_velocity_ms": float(avg_vel.mean()) if avg_vel.notna().any() else float("nan"),
            "peak_velocity_ms": float(peak_vel.max()) if peak_vel.notna().any() else float("nan"),
        })
    sessions_df = pd.DataFrame(rows).sort_values("rep_date").reset_index(drop=True)
 
    session_loads = fm.compute_session_loads(log_df)  # columns: rep_date, session_load, load_is_override, calculated_load
    sessions_df = sessions_df.merge(session_loads, on="rep_date", how="left")
    # Days with no load at all come out of the merge as NaN - normalise to a real bool.
    sessions_df["load_is_override"] = sessions_df["load_is_override"].eq(True)
 
    relative_loads = fm.compute_relative_session_load(log_df)  # columns: rep_date, relative_load_ratio, load_band
    sessions_df = sessions_df.merge(relative_loads, on="rep_date", how="left")
 
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
    and editing happens via the day-picker + edit form in the app instead.
    A load entered by hand is marked with an asterisk."""
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
                if pd.notna(sess["session_load"]):
                    marker = "*" if sess.get("load_is_override") else ""
                    load_line = f"<div class='cfm-load'>load {sess['session_load']:.0f}{marker}</div>"
                else:
                    load_line = ""
                title = str(sess.get("session_title") or "").strip()
                title_line = (
                    f"<div class='cfm-title' title='{_escape(title, quote=True)}'>{_escape(title)}</div>"
                    if title else ""
                )
                start_time = str(sess.get("session_time") or "").strip()
                time_line = f"<div class='cfm-mins'>{_escape(start_time)}</div>" if start_time else ""
                reps = int(sess["reps"])
                badge = f"{reps} reps" if reps > 0 else "Manual"
                mins_line = (
                    f"<div class='cfm-mins'>{sess['duration_min']:.0f} min</div>"
                    if reps > 0 or sess["duration_min"] > 0 else ""
                )
                band_class = LOAD_BAND_CLASS.get(sess.get("load_band"), "cfm-session")
                day_cells.append(
                    f"<td class='{band_class}'>"
                    f"<div class='cfm-daynum'>{day_num}</div>"
                    f"{title_line}"
                    f"<div class='cfm-badge'>{badge}</div>"
                    f"{time_line}"
                    f"{mins_line}"
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
.cfm-title {{ font-weight: 600; color: #047857; font-size: 0.78em; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; max-width: 100%; }}
.cfm-session {{ background: #f3f4f6; }}
.cfm-load-light {{ background: #eff6ff; }}
.cfm-load-moderate {{ background: #fefce8; }}
.cfm-load-heavy {{ background: #fef2f2; }}
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
 