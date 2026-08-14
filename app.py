"""
Climbing Fatigue Monitor - Streamlit app

Upload an accelerometer CSV recorded during a pull-up, confirm where the
"quiet hang" starts, and the app crops/filters the rep the same way the
original script did, then calculates and logs average + peak velocity
against a date so you can track fatigue across a session or over time.
It also compares today's velocity against your recent baseline to give a
daily readiness read, and lays reps out on a calendar.

Reps are logged as one of three categories:
    Pre-session  - 3 reps done before a session; today's Pre-session mean
                   is what "Training readiness" (vs the last 14 days) is
                   based on.
    Mid-session  - reps done through a session to track in-session fatigue;
                   this is what "Session readiness" is based on.
    Post-session - 3 reps done after a session; not analysed yet, logged
                   for a future session-load feature.

The log lives in a Google Sheet (not a local file), so it survives
Streamlit Community Cloud putting the app to sleep. See README.md for the
one-time Google Sheets setup.

Run locally with:
    pip install -r requirements.txt
    streamlit run app.py
"""

from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_gsheets import GSheetsConnection

import fatigue_processing as fp
import calendar_view as cal

GSHEETS_WORKSHEET = "fatigue_log"

LOG_COLUMNS = [
    "logged_at", "rep_date", "category", "note", "filename",
    "avg_velocity_ms", "peak_velocity_ms", "movement_duration_s",
    "quiet_start_s", "f95_Hz", "fc_Hz", "start_reason", "end_reason",
]

REP_CATEGORIES = ["Pre-session", "Mid-session", "Post-session"]
REP_CATEGORY_TARGETS = {"Pre-session": 3, "Post-session": 3}  # Mid-session has no fixed count

# Readiness bands: percentage difference of today's daily-mean velocity vs
# the mean of the previous READINESS_BASELINE_DAYS days' daily means.
#   diff_pct >  +READINESS_GOOD_BAND_PCT                       -> High performance
#   -READINESS_GOOD_BAND_PCT <= diff_pct <= +READINESS_GOOD_BAND_PCT -> Good performance
#   -READINESS_FATIGUE_PCT   <= diff_pct <  -READINESS_GOOD_BAND_PCT -> Low performance
#   diff_pct <  -READINESS_FATIGUE_PCT                          -> High fatigue
READINESS_BASELINE_DAYS = 14
READINESS_GOOD_BAND_PCT = 5.0
READINESS_FATIGUE_PCT = 15.0

st.set_page_config(page_title="Climbing Fatigue Monitor", page_icon="🧗", layout="centered")


# ==========================================================
# LOG HELPERS (backed by a Google Sheet, not a local file, so it
# survives Streamlit Community Cloud putting the app to sleep)
# ==========================================================

def get_gsheets_conn():
    return st.connection("gsheets", type=GSheetsConnection)


def get_spreadsheet_url() -> str:
    """Read the spreadsheet URL straight out of secrets ourselves rather
    than relying on GSheetsConnection's automatic secrets pickup - that
    auto-pickup is a known unreliable path for this library (several
    independent reports of "Spreadsheet must be specified" even with a
    correctly configured secrets.toml), so we pass it explicitly on every
    call instead."""
    try:
        return st.secrets["connections"]["gsheets"]["spreadsheet"]
    except (KeyError, FileNotFoundError):
        st.error(
            "Couldn't find `spreadsheet` under `[connections.gsheets]` in your "
            "secrets. Check `.streamlit/secrets.toml` locally, or the Secrets "
            "box in Streamlit Community Cloud's app settings."
        )
        st.stop()


def load_log() -> pd.DataFrame:
    conn = get_gsheets_conn()
    df = conn.read(spreadsheet=get_spreadsheet_url(), worksheet=GSHEETS_WORKSHEET, ttl=0)
    if df is None:
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = df.dropna(how="all")
    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[LOG_COLUMNS].reset_index(drop=True)


def save_log(log_df: pd.DataFrame):
    conn = get_gsheets_conn()
    conn.update(spreadsheet=get_spreadsheet_url(), worksheet=GSHEETS_WORKSHEET, data=log_df[LOG_COLUMNS])


def append_to_log(row: dict):
    current = load_log()
    updated = pd.concat([current, pd.DataFrame([row])], ignore_index=True)
    save_log(updated)


def clear_log():
    save_log(pd.DataFrame(columns=LOG_COLUMNS))


def compute_daily_summary(log_df: pd.DataFrame, category: str | None = None) -> pd.DataFrame:
    """One row per rep_date: the day's mean average-velocity and mean
    peak-velocity across reps logged that day, plus how many reps went
    into each mean. Pass `category` (e.g. "Pre-session") to restrict to
    just that category of rep."""
    df = log_df if category is None else log_df[log_df["category"] == category]
    daily = (
        df.groupby("rep_date", as_index=False)
        .agg(
            avg_velocity_ms=("avg_velocity_ms", "mean"),
            peak_velocity_ms=("peak_velocity_ms", "mean"),
            reps=("avg_velocity_ms", "size"),
        )
        .sort_values("rep_date")
        .reset_index(drop=True)
    )
    return daily


def categorize_diff_pct(diff_pct: float) -> str:
    """Shared banding used for the per-metric detail categories (not the
    combined readiness score, which has its own colour bands)."""
    if diff_pct > READINESS_GOOD_BAND_PCT:
        return "High performance"
    elif diff_pct >= -READINESS_GOOD_BAND_PCT:
        return "Good performance"
    elif diff_pct >= -READINESS_FATIGUE_PCT:
        return "Low performance"
    else:
        return "High fatigue"


def compute_readiness(daily_df: pd.DataFrame, metric_col: str, today: date) -> dict | None:
    """"Training readiness": compare today's daily-mean `metric_col`
    against the mean of that same daily-mean value over the previous
    READINESS_BASELINE_DAYS days (today itself excluded from the baseline
    window).

    Returns None if there's no data logged for `today`, or no baseline
    data in the trailing window - both cases mean readiness can't be
    computed yet, rather than being a category in their own right.
    """
    daily = daily_df.copy()
    daily["rep_date"] = pd.to_datetime(daily["rep_date"]).dt.date

    today_row = daily[daily["rep_date"] == today]
    if today_row.empty:
        return None

    window_start = today - timedelta(days=READINESS_BASELINE_DAYS)
    baseline_rows = daily[(daily["rep_date"] >= window_start) & (daily["rep_date"] < today)]
    if baseline_rows.empty:
        return None

    today_value = float(today_row[metric_col].iloc[0])
    baseline_value = float(baseline_rows[metric_col].mean())

    if baseline_value == 0:
        return None

    diff_pct = (today_value - baseline_value) / baseline_value * 100

    return {
        "today_value": today_value,
        "baseline_value": baseline_value,
        "diff_pct": diff_pct,
        "category": categorize_diff_pct(diff_pct),
        "baseline_desc": f"{len(baseline_rows)}-day baseline",
    }


def _rep_readiness_at(todays_sorted: pd.DataFrame, metric_col: str, idx: int) -> dict | None:
    """Compare the rep at position `idx` (0-based, chronological) within
    a single day's reps against the mean of the reps before it. Returns
    None for idx == 0, since the first rep of the day has nothing earlier
    to compare against."""
    if idx == 0:
        return None

    latest_value = float(todays_sorted[metric_col].iloc[idx])
    baseline_rows = todays_sorted.iloc[:idx]
    baseline_value = float(baseline_rows[metric_col].mean())

    if baseline_value == 0:
        return None

    diff_pct = (latest_value - baseline_value) / baseline_value * 100
    n = len(baseline_rows)

    return {
        "today_value": latest_value,
        "baseline_value": baseline_value,
        "diff_pct": diff_pct,
        "category": categorize_diff_pct(diff_pct),
        "baseline_desc": f"today's earlier {n} rep{'s' if n != 1 else ''}",
    }


def compute_session_readiness(log_df: pd.DataFrame, metric_col: str, today: date) -> dict | None:
    """"Session readiness": compare the most recently logged Mid-session
    rep today against the mean of today's *other* Mid-session reps so far.
    Pre-session and Post-session reps aren't part of this rolling
    within-session comparison. Returns None if fewer than 2 Mid-session
    reps are logged today.
    """
    todays = log_df.copy()
    todays["rep_date"] = pd.to_datetime(todays["rep_date"]).dt.date
    todays = todays[(todays["rep_date"] == today) & (todays["category"] == "Mid-session")]
    todays = todays.sort_values("logged_at").reset_index(drop=True)

    if len(todays) < 2:
        return None

    return _rep_readiness_at(todays, metric_col, len(todays) - 1)


def compute_session_readiness_history(log_df: pd.DataFrame, today: date) -> pd.DataFrame:
    """One row per Mid-session rep logged today, with that rep's combined
    readiness score vs the mean of the Mid-session reps logged earlier the
    same day - the data behind the "session readiness" bar chart. The
    first rep of the day has nothing earlier to compare against, so it's
    included as a neutral "baseline" bar (score 100, its own distinct
    grey colour) rather than being scored or left out - it's the
    reference point every later rep in the session gets compared to."""
    todays = log_df.copy()
    todays["rep_date"] = pd.to_datetime(todays["rep_date"]).dt.date
    todays = todays[(todays["rep_date"] == today) & (todays["category"] == "Mid-session")]
    todays = todays.sort_values("logged_at").reset_index(drop=True)

    rows = []
    for idx in range(len(todays)):
        if idx == 0:
            rows.append({
                "rep_number": 1,
                "logged_at": todays["logged_at"].iloc[0],
                "score": 100.0,
                "category": "baseline",
            })
            continue

        r_avg = _rep_readiness_at(todays, "avg_velocity_ms", idx)
        r_peak = _rep_readiness_at(todays, "peak_velocity_ms", idx)
        if r_avg is None or r_peak is None:
            continue
        score = compute_readiness_score(r_avg, r_peak)
        rows.append({
            "rep_number": idx + 1,
            "logged_at": todays["logged_at"].iloc[idx],
            "score": score["score"],
            "category": score["category"],
        })
    return pd.DataFrame(rows, columns=["rep_number", "logged_at", "score", "category"])


def compute_training_readiness_history(
    daily_df: pd.DataFrame, days_back: int = READINESS_BASELINE_DAYS
) -> pd.DataFrame:
    """One row per day, for the most recent `days_back` distinct days that
    have a computable training-readiness score (i.e. enough trailing
    baseline history) - the data behind the "training readiness" bar
    chart."""
    daily = daily_df.copy()
    daily["rep_date_parsed"] = pd.to_datetime(daily["rep_date"]).dt.date
    candidate_days = sorted(daily["rep_date_parsed"].unique())[-days_back:]

    rows = []
    for d in candidate_days:
        r_avg = compute_readiness(daily_df, "avg_velocity_ms", d)
        r_peak = compute_readiness(daily_df, "peak_velocity_ms", d)
        if r_avg is None or r_peak is None:
            continue
        score = compute_readiness_score(r_avg, r_peak)
        rows.append({"rep_date": d.isoformat(), "score": score["score"], "category": score["category"]})
    return pd.DataFrame(rows, columns=["rep_date", "score", "category"])


def compute_readiness_score(readiness_avg: dict, readiness_peak: dict) -> dict:
    """Combined readiness score: starts at 100, then the average- and
    peak-velocity percentage differences (whichever readiness mode they
    came from) are both added on. >100 means better than baseline overall,
    <100 means worse. Colour bands are independent of the per-metric
    High/Good/Low/Fatigue categories above."""
    score = 100 + readiness_avg["diff_pct"] + readiness_peak["diff_pct"]

    if score > 105:
        category = "pink"
    elif score >= 95:
        category = "green"
    elif score >= 85:
        category = "yellow"
    else:
        category = "red"

    return {"score": score, "category": category}


# ==========================================================
# CHART HELPERS
# ==========================================================

def full_trace_figure(trace: fp.FullTrace, quiet_start_s: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trace.t, y=trace.res_corr_g, mode="lines",
        name="Resultant accel (g, gravity-corrected)",
        line=dict(width=1.5),
    ))
    fig.add_vline(x=quiet_start_s, line_dash="dash", line_color="red")
    fig.update_layout(
        title="Full recording - drag the slider below to mark the start of quiet hanging",
        xaxis_title="Time (s)",
        yaxis_title="Resultant accel, gravity-corrected (g)",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def rep_velocity_figure(result: fp.RepResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.t_rel, y=result.v_mov_ms, mode="lines",
        name="Velocity", line=dict(width=2),
    ))
    fig.add_hline(y=result.peak_velocity_ms, line_dash="dot", line_color="green",
                  annotation_text=f"peak {result.peak_velocity_ms:.2f} m/s")
    fig.add_hline(y=result.avg_velocity_ms, line_dash="dot", line_color="orange",
                  annotation_text=f"avg {result.avg_velocity_ms:.2f} m/s")
    fig.update_layout(
        title="Cropped, filtered rep velocity",
        xaxis_title="Time from movement start (s)",
        yaxis_title="Velocity (m/s)",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def spectral_figure(result: fp.RepResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.f, y=result.pxx, mode="lines", name="PSD"))
    fig.add_vline(x=result.f95_Hz, line_dash="dash", line_color="red",
                  annotation_text=f"f95 {result.f95_Hz:.2f} Hz")
    fig.update_layout(
        title="Power spectral density (used to choose the filter cutoff)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="PSD",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def daily_summary_figure(daily_df: pd.DataFrame) -> go.Figure:
    hover = [f"{n} rep{'s' if n != 1 else ''}" for n in daily_df["reps"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily_df["rep_date"], y=daily_df["avg_velocity_ms"],
        name="Avg velocity (daily mean)",
        text=hover, hovertemplate="%{x}<br>avg %{y:.2f} m/s<br>%{text}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=daily_df["rep_date"], y=daily_df["peak_velocity_ms"],
        name="Peak velocity (daily mean)",
        text=hover, hovertemplate="%{x}<br>peak %{y:.2f} m/s<br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        title="Daily average velocity (Pre-session)",
        xaxis_title="Date",
        yaxis_title="Velocity (m/s)",
        barmode="group",
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


READINESS_SCORE_HEX = {
    "pink": "#ec4899",
    "green": "#22c55e",
    "yellow": "#eab308",
    "red": "#ef4444",
    "baseline": "#9ca3af",
}


def readiness_bar_figure(history_df: pd.DataFrame, x_col: str, x_title: str) -> go.Figure:
    colors = [READINESS_SCORE_HEX[c] for c in history_df["category"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=history_df[x_col], y=history_df["score"],
        marker_color=colors,
        hovertemplate="%{x}<br>score %{y:.1f}<extra></extra>",
    ))
    fig.add_hline(y=100, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Readiness score over time",
        xaxis_title=x_title,
        yaxis_title="Readiness score",
        height=340,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    return fig


def history_figure(log_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=log_df["logged_at"], y=log_df["peak_velocity_ms"],
        mode="lines+markers", name="Peak velocity (m/s)",
    ))
    fig.add_trace(go.Scatter(
        x=log_df["logged_at"], y=log_df["avg_velocity_ms"],
        mode="lines+markers", name="Average velocity (m/s)",
    ))
    fig.update_layout(
        title="Velocity over time",
        xaxis_title="Logged at",
        yaxis_title="Velocity (m/s)",
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ==========================================================
# READINESS DISPLAY
# ==========================================================

READINESS_ICON = {
    "High performance": "🚀",
    "Good performance": "✅",
    "Low performance": "⚠️",
    "High fatigue": "🔴",
}

READINESS_SCORE_ICON = {
    "pink": "🩷",
    "green": "🟢",
    "yellow": "🟡",
    "red": "🔴",
}


def show_readiness(label: str, readiness: dict):
    icon = READINESS_ICON[readiness["category"]]
    text = (
        f"{icon} **{label}: {readiness['category']}** "
        f"({readiness['diff_pct']:+.1f}% vs {readiness['baseline_desc']} - "
        f"today {readiness['today_value']:.2f} m/s, "
        f"baseline {readiness['baseline_value']:.2f} m/s)"
    )
    if readiness["category"] in ("High performance", "Good performance"):
        st.success(text)
    elif readiness["category"] == "Low performance":
        st.warning(text)
    else:
        st.error(text)


# ==========================================================
# MAIN APP
# ==========================================================

st.title("🧗 Climbing Fatigue Monitor")
st.caption(
    "Upload a pull-up accelerometer recording, confirm the quiet-hang start, "
    "and log average + peak velocity to track fatigue over a session."
)

log_df = load_log()

# ==========================================================
# TODAY'S READINESS
# ==========================================================

st.subheader("Today's readiness")

readiness_mode = st.selectbox(
    "Readiness mode",
    ["Training readiness (vs last 14 days)", "Session readiness (vs today's reps)"],
)

if log_df.empty:
    st.caption("Log some reps to see today's readiness.")
else:
    today = date.today()

    if readiness_mode.startswith("Training"):
        daily_df = compute_daily_summary(log_df, category="Pre-session")
        readiness_avg = compute_readiness(daily_df, "avg_velocity_ms", today)
        readiness_peak = compute_readiness(daily_df, "peak_velocity_ms", today)
        not_enough_msg = (
            f"Log 3 Pre-session reps today, plus some Pre-session history from "
            f"the last {READINESS_BASELINE_DAYS} days, to see training readiness."
        )
        history_df = compute_training_readiness_history(daily_df)
        history_x_col, history_x_title = "rep_date", "Date"
        history_empty_msg = (
            f"Not enough day-to-day Pre-session history yet to chart training "
            f"readiness - need at least one prior day within the last "
            f"{READINESS_BASELINE_DAYS} days."
        )
    else:
        readiness_avg = compute_session_readiness(log_df, "avg_velocity_ms", today)
        readiness_peak = compute_session_readiness(log_df, "peak_velocity_ms", today)
        not_enough_msg = "Log at least 2 Mid-session reps today to see session readiness."
        history_df = compute_session_readiness_history(log_df, today)
        history_x_col, history_x_title = "rep_number", "Rep number today"
        history_empty_msg = "Log a Mid-session rep today to see the session readiness chart."

    if readiness_avg is None or readiness_peak is None:
        st.caption(not_enough_msg)
    else:
        score = compute_readiness_score(readiness_avg, readiness_peak)
        icon = READINESS_SCORE_ICON[score["category"]]
        with st.expander(f"{icon}  Readiness score: {score['score']:.1f}"):
            show_readiness("Average velocity", readiness_avg)
            show_readiness("Peak velocity", readiness_peak)

    if history_df.empty:
        st.caption(history_empty_msg)
    else:
        st.plotly_chart(
            readiness_bar_figure(history_df, history_x_col, history_x_title),
            use_container_width=True,
        )

st.divider()

# ==========================================================
# UPLOAD & PROCESS A REP
# ==========================================================

uploaded_files = st.file_uploader(
    "Upload accelerometer CSV(s)", type=["csv"], accept_multiple_files=True,
)

if uploaded_files:
    file_names = [f.name for f in uploaded_files]
    active_name = st.selectbox("Which file are you working on?", file_names)
    active_file = next(f for f in uploaded_files if f.name == active_name)

    # Read the CSV fresh each time the active file is selected (cheap for
    # accelerometer-sized files, and keeps this simple to reason about).
    active_file.seek(0)
    try:
        df = pd.read_csv(active_file)
    except Exception as e:
        st.error(f"Couldn't read '{active_name}' as a CSV: {e}")
        st.stop()

    if df.shape[1] < 4:
        st.error(
            f"'{active_name}' has only {df.shape[1]} column(s) - expected at least 4 "
            "(time, gFx, gFy, gFz)."
        )
        st.stop()

    try:
        trace = fp.load_full_trace(df)
    except Exception as e:
        st.error(f"Couldn't process '{active_name}': {e}")
        st.stop()

    slider_key = f"quiet_start_{active_name}_{active_file.size}"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = round(fp.suggest_quiet_start_s(trace), 2)

    quiet_start_s = st.slider(
        "Quiet-hang start (s)",
        min_value=float(trace.t[0]),
        max_value=float(trace.t[-1]),
        step=0.05,
        key=slider_key,
    )

    st.plotly_chart(full_trace_figure(trace, quiet_start_s), use_container_width=True)

    rep_category = st.selectbox("Rep type", REP_CATEGORIES, index=1)  # default Mid-session

    col1, col2 = st.columns(2)
    with col1:
        rep_date = st.date_input("Date", value=date.today())
    with col2:
        note = st.text_input("Note (optional)", placeholder="e.g. session 2, rep 3")

    if rep_category in REP_CATEGORY_TARGETS:
        target = REP_CATEGORY_TARGETS[rep_category]
        day_str = rep_date.strftime("%Y-%m-%d")
        count_so_far = 0
        if not log_df.empty:
            count_so_far = int((
                (log_df["rep_date"] == day_str) & (log_df["category"] == rep_category)
            ).sum())
        st.caption(f"{rep_category} reps logged for {day_str}: {count_so_far} of {target}")

    process_clicked = st.button("Process rep", type="primary", use_container_width=True)

    result_key = f"result_{active_name}_{active_file.size}"

    if process_clicked:
        try:
            result = fp.process_rep(df, quiet_start_s)
            st.session_state[result_key] = result
        except ValueError as e:
            st.session_state.pop(result_key, None)
            st.error(str(e))

    if result_key in st.session_state:
        result = st.session_state[result_key]

        for w in result.warnings:
            st.warning(w)

        m1, m2, m3 = st.columns(3)
        m1.metric("Average velocity", f"{result.avg_velocity_ms:.2f} m/s")
        m2.metric("Peak velocity", f"{result.peak_velocity_ms:.2f} m/s")
        m3.metric("Rep duration", f"{result.movement_duration_s:.2f} s")

        st.plotly_chart(rep_velocity_figure(result), use_container_width=True)

        if st.button("Save to log", use_container_width=True):
            append_to_log({
                "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rep_date": rep_date.strftime("%Y-%m-%d"),
                "category": rep_category,
                "note": note,
                "filename": active_name,
                "avg_velocity_ms": round(result.avg_velocity_ms, 4),
                "peak_velocity_ms": round(result.peak_velocity_ms, 4),
                "movement_duration_s": round(result.movement_duration_s, 4),
                "quiet_start_s": round(result.quiet_start_s, 4),
                "f95_Hz": round(result.f95_Hz, 4),
                "fc_Hz": round(result.fc_Hz, 4),
                "start_reason": result.start_reason,
                "end_reason": result.end_reason,
            })
            st.success(f"Saved '{active_name}' to the log as a {rep_category.lower()} rep.")

        with st.expander("Processing details"):
            st.write(
                f"- Quiet-hang mean: `{result.quiet_mean_g:.5f}` g, "
                f"sd: `{result.quiet_sd_g:.5f}` g\n"
                f"- Movement-start threshold: `{result.start_thresh_g:.5f}` g "
                f"({result.start_reason})\n"
                f"- Movement end: {result.end_reason}\n"
                f"- Spectral cutoff: f95 = `{result.f95_Hz:.2f}` Hz, "
                f"filter cutoff = `{result.fc_Hz:.2f}` Hz"
            )
            st.plotly_chart(spectral_figure(result), use_container_width=True)

            cropped_df = pd.DataFrame({
                "TimeRel_s": result.t_rel,
                "Velocity_mps": result.v_mov_ms,
                "Resultant_corr_filt_g": result.res_mov_corr_g_f,
            })
            st.download_button(
                "Download cropped/filtered rep data (CSV)",
                data=cropped_df.to_csv(index=False),
                file_name=f"{Path(active_name).stem}_cropped_filtered.csv",
                mime="text/csv",
                use_container_width=True,
            )
else:
    st.info("Upload one or more CSV files to log a new rep.")

st.divider()

# ==========================================================
# HISTORY
# ==========================================================

st.subheader("History")

with st.expander("Restore or merge a backed-up log"):
    st.caption(
        "If your log ever gets wiped (e.g. a Streamlit Community Cloud app "
        "waking up after being asleep), upload a previously downloaded "
        "fatigue_log.csv here to merge it back in."
    )
    restore_file = st.file_uploader(
        "Upload a backed-up fatigue_log.csv", type=["csv"], key="restore_uploader",
    )
    if restore_file is not None:
        try:
            restore_df = pd.read_csv(restore_file)
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")
            restore_df = None

        if restore_df is not None:
            required_cols = {"logged_at", "rep_date", "avg_velocity_ms", "peak_velocity_ms"}
            missing = required_cols - set(restore_df.columns)
            if missing:
                st.error(
                    f"That doesn't look like a fatigue log - missing column(s): "
                    f"{', '.join(sorted(missing))}"
                )
            else:
                if "category" not in restore_df.columns or restore_df["category"].isna().all():
                    restore_df["category"] = "Pre-session"
                    st.caption(
                        "This file predates rep categories - treating all its rows "
                        "as Pre-session."
                    )
                st.write(f"Found {len(restore_df)} row(s) in the uploaded file.")
                if st.button("Merge into current log", use_container_width=True):
                    current = load_log()
                    for col in LOG_COLUMNS:
                        if col not in restore_df.columns:
                            restore_df[col] = pd.NA
                    merged = pd.concat([current, restore_df[LOG_COLUMNS]], ignore_index=True)
                    merged = merged.drop_duplicates()
                    save_log(merged)
                    st.success(f"Merged - log now has {len(merged)} row(s).")

log_df = load_log()  # reload - may include a rep just saved above, or just restored

if log_df.empty:
    st.caption("No reps logged yet - process a rep above and hit 'Save to log'.")
else:
    daily_df = compute_daily_summary(log_df, category="Pre-session")
    if daily_df.empty:
        st.caption("No Pre-session reps logged yet - the daily chart is based on those.")
    else:
        st.plotly_chart(daily_summary_figure(daily_df), use_container_width=True)

    with st.expander("Per-rep detail"):
        st.plotly_chart(history_figure(log_df), use_container_width=True)
        st.dataframe(
            log_df[["logged_at", "rep_date", "category", "note", "filename",
                    "avg_velocity_ms", "peak_velocity_ms", "movement_duration_s"]],
            use_container_width=True,
            hide_index=True,
        )

    dl_col, clear_col = st.columns(2)
    with dl_col:
        st.download_button(
            "Download full log (CSV)",
            data=log_df.to_csv(index=False),
            file_name="fatigue_log.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with clear_col:
        if st.button("Clear log", use_container_width=True):
            st.session_state["confirm_clear"] = True

    if st.session_state.get("confirm_clear"):
        st.warning("This deletes every logged rep. Are you sure?")
        yes_col, no_col = st.columns(2)
        if yes_col.button("Yes, delete everything", use_container_width=True):
            clear_log()
            st.session_state["confirm_clear"] = False
            st.rerun()
        if no_col.button("Cancel", use_container_width=True):
            st.session_state["confirm_clear"] = False

st.divider()

# ==========================================================
# CALENDAR
# ==========================================================

st.subheader("Calendar")

sessions_df = cal.compute_sessions(log_df)

if sessions_df.empty:
    st.caption("Log some reps to start building your calendar.")
else:
    weekly_df = cal.compute_weekly_summary(sessions_df)

    month_options = sorted(sessions_df["rep_date"].apply(lambda d: (d.year, d.month)).unique(), reverse=True)
    today_ym = (date.today().year, date.today().month)
    default_index = month_options.index(today_ym) if today_ym in month_options else 0
    month_labels = [date(y, m, 1).strftime("%B %Y") for (y, m) in month_options]

    selected_label = st.selectbox("Month", month_labels, index=default_index, key="calendar_month_select")
    year, month = month_options[month_labels.index(selected_label)]

    st.markdown(
        cal.build_calendar_html(year, month, sessions_df, weekly_df),
        unsafe_allow_html=True,
    )

    days_in_month = sessions_df[
        sessions_df["rep_date"].apply(lambda d: d.year == year and d.month == month)
    ]
    if not days_in_month.empty:
        day_options = sorted(days_in_month["rep_date"].tolist(), reverse=True)
        day_labels = [d.strftime("%a %d %b") for d in day_options]

        selected_day_label = st.selectbox("Session detail", day_labels, key="calendar_day_select")
        selected_day = day_options[day_labels.index(selected_day_label)]
        day_row = days_in_month[days_in_month["rep_date"] == selected_day].iloc[0]

        d1, d2, d3 = st.columns(3)
        d1.metric("Duration", f"{day_row['duration_min']:.0f} min")
        d2.metric("Avg velocity", f"{day_row['avg_velocity_ms']:.2f} m/s")
        d3.metric("Peak velocity", f"{day_row['peak_velocity_ms']:.2f} m/s")
        st.caption(
            f"{int(day_row['pre_reps'])} pre-session, {int(day_row['mid_reps'])} mid-session, "
            f"{int(day_row['post_reps'])} post-session rep(s)"
        )

        with st.expander("Reps logged this day"):
            day_log = log_df[log_df["rep_date"] == selected_day.isoformat()]
            st.dataframe(
                day_log[["logged_at", "category", "note",
                         "avg_velocity_ms", "peak_velocity_ms", "movement_duration_s"]],
                use_container_width=True,
                hide_index=True,
            )
