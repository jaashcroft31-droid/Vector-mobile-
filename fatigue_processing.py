"""
Core accelerometer processing for the climbing fatigue monitor.

This is a direct port of the logic in the original MATLAB-derived Python
script, with everything that can't run in a web app removed:
  - no tkinter file dialogs (Streamlit handles file upload instead)
  - no plt.ginput() manual click (the app uses a slider instead)
  - no plt.show() / blocking plots (the app renders Plotly charts instead)
  - no on-disk "processed" folder tree (results are kept in memory /
    offered as downloads instead, since a phone browser can't browse a
    server's filesystem)

The numerical steps (resultant-g, cropping, threshold-based start/end
detection, spectral cutoff selection, Butterworth filtering, drift-corrected
integration, time normalisation) are unchanged from the original script.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt, welch, detrend
from scipy.interpolate import PchipInterpolator

# ==========================================================
# SETTINGS (same defaults as the original script)
# ==========================================================

fs = 100
quiet_dur_s = 1.5
quiet_sd_factor = 7
vel_end_thresh_ms = 0.3
min_move_dur_s = 0.6
g_to_ms2 = 9.81
butter_order = 6
time_norm_points = 100


# ==========================================================
# LOW-LEVEL HELPERS (unchanged from the original script)
# ==========================================================

def compute_resultant_g(gfx, gfy, gfz):
    return np.sqrt(gfx**2 + gfy**2 + gfz**2)


def integrate_velocity(time_s, acceleration_ms2):
    return cumulative_trapezoid(acceleration_ms2, time_s, initial=0)


def calculate_f95(signal_g, fs):
    x = detrend(signal_g, type="constant")

    n = len(x)
    nfft = 2 ** int(np.ceil(np.log2(max(n, 2))))

    win = min(round(1.0 * fs), n)
    win = max(win, 16)
    noverlap = round(0.5 * win)

    f, pxx = welch(
        x,
        fs=fs,
        window="hann",
        nperseg=win,
        noverlap=noverlap,
        nfft=nfft,
    )

    cum_power = cumulative_trapezoid(pxx, f, initial=0)

    if np.max(cum_power) > 0:
        cum_power = cum_power / np.max(cum_power)

    idx95 = np.where(cum_power >= 0.95)[0]

    if len(idx95) == 0:
        f95 = min(12, fs / 2 - 1)
    else:
        f95 = f[idx95[0]]

    fc = max(f95, 0.5)
    fc = min(fc, fs / 2 - 1e-6)

    return f, pxx, cum_power, f95, fc


def butterworth_filter(signal, fs, fc, order):
    wn = fc / (fs / 2)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, signal)


def time_normalise(time_s, velocity_ms, n_points=100):
    tau = np.linspace(0, 1, n_points)
    time_norm = (time_s - time_s[0]) / max(np.finfo(float).eps, time_s[-1] - time_s[0])
    interpolator = PchipInterpolator(time_norm, velocity_ms)
    velocity_tn = interpolator(tau)
    return tau, velocity_tn


# ==========================================================
# STEP 1: full-trace resultant (used for the manual-start picker)
# ==========================================================

@dataclass
class FullTrace:
    t: np.ndarray
    res_corr_g: np.ndarray
    gFx: np.ndarray
    gFy: np.ndarray
    gFz: np.ndarray


def load_full_trace(df) -> FullTrace:
    """Take the raw dataframe (as loaded from the uploaded CSV) and compute
    the gravity-corrected resultant trace used to choose the quiet-hang
    start point. Mirrors the original script: the first column is assumed
    to be a time/index column and is not used for timing (t is rebuilt
    from sample count and the fixed `fs`); columns 2-4 are gFx, gFy, gFz.
    """
    gfx_col = df.columns[1]
    gfy_col = df.columns[2]
    gfz_col = df.columns[3]

    gFx = df[gfx_col].to_numpy(dtype=float)
    gFy = df[gfy_col].to_numpy(dtype=float)
    gFz = df[gfz_col].to_numpy(dtype=float)

    n = len(gFx)
    t = np.arange(n) / fs

    res_g = compute_resultant_g(gFx, gFy, gFz)
    res_corr_g = res_g - 1

    return FullTrace(t=t, res_corr_g=res_corr_g, gFx=gFx, gFy=gFy, gFz=gFz)


def suggest_quiet_start_s(trace: FullTrace) -> float:
    """Best-effort automatic suggestion for where quiet hanging begins:
    the first point where a rolling `quiet_dur_s`-long window has a
    standard deviation below a generous noise-floor threshold. This is
    only ever a *starting point* for the slider in the app - not a
    replacement for the user checking the chart - because a bad guess
    here just means the user drags the slider before confirming.
    """
    win = max(int(round(quiet_dur_s * fs)), 5)
    n = len(trace.res_corr_g)

    if n <= win:
        return float(trace.t[0])

    # rolling std via cumulative sums (fast, no extra deps)
    x = trace.res_corr_g
    csum = np.cumsum(np.insert(x, 0, 0))
    csum2 = np.cumsum(np.insert(x**2, 0, 0))
    count = win
    window_sum = csum[win:] - csum[:-win]
    window_sum2 = csum2[win:] - csum2[:-win]
    mean = window_sum / count
    var = np.maximum(window_sum2 / count - mean**2, 0)
    rolling_sd = np.sqrt(var)

    noise_floor = max(np.median(rolling_sd) * 1.5, 1e-4)
    quiet_idx = np.where(rolling_sd <= noise_floor)[0]

    if len(quiet_idx) == 0:
        return float(trace.t[0])

    return float(trace.t[quiet_idx[0]])


# ==========================================================
# STEP 2: crop / filter / integrate a single rep, given a chosen
# quiet-hang start time (this replaces the plt.ginput() step)
# ==========================================================

@dataclass
class RepResult:
    # movement-window signals (relative time, starting at 0)
    t_rel: np.ndarray
    v_mov_ms: np.ndarray
    res_mov_corr_g_raw: np.ndarray
    res_mov_corr_g_f: np.ndarray
    gFx_mov: np.ndarray
    gFy_mov: np.ndarray
    gFz_mov: np.ndarray

    # spectral
    f: np.ndarray
    pxx: np.ndarray
    f95_Hz: float
    fc_Hz: float

    # time-normalised velocity
    tau: np.ndarray
    v_tn: np.ndarray

    # headline numbers
    avg_velocity_ms: float
    peak_velocity_ms: float
    movement_duration_s: float

    # QC / provenance
    quiet_start_s: float
    quiet_mean_g: float
    quiet_sd_g: float
    start_thresh_g: float
    start_reason: str
    end_reason: str
    warnings: list = field(default_factory=list)


def process_rep(df, quiet_start_s: float) -> RepResult:
    trace = load_full_trace(df)
    t, gFx, gFy, gFz = trace.t, trace.gFx, trace.gFy, trace.gFz
    res_corr_g = trace.res_corr_g

    warnings = []

    manual_start_idx = int(np.argmin(np.abs(t - quiet_start_s)))

    res_c1_corr = res_corr_g[manual_start_idx:]
    t_c1 = t[manual_start_idx:]
    gFx_c1 = gFx[manual_start_idx:]
    gFy_c1 = gFy[manual_start_idx:]
    gFz_c1 = gFz[manual_start_idx:]

    n_c1 = len(res_c1_corr)
    win_quiet = round(quiet_dur_s * fs)

    if n_c1 < win_quiet + 10:
        raise ValueError(
            "Not enough data after the selected quiet-hang start - move the "
            "start point earlier, or check the recording is long enough."
        )

    quiet_seg = res_c1_corr[:win_quiet]
    quiet_mean = float(np.mean(quiet_seg))
    quiet_sd = float(np.std(quiet_seg, ddof=1))
    start_thresh = quiet_mean + quiet_sd_factor * quiet_sd

    above_thresh = np.where(res_c1_corr > start_thresh)[0]

    if len(above_thresh) == 0:
        move_start_rel = 0
        start_reason = "no_threshold_cross_used_first_sample"
        warnings.append(
            "Movement start was never detected above the quiet-hang threshold - "
            "using the very first sample after the selected start. Check the "
            "quiet-hang start point looks right."
        )
    else:
        move_start_rel = int(above_thresh[0])
        start_reason = "threshold_crossed"

    quiet_accel_mean = quiet_mean
    res_c1_corr_zeroed = res_c1_corr - quiet_accel_mean
    a_c1_ms2 = res_c1_corr_zeroed * g_to_ms2

    t_c1_rel = t_c1 - t_c1[0]
    v_c1_rel_ms = integrate_velocity(t_c1_rel, a_c1_ms2)
    v_c1_rel_ms = v_c1_rel_ms - v_c1_rel_ms[0]

    drift_line = np.linspace(v_c1_rel_ms[0], v_c1_rel_ms[-1], len(v_c1_rel_ms))
    velocity_for_end_detection = v_c1_rel_ms - drift_line

    min_move_samples = round(min_move_dur_s * fs)
    search_start_idx = move_start_rel + min_move_samples

    if search_start_idx >= len(velocity_for_end_detection):
        move_end_rel = len(velocity_for_end_detection) - 1
        end_reason = "search_start_beyond_file_used_last_sample"
        warnings.append(
            "The recording ended before the minimum movement duration was "
            "reached - using the last sample as the rep end. The result may "
            "be cut off."
        )
    else:
        search_region = velocity_for_end_detection[search_start_idx:]
        below_threshold_idx = np.where(search_region < vel_end_thresh_ms)[0]

        if len(below_threshold_idx) > 0:
            move_end_rel = search_start_idx + int(below_threshold_idx[0])
            end_reason = "first_velocity_below_0.3_after_buffer"
        else:
            move_end_rel = len(velocity_for_end_detection) - 1
            end_reason = "no_velocity_below_0.3_after_buffer_used_last_sample"
            warnings.append(
                "Velocity never dropped back below the 0.3 m/s end threshold - "
                "using the last sample as the rep end. The result may include "
                "extra data after the rep finished."
            )

    if move_end_rel <= move_start_rel:
        raise ValueError(
            "Detected movement end is not after movement start - try a "
            "different quiet-hang start point."
        )

    t_mov = t_c1[move_start_rel:move_end_rel + 1]
    t_rel = t_mov - t_mov[0]

    gFx_mov = gFx_c1[move_start_rel:move_end_rel + 1]
    gFy_mov = gFy_c1[move_start_rel:move_end_rel + 1]
    gFz_mov = gFz_c1[move_start_rel:move_end_rel + 1]

    res_mov_g_raw = compute_resultant_g(gFx_mov, gFy_mov, gFz_mov)
    res_mov_corr_g_raw = res_mov_g_raw - 1

    if len(res_mov_corr_g_raw) < 16:
        raise ValueError(
            "Cropped movement window is too short to filter (fewer than 16 "
            "samples) - try a different quiet-hang start point."
        )

    f, pxx, cum_power, f95, fc = calculate_f95(res_mov_corr_g_raw, fs)

    gFx_f = butterworth_filter(gFx_mov, fs, fc, butter_order)
    gFy_f = butterworth_filter(gFy_mov, fs, fc, butter_order)
    gFz_f = butterworth_filter(gFz_mov, fs, fc, butter_order)

    res_mov_g_f = compute_resultant_g(gFx_f, gFy_f, gFz_f)
    res_mov_corr_g_f = res_mov_g_f - 1

    a_mov_ms2 = res_mov_corr_g_f * g_to_ms2

    v_mov_ms = integrate_velocity(t_rel, a_mov_ms2)
    v_mov_ms = v_mov_ms - v_mov_ms[0]

    drift_line = np.linspace(v_mov_ms[0], v_mov_ms[-1], len(v_mov_ms))
    v_mov_ms = v_mov_ms - drift_line

    tau, v_tn = time_normalise(t_rel, v_mov_ms, time_norm_points)

    avg_velocity_ms = float(np.mean(v_mov_ms))
    peak_velocity_ms = float(np.max(v_mov_ms))

    return RepResult(
        t_rel=t_rel,
        v_mov_ms=v_mov_ms,
        res_mov_corr_g_raw=res_mov_corr_g_raw,
        res_mov_corr_g_f=res_mov_corr_g_f,
        gFx_mov=gFx_f,
        gFy_mov=gFy_f,
        gFz_mov=gFz_f,
        f=f,
        pxx=pxx,
        f95_Hz=float(f95),
        fc_Hz=float(fc),
        tau=tau,
        v_tn=v_tn,
        avg_velocity_ms=avg_velocity_ms,
        peak_velocity_ms=peak_velocity_ms,
        movement_duration_s=float(t_rel[-1]),
        quiet_start_s=float(quiet_start_s),
        quiet_mean_g=quiet_mean,
        quiet_sd_g=quiet_sd,
        start_thresh_g=float(start_thresh),
        start_reason=start_reason,
        end_reason=end_reason,
        warnings=warnings,
    )