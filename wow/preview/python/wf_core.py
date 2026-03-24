"""
wf_core — Wow & Flutter Analysis Engine

Single analysis engine for turntable FG signal analysis.
Supports two input types:
  - Audio PCM (FG signal from turntable motor)
  - Device text exports (e.g. shaknspin)

API:
  analyzeFull(data, sampleRate=None, inputType='audio')  → full result dict
  getPlotData(plotId, params={})                          → on-demand plot arrays

No plotting, no file I/O, no CLI. Consumers (SPA frontend, CLI wrapper)
handle presentation.

Signal processing pipeline (audio):
  1. Carrier frequency estimation via FFT peak detection
  2. Configurable bandpass prefilter (default 0.1×–1.9× carrier, Virtins spec)
  3. Hilbert analytic signal demodulation → instantaneous frequency
  4. Lowpass at 0.4×carrier (measurement BW + anti-alias)
  5. Edge trim (Hilbert transient + prefilter settling)
  6. Decimation to ~2×LP cutoff rate
  7. Metrics: AES6/DIN weighted + unweighted + drift
  8. Spectrum + peak detection + motor harmonic ID

Device pipeline enters at step 7 (deviation already available).
"""

import numpy as np
from scipy.signal import (butter, filtfilt, sosfiltfilt, bilinear,
                          lfilter, lfilter_zi, freqz, medfilt, hilbert,
                          resample_poly, decimate)
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks as _find_peaks


# ========================= INTERNAL CONSTANTS =========================

PREFILTER_BW_FACTOR = .45 #0.30
PREFILTER_ORDER = 2
SMOOTH_CYCLES = 1
OUTLIER_THRESH_PCT = 10.0

# Coupling marker significance multiplier (× median)
_COUPLING_SIG_MULT = 3.0


# ========================= MODULE STATE =========================
# Stashed by analyzeFull, consumed by getPlotData.
# Cleared on each new analyzeFull call.

_state = {}


def _clear_state():
    global _state
    _state = {}


# ========================= STATUS CALLBACK =========================

_status_callback = None


def set_status_callback(cb):
    """Set a callback function for progress updates. cb(message: str)."""
    global _status_callback
    _status_callback = cb


async def _status(msg):
    if _status_callback is not None:
        result = _status_callback(msg)
        if hasattr(result, '__await__'):
            await result


# ========================= SIGNAL PROCESSING =========================

def _estimate_carrier_freq(sig, fs):
    """Carrier frequency estimate using FFT peak detection on first 2s."""
    n_samp = min(len(sig), int(2.0 * fs))
    s = sig[:n_samp].astype(np.float64)
    s = s - np.mean(s)
    win = np.hanning(len(s))
    X = np.abs(np.fft.rfft(s * win))
    freqs = np.fft.rfftfreq(len(s), d=1.0 / fs)
    min_idx = max(1, int(20.0 / (fs / len(s))))
    peak_idx = min_idx + np.argmax(X[min_idx:])
    return freqs[peak_idx]


def _bandpass_prefilter(sig, fs, low, high, order=2):
    """Bandpass to clean up signal before zero-crossing detection."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, sig)


def _find_zero_crossings(sig, fs, hysteresis_frac=0.05, sinc_taps=32):
    """
    Positive-going zero crossings with hysteresis + linear interpolation.
    Vectorized via searchsorted.
    Returns array of crossing times in seconds.
    """
    sig = sig - np.mean(sig)
    threshold = hysteresis_frac * np.max(np.abs(sig))

    zc_mask = (sig[:-1] < 0) & (sig[1:] >= 0)
    zc_indices = np.where(zc_mask)[0]

    arm_mask = sig < -threshold
    arm_indices = np.where(arm_mask)[0]

    if len(zc_indices) == 0 or len(arm_indices) == 0:
        return np.array([])

    prev_zc = np.empty_like(zc_indices)
    prev_zc[0] = 0
    prev_zc[1:] = zc_indices[:-1]

    arm_search = np.searchsorted(arm_indices, prev_zc, side='left')
    valid = ((arm_search < len(arm_indices)) &
             (arm_indices[np.minimum(arm_search, len(arm_indices) - 1)] < zc_indices))
    crossing_indices = zc_indices[valid]

    margin = max(sinc_taps, 1)
    crossing_indices = crossing_indices[
        (crossing_indices >= margin) &
        (crossing_indices < len(sig) - margin - 1)
    ]

    s0 = sig[crossing_indices]
    s1 = sig[crossing_indices + 1]
    frac = -s0 / (s1 - s0)
    times = (crossing_indices.astype(np.float64) + frac) / fs
    return times


def _crossings_to_frequency(crossing_times):
    """Convert zero-crossing times to instantaneous frequency estimates."""
    periods = np.diff(crossing_times)
    midpoints = crossing_times[:-1] + periods / 2.0
    freqs = 1.0 / periods
    return midpoints, freqs


def _smooth_frequency(freqs, n_cycles):
    """Simple moving average over n_cycles."""
    if n_cycles <= 1:
        return freqs
    kernel = np.ones(n_cycles) / n_cycles
    smoothed = np.convolve(freqs, kernel, mode='same')
    half = n_cycles // 2
    for i in range(half):
        smoothed[i] = np.mean(freqs[:i + half + 1])
        smoothed[-(i + 1)] = np.mean(freqs[-(i + half + 1):])
    return smoothed


def _interpolate_to_uniform(times, freqs, output_rate=None, sinc_taps=16):
    """
    Resample non-uniform frequency samples to uniform time grid.
    CubicSpline — vectorized, fast in Pyodide.
    Returns (t_uniform, f_uniform, output_rate).
    """
    if output_rate is None:
        output_rate = len(times) / (times[-1] - times[0])

    dt_raw = (times[-1] - times[0]) / (len(times) - 1)
    margin = sinc_taps * dt_raw
    t_start = times[0] + margin
    t_end = times[-1] - margin
    t_uniform = np.arange(t_start, t_end, 1.0 / output_rate)

    cs = CubicSpline(times, freqs)
    f_uniform = cs(t_uniform)
    return t_uniform, f_uniform, output_rate


def _edge_trim(t_freq, freq, prefilter_bw_hz=None):
    """
    Edge trim combining time-based minimum (prefilter settling) with
    amplitude-based detection for large outliers.

    Time-based: trim at least 2/bw_hz seconds from each end — covers
    the prefilter's startup/shutdown transient regardless of carrier rate.

    Amplitude-based: also scan first/last 20 crossings for any that
    exceed 5×MAD, and extend the trim to cover them.
    """
    if len(t_freq) <= 40:
        return t_freq, freq

    n_xing = len(freq)

    # --- Time-based minimum trim (prefilter settling) ---
    time_trim = 0
    if prefilter_bw_hz is not None and prefilter_bw_hz > 0:
        settle_s = 2.0 / prefilter_bw_hz
        t_start = t_freq[0]
        t_end = t_freq[-1]
        # Count crossings within settle_s of each edge
        time_trim_start = int(np.searchsorted(t_freq, t_start + settle_s))
        time_trim_end = n_xing - int(np.searchsorted(t_freq, t_end - settle_s))
    else:
        time_trim_start = 0
        time_trim_end = 0

    # --- Amplitude-based trim (original logic) ---
    interior = freq[n_xing // 10: -n_xing // 10]
    f_med = np.median(interior)
    mad = np.median(np.abs(interior - f_med))
    thresh = 5.0 * mad / f_med * 100.0
    dev_from_med = np.abs(freq - f_med) / f_med * 100.0

    amp_trim_start = 0
    for i in range(min(20, n_xing)):
        if dev_from_med[i] >= thresh:
            amp_trim_start = i + 1

    amp_trim_end = 0
    for i in range(n_xing - 1, max(n_xing - 21, -1), -1):
        if dev_from_med[i] >= thresh:
            amp_trim_end = n_xing - i

    # Take the larger of time-based and amplitude-based
    trim_start = max(time_trim_start, amp_trim_start)
    trim_end = max(time_trim_end, amp_trim_end)

    if trim_start > 0 or trim_end > 0:
        end_idx = n_xing - trim_end if trim_end > 0 else n_xing
        t_freq = t_freq[trim_start:end_idx]
        freq = freq[trim_start:end_idx]

    return t_freq, freq


def _outlier_reject(t_freq, freq):
    """MAD-based adaptive outlier rejection."""
    f_median = np.median(freq)
    mad = np.median(np.abs(freq - f_median))
    mad_thresh_hz = max(8.0 * mad, f_median * 0.001)
    mad_thresh_hz = min(mad_thresh_hz, f_median * OUTLIER_THRESH_PCT / 100.0)
    outlier_mask = np.abs(freq - f_median) > mad_thresh_hz
    n_rejected = int(np.sum(outlier_mask))
    if n_rejected > 0:
        t_freq = t_freq[~outlier_mask]
        freq = freq[~outlier_mask]
    return t_freq, freq, n_rejected


def _median_despike(t_freq, freq, kernel=5):
    """Median despike for high-carrier signals (>500 Hz)."""
    f_carrier_est = (len(freq) / (t_freq[-1] - t_freq[0])
                     if len(t_freq) > 1 else 0)
    if f_carrier_est > 500 and len(freq) > kernel:
        mad_clean = np.median(np.abs(freq - np.median(freq)))
        freq_med = medfilt(freq, kernel_size=kernel)
        spike_thresh = 3.0 * mad_clean
        is_spike = np.abs(freq - freq_med) > spike_thresh
        if np.any(is_spike):
            freq = freq.copy()
            freq[is_spike] = freq_med[is_spike]
    return freq


# ========================= AES6 WEIGHTING FILTER =========================

def _make_aes6_weighting_filter(fs=1000.0):
    """
    AES6-2008 frequency weighting filter.

    Analog prototype optimized via Wurcer's methodology (brute grid +
    differential evolution, PTP error metric) against all 17 AES6 Table 1
    spec points.  Designed for fs=1000 Hz (post-SRC).

    Topology:
      - 3 zeros at DC (s^3)          -> 18 dB/oct highpass
      - Triple real pole at 0.6265 Hz -> HP-to-BP transition shaping
      - 1 real pole at 11.32 Hz      -> ~6 dB/oct LPF rolloff
      - 1 finite zero at 227.9 Hz    -> HF rolloff shaping

    Performance at fs=1000 Hz:
      PTP error vs Table 1:  1.013 dB
      17/17 spec points:     ALL PASS
      Min margin to tol:     1.32 dB (at 1.6 Hz)

    Returns (b, a) digital filter coefficients via bilinear transform.
    """
    POLES_HZ = [0.626524, 0.626524, 0.626524, 11.316542]
    ZERO_FINITE_HZ = 227.918719

    num_s = np.convolve([1.0, 0, 0, 0], [1, 2 * np.pi * ZERO_FINITE_HZ])
    den_s = np.array([1.0])
    for p in POLES_HZ:
        den_s = np.convolve(den_s, [1, 2 * np.pi * p])

    b, a = bilinear(num_s, den_s, fs=fs)
    w4 = 2 * np.pi * 4.0 / fs
    _, h_ref = freqz(b, a, worN=[w4])
    b /= np.abs(h_ref[0])
    return b, a


def _src_to_1khz(signal, fs):
    """
    Sample-rate convert deviation signal to 1 kHz.

    Odd-extension pads the signal before resample_poly (same technique
    scipy's filtfilt uses internally) — preserves both amplitude and
    derivative at boundaries, preventing FIR anti-aliasing ringing.

    Returns (signal_1k, 1000.0) or (signal, fs) if already at 1 kHz.
    """
    FS_TARGET = 1000
    if abs(fs - FS_TARGET) < 0.5:
        return signal, float(fs)

    from math import gcd
    g = gcd(int(FS_TARGET), int(round(fs)))
    up = int(FS_TARGET) // g
    down = int(round(fs)) // g

    # Odd extension (filtfilt-style): preserves both amplitude AND derivative
    # at boundary.  2*signal[0] - signal[N:0:-1] continues the slope smoothly.
    pad_n = min(int(0.5 * fs), len(signal) // 4)
    padded = np.concatenate([
        2 * signal[0] - signal[pad_n:0:-1],
        signal,
        2 * signal[-1] - signal[-2:-pad_n - 2:-1],
    ])

    resampled = resample_poly(padded, up, down)

    # Trim mirrored portions
    pad_out = int(round(pad_n * up / down))
    trimmed = resampled[pad_out:len(resampled) - pad_out]

    return trimmed, float(FS_TARGET)


# ========================= METRICS =========================

def _metric(value, confidence=0):
    """Create a metric dict: { value, confidence }."""
    return {'value': float(value), 'confidence': int(confidence)}


_DESPIKE_REF_K = 101    # wide medfilt kernel for MAD reference signal
_DESPIKE_N_MAD = 5      # threshold: flag samples > N × MAD from reference


def _despike_plot(dev_pct):
    """MAD-based despike for plot data. Operates at native rate, no SRC.

    Uses a wide median filter as a robust reference, then flags and replaces
    samples whose deviation from the reference exceeds N × MAD.  Handles
    clustered multi-impulse artifacts (vinyl pops/ticks) that simple medfilt
    cannot catch.

    Parameters
    ----------
    dev_pct : ndarray — deviation in percent

    Returns
    -------
    ndarray — despiked deviation in percent (same length, same rate)
    """
    if len(dev_pct) < _DESPIKE_REF_K:
        return dev_pct

    reference = medfilt(dev_pct, kernel_size=_DESPIKE_REF_K)
    residual = dev_pct - reference
    mad = np.median(np.abs(residual))

    if mad <= 0:
        return dev_pct

    thresh = _DESPIKE_N_MAD * mad
    is_spike = np.abs(residual) > thresh
    if not np.any(is_spike):
        return dev_pct

    out = dev_pct.copy()
    out[is_spike] = reference[is_spike]
    return out


def _compute_wf_metrics(deviation_frac, fs, carrier_freq, skip_seconds=0.0,
                        unweighted_bw='virtins'):
    """
    Compute all wow & flutter metrics from fractional deviation signal.

    Unweighted metrics computed at original sample rate (pre-SRC) to
    preserve full bandwidth up to 0.4×carrier.
    Weighted metrics computed at 1 kHz via SRC (AES6 weighting filter
    uses lfilter + lfilter_zi, causal, zero transient).

    Parameters:
        deviation_frac: fractional deviation signal (f - f_mean) / f_mean
        fs: sample rate of deviation_frac (Hz)
        carrier_freq: detected carrier frequency (Hz)
        skip_seconds: seconds to skip at start for filter settling
        unweighted_bw: bandwidth mode for unweighted metrics
            'virtins' (default): LP at 0.4×carrier, no HP — matches
                Virtins Multi-Instrument and satisfies AES6-2008 §6.1.1
                NOTE ("at least 0.2 Hz to 200 Hz").
            'aes6_min': BP 0.2–200 Hz — minimum AES6 NOTE range.

    Returns:
        standard: dict of standardized metrics (AES6/DIN/IEC)
        non_standard: dict of non-standardized metrics
        dev_unwtd: LP-filtered fractional deviation used for unweighted metrics
    """
    nyq = fs / 2.0
    n_total = len(deviation_frac)
    capture_dur = n_total / fs
    skip = int(skip_seconds * fs)

    # =================================================================
    # Unweighted (at original sample rate — preserves full bandwidth)
    # =================================================================
    if unweighted_bw == 'aes6_min':
        # AES6 §6.1.1 NOTE minimum: 0.2–200 Hz bandpass
        bp_lo = 0.2
        bp_hi = min(200.0, nyq * 0.95)
        if bp_hi > bp_lo:
            sos_u = butter(4, [bp_lo / nyq, bp_hi / nyq],
                           btype='band', output='sos')
            dev_unwtd = sosfiltfilt(sos_u, deviation_frac)
        else:
            dev_unwtd = deviation_frac.copy()
        unwtd_flutter_hi = bp_hi
    else:
        # Virtins spec: LP at 0.4×carrier, no HP (lower bound = 1/dur)
        lp_hi = min(0.4 * carrier_freq, nyq * 0.95)
        if lp_hi > 0:
            sos_u = butter(4, lp_hi / nyq, btype='low', output='sos')
            dev_unwtd = sosfiltfilt(sos_u, deviation_frac)
        else:
            dev_unwtd = deviation_frac.copy()
        unwtd_flutter_hi = lp_hi

    dev_u = dev_unwtd[skip:]
    unwtd_peak = float(np.percentile(np.abs(dev_u), 95) * 100.0)
    unwtd_rms = float(np.sqrt(np.mean(dev_u**2)) * 100.0)

    # --- Unweighted wow/flutter band separation (Virtins spec) ---
    # Wow: 0.5–6 Hz, Flutter: 6 Hz – unwtd_flutter_hi
    wow_cut = min(6.0, nyq * 0.95)
    unwtd_wow_rms = 0.0
    unwtd_flutter_rms = 0.0
    if wow_cut > 0.5:
        wow_lo = 0.5
        if wow_lo < nyq * 0.95 and wow_cut > wow_lo:
            sos_wow_u = butter(6, [wow_lo / nyq, wow_cut / nyq],
                               btype='band', output='sos')
            wow_sig_u = sosfiltfilt(sos_wow_u, dev_unwtd)
            unwtd_wow_rms = float(np.sqrt(np.mean(wow_sig_u[skip:]**2)) * 100.0)

        if unwtd_flutter_hi > wow_cut:
            sos_flutter_u = butter(6, [wow_cut / nyq, unwtd_flutter_hi / nyq],
                                   btype='band', output='sos')
            flutter_sig_u = sosfiltfilt(sos_flutter_u, dev_unwtd)
            unwtd_flutter_rms = float(np.sqrt(np.mean(flutter_sig_u[skip:]**2)) * 100.0)

    # --- Drift (non-standardized, at original sample rate) ---
    drift_rms = 0.0
    drift_lo = max(0.05, 1.0 / capture_dur)
    drift_hi = 0.5
    if drift_hi > drift_lo and drift_hi < nyq * 0.95:
        drift_taper_s = 2.0
        taper_n = min(int(drift_taper_s * fs), n_total // 4)
        taper_window = np.ones(n_total)
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(taper_n) / taper_n))
        taper_window[:taper_n] = ramp
        taper_window[-taper_n:] = ramp[::-1]

        drift_order = 10
        sos_drift = butter(drift_order,
                           [drift_lo / nyq, drift_hi / nyq],
                           btype='band', output='sos')
        drift_sig = sosfiltfilt(sos_drift, deviation_frac * taper_window)

        drift_skip = max(skip, taper_n, int(2.0 * fs))
        if n_total > 2 * drift_skip:
            drift_sig = drift_sig[drift_skip:-drift_skip]
            drift_rms = float(np.sqrt(np.mean(drift_sig**2)) * 100.0)

    # =================================================================
    # Weighted (SRC to 1 kHz for AES6 weighting filter)
    # =================================================================
    dev_w, fs_w = _src_to_1khz(deviation_frac, fs)
    nyq_w = fs_w / 2.0
    skip_w = int(skip_seconds * fs_w)

    b_w, a_w = _make_aes6_weighting_filter(fs_w)
    zi = lfilter_zi(b_w, a_w) * dev_w[0]
    dev_weighted, _ = lfilter(b_w, a_w, dev_w, zi=zi)
    dw = dev_weighted[skip_w:]
    wtd_peak = float(np.percentile(np.abs(dw), 95) * 100.0)
    wtd_rms = float(np.sqrt(np.mean(dw**2)) * 100.0)

    # --- Weighted wow/flutter (standardized) ---
    wow_cut_w = min(6.0, nyq_w * 0.95)
    wtd_wow_rms = 0.0
    wtd_flutter_rms = 0.0
    if wow_cut_w > 0.5:
        sos_wow = butter(6, wow_cut_w / nyq_w, btype='low', output='sos')
        wow_sig = sosfiltfilt(sos_wow, dev_weighted)
        wtd_wow_rms = float(np.sqrt(np.mean(wow_sig[skip_w:]**2)) * 100.0)

        sos_flutter = butter(6, wow_cut_w / nyq_w, btype='high', output='sos')
        flutter_sig = sosfiltfilt(sos_flutter, dev_weighted)
        wtd_flutter_rms = float(np.sqrt(np.mean(flutter_sig[skip_w:]**2)) * 100.0)

    # --- Confidence ---
    # Placeholder: all 0 (full confidence).
    # Future: heuristics based on duration, crossing density, SNR, etc.
    conf = 0

    standard = {
        'weighted_peak':        _metric(wtd_peak, conf),
        'weighted_rms':         _metric(wtd_rms, conf),
        'weighted_wow_rms':     _metric(wtd_wow_rms, conf),
        'weighted_flutter_rms': _metric(wtd_flutter_rms, conf),
        'unweighted_peak':      _metric(unwtd_peak, conf),
        'unweighted_rms':       _metric(unwtd_rms, conf),
    }

    non_standard = {
        'unweighted_wow_rms':     _metric(unwtd_wow_rms, conf),
        'unweighted_flutter_rms': _metric(unwtd_flutter_rms, conf),
        'drift_rms':              _metric(drift_rms, conf),
    }

    return standard, non_standard, dev_unwtd


# ========================= SPECTRUM =========================

def _compute_spectrum(deviation_pct, output_rate, max_freq=50.0,
                      max_peaks=16, peak_threshold=0.02):
    """
    Compute deviation spectrum and detect peaks.
    max_freq: upper frequency limit (Hz) — prefilter bw_hz for audio, 50 for device.
    Returns spectrum dict (freqs, amplitude, peaks without identity/coupling).
    Arrays are truncated to max_freq.
    """
    N = len(deviation_pct)
    win = np.hanning(N)
    X = np.fft.rfft(deviation_pct * win)
    freqs = np.fft.rfftfreq(N, d=1.0 / output_rate)

    amp = np.abs(X) * np.sqrt(2.0) / (np.sum(win) * np.sqrt(freqs[1]))
    fft_amp_bin = np.abs(X) * 2.0 / np.sum(win)

    # Truncate to valid range
    max_f = min(max_freq, output_rate / 2)
    mask = freqs <= max_f
    freqs = freqs[mask]
    amp = amp[mask]
    fft_amp_bin = fft_amp_bin[mask]

    if len(amp) > 1:
        pk_idx, _ = _find_peaks(amp, height=np.max(amp[1:]) * peak_threshold)
        pk_idx = [p for p in pk_idx if freqs[p] > 0.3]
        pk_idx = sorted(pk_idx, key=lambda p: amp[p], reverse=True)[:max_peaks]
    else:
        pk_idx = []

    peaks = []
    for p in sorted(pk_idx, key=lambda p: freqs[p]):
        bin_rms = float(fft_amp_bin[p] / np.sqrt(2))
        peaks.append({
            'freq': float(freqs[p]),
            'amplitude': float(amp[p]),
            'rms': bin_rms,
            'fft_bin_index': int(p),
            'label': None,
            'am_coupled': False,
            'coupling_strength': None,
        })

    return {
        'freqs': freqs.tolist(),
        'amplitude': amp.tolist(),
        'peaks': peaks,
        'coupling_threshold': None,
    }


# ========================= RPM AUTO-DETECTION =========================

# Standard platter speeds: (nominal RPM, rotation frequency Hz)
_STANDARD_SPEEDS = [
    (16.67, 16.67 / 60.0),   # half-speed 33⅓
    (22.50, 22.50 / 60.0),   # half-speed 45
    (33.33, 33.33 / 60.0),   # standard LP
    (45.00, 45.00 / 60.0),   # standard single
    (78.26, 78.26 / 60.0),   # shellac
]

_RPM_TOLERANCE = 0.05        # ±5% of nominal f_rot
_RPM_MIN_FREQ = 0.20         # Hz — below this is DC drift, not rotation
_RPM_MAX_FREQ = 2.0          # Hz — above this is not a platter speed
_RPM_HIGH_SNR = 10.0         # peak / median for high confidence
_RPM_MED_SNR = 5.0           # peak / median for medium confidence
_RPM_STD_LOW_SNR = 3.0       # accept lower SNR if very close to standard speed
_RPM_STD_CLOSE_TOL = 0.01    # ±1% for low-SNR standard match
_RPM_MIN_DURATION = 4.0      # seconds — minimum file length for detection


def _detect_rpm(spectrum_freqs, spectrum_amp, duration):
    """
    Detect platter RPM from FM deviation spectrum.

    Checks amplitude at each standard rotation frequency directly.
    Returns the standard speed with the best SNR above threshold.

    Returns dict: {value, source, confidence, f_rot_measured} or
    {value: None, source: None, confidence: None, f_rot_measured: None}
    """
    no_detect = {'value': None, 'source': None, 'confidence': None,
                 'f_rot_measured': None}

    if duration < _RPM_MIN_DURATION:
        return no_detect

    freqs = np.asarray(spectrum_freqs)
    amp = np.asarray(spectrum_amp)

    if len(freqs) < 3:
        return no_detect

    df = float(freqs[1] - freqs[0])

    # Compute noise floor excluding bins near standard speeds
    det_mask = (freqs >= _RPM_MIN_FREQ) & (freqs <= _RPM_MAX_FREQ)
    noise_mask = det_mask.copy()
    for _, nominal_frot in _STANDARD_SPEEDS:
        bin_idx = int(round((nominal_frot - freqs[0]) / df))
        exclude = max(1, int(round(nominal_frot * _RPM_TOLERANCE / df)))
        lo = max(0, bin_idx - exclude)
        hi = min(len(freqs), bin_idx + exclude + 1)
        noise_mask[lo:hi] = False
    if np.any(noise_mask):
        median_noise = float(np.median(amp[noise_mask]))
    else:
        median_noise = float(np.median(amp[det_mask]))
    if median_noise <= 0:
        return no_detect

    best_score = 0
    best_result = None

    for nominal_rpm, nominal_frot in _STANDARD_SPEEDS:
        # Find nearest bin and check ±1 bin for peak
        bin_idx = int(round((nominal_frot - freqs[0]) / df))
        if bin_idx < 1 or bin_idx >= len(amp) - 1:
            continue
        peak_amp = float(max(amp[bin_idx - 1], amp[bin_idx], amp[bin_idx + 1]))
        peak_bin = bin_idx + int(np.argmax(amp[bin_idx - 1:bin_idx + 2])) - 1
        snr = peak_amp / median_noise
        closeness = 1.0 - abs(float(freqs[peak_bin]) - nominal_frot) / nominal_frot
        score = snr * max(closeness, 0.0)

        if snr >= _RPM_STD_LOW_SNR and score > best_score:
            best_score = score
            confidence = min(1.0, snr / _RPM_HIGH_SNR)
            best_result = {
                'value': nominal_rpm,
                'source': 'detected',
                'confidence': round(confidence, 2),
                'f_rot_measured': round(float(freqs[peak_bin]), 4),
            }

    return best_result if best_result is not None else no_detect


# ========================= MOTOR HARMONIC IDENTIFICATION =========================

def _identify_motor_harmonics(peaks, f_rot, motor_slots=None,
                               motor_poles=None, drive_ratio=1.0):
    """
    Tag peaks with motor harmonic labels.

    f_rot: rotation frequency (rpm / 60). If None, no labeling is done.
    motor_slots/motor_poles: enables electrical/slot/ripple labels.
    drive_ratio: motor-to-platter speed ratio for non-direct-drive.
        Motor frequencies at the platter are multiplied by drive_ratio.

    Mutates peaks in-place (sets 'label' field).
    """
    if f_rot is None or f_rot <= 0:
        return  # No rpm provided, can't label anything

    # Motor frequencies adjusted for drive ratio
    if motor_poles is not None:
        pole_pairs = motor_poles // 2
        f_elec = pole_pairs * f_rot * drive_ratio
        f_ripple = 3 * f_elec
    else:
        f_elec = None
        f_ripple = None

    if motor_slots is not None:
        f_slot = motor_slots * f_rot * drive_ratio
    else:
        f_slot = None

    # Tolerance: scale with rotation freq
    tol = max(f_rot * 0.3, 0.05)

    for peak in peaks:
        f = peak['freq']
        label = None

        # Most specific first: torque ripple
        if f_ripple is not None:
            for n in range(1, 5):
                if abs(f - n * f_ripple) < tol:
                    label = f'{n}× torque ripple' if n > 1 else 'torque ripple'
                    break

        # Slot passing
        if label is None and f_slot is not None:
            for n in range(1, 5):
                if abs(f - n * f_slot) < tol:
                    label = f'{n}× slot' if n > 1 else 'slot passing'
                    break

        # Electrical
        if label is None and f_elec is not None:
            for n in range(1, 10):
                if abs(f - n * f_elec) < tol:
                    label = f'{n}× electrical' if n > 1 else 'electrical'
                    break

        # Motor rotation harmonics (non-direct-drive only)
        if label is None and drive_ratio != 1.0:
            f_motor_rot = f_rot * drive_ratio
            for n in range(1, 20):
                if abs(f - n * f_motor_rot) < tol:
                    label = f'{n}× motor rot' if n > 1 else 'motor rot'
                    break

        # Platter rotation harmonics (always available)
        if label is None:
            for n in range(1, 20):
                if abs(f - n * f_rot) < tol:
                    label = f'{n}× rotation' if n > 1 else 'rotation'
                    break

        peak['label'] = label


# ========================= AM/FM COUPLING =========================

def _compute_coupling_at_freq(am_full, fm_full, fs, freq):
    """
    Compute coupling strength at one frequency using sosfiltfilt Butterworth
    bandpass — ported directly from fg_coupling_strength.py.

    Returns: strength (R × sig), R, sig, phase_deg
    """
    nyq = fs / 2.0
    bw = max(0.15, freq * 0.3)
    lo = max(0.2, freq - bw)
    hi = min(freq + bw, nyq * 0.9)
    if hi <= lo:
        return 0.0, 0.0, 0.0, 0.0

    sos = butter(4, [lo / nyq, hi / nyq], btype='band', output='sos')
    am_band = sosfiltfilt(sos, am_full)
    fm_band = sosfiltfilt(sos, fm_full)

    # Skip edges (3s each side)
    skip = int(3.0 * fs)
    if len(am_band) <= 2 * skip:
        return 0.0, 0.0, 0.0, 0.0
    am_band = am_band[skip:-skip]
    fm_band = fm_band[skip:-skip]

    # Signal amplitude: geometric mean of AM and FM RMS (%)
    am_rms = np.std(am_band)
    fm_rms = np.std(fm_band)
    sig = np.sqrt(am_rms * fm_rms)

    if sig < 1e-12:
        return 0.0, 0.0, 0.0, 0.0

    # Phase lock R
    am_a = hilbert(am_band)
    fm_a = hilbert(fm_band)
    dphi = np.angle(fm_a) - np.angle(am_a)
    resultant = np.mean(np.exp(1j * dphi))
    R = float(np.abs(resultant))
    phase_deg = float(np.degrees(np.angle(resultant)))

    return R * sig, R, sig, phase_deg


def _find_coupling_freqs(am_full, fm_full, fs, max_freq=50.0, n_peaks=8):
    """
    Find data-driven test frequencies from peaks in both AM and FM spectra.
    Ported from fg_am_fm_phase.py collect_sweep_freqs / find_spectral_peaks.

    Returns sorted, deduplicated list of frequencies.
    """
    def _spectral_peaks(signal, fs, min_freq=0.3, max_freq=50.0, n_peaks=8):
        N = len(signal)
        win = np.hanning(N)
        sig_ac = signal - np.mean(signal)
        spec = np.abs(np.fft.rfft(sig_ac * win)) * 2.0 / np.sum(win)
        freqs = np.fft.rfftfreq(N, d=1.0 / fs)

        mask = (freqs >= min_freq) & (freqs <= max_freq)
        spec_m = spec.copy()
        spec_m[~mask] = 0

        threshold = np.max(spec_m) * 0.05
        dist = max(1, int(0.15 / (freqs[1] - freqs[0])))
        pk_idx, _ = _find_peaks(spec_m, height=threshold, distance=dist)

        pk_idx = sorted(pk_idx, key=lambda i: spec[i], reverse=True)[:n_peaks]
        return sorted([float(freqs[i]) for i in pk_idx])

    am_peaks = _spectral_peaks(am_full, fs, max_freq=max_freq, n_peaks=n_peaks)
    fm_peaks = _spectral_peaks(fm_full, fs, max_freq=max_freq, n_peaks=n_peaks)

    all_freqs = sorted(set(am_peaks + fm_peaks))

    # Deduplicate: merge frequencies within 0.1 Hz
    merged = []
    for f in all_freqs:
        if not merged or abs(f - merged[-1]) > 0.1:
            merged.append(f)
        else:
            merged[-1] = (merged[-1] + f) / 2.0

    return sorted(merged)


def _compute_coupling_markers(peaks, am_full, fm_full, fs):
    """
    Compute coupling strength at data-driven frequencies found in both
    AM and FM spectra, plus rotation harmonics when rpm is known.
    Ported from fg_coupling_strength.py.

    Mutates peaks in-place (sets coupling_strength, am_coupled).
    Returns coupling_threshold.
    """
    # Stash full signals for on-demand lissajous
    _state['_am_full'] = am_full
    _state['_fm_full'] = fm_full

    # Find test frequencies from AM/FM spectra
    test_freqs = _find_coupling_freqs(am_full, fm_full, fs)

    # Compute coupling at each test frequency
    freq_strengths = {}  # freq -> (strength, R, sig, phase)
    for freq in test_freqs:
        strength, R, sig, phase = _compute_coupling_at_freq(
            am_full, fm_full, fs, freq)
        freq_strengths[freq] = (strength, R, sig, phase)

    # Significance threshold: 3× median of all coupling values.
    # This matches fg_coupling_strength.py exactly.
    all_strengths = [v[0] for v in freq_strengths.values()]
    if len(all_strengths) > 1:
        median_val = float(np.median(all_strengths))
        threshold = _COUPLING_SIG_MULT * median_val
    else:
        threshold = 0.0

    # Match each spectrum peak to the nearest test frequency (within tolerance)
    # and assign its coupling values.
    for peak in peaks:
        pf = peak['freq']
        best_freq = None
        best_dist = float('inf')
        for tf in test_freqs:
            d = abs(pf - tf)
            if d < best_dist:
                best_dist = d
                best_freq = tf
        # Match tolerance: within 0.15 Hz or 15% of freq, whichever is larger
        tol = max(0.15, pf * 0.15)
        if best_freq is not None and best_dist <= tol:
            s, R, sig, phase = freq_strengths[best_freq]
            peak['coupling_strength'] = float(s)
            peak['am_coupled'] = (s > threshold) if threshold > 0 else False
        else:
            peak['coupling_strength'] = None
            peak['am_coupled'] = False

    return float(threshold) if threshold > 0 else None


# ========================= DEVICE INPUT =========================

def _detect_device_format(text_data):
    """
    Detect device format from text content.
    Returns format string or None.
    """
    # shaknspin: semicolon-delimited, header contains known keys
    lines = text_data.split('\n')[:25]
    shaknspin_keys = {'Session', 'Avg Speed', 'W&F peak', 'W&F DIN'}
    found = 0
    for line in lines:
        parts = line.strip().split(';')
        if len(parts) >= 2:
            key = parts[0].strip().rstrip(':')
            if key in shaknspin_keys:
                found += 1
    if found >= 2:
        return 'shaknspin'

    return None


def _parse_shaknspin_text(text_data):
    """
    Parse shaknspin text export from string.

    Returns dict:
        time_s: numpy array of time in seconds
        deviation_pct: numpy array of speed deviation in %
        sample_rate: float (Hz)
        f_mean: float (mean rotation frequency in Hz)
        metadata: dict with header fields
        device_label: str (session/serial for display)
    """
    lines = text_data.split('\n')

    # Header
    header = {}
    for line in lines[:25]:
        parts = line.strip().split(';')
        if len(parts) == 2:
            header[parts[0].strip().rstrip(':')] = parts[1].strip()

    # Time-domain trace: columns 0–1 (Time;Speed) are present in every data
    # row.  4-column rows carry extra Freq;Magnitude for the spectrum but
    # the speed trace is continuous across both sections (0 – ~8000 ms).
    td_time = []
    td_speed = []
    for line in lines[26:]:
        parts = line.strip().split(';')
        if len(parts) >= 2:
            try:
                td_time.append(float(parts[0]))
                td_speed.append(float(parts[1]))
            except ValueError:
                continue

    time_ms = np.array(td_time)
    time_s = time_ms / 1000.0
    speed_rpm = np.array(td_speed)

    if len(time_ms) < 2:
        raise ValueError("shaknspin data has fewer than 2 time samples")

    fs = 1000.0 / (time_ms[1] - time_ms[0])
    nominal_rpm = np.mean(speed_rpm)
    f_mean = nominal_rpm / 60.0
    deviation_pct = (speed_rpm - nominal_rpm) / nominal_rpm * 100.0

    # Device label: serial number if available, fall back to session
    device_label = header.get('Device', None)
    if device_label is None:
        device_label = header.get('Session', None)
        if device_label and '.CSV' in device_label:
            device_label = device_label.replace('.CSV', '')

    return {
        'time_s': time_s,
        'deviation_pct': deviation_pct,
        'sample_rate': fs,
        'f_mean': f_mean,
        'metadata': header,
        'device_label': device_label,
    }


def _parse_device_data(text_data, fmt):
    """Dispatch to format-specific parser."""
    if fmt == 'shaknspin':
        return _parse_shaknspin_text(text_data)
    raise ValueError(f"Unknown device format: {fmt}")


# ========================= MAIN API =========================

def analyzeFull(data, sampleRate=None, inputType='audio',
                rpm=None, motor_slots=None, motor_poles=None,
                drive_ratio=1.0, fm_bw=None):
    """
    Single entry point for all analysis. Sync wrapper — detects whether
    an event loop is running (Pyodide) and returns a coroutine for await,
    otherwise runs synchronously (CLI).

    Parameters:
        data: PCM float64 array (audio) or text string (device)
        sampleRate: required for audio, ignored for device
        inputType: 'audio' or 'device'
        rpm: platter/transport RPM (optional). Enables polar plot and
             rotation harmonic labeling. For device input, extracted
             from data automatically but can be overridden.
        motor_slots: number of motor stator slots (optional). Enables
                     slot passing harmonic labels. Requires rpm.
        motor_poles: number of motor poles (optional). Enables electrical
                     and torque ripple harmonic labels. Requires rpm.
        drive_ratio: motor-to-platter speed ratio for non-direct-drive
                     (default 1.0 = direct drive).
        fm_bw: FM measurement bandwidth in Hz, or 'aes_min' for 200 Hz,
               or None (default) for max (0.4 × carrier).  Clamped to
               0.4 × carrier internally.

    Returns structured result dict per SPA integration plan.
    """
    import asyncio

    coro = _analyzeFull_async(data, sampleRate=sampleRate, inputType=inputType,
                               rpm=rpm, motor_slots=motor_slots,
                               motor_poles=motor_poles, drive_ratio=drive_ratio,
                               fm_bw=fm_bw)

    # In Pyodide (or any running event loop), return the coroutine for await.
    # In CLI (no event loop), run synchronously.
    try:
        loop = asyncio.get_running_loop()
        return coro  # caller must await
    except RuntimeError:
        return asyncio.run(coro)


async def _analyzeFull_async(data, sampleRate=None, inputType='audio',
                              rpm=None, motor_slots=None, motor_poles=None,
                              drive_ratio=1.0, fm_bw=None):
    """Async implementation of analyzeFull."""
    _clear_state()

    # Stash motor params for use in sub-functions
    _state['_rpm'] = rpm
    _state['_motor_slots'] = motor_slots
    _state['_motor_poles'] = motor_poles
    _state['_drive_ratio'] = drive_ratio

    if inputType == 'device':
        return await _analyze_device(data, rpm=rpm)
    else:
        if sampleRate is None:
            raise ValueError("sampleRate is required for audio input")
        return await _analyze_audio(data, sampleRate, fm_bw=fm_bw)


async def _analyze_audio(pcm_data, sample_rate, fm_bw=None):
    """Full audio pipeline — Hilbert demod with configurable measurement BW."""
    fs = sample_rate
    sig = np.asarray(pcm_data, dtype=np.float64)
    nyq = fs / 2.0

    # 0. Carrier amplitude gate — trim leading/trailing silence
    #    ZC with hysteresis naturally ignored silence; Hilbert demod does not
    #    (bandpass rings at carrier freq).  Detect where carrier is actually
    #    present via short-window RMS and slice the signal down.
    gate_win = max(int(0.01 * fs), 1)            # 10 ms windows
    n_full = (len(sig) // gate_win) * gate_win
    if n_full > gate_win:
        rms = np.sqrt(np.mean(
            sig[:n_full].reshape(-1, gate_win) ** 2, axis=1))
        gate_thresh = 0.10 * np.max(rms)         # 10 % of peak RMS
        above = np.where(rms >= gate_thresh)[0]
        if len(above) > 0:
            first_sample = above[0] * gate_win
            last_sample = min((above[-1] + 1) * gate_win, len(sig))
            sig = sig[first_sample:last_sample]

    # 0b. SRC to 48 kHz — reduces Hilbert FFT size, processing time ~O(N log N)
    #     Max carrier 5 kHz → prefilter upper edge 9.5 kHz → 48 kHz is safe.
    _TARGET_FS = 48000
    if fs != _TARGET_FS:
        from math import gcd
        _up = _TARGET_FS
        _down = int(fs)
        _g = gcd(_up, _down)
        sig = resample_poly(sig, _up // _g, _down // _g)
        fs = float(_TARGET_FS)
        nyq = fs / 2.0

    duration = len(sig) / fs
    await _status(f"Loaded: {duration:.1f}s at {int(fs)} Hz")

    # 1. Carrier frequency
    await _status("Detecting carrier frequency...")
    f_est = _estimate_carrier_freq(sig, fs)

    # 2. Bandpass prefilter — fixed at 0.1×–1.9× carrier
    await _status("Applying prefilter...")
    if f_est > 0:
        bp_low = max(f_est * 0.1, 1.0)
        bp_high = min(f_est * 1.9, nyq * 0.95)
        b, a = butter(PREFILTER_ORDER, [bp_low / nyq, bp_high / nyq], btype='band')
        sig_filtered = filtfilt(b, a, sig)
    else:
        sig_filtered = sig

    # 3. Hilbert demod → instantaneous frequency
    await _status("Hilbert demodulation...")
    analytic = hilbert(sig_filtered)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) * fs / (2.0 * np.pi)
    del analytic, phase

    # 4. Lowpass — FM measurement bandwidth
    #    Resolve fm_bw: None → max (0.4×carrier), 'aes_min' → 200 Hz, numeric → Hz
    #    Always clamped to 0.4×carrier (Nyquist of FM system)
    carrier_max = 0.4 * f_est
    if fm_bw is None:
        lp_cut = carrier_max
    elif fm_bw == 'aes_min':
        lp_cut = min(200.0, carrier_max)
    else:
        lp_cut = min(float(fm_bw), carrier_max)
    lp_cut = min(lp_cut, nyq * 0.95)

    if lp_cut > 0:
        sos_lp = butter(4, lp_cut / nyq, btype='low', output='sos')
        inst_freq = sosfiltfilt(sos_lp, inst_freq)

    # 5. Edge trim — Hilbert transient settling time
    #    The FFT-based Hilbert has edge artifacts that decay over a few cycles
    #    of the lowest frequency present.  After prefilter, f_low = 0.1×carrier.
    #    We trim 3 cycles × 2× safety margin = 6 / f_low from each edge.
    await _status("Trimming edges...")
    n_if = len(inst_freq)

    f_low = max(f_est * 0.1, 1.0)                      # lowest freq after prefilter
    edge_time = 6.0 / f_low                             # 3 cycles × 2× margin
    edge_samples = max(int(edge_time * fs), int(0.01 * fs))  # floor 10 ms
    edge_samples = min(edge_samples, n_if // 4)         # never trim more than 25% per side

    trim_start = edge_samples
    trim_end = edge_samples
    inst_freq = inst_freq[trim_start:-trim_end] if trim_end > 0 else inst_freq

    edge_time_offset = trim_start / fs  # seconds into original file

    f_mean = float(np.mean(inst_freq))
    deviation_frac = (inst_freq - f_mean) / f_mean
    del inst_freq

    # 7. Decimate — LP already serves as AA filter
    min_rate = max(2.0 * lp_cut, 1000.0)
    dec_factor = max(1, int(fs / min_rate))
    output_rate = float(fs / dec_factor)
    deviation_frac = deviation_frac[::dec_factor]

    t_uniform = np.arange(len(deviation_frac)) / output_rate + edge_time_offset

    # 7. Metrics — returns LP-filtered deviation used for unweighted measurement
    await _status("Computing metrics...")
    standard, non_standard, dev_unwtd = _compute_wf_metrics(deviation_frac, output_rate, f_est)

    # Plot data: MAD despike at native rate
    deviation_pct = _despike_plot(dev_unwtd * 100.0)

    # 8. Spectrum + peaks
    await _status("Computing spectrum...")
    spec_max_freq = lp_cut if lp_cut > 0 else 50.0
    spectrum = _compute_spectrum(deviation_pct, output_rate,
                                 max_freq=spec_max_freq)

    # RPM: use user-provided, or auto-detect from spectrum
    user_rpm = _state.get('_rpm')
    if user_rpm is not None:
        rpm = user_rpm
        rpm_info = {
            'value': float(user_rpm),
            'source': 'user',
            'confidence': 1.0,
            'f_rot_measured': round(user_rpm / 60.0, 4),
        }
    else:
        rpm_info = _detect_rpm(spectrum['freqs'], spectrum['amplitude'], duration)
        rpm = rpm_info['value']

    f_rot = rpm / 60.0 if rpm is not None else None
    _state['_rpm'] = rpm
    _state['_f_rot'] = f_rot

    # Motor harmonic labels (conditional on rpm)
    _identify_motor_harmonics(spectrum['peaks'], f_rot,
                               motor_slots=_state.get('_motor_slots'),
                               motor_poles=_state.get('_motor_poles'),
                               drive_ratio=_state.get('_drive_ratio', 1.0))

    # 9. AM/FM coupling — disabled pending rework
    _state['_am_envelope'] = None
    _state['_fm_deviation'] = deviation_pct
    spectrum['coupling_threshold'] = None
    _state['_coupling_threshold'] = None

    # Stash state for getPlotData
    _state['_deviation_pct'] = deviation_pct
    _state['_t_uniform'] = t_uniform
    _state['_output_rate'] = output_rate
    _state['_f_mean'] = f_mean
    _state['_f_rot'] = f_rot
    _state['_input_type'] = 'audio'

    # Available on-demand plots
    available = {
        'histogram': {},
        'harmonic_extract': {},
        'lissajous': {},
    }

    # Polar requires rpm (need rotation period to define one revolution)
    if f_rot is not None and f_rot > 0:
        sec_per_rev = 1.0 / f_rot
        samples_per_rev = int(round(sec_per_rev * output_rate))
        max_revolutions = len(deviation_pct) // samples_per_rev if samples_per_rev > 0 else 0
        available['polar'] = {'max_revolutions': max_revolutions}

    await _status("Complete")

    # Build fm_bw options: 200 Hz steps from AES min to below max
    fm_bw_max = int(carrier_max)
    fm_bw_options = list(range(200, fm_bw_max, 200)) if fm_bw_max > 200 else []

    return {
        'metrics': {
            'f_mean': f_mean,
            'carrier_freq': float(f_est),
            'fm_bw': {
                'value': float(lp_cut),
                'max': float(carrier_max),
                'options': fm_bw_options,
            },
            'rpm': rpm_info,
            'duration': float(duration),
            'input_type': 'audio',
            'device_format': None,
            'device_label': None,
            'standard': standard,
            'non_standard': non_standard,
        },
        'plots': {
            'dev_time': {
                't': t_uniform.tolist(),
                'deviation_pct': deviation_pct.tolist(),
            },
            'spectrum': spectrum,
        },
        'available': available,
    }


async def _analyze_device(text_data, rpm=None):
    """Device input pipeline — enters at deviation stage."""
    await _status("Detecting device format...")
    fmt = _detect_device_format(text_data)
    if fmt is None:
        raise ValueError(
            "Unable to detect device format from text data. "
            "Supported formats: shaknspin"
        )

    await _status(f"Parsing {fmt} data...")
    parsed = _parse_device_data(text_data, fmt)

    time_s = parsed['time_s']
    deviation_pct = parsed['deviation_pct']
    fs = parsed['sample_rate']
    f_mean = parsed['f_mean']
    duration = float(time_s[-1] - time_s[0])
    deviation_frac = deviation_pct / 100.0

    # Metrics
    await _status("Computing metrics...")
    # Device path has no carrier — use 500 Hz for weighting filter
    standard, non_standard, dev_unwtd = _compute_wf_metrics(deviation_frac, fs, carrier_freq=500.0)

    # Plot data: MAD despike at native rate
    deviation_pct = _despike_plot(dev_unwtd * 100.0)

    # Spectrum
    await _status("Computing spectrum...")
    spectrum = _compute_spectrum(deviation_pct, fs, max_freq=50.0)

    # RPM: use override if provided, otherwise detect from spectrum
    if rpm is None:
        rpm_info = _detect_rpm(spectrum['freqs'], spectrum['amplitude'], duration)
    else:
        rpm_info = {
            'value': float(rpm),
            'source': 'user',
            'confidence': 1.0,
            'f_rot_measured': float(rpm) / 60.0,
        }

    if rpm_info['value'] is not None:
        rpm = rpm_info['value']
        f_rot = rpm / 60.0
    else:
        rpm = None
        f_rot = None
    _state['_rpm'] = rpm

    # Motor harmonic labels
    _identify_motor_harmonics(spectrum['peaks'], f_rot,
                               motor_slots=_state.get('_motor_slots'),
                               motor_poles=_state.get('_motor_poles'),
                               drive_ratio=_state.get('_drive_ratio', 1.0))

    # No coupling for device input
    # spectrum peaks already have coupling_strength=None, am_coupled=False

    # Stash state for getPlotData
    _state['_deviation_pct'] = deviation_pct
    _state['_t_uniform'] = time_s
    _state['_output_rate'] = fs
    _state['_f_mean'] = f_mean
    _state['_f_rot'] = f_rot
    _state['_input_type'] = 'device'

    # Available on-demand plots
    available = {
        'histogram': {},
        'harmonic_extract': {},
        # No lissajous for device input
    }

    # Polar requires rpm (always available for device since we have RPM)
    if f_rot is not None and f_rot > 0:
        sec_per_rev = 1.0 / f_rot
        samples_per_rev = int(round(sec_per_rev * fs))
        max_revolutions = len(deviation_pct) // samples_per_rev if samples_per_rev > 0 else 0
        available['polar'] = {'max_revolutions': max_revolutions}

    await _status("Complete")

    return {
        'metrics': {
            'f_mean': None,
            'carrier_freq': 500.0,
            'rpm': rpm_info,
            'f_rot': float(f_rot) if f_rot else None,
            'duration': duration,
            'input_type': 'device',
            'device_format': fmt,
            'device_label': parsed.get('device_label'),
            'standard': standard,
            'non_standard': non_standard,
        },
        'plots': {
            'dev_time': {
                't': time_s.tolist(),
                'deviation_pct': deviation_pct.tolist(),
            },
            'spectrum': spectrum,
        },
        'available': available,
    }


# ========================= ON-DEMAND PLOT DATA =========================

def getPlotData(plotId, params=None):
    """
    Returns arrays for the requested on-demand plot.
    Requires analyzeFull to have been called first.

    Parameters:
        plotId: str — one of 'polar', 'histogram', 'harmonic_extract', 'lissajous'
        params: dict — plot-specific parameters

    Returns plot-specific dict.
    """
    if params is None:
        params = {}

    if not _state:
        raise RuntimeError("No analysis data. Call analyzeFull first.")

    if plotId == 'polar':
        return _plot_polar(params)
    elif plotId == 'histogram':
        return _plot_histogram(params)
    elif plotId == 'harmonic_extract':
        return _plot_harmonic_extract(params)
    elif plotId == 'lissajous':
        return _plot_lissajous(params)
    else:
        raise ValueError(f"Unknown plotId: {plotId}")


def _plot_polar(params):
    """
    Polar plot data. Starts at revolution 0 (no skip).

    params:
        revolutions: int (default 2)
        polar_lp: LP filter cutoff in Hz (default 60). Clamped to
                  Nyquist of the output sample rate.  Smooths the polar
                  trace for visual clarity without affecting metrics.
    """
    f_rot = _state.get('_f_rot')
    if f_rot is None or f_rot <= 0:
        raise ValueError("Polar plot requires rpm (rotation frequency unknown)")

    revolutions = params.get('revolutions', 2)
    deviation_pct = _state['_deviation_pct']
    output_rate = _state['_output_rate']
    f_mean = _state['_f_mean']

    # Polar LP — 3rd order Butterworth, filtfilt = 6-pole zero-phase
    polar_lp = float(params.get('polar_lp', 60))
    polar_lp = max(1.0, min(polar_lp, output_rate * 0.45))
    b_lp, a_lp = butter(3, polar_lp / (output_rate / 2.0), btype='low')
    deviation_pct = filtfilt(b_lp, a_lp, deviation_pct)

    sec_per_rev = 1.0 / f_rot
    samples_per_rev = int(round(sec_per_rev * output_rate))

    inst_freq = f_mean * (1.0 + deviation_pct / 100.0)

    # No skip — preconditioning handles filter artifacts
    start_idx = 0
    end_idx = min(revolutions * samples_per_rev, len(inst_freq))
    actual_revs = (end_idx - start_idx) // samples_per_rev

    # Theta: one revolution = 2π
    theta = np.linspace(0, 2 * np.pi, samples_per_rev, endpoint=False)

    revs_data = []
    for rev in range(actual_revs):
        idx_start = start_idx + rev * samples_per_rev
        idx_end = idx_start + samples_per_rev
        if idx_end > len(inst_freq):
            break
        revs_data.append({
            'angle': theta.tolist(),
            'radius': inst_freq[idx_start:idx_end].tolist(),
        })

    return {
        'revolutions': revs_data,
        'f_mean': f_mean,
        'n_revolutions': len(revs_data),
    }


def _plot_histogram(params):
    """Deviation histogram data."""
    deviation_pct = _state['_deviation_pct']
    counts, bin_edges = np.histogram(deviation_pct, bins=256, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    return {
        'bins': bin_centers.tolist(),
        'counts': counts.tolist(),
        'bin_edges': bin_edges.tolist(),
    }


def _plot_harmonic_extract(params):
    """
    Bandpassed deviation at each requested frequency.

    params:
        freqs: list of float (Hz)
    """
    freqs = params.get('freqs', [])
    if not freqs:
        return {'components': []}

    deviation_pct = _state['_deviation_pct']
    output_rate = _state['_output_rate']
    f_mean = _state['_f_mean']

    components = []
    for freq in freqs:
        nyquist = output_rate / 2.0
        if freq >= nyquist:
            components.append([])
            continue

        bw = min(f_mean * 0.8, max(0.3, freq * 0.05))
        lo = max(freq - bw / 2, 0.05)
        hi = min(freq + bw / 2, nyquist * 0.95)

        if hi <= lo:
            components.append([])
            continue

        sos = butter(4, [lo / nyquist, hi / nyquist],
                     btype='bandpass', output='sos')
        comp = sosfiltfilt(sos, deviation_pct)
        components.append(comp.tolist())

    return {'components': components}


def _plot_lissajous(params):
    """
    AM/FM Lissajous data at one frequency. Audio only.
    Uses sosfiltfilt Butterworth bandpass — same method as coupling markers.

    params:
        freq: float (Hz)
    """
    if _state.get('_input_type') != 'audio':
        raise ValueError("Lissajous requires audio input (AM envelope needed)")

    freq = params.get('freq')
    if freq is None:
        raise ValueError("freq parameter required for lissajous")

    am_full = _state.get('_am_full')
    fm_full = _state.get('_fm_full')
    fs = _state['_output_rate']

    if am_full is None or fm_full is None:
        raise RuntimeError("Coupling data not available. "
                          "Was analyzeFull called with audio input?")

    # sosfiltfilt Butterworth bandpass — matches coupling_strength()
    nyq = fs / 2.0
    bw = max(0.15, freq * 0.3)
    lo = max(0.2, freq - bw)
    hi = min(freq + bw, nyq * 0.9)
    if hi <= lo:
        raise ValueError(f"Bandpass range invalid for freq={freq}")

    sos = butter(4, [lo / nyq, hi / nyq], btype='band', output='sos')
    am_band = sosfiltfilt(sos, am_full)
    fm_band = sosfiltfilt(sos, fm_full)

    # Skip edges
    skip = int(3.0 * fs)
    if len(am_band) > 2 * skip:
        am_band = am_band[skip:-skip]
        fm_band = fm_band[skip:-skip]

    # Normalize to peak
    am_peak = np.max(np.abs(am_band)) if np.max(np.abs(am_band)) > 0 else 1.0
    fm_peak = np.max(np.abs(fm_band)) if np.max(np.abs(fm_band)) > 0 else 1.0
    am_norm = am_band / am_peak
    fm_norm = fm_band / fm_peak

    # Phase stats
    am_a = hilbert(am_band)
    fm_a = hilbert(fm_band)
    dphi = np.angle(fm_a) - np.angle(am_a)
    resultant = np.mean(np.exp(1j * dphi))
    R = float(np.abs(resultant))
    phase = float(np.degrees(np.angle(resultant)))

    am_rms = np.std(am_band)
    fm_rms = np.std(fm_band)
    strength = float(R * np.sqrt(am_rms * fm_rms))

    # Check significance against the threshold computed during analyzeFull
    coupling_threshold = _state.get('_coupling_threshold', None)
    significant = (coupling_threshold is not None and
                   strength > coupling_threshold)

    return {
        'am_norm': am_norm.tolist(),
        'fm_norm': fm_norm.tolist(),
        'R': R,
        'phase': phase,
        'strength': strength,
        'significant': significant,
    }
