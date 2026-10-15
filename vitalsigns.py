import numpy as np
import scipy.signal as signal

import capture_params
import spectral
import vital_bands

# Minimum peak-to-average power ratio in the band for a window to count.
# In the default 15 s heart window a clean sinusoid measures about 12 and pure
# noise about 4, so 6 separates them; longer windows score higher.
DEFAULT_CONF_THRESH = 6.0

class SimpleKalman:
    def __init__(self, q=1e-5, r=1e-2):
        self.q = q # process noise covariance
        self.r = r # measurement noise covariance
        self.x = 0 # estimated value
        self.p = 1 # estimated error covariance
        self.first = True

    def update(self, measurement):
        if self.first:
            self.x = measurement
            self.first = False
            return self.x
        self.p = self.p + self.q
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * self.p
        return self.x

    def filter(self, data):
        return np.array([self.update(z) for z in data])

def dc_offset_correction(iq_sig, method='mean'):
    """
    Removes DC offset from complex IQ signal.
    Methods: 'mean', 'moving_average'
    """
    if method == 'mean':
        return iq_sig - np.mean(iq_sig)
    elif method == 'moving_average':
        # Simple moving average over 1-2 seconds of data
        # Assuming fps is around 15-20, window size 32-64
        win_size = 64
        if len(iq_sig) <= win_size:
            return iq_sig - np.mean(iq_sig)
        
        # Real and Imag separately
        i = np.real(iq_sig)
        q = np.imag(iq_sig)
        
        # Using uniform filter for fast moving average
        from scipy.ndimage import uniform_filter1d
        i_ma = uniform_filter1d(i, size=win_size, mode='nearest')
        q_ma = uniform_filter1d(q, size=win_size, mode='nearest')
        
        return (i - i_ma) + 1j * (q - q_ma)
    else:
        return iq_sig

def extract_phase_dacm(iq_sig, wave_len):
    """
    Extracts phase using Differentiate and Cross Multiply (DACM).
    phi[n] = sum( (I[n]*dQ[n] - Q[n]*dI[n]) / (I[n]^2 + Q[n]^2) )
    """
    i = np.real(iq_sig)
    q = np.imag(iq_sig)
    
    di = np.diff(i)
    dq = np.diff(q)
    
    # Denominator
    den = i[:-1]**2 + q[:-1]**2 + 1e-9
    
    d_phi = (i[:-1] * dq - q[:-1] * di) / den
    
    # Smoothing d_phi to reduce noise from differentiation
    if len(d_phi) > 7:
        d_phi = signal.medfilt(d_phi, kernel_size=3)
    
    # Extended DACM usually involves handling the atan2 jump, 
    # but the derivative approach already handles small changes.
    # We sum it up to get the unwrapped phase.
    ph = np.zeros_like(i)
    ph[1:] = np.cumsum(d_phi)
    
    displacement = ph * (wave_len / (4 * np.pi))
    return displacement

def wiener_denoise(x):
    """
    Two passes of a short Wiener filter.

    This was called haar_denoise and the checkbox still said "DWT Denoising", but there
    is no wavelet anywhere in it - it is a Wiener filter, and the power-of-two padding
    it did was left over from a wavelet implementation that never arrived. Named for
    what it does so nobody reads the results as wavelet denoising.
    """
    return signal.wiener(signal.wiener(np.asarray(x, dtype=float), mysize=3), mysize=3)

# --- breaths counted one by one ------------------------------------------------------------
# A spectral peak measures a rhythm, not breaths: irregular breathing (3-9 s between breaths)
# spreads over the band and its peak falls on a harmonic or on the slowest gap. A seated
# person's minute on the AWR2243 cascade (2026-10-09): 9 breaths counted by the person, the
# spectrum said 6.8 /min (or 16.5, the second harmonic, in some range bins), the peaks of the
# chest displacement 9.
BREATH_MIN_SNR = 6.0         # a breath stands this many times the band's noise above its troughs
BREATH_REL = 0.2             # ... and this fraction of the signal's typical swing (5th-95th %):
                             # 0.3 missed a shallow breath of that minute, 0.1 counted 11 of 9


def breathing_noise_rms(disp, fps, resp_band, above_hz):
    """
    The RMS that the noise alone gives a breathing-band signal: the white floor of disp
    measured above above_hz (the heart band's top), spread over the breathing band's width.
    NaN when the frame rate leaves no room above above_hz.
    """
    x = np.asarray(disp, float)
    lo, hi = above_hz + 0.5, 0.8 * 0.5 * fps
    if x.size < 16 or hi - lo < 0.5:
        return float("nan")
    f, p = signal.welch(x - x.mean(), fps, nperseg=int(min(x.size, max(16, 4 * fps))))
    m = (f >= lo) & (f <= hi)
    if m.sum() < 3:
        return float("nan")
    return float(np.sqrt(np.median(p[m]) * (resp_band[1] - resp_band[0])))


def count_breaths(resp, fps, resp_band, noise_rms=None, swing=None):
    """
    Breaths as the peaks of a breathing-band displacement (inhalations), the way a person
    counts them: dict(n, times [s], intervals [s], rate [/min, from the mean interval], cv
    [interval spread], snr [median peak prominence / noise_rms], ok). A peak counts when it
    stands BREATH_REL of the typical swing and BREATH_MIN_SNR times noise_rms above its
    troughs, at least 0.8 of the fastest breath of the band after the one before (swing:
    the typical swing of a longer stretch, for counting in a short window of it). ok: two
    breaths or more, standing BREATH_MIN_SNR above the noise - band-passed noise alone has
    peaks too, but 1-3 times its own RMS high.
    """
    x = np.asarray(resp, float)
    out = dict(n=0, times=np.zeros(0), intervals=np.zeros(0), rate=float("nan"),
               cv=float("nan"), snr=float("nan"), ok=False)
    if x.size < 3 or not np.isfinite(x).all():
        return out
    if swing is None:
        swing = float(np.percentile(x, 95) - np.percentile(x, 5))
    prom = BREATH_REL * float(swing)
    has_noise = noise_rms is not None and np.isfinite(noise_rms) and noise_rms > 0
    if has_noise:
        prom = max(prom, BREATH_MIN_SNR * noise_rms)
    if prom <= 0:
        return out
    pk, props = signal.find_peaks(x, distance=max(1, int(0.8 * fps / resp_band[1])),
                                  prominence=prom)
    out["n"] = int(pk.size)
    out["times"] = pk / fps
    if pk.size:
        out["snr"] = (float(np.median(props["prominences"]) / noise_rms) if has_noise
                      else float("nan"))
    if pk.size >= 2:
        iv = np.diff(pk) / fps
        out["intervals"] = iv
        out["rate"] = float(60.0 / iv.mean())
        out["cv"] = float(iv.std() / iv.mean())
    out["ok"] = bool(pk.size >= 2 and (not has_noise or out["snr"] >= BREATH_MIN_SNR))
    return out


def breathing_rate_over(times, t0, t1, min_cover=0.5):
    """
    The breathing rate [/min] that breaths at `times` [s] give over the span [t0, t1):
    every interval between two breaths is one cycle, counted for the part of it inside
    the span. A pause inside counts as the slow breath it is, so three breaths bunched
    in a window that also holds a pause do not make it fast (the mean interval of the
    window's own peaks read 20 /min where its minute breathed 7). NaN when the intervals
    cover less than min_cover of the span.
    """
    ts = np.asarray(times, float)
    if ts.size < 2 or t1 <= t0:
        return float("nan")
    a, b = ts[:-1], ts[1:]
    inside = np.clip(np.minimum(b, t1) - np.maximum(a, t0), 0.0, None)
    cover = float(inside.sum())
    if cover < min_cover * (t1 - t0):
        return float("nan")
    return float(60.0 * (inside / (b - a)).sum() / cover)


def detect_damaged_frames(cube, min_run=64):
    """
    Frames the recorder filled with zeros because LVDS packets were lost.

    The DCA1000 substitutes zeros for packets it never received, and nothing downstream
    can tell those apart from a real reading - in the phase record they appear as a
    sudden jump and back.

    What marks a lost packet is a long UNBROKEN run of zeros, not the count of them. An
    earlier version simply required 5 % of a frame to be zero, which flagged every frame
    of a REAL capture: a real-valued ADC signal crosses zero constantly, and one such
    recording sat at 50 % exact zeros with no packet loss at all. A genuine signal never
    holds exactly zero for dozens of consecutive samples; a zero-filled packet is
    hundreds long.

    Returns a boolean array, one entry per frame.
    """
    arr = np.asarray(cube)
    if arr.ndim < 2 or arr.shape[0] == 0:
        return np.zeros(arr.shape[0] if arr.ndim else 0, dtype=bool)
    flat = arr.reshape(arr.shape[0], -1)
    zero = (flat == 0)
    if not zero.any():
        return np.zeros(flat.shape[0], dtype=bool)

    # A frame only needs checking when it holds at least min_run zeros in total.
    out = np.zeros(flat.shape[0], dtype=bool)
    candidates = np.flatnonzero(zero.sum(axis=1) >= min_run)
    for i in candidates:
        row = zero[i]
        # Run lengths from the gaps between non-zero samples.
        edges = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8), [0]))))
        if edges.size and np.max(edges[1::2] - edges[0::2]) >= min_run:
            out[i] = True
    return out


def damaged_spans(damaged):
    """Contiguous [start, end) runs of True, for shading or dashing a plot."""
    d = np.asarray(damaged, dtype=bool)
    if d.size == 0 or not d.any():
        return []
    edges = np.diff(np.concatenate(([0], d.view(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def get_peak_conf(sig, fs, f_lim, br_freq=None):
    """
    Dominant frequency in f_lim and how far it stands above the rest of the band.

    The confidence used to be peak_bin / sum_of_band. That is a share of the band
    energy, so it halves every time the FFT length doubles: a noiseless sinusoid scored
    0.192 at nfft=2048, 0.098 at 4096 and 0.049 at 8192 - all for the same signal. With
    the default threshold of 0.1 and nfft=4096 even a perfect sinusoid was rejected, so
    the sliding-window statistics on the Vital Signs tab never showed anything.

    It is now the peak-to-average power ratio over the band, which does not depend on
    the FFT length: about 12 for a clean sinusoid at any nfft, about 4 for pure noise.
    (ai_vital_analysis.find_best_peak already defines it this way.)
    """
    # nfft must be at least nperseg, or scipy raises - a 120 s window at 40 fps is
    # 4800 samples and used to abort the whole run.
    n = np.shape(sig)[-1]
    nfft = max(4096, int(1 << (n - 1).bit_length()))
    # Using Hann window for better spectral leakage control; several receivers (RX, n):
    # their spectra summed
    f, pxx = signal.welch(sig, fs=fs, nperseg=n, nfft=nfft, window='hann', axis=-1)
    if pxx.ndim > 1:
        pxx = pxx.reshape(-1, pxx.shape[-1]).sum(axis=0)
    mask = (f >= f_lim[0]) & (f <= f_lim[1])
    if not np.any(mask): 
        return 0.0, 0.0, f, pxx
    f_roi, pxx_roi = f[mask], pxx[mask].copy()
    
    # Suppress edges of the band to avoid picking filter artifacts or DC/Breathing residue
    # 0.02 Hz buffer on each side
    df = f[1] - f[0]
    n_buff = int(0.02 / df)
    if len(pxx_roi) > 2 * n_buff:
        pxx_roi[:n_buff] *= 0.1
        pxx_roi[-n_buff:] *= 0.1
    
    # Harmonic suppression if br_freq is provided
    if br_freq is not None:
        # Penalize frequencies near 2*br_freq and 3*br_freq
        for harm in [2, 3]:
            h_freq = harm * br_freq
            # Define a notch around harmonic
            harm_mask = (f_roi >= h_freq - 0.05) & (f_roi <= h_freq + 0.05)
            pxx_roi[harm_mask] *= 0.1 # Aggressive suppression
            
    # A rate is a peak: no lower than its neighbours on the WHOLE spectrum. At the low
    # edge of the band a spectrum still rising towards 0 Hz - drift, a movement - won
    # even through the edge penalty: on a 30 s take with a move in it, 5.1 /min came out
    # where the chest went 14 /min. The strongest true peak wins; argmax only if none.
    k_all = np.flatnonzero(mask)
    peak = ((pxx[k_all] >= pxx[np.maximum(k_all - 1, 0)])
            & (pxx[k_all] >= pxx[np.minimum(k_all + 1, len(pxx) - 1)]))
    idx = int(np.argmax(np.where(peak, pxx_roi, -np.inf))) if peak.any() else int(np.argmax(pxx_roi))
    conf = float(pxx_roi[idx] / (np.mean(pxx_roi) + 1e-30))
    # Interpolated on the spectrum as measured, not the penalised copy: the 0.1 steps
    # at the band edges and harmonic notches would drag the fitted vertex.
    f_peak, _ = spectral.refine_peak(f, pxx, int(np.flatnonzero(mask)[idx]))
    return f_peak, conf, f, pxx

def sliding(sig, fs, w_s, step_s, f_lim, br_t=None, br_v=None):
    """
    Peak frequency per sliding window.

    br_t / br_v: the breathing rate over time, used to suppress its harmonics inside the
    heart band. These must be matched by TIME. The heart windows are shorter than the
    breathing windows (15 s vs 30 s by default), so window i of one is not window i of
    the other - the old code indexed br_v[i] directly, which paired each heart window
    with a breathing estimate 7.5 s away and left the last 15 of 46 windows with no
    suppression at all, because the breathing series is shorter.
    """
    nw, ns = int(w_s * fs), max(1, int(step_s * fs))
    n_sig = np.shape(sig)[-1]                  # sig: (n,) or several receivers (RX, n)
    if n_sig < nw:
        # Fallback for short signals: use whatever we have
        nw = n_sig
        if nw < 10: return np.array([]), np.array([]), np.array([])
        
    ts, vs, cs = [], [], []
    range_end = n_sig - nw + 1
    if range_end <= 0: range_end = 1

    have_br = br_t is not None and br_v is not None and len(br_t) > 0 and len(br_v) > 0

    for start in range(0, range_end, ns):
        w_sig = signal.detrend(sig[..., start : start + nw], axis=-1)
        t_centre = (start + nw / 2) / fs

        curr_br = None
        if have_br:
            # Nearest breathing estimate in time; np.interp clamps at both ends, so the
            # windows beyond the breathing series keep using its last value.
            curr_br = float(np.interp(t_centre, br_t, br_v)) / 60.0

        f_p, c, _, _ = get_peak_conf(w_sig, fs, f_lim, br_freq=curr_br)
        ts.append(t_centre)
        vs.append(f_p * 60.0)
        cs.append(c)
    return np.array(ts), np.array(vs), np.array(cs)

def calc_stats(vs, cs, th):
    valid = cs >= th
    if not np.any(valid): 
        return None, None, None, None
    vv = vs[valid]
    return vv[-1], np.mean(vv), np.min(vv), np.max(vv)

def select_best_component(components, fs):
    """Select component with strongest HR peak in the 0.8 - 3.0 Hz range"""
    best_score = -1
    best_idx = 0
    for i in range(components.shape[1]):
        sig = components[:, i]
        f, pxx = signal.welch(sig, fs=fs, nperseg=len(sig))
        mask_hr = (f >= 0.8) & (f <= 3.0)
        if not np.any(mask_hr): continue
        hr_energy = np.sum(pxx[mask_hr])
        total_energy = np.sum(pxx) + 1e-9
        score = hr_energy / total_energy
        if score > best_score:
            best_score = score
            best_idx = i
    return components[:, best_idx]

def pca_fusion(signals, fs):
    """PCA method with smart component selection"""
    means = np.mean(signals, axis=0, keepdims=True)
    stds = np.std(signals, axis=0, keepdims=True) + 1e-9
    norm_sigs = (signals - means) / stds
    try:
        u, s, vh = np.linalg.svd(norm_sigs, full_matrices=False)
        comps = norm_sigs @ vh.T
        return select_best_component(comps, fs)
    except:
        return signals[:, 0]

def ica_fusion(signals, fs):
    """Basic ICA method with smart component selection"""
    # Fixed seed: FastICA starts from a random vector, so an unseeded run gave a
    # different heart rate every time the same recording was analysed.
    rng = np.random.default_rng(0)
    from scipy import linalg
    X = signals.T 
    X_mean = X.mean(axis=1, keepdims=True)
    X_centered = X - X_mean
    cov = np.cov(X_centered)
    try:
        d, E = linalg.eigh(cov)
        D_inv = np.diag(1.0 / np.sqrt(d + 1e-9))
        V = E @ D_inv @ E.T
        X_white = V @ X_centered
        
        def g(x): return np.tanh(x)
        def g_prime(x): return 1 - np.tanh(x)**2
        
        n_components = X.shape[0]
        W = np.zeros((n_components, n_components))
        for i in range(n_components):
            w = rng.random(n_components)
            for _ in range(30):
                w_new = (X_white * g(w @ X_white)).mean(axis=1) - g_prime(w @ X_white).mean() * w
                if i > 0: w_new -= (w_new @ W[:i].T) @ W[:i]
                w_new /= (np.linalg.norm(w_new) + 1e-9)
                if np.abs(np.abs(w @ w_new) - 1) < 1e-4: break
                w = w_new
            W[i, :] = w
        S = (W @ X_white).T
        return select_best_component(S, fs)
    except:
        return signals[:, 0]

def extract_phase_signal(complex_sig, wave_len, fps, auto_rx=True, rx_idx=0, dc_method='mean', phase_method='unwrap'):
    """
    Extracts phase from complex signal (F, RX) and converts to displacement.
    """
    if phase_method in ['pca', 'ica']:
        all_disps = []
        for i in range(complex_sig.shape[1]):
            s_rx = dc_offset_correction(complex_sig[:, i], method=dc_method)
            ph = np.unwrap(np.angle(s_rx))
            all_disps.append(ph * (wave_len / (4 * np.pi)))
        all_disps = np.array(all_disps).T
        if phase_method == 'pca':
            return pca_fusion(all_disps, fps), -1
        else:
            return ica_fusion(all_disps, fps), -1

    if auto_rx:
        best_rx = 0
        max_rx_e = -1
        for i in range(complex_sig.shape[1]):
            s_rx = complex_sig[:, i]
            # DC correction before energy check
            s_rx_corr = dc_offset_correction(s_rx, method=dc_method)
            # Simple energy check for RX selection
            _, pxx = signal.welch(np.abs(s_rx_corr), fs=fps, nperseg=min(len(s_rx_corr), 256))
            energy = np.sum(pxx)
            if energy > max_rx_e:
                max_rx_e = energy
                best_rx = i
        rx_idx = best_rx
    
    rx_idx = max(0, min(rx_idx, complex_sig.shape[1] - 1))
    s_t = complex_sig[:, rx_idx]
    
    # 1. DC Correction
    s_t = dc_offset_correction(s_t, method=dc_method)
    
    # 2. Phase Extraction
    if phase_method == 'dacm':
        displacement = extract_phase_dacm(s_t, wave_len)
    else: # unwrap
        ph = np.unwrap(np.angle(s_t))
        displacement = ph * (wave_len / (4 * np.pi))
        
    return displacement, rx_idx

DEFAULT_HR_WIN_S = 30.0      # heart windows: the radar's heart line needs more than 15 s
TRACK_STEP_S = 1.0           # the heart path's own step, at most

# The Vital Signs view's settings as it opens: DCA_processing takes its defaults from here,
# tools.vitals_check runs the chain with them. The bands are the shared ones (vital_bands).
DEFAULT_CONFIG = dict(auto_bin=True, bin_idx=0, roi_hw=1, roi_mode="Best bin in ROI",
                      auto_rx=True, rx_idx=0, dc_method="mean", phase_method="unwrap",
                      clutter_removal=False, kalman_filter=False, wiener_denoising=False,
                      win_br=30.0, win_hr=DEFAULT_HR_WIN_S, win_step=1.0,
                      conf_thresh=DEFAULT_CONF_THRESH)


def heart_rate_track(d, fps, bands, win_s, step_s, br_t=None, br_v=None):
    """
    (times, rates, confidences, grid, spectra) of the heart in displacement d, window
    by window: moving_hr.heart_spectra and heart_track (harmonic sum, Viterbi path),
    the harmonics of each window's breathing rate (br_t, br_v) notched out.

    The tracker's confidence (its path over the window's average, moving_hr.CONF_MIN
    for an estimate) is scaled so that DEFAULT_CONF_THRESH means the same: one
    Min peak/avg setting judges the breathing and the heart windows alike.
    """
    import moving_hr
    d = np.asarray(d, float)
    # The path is followed in steps of at most TRACK_STEP_S whatever step the caller
    # reports at: the step sets how far the rate may move between windows (3 bpm/s x
    # step), and at 6 s steps the path lost the heart - 13-15 instead of 1-3 bpm off on
    # two recordings. Coarser steps are read off the fine path.
    fine = min(float(step_s), TRACK_STEP_S)
    wins = moving_hr._windows(len(d), fps, win_s, fine)
    tc = np.array([(a + b) / 2.0 / fps for a, b in wins])
    br = np.interp(tc, br_t, br_v) if br_t is not None and len(br_t) else np.full(tc.size, np.nan)
    grid, spec = moving_hr.heart_spectra(d - d.mean(), fps, bands, wins, br_bpm=br)
    hr, conf, _ = moving_hr.heart_track(spec, grid, fine)
    conf = conf * (DEFAULT_CONF_THRESH / moving_hr.CONF_MIN)
    hr = np.asarray(hr, float)
    if fine < step_s and len(wins) > 1:
        keep = [0]
        for i in range(1, len(tc)):
            if tc[i] - tc[keep[-1]] >= step_s - 1e-9:
                keep.append(i)
        keep = np.asarray(keep)
        tc, hr, conf, spec = tc[keep], hr[keep], conf[keep], spec[keep]
    return tc, hr, conf, grid, spec


HEART_BINS_HW = 1            # the heart from the chosen bin and this many bins either side
ARC_SNR_MIN = 1.5            # a channel's arc over its noise; noise alone gives 1.00 +- 0.02
#                              (at 3 a weak person's neighbours fell out: 15 instead of 11 bpm)


def arc_snr(z):
    """
    How far a channel's samples z stay from their centre, over its noise: 1 for noise
    alone, 6-180 for a seated chest. The noise from the top 30 % of the band, where a
    chest has nothing (the median of the periodogram there, robust to a stray line); a
    difference of neighbouring samples took the chest's own motion for noise at 20 fps.
    """
    z = np.asarray(z, complex)
    if z.size < 16:
        return 0.0
    zc = z - z.mean()
    p = np.abs(np.fft.fft(zc)) ** 2 / z.size              # E = the noise power per sample
    hi = np.abs(np.fft.fftfreq(z.size)) > 0.35
    noise = np.sqrt(np.median(p[hi]) / np.log(2.0))        # the median of exponential power
    return float(np.median(np.abs(zc)) / (np.sqrt(np.log(2.0)) * noise + 1e-30))


def heart_channels(sig, centre, wave_len, half=HEART_BINS_HW, dc_method='mean', rxs=None):
    """
    (channels, F): the centred, unwrapped displacement of the receivers rxs (all: None)
    in the bins centre-half .. centre+half of sig (F, RX, R) - the centre's always, a
    neighbour's where its arc stands above its noise (arc_snr >= ARC_SNR_MIN). Their mean
    is the heart's signal: the chest spans more than one range bin, and the average over
    the receivers and the neighbouring bins was 1.2 bpm off a pulse oximeter on five
    seated recordings, the receivers of one bin 1.7 (combining their complex signals
    first, or their spectra, was worse).
    """
    F, RX, R = sig.shape
    rxs = range(RX) if rxs is None else rxs
    out = []
    for b in range(max(0, centre - half), min(R, centre + half + 1)):
        for i in rxs:
            z = sig[:, i, b]
            if b != centre and arc_snr(z) < ARC_SNR_MIN:
                continue
            out.append(signal.detrend(np.unwrap(np.angle(dc_offset_correction(z, method=dc_method))))
                       * (wave_len / (4 * np.pi)))
    return np.array(out)


AUTO_BIN_WINDOW_S = 10.0     # Auto Bin: breathing energy judged window by window
AUTO_BIN_MIN_RANGE_M = 0.3   # nearer is the TX-RX coupling, not a chest


def auto_bin(rfft_cube, fps, f_br_lim, r_axis=None, window_s=AUTO_BIN_WINDOW_S,
             min_range_m=AUTO_BIN_MIN_RANGE_M, bands=None, damaged=None):
    """
    The range bin of the breathing person.

    With `bands` and a range axis: the bin whose slow-time signal holds the most
    breathing-band power above its own noise, summed over the receivers - Fastcheck's
    choice (fastcheck.breathing_bin). Against the pulse oximeter of a somnograph on six
    seated recordings it found the chest where the breathing-band energy of the
    signal's magnitude took a reflection 0.1-0.2 m off it.

    Otherwise, or when no bin breathes clearly above its noise: the bin where the
    breathing-band energy of the magnitude PERSISTS - the band energy in windows of
    `window_s` (half overlapping), median over the windows. The energy of the whole
    recording let a 2-second movement beat a person breathing for 20 s at another range.
    Bins nearer than min_range_m (the antenna coupling) are never chosen; without a
    range axis the first two bins are skipped.
    """
    if bands is not None and r_axis is not None and len(r_axis) == rfft_cube.shape[-1]:
        import fastcheck
        if rfft_cube.shape[0] / fps >= fastcheck.MIN_VITAL_S and fps > 2 * bands.resp_hz[1]:
            best, _, _ = fastcheck.breathing_bin(np.mean(rfft_cube, axis=1), fps, bands,
                                                 np.asarray(r_axis), damaged, min_range_m)
            if best is not None:
                return int(best)
    sig = np.mean(rfft_cube, axis=(1, 2))                    # (F, R)
    F, R = sig.shape
    n = max(8, int(round(window_s * fps)))
    if F < 2 * n:
        n, starts = F, [0]                                    # too short to split: all of it
    else:
        starts = list(range(0, F - n + 1, max(1, n // 2)))
    energies = []
    for a in starts:
        s = sig[a:a + n] - sig[a:a + n].mean(axis=0)
        f_a, pxx = signal.welch(np.abs(s), fs=fps, nperseg=min(len(s), 256), axis=0)
        band = (f_a >= f_br_lim[0]) & (f_a <= f_br_lim[1])
        energies.append(pxx[band].sum(axis=0))
    e = np.median(np.array(energies), axis=0)
    first = 2
    if r_axis is not None and len(r_axis) == R and np.any(np.asarray(r_axis) >= min_range_m):
        first = int(np.argmax(np.asarray(r_axis) >= min_range_m))
    if first >= R:
        first = min(2, R - 1)
    return int(first + np.argmax(e[first:]))


def process_vital_signs_logic(
    rfft_cube, 
    params, 
    fps, 
    r_axis,
    config
):
    """
    Core logic for vital signs processing.
    
    config: dict with keys:
        - auto_bin (bool)
        - bin_idx (int)
        - roi_hw (int)
        - roi_mode (str)
        - auto_rx (bool)
        - rx_idx (int)
        - clutter_removal (bool)
        - kalman_filter (bool)
        - wiener_denoising (bool)
        - win_br (float)
        - win_hr (float)
        - win_step (float)
        - conf_thresh (float)
        - resp_bpm, heart_bpm ((low, high), optional): the search bands; default are
          the shared ones in vital_bands.py
        - damaged_frames (bool array, optional)
    """
    F, C, RX, R = rfft_cube.shape
    bands = vital_bands.from_config(config)
    f_br_lim, f_hr_lim = bands.resp_hz, bands.heart_hz

    # 1. Range Bin Selection (if auto)
    if config.get('auto_bin', False):
        # the lost-frame mask stays presentation only (test_losses_do_not_change...)
        best_bin = auto_bin(rfft_cube, fps, f_br_lim, r_axis, bands=bands)
    else:
        best_bin = config.get('bin_idx', 0)
    
    best_bin = max(0, min(best_bin, R - 1))
    dist_m = r_axis[best_bin]

    # 2. ROI Definition
    roi_hw = config.get('roi_hw', 1)
    roi_bins = np.arange(max(0, best_bin - roi_hw), min(R, best_bin + roi_hw + 1))
    r_start, r_end = r_axis[roi_bins[0]], r_axis[roi_bins[-1]]

    # 3. Wave Length: in the middle of the ADC window (capture_params.wavelength_m,
    # roadmap 1.9), the same in every view; 77 GHz only when the log has no frequency.
    wave_len = capture_params.wavelength_m(params, default_ghz=77.0)
    
    # 3b. Clutter Removal (optional, usually done before calling this but keeping config)
    if config.get('clutter_removal', False):
         rfft_cube = rfft_cube - np.mean(rfft_cube, axis=0, keepdims=True)

    roi_mode = config.get('roi_mode', "Best bin in ROI")
    auto_rx = config.get('auto_rx', True)
    rx_idx = config.get('rx_idx', 0)
    dc_method = config.get('dc_method', 'mean')
    phase_method = config.get('phase_method', 'unwrap')
    
    actual_best_rx = rx_idx

    # 4. Phase Extraction based on ROI mode
    bin_used = None             # the one bin the phase comes from (not for the weighted ROI)
    if roi_mode == "Single bin":
        bin_used = best_bin
        sig_complex = np.mean(rfft_cube[:, :, :, best_bin], axis=1) # (F, RX)
        displacement, actual_best_rx = extract_phase_signal(sig_complex, wave_len, fps, auto_rx, rx_idx, dc_method, phase_method)
    elif roi_mode == "Best bin in ROI":
        # the neighbour that breathes most above its noise (Auto Bin's measure), else the
        # chosen bin itself. The total power of the magnitude moved a well chosen bin onto
        # a neighbour with more drift: 11 instead of 1 bpm off the pulse on one recording.
        best_roi_idx = int(np.flatnonzero(roi_bins == best_bin)[0])
        if rfft_cube.shape[0] >= 8 * fps and fps > 2 * f_br_lim[1]:
            import fastcheck
            k, _, _ = fastcheck.breathing_bin(np.mean(rfft_cube[:, :, :, roi_bins], axis=1), fps,
                                              bands, np.asarray(r_axis)[roi_bins],
                                              None, min_range_m=0.0,
                                              nyquist_last=False)
            if k is not None:
                best_roi_idx = int(k)
        actual_best_bin = roi_bins[best_roi_idx]
        bin_used = actual_best_bin
        sig_complex = np.mean(rfft_cube[:, :, :, actual_best_bin], axis=1)
        displacement, actual_best_rx = extract_phase_signal(sig_complex, wave_len, fps, auto_rx, rx_idx, dc_method, phase_method)
    else: # Weighted ROI average
        disps = []
        weights = []
        rx_indices = []
        for b in roi_bins:
            sig_b = np.mean(rfft_cube[:, :, :, b], axis=1)
            d_b, a_rx = extract_phase_signal(sig_b, wave_len, fps, auto_rx, rx_idx, dc_method, phase_method)
            disps.append(d_b)
            # Weight by signal amplitude after DC correction
            sig_b_corr = np.mean([dc_offset_correction(sig_b[:, i], method=dc_method) for i in range(sig_b.shape[1])], axis=0)
            weights.append(np.mean(np.abs(sig_b_corr)))
            rx_indices.append(a_rx)
        weights = np.array(weights)
        weights /= (np.sum(weights) + 1e-9)
        displacement = np.sum([d*w for d,w in zip(disps, weights)], axis=0)
        actual_best_rx = int(np.round(np.mean(rx_indices))) # Heuristic

    disp_detrend = signal.detrend(displacement)
    disp_filtered = disp_detrend.copy()

    if config.get('kalman_filter', False):
        kf = SimpleKalman(q=1e-4, r=1e-2)
        disp_filtered = kf.filter(disp_detrend)

    if config.get('wiener_denoising', False):
        disp_filtered = wiener_denoise(disp_filtered)

    # 4b. Every receiver's own displacement at that bin, unwrapped one by one: the
    # breathing from their spectra summed. The heart from the mean of theirs and of the
    # neighbouring bins' receivers (heart_channels) - "Single bin": that bin alone.
    rx_disps = None
    heart_disps = None
    if bin_used is not None and phase_method == 'unwrap':
        rxs = (list(range(rfft_cube.shape[2])) if auto_rx
               else [max(0, min(rx_idx, rfft_cube.shape[2] - 1))])

        def _post(d_i):
            if config.get('kalman_filter', False):
                d_i = SimpleKalman(q=1e-4, r=1e-2).filter(d_i)
            if config.get('wiener_denoising', False):
                d_i = wiener_denoise(d_i)
            return d_i
        rx_disps = np.array([_post(d) for d in heart_channels(
            np.mean(rfft_cube[:, :, :, bin_used:bin_used + 1], axis=1), 0, wave_len, half=0,
            dc_method=dc_method, rxs=rxs)])                                    # (RX, F)
        hw = config.get('heart_bins_hw', HEART_BINS_HW) if roi_mode == "Best bin in ROI" else 0
        if hw > 0:
            lo, hi = max(0, bin_used - hw), min(R, bin_used + hw + 1)
            heart_disps = np.array([_post(d) for d in heart_channels(
                np.mean(rfft_cube[:, :, :, lo:hi], axis=1), bin_used - lo, wave_len, half=hw,
                dc_method=dc_method, rxs=rxs)])
        else:
            heart_disps = rx_disps

    # 5. Filtering for BR/HR, in the bands chosen above (vital_bands.py). A narrower
    # heart band - e.g. the resting preset, 42-102 bpm - keeps a breathing harmonic or a
    # movement artefact at the top of the band from being picked as the heart rate.
    # Second-order sections, not transfer-function coefficients. Breathing is a very low
    # frequency relative to the sampling rate, and butter(..., [b, a]) loses the poles to
    # round-off as that ratio grows: at 100 fps the 0.083-0.667 Hz bandpass comes back
    # with poles at |z| = 1.0049 - outside the unit circle, so the filter is unstable -
    # and a 4 mm breathing sine came out 2.7x too big; at 200 fps the output reached
    # 1e90. A 100 fps vital-signs capture is entirely normal, so this was not a corner
    # case. SOS gives the right amplitude at every rate tested.
    sos_br = signal.butter(4, [f_br_lim[0], f_br_lim[1]], 'bandpass', fs=fps, output='sos')
    sig_br = signal.sosfiltfilt(sos_br, disp_filtered)
    d_heart = heart_disps.mean(axis=0) if heart_disps is not None else disp_filtered
    with_hr = fps > 2 * f_hr_lim[1]
    sig_hr = (signal.sosfiltfilt(
        signal.butter(4, [f_hr_lim[0], f_hr_lim[1]], 'bandpass', fs=fps, output='sos'), d_heart)
        if with_hr else np.zeros_like(d_heart))

    # 6. Sliding Window Analysis
    w_br = config.get('win_br', 30.0)
    w_hr = config.get('win_hr', DEFAULT_HR_WIN_S)
    step = config.get('win_step', 1.0)

    # breathing: the receivers' spectra summed, window by window - one receiver alone
    # missed the rate where its own phase was the noisy one
    br_in = signal.sosfiltfilt(sos_br, rx_disps, axis=-1) if rx_disps is not None else sig_br
    rr_t, rr_v, rr_c = sliding(br_in, fps, w_br, step, f_br_lim)

    # heart: followed as the phase-12 pipeline does it (moving_hr) - the heart band and
    # the envelope of its 5-25 Hz content, the breathing harmonics notched, a harmonic
    # sum and a Viterbi path through the windows - on the receivers' mean displacement.
    # The largest peak of each 15 s window was 8 bpm off a pulse oximeter on average and
    # passed the confidence bar in fewer than half the windows.
    if with_hr:
        hr_t, hr_v, hr_c, hr_grid, hr_spec = heart_rate_track(d_heart, fps, bands, w_hr, step,
                                                             rr_t, rr_v)
    else:
        hr_t, hr_v, hr_c, hr_grid, hr_spec = np.array([]), np.array([]), np.array([]), None, None

    # 7. Statistics
    conf_th = config.get('conf_thresh', DEFAULT_CONF_THRESH)
    rr_curr, rr_mean, rr_min, rr_max = calc_stats(rr_v, rr_c, conf_th)
    hr_curr, hr_mean, hr_min, hr_max = calc_stats(hr_v, hr_c, conf_th)

    # 8. Range Profile
    rp_avg = np.mean(np.abs(rfft_cube), axis=(0, 1, 2))
    rp_db = 20 * np.log10(rp_avg + 1e-9)

    # 9. Spectral plots (last window): the breathing spectrum the rate came from (summed
    # over the receivers), and the heart's harmonic sum where the track ended
    n_br_win = int(w_br * fps)
    f_p_br, c_br, f_ax_br, psd_br = get_peak_conf(
        signal.detrend(br_in[..., -n_br_win:] if br_in.shape[-1] >= n_br_win else br_in, axis=-1),
        fps, f_br_lim)
    if hr_spec is not None and len(hr_v):
        f_p_hr, f_ax_hr, psd_hr = hr_v[-1] / 60.0, hr_grid / 60.0, hr_spec[-1]
    else:
        f_p_hr, f_ax_hr, psd_hr = float("nan"), np.array([f_hr_lim[0], f_hr_lim[1]]), np.zeros(2)

    # Which frames were zero-filled by the recorder. The GUI passes the mask measured on
    # the raw cube; failing that, a fully zero-filled frame survives the range FFT as
    # exact zeros, so it can still be spotted here.
    damaged = config.get('damaged_frames')
    if damaged is None:
        damaged = detect_damaged_frames(rfft_cube)
    damaged = np.asarray(damaged, dtype=bool)
    if damaged.size != len(displacement):
        damaged = np.zeros(len(displacement), dtype=bool)

    results = {
        'damaged_frames': damaged,
        'damaged_fraction': float(damaged.mean()) if damaged.size else 0.0,
        'fps': fps,
        'time': np.arange(len(displacement)) / fps,
        'phase': disp_detrend,
        'phase_filtered': disp_filtered if (config.get('kalman_filter') or config.get('wiener_denoising')) else None,
        'sig_br': sig_br,
        'sig_hr': sig_hr,
        'rr_times': rr_t, 'rr_values': rr_v, 'rr_confs': rr_c,
        'hr_times': hr_t, 'hr_values': hr_v, 'hr_confs': hr_c,
        'f_axis_br': f_ax_br, 'psd_br': psd_br, 'f_axis_hr': f_ax_hr, 'psd_hr': psd_hr,
        'br_bpm': f_p_br * 60.0,
        'hr_bpm': f_p_hr * 60.0,
        'rr_stats': (rr_curr, rr_mean, rr_min, rr_max),
        'hr_stats': (hr_curr, hr_mean, hr_min, hr_max),
        'r_axis': r_axis,
        'rp_db': rp_db,
        'best_bin': best_bin,
        'dist_m': dist_m,
        'roi_dist_range': (r_start, r_end),
        'roi_bins': roi_bins,
        'f_resp_range': f_br_lim,
        'f_heart_range': f_hr_lim,
        'conf_th': conf_th,
        'actual_best_rx': actual_best_rx,
        'bin_used': bin_used,
        'heart_channels': 0 if heart_disps is None else len(heart_disps)
    }
    
    return results
