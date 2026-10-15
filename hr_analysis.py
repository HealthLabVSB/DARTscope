import copy
import numpy as np
import scipy.signal as signal
import scipy.linalg as linalg
import json
import csv
from typing import Dict, List, Any, Optional

import spectral
import vital_bands

# Minimum peak-to-average power ratio in a band for an estimate to count as valid.
# Over a 15 s window a clean sinusoid measures about 12 and pure noise about 4;
# the figure does not move with the frame rate, and rises with window length.
MIN_PEAK_TO_AVG = 6.0


def band_peak_index(f: np.ndarray, pxx: np.ndarray, mask: np.ndarray) -> int:
    """
    Index (into f) of the strongest TRUE peak inside the band mask - no lower than its
    neighbours on the whole spectrum - or of the band's maximum when it holds none. The
    plain maximum took the band's edge where the spectrum only rose towards a stronger
    line outside it (breathing below the heart band, drift below the breathing band).
    """
    k = np.flatnonzero(mask)
    peak = (pxx[k] >= pxx[np.maximum(k - 1, 0)]) & (pxx[k] >= pxx[np.minimum(k + 1, len(pxx) - 1)])
    if peak.any():
        return int(k[peak][np.argmax(pxx[k][peak])])
    return int(k[np.argmax(pxx[k])])


class HRAnalysisLogic:
    def __init__(self):
        self.default_filters = {
            "respiration_branch": {
                "enabled": True,
                "type": "bandpass",
                "low_hz": 0.1,
                "high_hz": 0.6,
                "order": 4,
                "mode": "zero_phase"
            },
            "cardiac_hr_branch": {
                "enabled": True,
                "type": "bandpass",
                "low_hz": 0.8,
                "high_hz": 3.0,
                "order": 4,
                "mode": "zero_phase"
            },
            "cardiac_waveform_branch": {
                "enabled": True,
                "type": "bandpass",
                "low_hz": 0.8,
                "high_hz": 8.0,
                "order": 4,
                "mode": "zero_phase"
            },
            "notch_resp_harmonics": {
                "enabled": False,
                "frequencies_hz": [],
                "bandwidth_hz": 0.05
            },
            "smoothing": {
                "enabled": False,
                "type": "moving_average",
                "window_sec": 3.0
            },
            "tracker": {
                "enabled": True,
                "type": "rate_limited",
                "max_bpm_change_per_sec": 3.0
            }
        }
        self._clamp_warned = set()
        # The breathing and heart bands above are replaced by the shared ones.
        self.apply_bands(vital_bands.current(), defaults=True)
        # deepcopy: set_filter_params() does filters[name].update(...), which with a
        # shallow copy writes straight through into default_filters - after one change
        # the "defaults" were no longer the defaults.
        self.filters = copy.deepcopy(self.default_filters)

    def apply_bands(self, bands, defaults: bool = False):
        """
        Set the breathing and heart branches (and the waveform branch's low edge) from
        the shared bands (vital_bands.py), keeping order, mode and everything else.
        defaults=True writes them into default_filters, as at start-up.
        """
        target = self.default_filters if defaults else self.filters
        target["respiration_branch"].update(low_hz=bands.resp_hz[0], high_hz=bands.resp_hz[1])
        target["cardiac_hr_branch"].update(low_hz=bands.heart_hz[0], high_hz=bands.heart_hz[1])
        target["cardiac_waveform_branch"]["low_hz"] = bands.heart_hz[0]
        self.resp_bpm_valid = tuple(bands.resp_bpm)

    def get_filter_params(self, name: str) -> Dict:
        return self.filters.get(name, {})

    def set_filter_params(self, name: str, params: Dict):
        if name in self.filters:
            self.filters[name].update(params)

    def apply_filter(self, sig: np.ndarray, fs: float, filter_cfg: Dict) -> np.ndarray:
        """
        Apply one configured filter branch.

        Cut-offs are clamped below Nyquist. The cardiac waveform branch reaches 8 Hz,
        which is not representable below 16 frames per second - and 15 fps is this
        application's own fallback frame rate. butter() then raised "critical
        frequencies must be 0 < Wn < 1", the exception was swallowed, and the RAW,
        UNFILTERED displacement was returned and analysed as if it had been filtered.
        A silently unfiltered signal is dominated by breathing, so the heart-rate
        estimate taken from it was meaningless without anything saying so.
        """
        if not filter_cfg.get("enabled", True):
            return sig
        
        f_type = filter_cfg.get("type", "bandpass")
        order = filter_cfg.get("order", 4)
        mode = filter_cfg.get("mode", "zero_phase")
        nyq = fs / 2.0
        if nyq <= 0:
            return sig

        def _clamp(hz, name):
            """Keep a cut-off strictly inside (0, Nyquist), and say when it had to move."""
            limit = 0.9 * nyq
            if hz >= limit:
                key = (name, round(hz, 3), round(fs, 3))
                if key not in self._clamp_warned:
                    self._clamp_warned.add(key)
                    print(f"[HR] {name} cut-off {hz:g} Hz is above what {fs:g} fps can "
                          f"represent; limited to {limit:.2f} Hz. Record faster to use "
                          f"the full band.")
                return limit
            return max(hz, 1e-4)

        try:
            if f_type == "bandpass":
                low = _clamp(filter_cfg["low_hz"], "band low") / nyq
                high = _clamp(filter_cfg["high_hz"], "band high") / nyq
                if low >= high:
                    print(f"[HR] Band {filter_cfg['low_hz']:g}-{filter_cfg['high_hz']:g} Hz "
                          f"does not fit below Nyquist at {fs:g} fps; signal left unfiltered.")
                    return sig
                sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
            elif f_type == "lowpass":
                cutoff = _clamp(filter_cfg["high_hz"], "lowpass") / nyq
                sos = signal.butter(order, cutoff, btype='low', output='sos')
            elif f_type == "highpass":
                cutoff = _clamp(filter_cfg["low_hz"], "highpass") / nyq
                sos = signal.butter(order, cutoff, btype='high', output='sos')
            elif f_type == "notch":
                freq = _clamp(filter_cfg["freq_hz"], "notch")
                bw = filter_cfg.get("bandwidth_hz", 0.05)
                sos = signal.tf2sos(*signal.iirnotch(freq / nyq, freq / bw))
            else:
                return sig

            # Second-order sections throughout: the respiration branch is 0.1-0.6 Hz,
            # which against a 100 fps sampling rate is where transfer-function
            # coefficients lose their poles to round-off and the filter goes unstable.
            if mode == "zero_phase":
                return signal.sosfiltfilt(sos, sig)
            else:
                return signal.sosfilt(sos, sig)
        except Exception as e:
            print(f"Filter error: {e}")
            return sig

    def static_clutter_removal(self, rfft_cube: np.ndarray) -> np.ndarray:
        """Subtract mean across frames to remove static reflections"""
        return rfft_cube - np.mean(rfft_cube, axis=0, keepdims=True)

    def extract_phase_baseline(self, complex_sig: np.ndarray) -> np.ndarray:
        """4.1 Arctangent + phase unwrap baseline"""
        phase = np.angle(complex_sig)
        unwrapped_phase = np.unwrap(phase)
        return unwrapped_phase

    def extract_phase_circle_fitting(self, complex_sig: np.ndarray) -> np.ndarray:
        """4.2 Circle fitting (Pratt algorithm) + unwrap"""
        x = np.real(complex_sig)
        y = np.imag(complex_sig)
        
        # Centroid
        mx = np.mean(x)
        my = np.mean(y)
        xi = x - mx
        yi = y - my
        zi = xi**2 + yi**2
        
        # Moments
        m_z = np.mean(zi)
        m_xz = np.mean(xi * zi)
        m_yz = np.mean(yi * zi)
        m_zz = np.mean(zi**2)
        
        # Matrix for Eigenproblem
        M = np.array([
            [m_zz, m_xz, m_yz, m_z],
            [m_xz, np.mean(xi**2), np.mean(xi * yi), 0],
            [m_yz, np.mean(xi * yi), np.mean(yi**2), 0],
            [m_z,  0, 0, 1]
        ])
        
        B = np.array([
            [0, 0, 0, -2],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-2, 0, 0, 0]
        ])
        
        try:
            # Solve generalized eigenproblem M * a = lambda * B * a
            vals, vecs = linalg.eig(M, B)
            # Smallest positive eigenvalue. Select it by index rather than by comparing
            # the complex eigenvalue array against a real scalar: that equality only
            # matched when the imaginary part was exactly 0.0, so the tiniest rounding
            # raised IndexError and the blanket except below silently returned the plain
            # centroid phase - while the GUI still reported "circle fitting".
            real_vals = np.real(vals)
            positive = np.flatnonzero(np.isfinite(real_vals) & (real_vals > 0))
            if positive.size == 0:
                print("[hr_analysis] Circle fit found no positive eigenvalue; "
                      "falling back to centroid phase.")
                return np.unwrap(np.angle(xi + 1j*yi))

            idx = positive[np.argmin(real_vals[positive])]
            v = np.real(vecs[:, idx])
            if abs(v[0]) < 1e-12:
                print("[hr_analysis] Circle fit is degenerate (radius term ~ 0); "
                      "falling back to centroid phase.")
                return np.unwrap(np.angle(xi + 1j*yi))

            # Center (a, b)
            a_fit = -v[1] / (2 * v[0]) + mx
            b_fit = -v[2] / (2 * v[0]) + my

            corrected_sig = (x - a_fit) + 1j * (y - b_fit)
            return np.unwrap(np.angle(corrected_sig))
        except Exception as e:
            print(f"[hr_analysis] Circle fit failed ({type(e).__name__}: {e}); "
                  "falling back to centroid phase.")
            return np.unwrap(np.angle(xi + 1j*yi))

    def extract_phase_dacm(self, complex_sig: np.ndarray) -> np.ndarray:
        """4.3 DACM / differential phase metoda"""
        i = np.real(complex_sig)
        q = np.imag(complex_sig)
        di = np.diff(i, prepend=i[0])
        dq = np.diff(q, prepend=q[0])
        den = i**2 + q**2 + 1e-9
        d_phi = (i * dq - q * di) / den
        return np.cumsum(d_phi)

    def estimate_hr(self, sig: np.ndarray, fs: float, method_name: str, source: str) -> Dict:
        """Helper to estimate HR from a filtered signal using multiple methods"""
        nfft = max(2048, len(sig))
        f, pxx = signal.welch(sig, fs=fs, nperseg=len(sig), nfft=nfft)
        
        hr_cfg = self.filters["cardiac_hr_branch"]
        mask = (f >= hr_cfg["low_hz"]) & (f <= hr_cfg["high_hz"])
        
        if not np.any(mask):
            return self._empty_result(method_name, source)
        
        f_roi = f[mask]
        pxx_roi = pxx[mask]
        
        # 1. FFT Peak Method (the strongest true peak of the band)
        k_best = band_peak_index(f, pxx, mask)
        best_idx = int(np.flatnonzero(np.flatnonzero(mask) == k_best)[0])
        dom_freq_fft, _ = spectral.refine_peak(f, pxx, k_best)
        hr_fft = dom_freq_fft * 60.0
        
        # 2. Autocorrelation Method
        # Detrend and normalize
        sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-9)
        # FFT-based autocorrelation (O(n log n)) - equivalent to the non-negative-lag
        # half of np.correlate(sig_norm, sig_norm, mode='full'), which is O(n^2).
        n_sig = sig_norm.size
        n_fft_ac = 1 << (2 * n_sig - 1).bit_length()
        spec_ac = np.fft.rfft(sig_norm, n=n_fft_ac)
        autocorr = np.fft.irfft(spec_ac * np.conj(spec_ac), n=n_fft_ac)[:n_sig]
        
        # Search range for lags
        min_lag = int(fs / hr_cfg["high_hz"])
        max_lag = int(fs / hr_cfg["low_hz"])
        
        if len(autocorr) > max_lag:
            ac_roi = autocorr[min_lag:max_lag]
            if len(ac_roi) > 0:
                k = int(np.argmax(ac_roi))
                lag = k + min_lag
                # Interpolate the peak. At 20 fps the integer lags around 72 bpm are
                # 4-5 bpm apart (lag 16 = 75, 17 = 70.6, 18 = 66.7), so without this the
                # autocorrelation estimate is quantised far more coarsely than the
                # 10 bpm consensus check that compares against it.
                if 0 < k < len(ac_roi) - 1:
                    y0, y1, y2 = ac_roi[k - 1], ac_roi[k], ac_roi[k + 1]
                    denom = y0 - 2 * y1 + y2
                    if abs(denom) > 1e-30:
                        lag = lag + 0.5 * (y0 - y2) / denom
                hr_ac = 60.0 * fs / max(lag, 1e-9)
            else:
                hr_ac = hr_fft
        else:
            hr_ac = hr_fft

        # 3. Peak-to-Peak (Time Domain)
        # find_peaks with only a distance limit counts every noise wiggle between beats,
        # which on a clean 72 bpm test signal reported 112. Requiring the peak to stand
        # out from the waveform, and taking the median interval rather than the mean,
        # makes it robust to an extra or a missed beat.
        peaks, _ = signal.find_peaks(sig, distance=max(1, int(fs / hr_cfg["high_hz"])),
                                     prominence=0.5 * np.std(sig))
        if len(peaks) >= 3:
            # Between the samples: at 20 fps a 96 bpm beat is 12.5 samples long, and
            # integer intervals read it as 100 or 92.3 bpm; 140 bpm came out 6.7 low.
            pos = peaks + np.array([spectral.peak_offset(sig, k, "parabolic") for k in peaks])
            intervals = np.diff(pos) / fs
            hr_p2p = 60.0 / max(float(np.median(intervals)), 1e-9)
        else:
            hr_p2p = hr_fft

        # Consensus HR
        # For now, let's use FFT as primary but check consistency
        hr_bpm = hr_fft
        hr_methods = {"fft": hr_fft, "autocorr": hr_ac, "p2p": hr_p2p}
        
        # Confidence/Quality score
        in_band_energy = np.sum(pxx_roi)
        # estimate_hr is always handed a signal that has already been bandpassed to this
        # very band, so "in-band energy / total energy" was ~1 whatever the input: it
        # measured 0.9998 on a clean recording and 0.982 on pure noise, and the
        # quality > 0.4 test could never fail. Both numbers are now peak-to-average
        # power in the band, which does not depend on the FFT length and does tell the
        # two apart (about 12 for a clean pulse in a 15 s window, about 4 for noise).
        band_mean = np.mean(pxx_roi) + 1e-30
        confidence = pxx_roi[best_idx] / band_mean
        quality = confidence
        
        # Validity check - consensus within 10 BPM
        is_consistent = abs(hr_fft - hr_ac) < 10 or abs(hr_fft - hr_p2p) < 10
        valid = (quality >= MIN_PEAK_TO_AVG) and is_consistent
        
        return {
            "method": method_name,
            "source": source,
            "hr_bpm": float(hr_bpm),
            "dominant_freq_hz": float(dom_freq_fft),
            "confidence": float(confidence),
            "quality": float(quality),
            "valid": bool(valid),
            "hr_methods": hr_methods,
            "note": "" if is_consistent else "Low consensus between methods",
            "debug": {"pxx": pxx.tolist(), "f": f.tolist()}
        }

    def _empty_result(self, method: str, source: str) -> Dict:
        return {
            "method": method,
            "source": source,
            "hr_bpm": 0,
            "dominant_freq_hz": 0,
            "confidence": 0,
            "quality": 0,
            "valid": False,
            "debug": {}
        }

    def select_best_component(self, components: np.ndarray, fs: float) -> np.ndarray:
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

    def pca_fusion(self, signals: np.ndarray, fs: float) -> np.ndarray:
        """4.5 PCA method with smart component selection"""
        means = np.mean(signals, axis=0, keepdims=True)
        stds = np.std(signals, axis=0, keepdims=True) + 1e-9
        norm_sigs = (signals - means) / stds
        
        u, s, vh = np.linalg.svd(norm_sigs, full_matrices=False)
        comps = norm_sigs @ vh.T
        return self.select_best_component(comps, fs)

    def ica_fusion(self, signals: np.ndarray, fs: float) -> np.ndarray:
        """4.6 ICA method with smart component selection"""
        # Fixed seed: FastICA starts from a random vector, so an unseeded run gave a
        # different heart rate every time the same recording was analysed.
        rng = np.random.default_rng(0)
        X = signals.T 
        X_mean = X.mean(axis=1, keepdims=True)
        X_centered = X - X_mean
        
        cov = np.cov(X_centered)
        try:
            d, E = linalg.eigh(cov)
        except:
            return signals[:, 0] # Fallback
            
        D_inv = np.diag(1.0 / np.sqrt(d + 1e-9))
        V = E @ D_inv @ E.T
        X_white = V @ X_centered
        
        def g(x): return np.tanh(x)
        def g_prime(x): return 1 - np.tanh(x)**2
        
        n_components = X.shape[0]
        W = np.zeros((n_components, n_components))
        
        for i in range(n_components):
            w = rng.random(n_components)
            for _ in range(50): # Fewer iterations for speed
                w_new = (X_white * g(w @ X_white)).mean(axis=1) - g_prime(w @ X_white).mean() * w
                if i > 0:
                    w_new -= (w_new @ W[:i].T) @ W[:i]
                w_new /= (np.linalg.norm(w_new) + 1e-9)
                if np.abs(np.abs(w @ w_new) - 1) < 1e-4:
                    break
                w = w_new
            W[i, :] = w
            
        S = (W @ X_white).T
        return self.select_best_component(S, fs)

    @staticmethod
    def _ricker(points: int, a: float) -> np.ndarray:
        """Ricker (Mexican hat) wavelet, L2-normalised - scipy removed signal.ricker."""
        t = np.arange(points) - (points - 1.0) / 2.0
        return (2.0 / (np.sqrt(3.0 * a) * np.pi ** 0.25)) * \
               (1.0 - (t / a) ** 2) * np.exp(-t ** 2 / (2.0 * a ** 2))

    def wavelet_hr_estimate(self, sig: np.ndarray, fs: float) -> Dict:
        """
        4.4 Wavelet method using a Ricker filter bank.

        Two things were wrong here. It called signal.cwt and signal.ricker, which SciPy
        removed in 1.15 (this project runs 1.17), so the method raised AttributeError
        the moment anything called it. And the scale for a target frequency was
        0.25*fs/f, while the Ricker wavelet's spectral peak actually sits at
        sqrt(2)/(2*pi*a) = 0.2251/a - measured, not quoted - an 11 % error that would
        have reported a 72 bpm pulse as about 80.
        """
        RICKER_PEAK = np.sqrt(2.0) / (2.0 * np.pi)      # 0.22508, measured
        freqs_target = np.linspace(0.6, 4.0, 120)       # ~1.7 bpm grid
        scales = (RICKER_PEAK * fs) / freqs_target

        x = np.asarray(sig, dtype=float)
        avg_power = np.empty(len(scales))
        for i, a in enumerate(scales):
            n = min(len(x), int(10 * a) | 1)             # odd length, ~10 scales wide
            if n < 3:
                avg_power[i] = 0.0
                continue
            conv = np.convolve(x, self._ricker(n, a), mode='same')
            # Divide by the scale: with an L2-normalised wavelet the response to a
            # sinusoid still grows as sqrt(a), which on its own biases the answer
            # towards low frequencies (a 96 bpm pulse came out at 86). This makes the
            # response comparable across scales.
            avg_power[i] = float(np.mean(conv ** 2) / a)

        dom_freq = float(freqs_target[int(np.argmax(avg_power))])

        res = self.estimate_hr(sig, fs, "Wavelet", "source")
        res["hr_bpm"] = float(dom_freq * 60.0)
        res["dominant_freq_hz"] = dom_freq
        return res

    def compute_quality_metrics(self, sig: np.ndarray, fs: float, hr_bpm: float, respiration_sig: Optional[np.ndarray] = None) -> Dict:
        """9. Quality score calculation with enhanced metrics"""
        nfft = max(2048, len(sig))
        f, pxx = signal.welch(sig, fs=fs, nperseg=len(sig), nfft=nfft)
        
        hr_freq = hr_bpm / 60.0
        bw = 0.1
        mask_heart = (f >= hr_freq - bw) & (f <= hr_freq + bw)
        # Respiratory band 0.1 - 0.6 Hz
        mask_resp = (f >= 0.1) & (f <= 0.6)
        # Heart band 0.8 - 3.0 Hz
        mask_heart_band = (f >= 0.8) & (f <= 3.0)
        # Noise band (everything else within meaningful range)
        mask_noise = (f >= 0.1) & (f <= 5.0) & (~mask_heart) & (~mask_resp)
        
        heart_energy = np.sum(pxx[mask_heart])
        noise_energy = np.sum(pxx[mask_noise]) + 1e-9
        resp_energy = np.sum(pxx[mask_resp]) + 1e-9
        snr = 10 * np.log10(heart_energy / noise_energy)
        
        # Peak dominance: ratio of highest peak to second highest in heart band
        pxx_heart = pxx[mask_heart_band]
        f_heart = f[mask_heart_band]
        if len(pxx_heart) > 0:
            peaks, props = signal.find_peaks(pxx_heart, height=np.max(pxx_heart)*0.1)
            if len(peaks) >= 2:
                sorted_heights = np.sort(props['peak_heights'])
                peak_dominance = sorted_heights[-1] / (sorted_heights[-2] + 1e-9)
            elif len(peaks) == 1:
                peak_dominance = 2.0 # Good enough
            else:
                peak_dominance = 1.0
        else:
            peak_dominance = 1.0

        # Harmonic penalty
        harmonic_penalty = 1.0
        if respiration_sig is not None:
            br_data = self.estimate_breathing_rate(respiration_sig, fs)
            br_hz = br_data["br_hz"]
            for h in [2, 3, 4]:
                if abs(hr_freq - h*br_hz) < 0.1:
                    harmonic_penalty = 0.5 # Possible harmonic interference
                    break

        # The signal reaching this function is already bandpassed to the heart band, so
        # in-band/total is ~1 for anything: it is reported for information but no longer
        # carries weight in the score.
        in_band = np.sum(pxx[mask_heart_band])
        total = np.sum(pxx) + 1e-9
        ratio = in_band / total
        
        # Stability - how smooth the waveform is, sample to sample, relative to its own
        # amplitude. This used to be 1/std(diff(sig)) on a displacement in metres, so it
        # came out around 1e4 and min(1, stability/100) was pinned at 1.0 for every
        # recording; feeding the same signal in millimetres instead changed the final
        # score from 0.90 to 0.73. A ratio of two quantities in the same unit cannot do
        # that.
        stability = float(np.std(sig) / (np.std(np.diff(sig)) + 1e-30))
        
        motion_penalty = 0.0
        if np.max(np.abs(np.diff(sig))) > np.std(sig) * 5:
            motion_penalty = 0.5
            
        # Combine metrics into final score
        # Weights add up to 1.0. The stability term is normalised against the number of
        # samples per heart beat: a clean sine sampled n times per period has
        # std(sig)/std(diff) of about n/(2*pi), so the reference is the ideal value for
        # this frame rate and rate rather than an arbitrary constant.
        samples_per_beat = fs / max(hr_freq, 1e-6)
        stability_ref = max(1.0, samples_per_beat / (2 * np.pi))
        quality = (min(1.0, max(0.0, snr) / 20.0) * 0.35 +
                   min(1.0, (peak_dominance - 1) / 2) * 0.25 +
                   min(1.0, stability / stability_ref) * 0.15 +
                   (1 - motion_penalty) * 0.15 +
                   harmonic_penalty * 0.10)
        
        return {
            "heart_snr": float(snr),
            "peak_dominance": float(peak_dominance),
            "in_out_ratio": float(ratio),
            "stability": float(stability),
            "motion_penalty": float(motion_penalty),
            "harmonic_penalty": float(harmonic_penalty),
            "quality_score": float(max(0.0, min(1.0, quality)))
        }

    def estimate_breathing_rate(self, sig: np.ndarray, fs: float) -> Dict:
        """Estimate breathing rate from filtered signal"""
        nfft = max(2048, len(sig))
        f, pxx = signal.welch(sig, fs=fs, nperseg=len(sig), nfft=nfft)
        
        br_cfg = self.filters["respiration_branch"]
        mask = (f >= br_cfg["low_hz"]) & (f <= br_cfg["high_hz"])
        
        if not np.any(mask):
            return {
                "br_bpm": 0.0,
                "br_hz": 0.0,
                "confidence": 0.0,
                "valid": False
            }
        
        f_roi = f[mask]
        pxx_roi = pxx[mask]
        k_best = band_peak_index(f, pxx, mask)
        best_idx = int(np.flatnonzero(np.flatnonzero(mask) == k_best)[0])
        dom_freq, _ = spectral.refine_peak(f, pxx, k_best)
        br_bpm = dom_freq * 60.0
        
        # Peak-to-average, not a share of the band energy: the old metric scaled with
        # the FFT length, so the SAME 15 bpm breathing signal was "valid" at 20 fps
        # (0.364) and "invalid" at 15 and 10 fps (0.292, 0.194) against a fixed 0.3 bar.
        confidence = float(pxx_roi[best_idx] / (np.mean(pxx_roi) + 1e-30))
        
        # Validity: inside the breathing band (vital_bands.py)
        lo, hi = getattr(self, "resp_bpm_valid", vital_bands.current().resp_bpm)
        valid = (lo <= br_bpm <= hi) and (confidence >= MIN_PEAK_TO_AVG)
        
        return {
            "br_bpm": float(br_bpm),
            "br_hz": float(dom_freq),
            "confidence": float(confidence),
            "valid": bool(valid)
        }

    def detect_artifacts(self, sig: np.ndarray) -> Dict:
        """10. Detekce artefaktů"""
        diff = np.abs(np.diff(sig))
        jumps = diff > np.std(sig) * 3
        return {
            "artifact_detected": bool(np.any(jumps)),
            "jump_count": int(np.sum(jumps)),
            "is_valid": bool(not np.any(jumps))
        }

    def vmd_hr_estimate(self, sig: np.ndarray, fs: float) -> Dict:
        """4.7 Variational Mode Decomposition (Simplified)"""
        # Basic VMD inspired implementation
        # For radar HR, we typically look for 3 modes: Respiratory, Cardiac, Noise
        alpha = 2000       # bandwidth constraint
        tau = 0            # noise-tolerance
        K = 3              # number of modes
        DC = 0             # no DC part
        init = 1           # initialize omegas uniformly
        tol = 1e-7
        
        T = len(sig)
        f_hat = np.fft.fft(sig)
        freqs = np.fft.fftfreq(T)
        
        # Initialization
        u_hat = np.zeros((K, T), dtype=complex)
        omega = np.zeros((K,))
        if init == 1:
            for k in range(K):
                omega[k] = 0.5/K * (k)
                
        # Main loop
        for _ in range(100):
            for k in range(K):
                # Update u_hat
                sum_u = np.sum(u_hat, axis=0) - u_hat[k,:]
                res = f_hat - sum_u
                u_hat[k,:] = res / (1 + alpha * (freqs - omega[k])**2)
                
                # Update omega
                idx = np.arange(T)
                u_sq = np.abs(u_hat[k,:])**2
                omega[k] = np.sum(freqs * u_sq) / np.sum(u_sq)
                
        # Inverse FFT to get modes
        u = np.real(np.fft.ifft(u_hat, axis=1))
        
        # Identify cardiac mode (usually the one with freq in 0.8-3.0 Hz range)
        cardiac_mode_idx = 0
        best_dist = 1e9
        target_center = 1.5 / (fs/2) # normalized
        for k in range(K):
            dist = np.abs(omega[k]*fs - 1.5)
            if dist < best_dist:
                best_dist = dist
                cardiac_mode_idx = k
                
        cardiac_sig = u[cardiac_mode_idx, :]
        return self.estimate_hr(cardiac_sig, fs, "VMD", "source")

    def sliding_window_analysis(self, sig: np.ndarray, fs: float, win_sec: float, overlap_pct: float, method_func, source: str) -> Dict:
        """Analyze HR in sliding windows and return trend and statistics"""
        win_len = int(win_sec * fs)
        step = int(win_len * (1 - overlap_pct / 100.0))
        if step < 1: step = 1
        
        results = []
        for start in range(0, len(sig) - win_len + 1, step):
            window = sig[start : start + win_len]
            t_center = (start + win_len / 2) / fs
            # Pass only the method name if needed, or wrap method_func
            res = method_func(window, fs, "Sliding", source)
            res["time_center"] = t_center
            results.append(res)
            
        if not results:
            return {"trend": [], "stats": {}}
            
        valid_hrs = [r["hr_bpm"] for r in results if r["valid"]]
        stats = {}
        if valid_hrs:
            stats = {
                "mean": float(np.mean(valid_hrs)),
                "median": float(np.median(valid_hrs)),
                "min": float(np.min(valid_hrs)),
                "max": float(np.max(valid_hrs)),
                "std": float(np.std(valid_hrs)),
                "count_valid": len(valid_hrs),
                "total_count": len(results)
            }
        
        return {
            "trend": results,
            "stats": stats
        }
    def generate_synthetic_data(self, fs: float, duration_sec: float) -> tuple:
        """
        Synthetic rfft_cube for testing: a chest at bin 10 (1.6 m) seen by four receivers,
        each with its own heart rate - RX1 60, RX2 80, RX3 100 bpm, RX4 none - over the same
        breathing (15 /min), with a static reflection in the bin and the antenna coupling
        at bin 0. Real sizes at 77 GHz: 2 mm of breathing and 0.2 mm of heart are 6.4 and
        0.6 rad of phase. 0.1 m of breathing (320 rad, 25 rad from one frame to the next)
        was aliased beyond any unwrapping: RX3 came out at 65 bpm.
        """
        rng = np.random.default_rng(0)
        n_samples = int(fs * duration_sec)
        t = np.arange(n_samples) / fs

        # 4 RX, 1 chirp per frame (simplified), 64 bins
        n_rx = 4
        n_chirps = 1
        n_bins = 64
        target_bin = 10
        lam = 3e8 / 77e9

        # HR frequencies for RX1, RX2, RX3, RX4: 60, 80, 100, noise
        hr_bpms = [60, 80, 100, 0]

        rfft_cube = np.zeros((n_samples, n_chirps, n_rx, n_bins), dtype=complex)
        br_disp = 2e-3 * np.sin(2 * np.pi * 0.25 * t)
        for i in range(n_rx):
            if hr_bpms[i] > 0:
                disp = br_disp + 0.2e-3 * np.sin(2 * np.pi * hr_bpms[i] / 60.0 * t)
            else:
                disp = br_disp + 0.05e-3 * rng.standard_normal(n_samples)     # RX4: no heart
            chest = np.exp(1j * (4 * np.pi * disp / lam + rng.uniform(0, 2 * np.pi)))
            chest *= 1.0 + 0.05 * rng.standard_normal(n_samples)              # amplitude noise
            rfft_cube[:, 0, i, target_bin] = chest + 3.0 * np.exp(1j * rng.uniform(0, 2 * np.pi))
            rfft_cube[:, 0, i, 0] = 30.0                                      # the coupling at 0 m

        # Add noise to all bins
        rfft_cube += 0.01 * (rng.standard_normal(rfft_cube.shape) + 1j * rng.standard_normal(rfft_cube.shape))

        params = {'startFreq_GHz': 77.0}
        r_axis = np.linspace(0, 10, n_bins)
        rp = np.zeros((n_bins, n_samples)) # dummy
        
        return rfft_cube, params, fs, r_axis, rp
