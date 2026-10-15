import numpy as np
import scipy.signal as signal
from scipy import linalg
from typing import Dict, List, Any, Optional, Tuple
import json
import csv
import time
import os
from datetime import datetime

import spectral
import vital_bands

import importlib.util

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
PCA = FastICA = KMeans = GaussianMixture = IsolationForest = StandardScaler = None


def _load_sklearn() -> bool:
    """scikit-learn, imported when a method first needs it: with pandas behind it the
    import took 4 s of DARTscope's start, and the default rule-based scoring uses none."""
    global PCA, FastICA, KMeans, GaussianMixture, IsolationForest, StandardScaler, SKLEARN_AVAILABLE
    if PCA is None and SKLEARN_AVAILABLE:
        try:
            from sklearn.decomposition import PCA, FastICA
            from sklearn.cluster import KMeans
            from sklearn.mixture import GaussianMixture
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            SKLEARN_AVAILABLE = False
    return SKLEARN_AVAILABLE

RR_MIN_PEAK_TO_AVG = 6.0    # a breathing window counts above this (vitalsigns.DEFAULT_CONF_THRESH)


class AIVitalAnalysis:
    def __init__(self):
        self.last_results = {}
        self.feature_names = [
            "hr_snr", "rr_snr", "phase_stability", "rx_coherence",
            "motion_score", "spectral_entropy", "resp_harmonic_overlap",
            "amp_stability", "phase_drift", "hr_peak_hz", "rr_peak_hz"
        ]
        self.reference_data = None # Dictionary with 'time', 'signal', 'hr', 'fs', etc.
        self.pulse_data = None
        self.reference_sync = {}
        self.c = 299792458

    @staticmethod
    def parse_reference_timestamp(value):
        """Parse the absolute timestamp used by the PPG/Pulse exports."""
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        # The export uses milliseconds, while ``%f`` accepts microseconds.
        text = text.replace(',', '.')
        for fmt in ("%d.%m.%Y %H:%M:%S.%f", "%d.%m.%Y %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt).timestamp()
            except ValueError:
                continue
        return None

    def find_best_peak(self, freqs: np.ndarray, pxx: np.ndarray, band: Tuple[float, float]) -> Tuple[float, float, float]:
        """
        Finds the best peak in the given frequency band.
        Avoids picking the edge of the band if a valid internal peak exists.
        Returns (peak_hz, bpm, snr).
        """
        idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
        if len(idx) == 0:
            return 0.0, 0.0, 0.0
        
        # The local peaks in the band, judged on the WHOLE spectrum: find_peaks on the
        # band alone never returns its first bin, so a breathing peak right at the bottom
        # of the band (6 /min of a 5-40 /min band) went missing and a higher one was taken
        # (10-32 /min where the chest and the reference went 6)
        is_peak = (pxx[idx] >= pxx[np.maximum(idx - 1, 0)]) & (pxx[idx] >= pxx[np.minimum(idx + 1, len(pxx) - 1)])
        peaks = np.flatnonzero(is_peak)
        
        if len(peaks) > 0:
            # Pick the highest peak among local peaks
            k = idx[peaks[np.argmax(pxx[idx[peaks]])]]
        else:
            # Fallback to argmax if no local peaks found (e.g. monotonic)
            k = idx[np.argmax(pxx[idx])]
        # Between the bins rather than on one: a Welch bin at 100 fps and 20 s is
        # 3 bpm wide (see spectral.py for how much the fit recovers).
        peak_f, _ = spectral.refine_peak(freqs, pxx, k)
        peak_val = pxx[k]
        
        # Calculate SNR (Peak vs Mean of the band); the guard far below any spectrum: 1e-9
        # was the size of a heart line's spectrum in metres and halved its ratio
        snr = peak_val / (np.mean(pxx[idx]) + 1e-30)
        return float(peak_f), float(peak_f * 60), float(snr)

    def compute_weighted_median(self, values: np.ndarray, weights: np.ndarray) -> float:
        """Computes weighted median of values."""
        if len(values) == 0:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        
        # Sort values and weights
        sort_idx = np.argsort(values)
        sorted_values = values[sort_idx]
        sorted_weights = weights[sort_idx]
        
        cumsum_weights = np.cumsum(sorted_weights)
        total_weight = np.sum(sorted_weights)
        if total_weight == 0: return float(np.median(values))
        
        cutoff = total_weight / 2.0
        median_idx = np.where(cumsum_weights >= cutoff)[0][0]
        return float(sorted_values[median_idx])

    def dc_offset_correction(self, iq, method="mean"):
        """Odstranění DC složky z IQ signálu."""
        iq = np.nan_to_num(iq)
        if method == "mean":
            return iq - np.mean(iq)
        elif method == "moving_average":
            # Okno cca 2 sekundy pro sledování driftu
            window = max(11, len(iq) // 10)
            if window % 2 == 0: window += 1
            if len(iq) < window: return iq - np.mean(iq)
            real_dc = signal.medfilt(np.real(iq), window)
            imag_dc = signal.medfilt(np.imag(iq), window)
            return iq - (real_dc + 1j * imag_dc)
        return iq

    def extract_phase(self, iq, method="unwrap"):
        """Extrakce fáze z IQ signálu."""
        if method == "unwrap":
            return np.unwrap(np.angle(iq))
        elif method == "dacm":
            # Extended DACM algorithm
            I = np.real(iq)
            Q = np.imag(iq)
            dI = np.diff(I, prepend=I[0])
            dQ = np.diff(Q, prepend=Q[0])
            denom = I**2 + Q**2
            denom[denom == 0] = 1e-9
            dphi = (I * dQ - Q * dI) / denom
            return np.cumsum(dphi)
        elif method == "circlefit":
            # Real Pratt algebraic circle fit, shared with the HR Analysis tab.
            # This used to be plain mean-centering despite being labelled "circlefit",
            # so the two tabs applied different processing under the same method name
            # and their results were not comparable.
            return self._phase_circle_fit(iq)
        return np.unwrap(np.angle(iq))

    @staticmethod
    def _phase_circle_fit(iq):
        """
        Pratt algebraic circle fit of the IQ trajectory, then unwrapped phase about the
        fitted centre. Falls back to the centroid (and says so) when the fit degenerates.
        """
        x = np.real(iq).astype(float)
        y = np.imag(iq).astype(float)
        mx, my = np.mean(x), np.mean(y)
        xi, yi = x - mx, y - my
        zi = xi**2 + yi**2

        M = np.array([
            [np.mean(zi**2), np.mean(xi * zi), np.mean(yi * zi), np.mean(zi)],
            [np.mean(xi * zi), np.mean(xi**2), np.mean(xi * yi), 0.0],
            [np.mean(yi * zi), np.mean(xi * yi), np.mean(yi**2), 0.0],
            [np.mean(zi), 0.0, 0.0, 1.0],
        ])
        B = np.array([
            [0.0, 0.0, 0.0, -2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-2.0, 0.0, 0.0, 0.0],
        ])

        try:
            vals, vecs = linalg.eig(M, B)
            real_vals = np.real(vals)
            positive = np.flatnonzero(np.isfinite(real_vals) & (real_vals > 0))
            if positive.size == 0:
                raise ValueError("no positive eigenvalue")
            v = np.real(vecs[:, positive[np.argmin(real_vals[positive])]])
            if abs(v[0]) < 1e-12:
                raise ValueError("degenerate fit")
            a_fit = -v[1] / (2 * v[0]) + mx
            b_fit = -v[2] / (2 * v[0]) + my
            return np.unwrap(np.angle((x - a_fit) + 1j * (y - b_fit)))
        except Exception as e:
            print(f"[ai_vital] Circle fit failed ({e}); falling back to centroid phase.")
            return np.unwrap(np.angle(xi + 1j * yi))

    def phase_to_displacement(self, phase, wavelength):
        """Převod fáze na posun v metrech."""
        return (phase * wavelength) / (4 * np.pi)

    def safe_bandpass(self, sig, fs, low, high, order=4):
        """Robustní bandpass filtr s kontrolou Nyquista."""
        if fs <= 0 or len(sig) < 3 * order:
            return np.nan_to_num(sig)
        
        nyq = 0.5 * fs
        low_f = max(0.01, low)
        high_f = min(high, 0.9 * nyq)
        
        if low_f >= high_f:
            return np.nan_to_num(sig)
        
        try:
            # SOS, not (b, a): at 100 fps a 0.1-0.6 Hz bandpass built as transfer-function
            # coefficients has poles outside the unit circle and blows up.
            sos = signal.butter(order, [low_f, high_f], btype='bandpass', fs=fs, output='sos')
            filtered = signal.sosfiltfilt(sos, sig)
            return np.nan_to_num(filtered)
        except Exception:
            return np.nan_to_num(sig)

    def compute_features(self, iq_sig: np.ndarray, fs: float,
                         resp_band=None, hr_band=None,
                         pulse_band=None, neighboring_iq=None,
                         phase_method="unwrap") -> Dict[str, float]:
        """Extrakce příznaků pro jeden IQ signál. Pásma v Hz, výchozí z vital_bands."""
        bands = vital_bands.current()
        resp_band = resp_band or bands.resp_hz
        hr_band = hr_band or bands.heart_hz
        pulse_band = pulse_band or (bands.heart_hz[0], max(8.0, bands.heart_hz[1]))
        if len(iq_sig) < 2 * fs:
             return {name: 0.0 for name in self.feature_names}
        
        iq_corr = self.dc_offset_correction(iq_sig, method="mean")
        phase = self.extract_phase(iq_corr, method=phase_method)
        phase = np.nan_to_num(phase)
        
        nfft = max(1024, len(phase))
        nperseg = min(len(phase), 512)
        freqs, pxx = signal.welch(phase, fs, nperseg=nperseg, nfft=nfft)
        
        features = {}
        
        # RR SNR
        rr_f, rr_bpm, rr_snr = self.find_best_peak(freqs, pxx, resp_band)
        features['rr_peak_hz'] = rr_f
        features['rr_snr'] = 10 * np.log10(rr_snr) if rr_snr > 0 else -20

        # HR SNR with harmonic penalty
        idx_hr = np.where((freqs >= hr_band[0]) & (freqs <= hr_band[1]))[0]
        if len(idx_hr) > 0:
            rr_hz = features.get('rr_peak_hz', 0)
            harmonics = [2*rr_hz, 3*rr_hz, 4*rr_hz]
            pxx_clean = pxx.copy()
            for h in harmonics:
                if hr_band[0]-0.1 <= h <= hr_band[1]+0.1:
                    h_idx = np.argmin(np.abs(freqs - h))
                    w = max(1, int(0.05 / (freqs[1]-freqs[0])))
                    pxx_clean[max(0, h_idx-w):min(len(pxx), h_idx+w)] *= 0.1

            hr_f, hr_bpm, hr_snr_val = self.find_best_peak(freqs, pxx_clean, hr_band)
            features['hr_peak_hz'] = hr_f
            
            # Re-calculate SNR based on cleaned PSD
            idx_hr_clean = np.where((freqs >= hr_band[0]) & (freqs <= hr_band[1]))[0]
            hr_power = pxx_clean[np.argmin(np.abs(freqs - hr_f))]
            noise_p = (np.sum(pxx_clean[idx_hr_clean]) - hr_power) / (len(idx_hr_clean)-1 + 1e-9)
            features['hr_snr'] = 10 * np.log10(hr_power / (noise_p + 1e-9))
        else:
            features['hr_snr'] = -20
            features['hr_peak_hz'] = 0

        features['phase_stability'] = 1.0 / (np.var(np.diff(phase)) + 1e-6)
        diff_phase = np.abs(np.diff(phase))
        features['motion_score'] = np.sum(diff_phase > np.std(diff_phase) * 4) / len(diff_phase)
        psd_norm = pxx / (np.sum(pxx) + 1e-9)
        features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-9))
        amp = np.abs(iq_sig)
        features['amp_stability'] = np.mean(amp) / (np.std(amp) + 1e-9)
        features['phase_drift'] = np.abs(phase[-1] - phase[0]) / len(phase)

        rr_hz = features.get('rr_peak_hz', 0)
        hr_hz = features.get('hr_peak_hz', 0)
        overlap = 0.0
        for h_mult in [2, 3, 4]:
            if abs(hr_hz - h_mult*rr_hz) < 0.08:
                overlap = 1.0
                break
        features['resp_harmonic_overlap'] = overlap
            
        if neighboring_iq is not None and len(neighboring_iq) > 0:
            coh = []
            for n_iq in neighboring_iq:
                n_phase = np.nan_to_num(np.unwrap(np.angle(n_iq)))
                c_matrix = np.corrcoef(phase, n_phase)
                c_val = np.abs(c_matrix[0, 1]) if not np.any(np.isnan(c_matrix)) else 0.0
                coh.append(c_val)
            features['rx_coherence'] = np.mean(coh)
        else:
            features['rx_coherence'] = 0.0

        return features

    def compute_ai_score(self, features: Dict[str, float], weights: Dict[str, float]) -> float:
        """Kombinace příznaků do jednoho skóre."""
        score = (
            weights.get('hr_snr', 1.0) * (features.get('hr_snr', 0) + 10) / 20.0 + 
            weights.get('rr_snr', 1.0) * (features.get('rr_snr', 0) + 10) / 20.0 +
            weights.get('phase_stability', 0.5) * np.tanh(features.get('phase_stability', 0) / 10.0) +
            weights.get('rx_coherence', 0.5) * features.get('rx_coherence', 0) +
            weights.get('amp_stability', 0.2) * np.tanh(features.get('amp_stability', 0) / 5.0) -
            weights.get('motion_penalty', 2.0) * features.get('motion_score', 0) -
            weights.get('entropy_penalty', 0.5) * (features.get('spectral_entropy', 5) / 10.0) -
            weights.get('harmonic_penalty', 1.0) * features.get('resp_harmonic_overlap', 0)
        )
        return float(score)

    def fuse_signals(self, signals: np.ndarray, fs: float, method: str, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, str]:
        """ROI Fusion. Returns (fused_signal, type) where type is 'complex_iq', 'phase', or 'displacement'."""
        if signals.ndim == 1:
            return signals, "complex_iq"
        
        n_samples, n_channels = signals.shape
        if n_channels == 1:
            return signals[:, 0], "complex_iq"
        
        if method == "best bin":
            return signals[:, 0], "complex_iq"
        
        elif method == "weighted complex":
            if weights is None or len(weights) != n_channels:
                weights = np.ones(n_channels)
            w = np.maximum(weights, 0)
            if np.sum(w) == 0: w = np.ones(n_channels)
            w = w / np.sum(w)
            return np.dot(signals, w), "complex_iq"
            
        elif method == "weighted displacement":
            # every channel's own arc centred (dc_offset_correction) before its phase, as the
            # single-channel path does: with the static part left in, the phase is a short
            # arc around it and the heart all but vanished (19 instead of 1 bpm off a pulse)
            phases = np.zeros((n_samples, n_channels))
            for i in range(n_channels):
                phases[:, i] = self.extract_phase(self.dc_offset_correction(signals[:, i]))
            
            if weights is None or len(weights) != n_channels:
                weights = np.ones(n_channels)
            w = np.maximum(weights, 0)
            if np.sum(w) == 0: w = np.ones(n_channels)
            w = w / np.sum(w)
            return np.dot(phases, w), "phase"
            
        elif method == "PCA fusion" and _load_sklearn():
            try:
                phases = np.zeros((n_samples, n_channels))
                for i in range(n_channels):
                    phases[:, i] = self.extract_phase(self.dc_offset_correction(signals[:, i]))
                pca = PCA(n_components=1)
                fused = pca.fit_transform(phases).flatten()
                return fused, "phase"
            except Exception:
                return np.mean(signals, axis=1), "complex_iq"
            
        elif method == "ICA fusion" and _load_sklearn():
            try:
                phases = np.zeros((n_samples, n_channels))
                for i in range(n_channels):
                    phases[:, i] = self.extract_phase(self.dc_offset_correction(signals[:, i]))
                ica = FastICA(n_components=min(3, n_channels), random_state=42)
                components = ica.fit_transform(phases)
                
                best_idx = 0
                max_snr = -100
                for i in range(components.shape[1]):
                    f, p = signal.welch(components[:, i], fs, nperseg=min(len(components[:, i]), 512))
                    idx = np.where((f >= 0.8) & (f <= 3.0))[0]
                    if len(idx) > 0:
                        snr = np.max(p[idx]) / (np.mean(p) + 1e-9)
                        if snr > max_snr:
                            max_snr = snr
                            best_idx = i
                return components[:, best_idx], "phase"
            except Exception:
                return np.mean(signals, axis=1), "complex_iq"

        return np.mean(signals, axis=1), "complex_iq"

    def centred_displacements(self, iq: np.ndarray, wavelength: float = 1.0) -> np.ndarray:
        """(channels, frames): every channel's arc centred, its phase unwrapped and turned
        into displacement (wavelength / 4 pi; 1.0: radians / 4 pi)."""
        iq = np.atleast_2d(np.asarray(iq).T).T if np.ndim(iq) == 1 else np.asarray(iq)
        return np.array([signal.detrend(self.extract_phase(self.dc_offset_correction(iq[:, i])))
                         * wavelength / (4 * np.pi) for i in range(iq.shape[1])])

    def bin_trends(self, iq: np.ndarray, fs: float, win_sec: float, overlap_pct: float,
                   wavelength: float = 1.0, heart_iq=None, centre: int = 0):
        """estimate_trends of one range bin from its receivers' signals iq (frames, RX):
        the breathing from their spectra summed, the heart from their mean displacement -
        or, with heart_iq (frames, RX, bins) around it (the bin at index centre), from the
        mean over its receivers and those of its neighbours (vitalsigns.heart_channels)."""
        d = self.centred_displacements(iq, wavelength)
        heart = None
        if heart_iq is not None and np.ndim(heart_iq) == 3:
            import vitalsigns
            heart = vitalsigns.heart_channels(np.asarray(heart_iq), centre, wavelength,
                                              half=np.shape(heart_iq)[2]).mean(axis=0)
        return self.estimate_trends(d.mean(axis=0), fs, win_sec, overlap_pct, rx_disps=d,
                                    heart_disp=heart)

    def estimate_trends(self, sig: np.ndarray, fs: float, win_sec: float, overlap_pct: float,
                        resp_band=None, hr_band=None, rx_disps=None, heart_disp=None):
        """
        Sliding window analysis for HR/RR trends with robust logic. rx_disps (channels,
        frames): the receivers' own displacements - the breathing rate then comes from
        their spectra summed, as in the Vital Signs view (one fused signal was 2.5 /min
        off the reference against 1.5). heart_disp (frames,): the displacement the heart is
        followed in, when not sig itself (the receivers of the neighbouring bins too).
        """
        resp_band = resp_band or vital_bands.current().resp_hz
        hr_band = hr_band or vital_bands.current().heart_hz
        win_len = int(win_sec * fs)
        if win_len <= 0 or win_len > len(sig):
            win_len = len(sig)
            
        step = max(1, int(win_len * (1 - overlap_pct / 100)))
        results = []
        
        # Determine if sig is IQ or phase
        if np.iscomplexobj(sig):
            phase = self.extract_phase(self.dc_offset_correction(sig))
        else:
            phase = sig
        
        # the receivers' displacements in the breathing band, as the Vital Signs view takes
        # them: the drift of a whole window otherwise sits at the bottom of the band
        rx_br = None
        if rx_disps is not None and np.shape(rx_disps)[-1] == len(sig) and resp_band[1] < 0.5 * fs:
            sos = signal.butter(4, [resp_band[0], resp_band[1]], 'bandpass', fs=fs, output='sos')
            rx_br = signal.sosfiltfilt(sos, np.asarray(rx_disps, float), axis=-1)
        starts = list(range(0, len(sig) - win_len + 1, step))
        for start in starts:
            seg_phase = phase[start:start+win_len]
            seg_phase = seg_phase - np.mean(seg_phase)
            t_center = (start + win_len/2) / fs
            
            # One Hann periodogram of the whole window, zero-padded: Welch segments of 512
            # samples are 5 s at 100 fps, bins of 12 /min - a breathing rate could only
            # come out near 0, 12 or 24 /min (4 /min off the reference on average)
            nfft = max(4096, 1 << (win_len - 1).bit_length())
            f, p = signal.welch(seg_phase, fs, nperseg=len(seg_phase), nfft=nfft)
            p_br = p
            if rx_br is not None:
                seg_rx = signal.detrend(rx_br[:, start:start + win_len], axis=-1)
                _, p_rx = signal.welch(seg_rx, fs, nperseg=win_len, nfft=nfft, axis=-1)
                p_br = p_rx.sum(axis=0)
            
            # RR
            rr_f, rr_bpm, rr_snr = self.find_best_peak(f, p_br, resp_band)
            rr_conf = rr_snr
            
            # HR with harmonic suppression
            p_hr = p.copy()
            if rr_f > 0:
                for h_mult in [2, 3]:
                    h_f = rr_f * h_mult
                    h_idx = np.argmin(np.abs(f - h_f))
                    w = max(1, int(0.05 / (f[1]-f[0])))
                    p_hr[max(0, h_idx-w):min(len(p), h_idx+w)] *= 0.1
            
            hr_f, hr_bpm, hr_snr = self.find_best_peak(f, p_hr, hr_band)
            hr_conf = hr_snr
            valid = hr_conf > 3.0

            results.append({
                't_s': t_center,
                'hr_bpm': hr_bpm,
                'rr_bpm': rr_bpm,
                'hr_conf': float(np.clip(hr_conf / 15.0, 0, 1)),
                'rr_conf': float(np.clip(rr_conf / 15.0, 0, 1)),
                'quality_score': float(np.clip((hr_conf + rr_conf) / 30.0, 0, 1)),
                'valid': bool(valid),
                # the breathing counts where its peak passes Min peak/avg, as in the Vital
                # Signs view: on six recordings those windows were 0.7 /min off the
                # reference, all of them 1.2
                'rr_valid': bool(rr_conf >= RR_MIN_PEAK_TO_AVG)
            })

        # The heart followed through the windows as the Vital Signs view does it
        # (vitalsigns.heart_rate_track: harmonic sum, Viterbi path), not the largest
        # peak of each window held back by a rate limiter: on six seated recordings
        # against a somnograph that was 12 bpm off the pulse, in 10 s windows.
        if results and hr_band[1] < 0.5 * fs:
            import vitalsigns
            bands = vital_bands.Bands(tuple(60.0 * np.asarray(resp_band)), tuple(60.0 * np.asarray(hr_band)))
            disp = np.real(phase) if np.iscomplexobj(phase) else np.asarray(phase, float)
            if heart_disp is not None and np.shape(heart_disp) == np.shape(disp):
                disp = np.asarray(heart_disp, float)
            br_t = np.array([r['t_s'] for r in results])
            br_v = np.array([r['rr_bpm'] for r in results])
            # the path at its own fine step, read off at each window's centre
            tc, hr, conf, _, _ = vitalsigns.heart_rate_track(
                disp, fs, bands, win_len / fs, min(step / fs, vitalsigns.TRACK_STEP_S), br_t, br_v)
            for r in results:
                k = int(np.argmin(np.abs(tc - r['t_s'])))
                h, c = hr[k], conf[k]
                r['hr_bpm'] = float(h)
                r['hr_conf'] = float(np.clip(c / 15.0, 0, 1))
                r['quality_score'] = float(np.clip((c + r['rr_conf'] * 15.0) / 30.0, 0, 1))
                r['valid'] = bool(c >= vitalsigns.DEFAULT_CONF_THRESH)
        return results

    def run_unsupervised_scoring(self, feature_matrix: np.ndarray, method: str):
        if not _load_sklearn() or feature_matrix.shape[0] < 2:
            return np.zeros(feature_matrix.shape[0])

        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(np.nan_to_num(feature_matrix))
            
            if method == "PCA scoring":
                pca = PCA(n_components=1)
                score = pca.fit_transform(X_scaled).flatten()
                if np.corrcoef(score, feature_matrix[:, 0])[0, 1] < 0: score = -score
                return score
            elif method == "KMeans clustering":
                kmeans = KMeans(n_clusters=min(3, X_scaled.shape[0]), n_init=10)
                kmeans.fit(X_scaled)
                best_c = np.argmax([np.mean(feature_matrix[kmeans.labels_ == i, 0]) for i in range(kmeans.n_clusters)])
                return -np.linalg.norm(X_scaled - kmeans.cluster_centers_[best_c], axis=1)
            elif method == "Gaussian Mixture":
                gmm = GaussianMixture(n_components=min(2, X_scaled.shape[0]))
                gmm.fit(X_scaled)
                return gmm.score_samples(X_scaled)
            elif method == "Isolation Forest":
                iso = IsolationForest(contamination=0.1, random_state=42)
                return iso.fit_predict(X_scaled).astype(float)
        except Exception: pass
        return np.zeros(feature_matrix.shape[0])

    # --- Robust Reference PPG Module ---

    def parse_pleth_txt_file(self, path, default_fs=128.0, signal_kind="ppg"):
        """Specifický parser pro Pleth TXT formát s metadaty."""
        try:
            metadata = {}
            data_start_found = False
            ppg_raw = []
            timestamps = []
            
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            data_line_idx = -1
            for i, line in enumerate(lines):
                if line.startswith("Signal Type:"): metadata['signal_type'] = line.split(":", 1)[1].strip()
                elif line.startswith("Start Time:"): metadata['start_time'] = line.split(":", 1)[1].strip()
                elif line.startswith("Sample Rate:"): metadata['fs'] = float(line.split(":", 1)[1].strip())
                elif line.startswith("Length:"): metadata['length'] = int(line.split(":", 1)[1].strip())
                elif line.startswith("Unit:"): metadata['unit'] = line.split(":", 1)[1].strip()
                elif line.startswith("Data:"):
                    data_line_idx = i + 1
                    data_start_found = True
                    break
            
            if not data_start_found:
                raise ValueError("Could not find the 'Data:' section.")
            
            fs = metadata.get('fs', default_fs)
            
            valid_values = []
            valid_timestamps = []
            for line in lines[data_line_idx:]:
                line = line.strip()
                if not line: continue
                parts = line.split(';')
                if len(parts) < 2: continue
                
                # Timestamp: DD.MM.YYYY HH:MM:SS,mmm
                ts_str = parts[0].strip()
                try:
                    # Timestamp formát: DD.MM.YYYY HH:MM:SS,mmm
                    # V Pythonu %f bere mikrosekundy (6 cifer), mmm jsou milisekundy (3 cifry).
                    # Musíme zajistit správné parsování milisekund.
                    ts_value = self.parse_reference_timestamp(ts_str)
                    if ts_value is None:
                        continue
                    value = float(parts[1].strip().replace(',', '.'))
                    valid_timestamps.append(ts_value)
                    valid_values.append(value)
                except (TypeError, ValueError):
                    continue
            
            ppg_raw = np.asarray(valid_values, dtype=float)
            timestamps = np.asarray(valid_timestamps, dtype=float)
            
            if len(ppg_raw) == 0:
                raise ValueError("Signal is empty.")
                
            if 'length' in metadata and len(ppg_raw) != metadata['length']:
                print(f"[AI Reference] Warning: Počet vzorků ({len(ppg_raw)}) neodpovídá Length ({metadata['length']})")
        
            if fs <= 0:
                raise ValueError(f"Invalid sampling rate: {fs}")

            time_s = np.arange(len(ppg_raw)) / fs
            
            kind = str(signal_kind or "ppg").lower()
            self.reference_data = {
                "source_type": "pleth_txt",
                "path": path,
                "signal_type": metadata.get('signal_type', 'Pleth'),
                "start_time": metadata.get('start_time', ''),
                "fs": fs,
                "time": time_s,
                "timestamp": timestamps,
                "signal": ppg_raw,
                "hr_ref": ppg_raw.copy() if kind == "pulse" else np.array([]),
                "signal_kind": kind,
                "has_absolute_timestamps": bool(len(timestamps) == len(ppg_raw) and len(timestamps) > 0),
                "metadata": metadata,
                "ppg_clean": None,
                "ppg_filtered": None,
                "peaks": None,
                "peak_times": None,
                "time_hr": None,
                "hr_calc": None,
                "sqi": None
            }
            return True
        except Exception as e:
            print(f"[AI Reference] Pleth TXT parser error: {e}")
            raise e

    def auto_detect_delimiter(self, path):
        try:
            with open(path, 'r') as f:
                head = f.read(2048)
                # A comma is part of the required millisecond timestamp, so
                # prefer the semicolon when it separates the value column.
                if ';' in head:
                    return ';'
                dialect = csv.Sniffer().sniff(head, delimiters=',;\t ')
                return dialect.delimiter
        except:
            return ','

    def load_reference_txt_csv(self, path, config=None, signal_kind="ppg"):
        if not os.path.exists(path): return False
        
        # 1. Zkusit Pleth TXT parser nejdříve
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                head = f.read(1024)
                if "Signal Type:" in head and "Sample Rate:" in head and "Data:" in head:
                    default_fs = (config or {}).get('fs', 4.0 if signal_kind == 'pulse' else 128.0)
                    return self.parse_pleth_txt_file(path, default_fs=default_fs, signal_kind=signal_kind)
        except:
            pass

        # 2. Fallback na obecný CSV/TXT parser
        cfg = config or {}
        signal_kind = str(signal_kind or cfg.get('signal_kind', 'ppg')).lower()
        delim = cfg.get('delimiter', 'auto')
        if delim == 'auto': delim = self.auto_detect_delimiter(path)
            
        try:
            raw_data = []
            headers = []
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read a sample for sniffing
                sample = f.read(4096)
                f.seek(0)
                
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=',;\t ')
                    has_header = csv.Sniffer().has_header(sample)
                except:
                    dialect = None
                    has_header = False
                
                final_delim = delim if delim != 'auto' else (dialect.delimiter if dialect else ',')
                
                if has_header:
                    reader = csv.DictReader(f, delimiter=final_delim)
                    headers = reader.fieldnames
                    for row in reader: raw_data.append(row)
                else:
                    reader = csv.reader(f, delimiter=final_delim)
                    for row in reader: raw_data.append(row)
            
            if not raw_data: return False
            
            t_col, s_col, h_col = None, None, None
            time_headers = ['time', 'sekund', 't_s', 'timestamp', 'seconds', 'cas', 'time_s']
            sig_headers = ['ppg', 'pleth', 'signal', 'value', 'plethysmogram', 'pulse', 'data']
            hr_headers = ['hr', 'bpm', 'tep', 'heart_rate', 'heartrate', 'puls']

            if headers:
                for h in headers:
                    l = h.lower()
                    if any(x == l or x in l for x in time_headers): t_col = h
                    if any(x == l or x in l for x in sig_headers): s_col = h
                    if any(x == l or x in l for x in hr_headers): h_col = h
            
            res = {'time': [], 'signal': [], 'hr': [], 'timestamp': []}
            for row in raw_data:
                try:
                    if headers:
                        if t_col:
                            raw_time = row[t_col]
                            parsed_ts = self.parse_reference_timestamp(raw_time)
                            if parsed_ts is not None:
                                res['timestamp'].append(parsed_ts)
                            else:
                                res['time'].append(float(str(raw_time).replace(',', '.')))
                        if s_col: res['signal'].append(float(str(row[s_col]).replace(',', '.')))
                        if h_col: res['hr'].append(float(str(row[h_col]).replace(',', '.')))
                    else:
                        # Timestamped exports commonly have a text timestamp in column 1.
                        parsed_ts = self.parse_reference_timestamp(row[0]) if row else None
                        if parsed_ts is not None:
                            res['timestamp'].append(parsed_ts)
                            values = []
                            for x in row[1:]:
                                try:
                                    values.append(float(str(x).replace(',', '.')))
                                except (TypeError, ValueError):
                                    continue
                            if values:
                                res['signal'].append(values[0])
                                if len(values) > 1: res['hr'].append(values[1])
                        else:
                            v = []
                            for x in row:
                                try: v.append(float(str(x).replace(',', '.')))
                                except (TypeError, ValueError): continue
                            if len(v) >= 1: res['time'].append(v[0])
                            if len(v) >= 2: res['signal'].append(v[1])
                            if len(v) >= 3: res['hr'].append(v[2])
                except (TypeError, ValueError, IndexError):
                    continue
                
            for k in res: res[k] = np.asarray(res[k], dtype=float)
            
            # If still none found, try to guess signal column
            if len(res['signal']) == 0 and len(res['hr']) == 0 and headers:
                for h in headers:
                    try:
                        vals = [float(str(r[h]).replace(',', '.')) for r in raw_data[:10]]
                        if len(vals) > 5:
                            s_col = h
                            res['signal'] = np.array([float(str(r[h]).replace(',', '.')) for r in raw_data if h in r])
                            break
                    except: continue

            value_count = max(len(res['signal']), len(res['hr']), len(res['timestamp']))
            fs_default = float(cfg.get('fs', 4.0 if signal_kind == 'pulse' else 128.0))
            if len(res['timestamp']) >= 2:
                # Absolute timestamped files are the source of truth. Local time is
                # kept only for processing windows before radar synchronization.
                order = np.argsort(res['timestamp'], kind='stable')
                res['timestamp'] = res['timestamp'][order]
                if len(res['signal']) == len(order): res['signal'] = res['signal'][order]
                if len(res['hr']) == len(order): res['hr'] = res['hr'][order]
                diffs = np.diff(res['timestamp'])
                valid_diffs = diffs[diffs > 0]
                fs = 1.0 / np.median(valid_diffs) if len(valid_diffs) else fs_default
                res['time'] = np.arange(len(res['timestamp']), dtype=float) / fs
            elif len(res['time']) < 2:
                fs = fs_default
                res['time'] = np.arange(value_count, dtype=float) / fs
            else:
                diffs = np.diff(res['time'])
                valid_diffs = diffs[diffs > 0]
                fs = 1.0 / np.median(valid_diffs) if len(valid_diffs) > 0 else fs_default

            # Pulse files may label their value column as Pulse/HR, while PPG
            # files use Signal. Keep both representations for compatibility.
            values = res['hr'] if signal_kind == 'pulse' and len(res['hr']) else res['signal']
            if len(res['timestamp']) and len(values) != len(res['timestamp']):
                n = min(len(values), len(res['timestamp']))
                values = values[:n]
                res['timestamp'] = res['timestamp'][:n]
                res['time'] = np.arange(n, dtype=float) / fs

            self.reference_data = {
                'time': res['time'], 'signal': values if signal_kind == 'pulse' else res['signal'],
                'hr_ref': values if signal_kind == 'pulse' else res['hr'],
                'timestamp': res['timestamp'],
                'has_absolute_timestamps': len(res['timestamp']) == len(values) and len(values) > 0,
                'signal_kind': signal_kind,
                'fs': fs, 'path': path, 'sqi': None, 'hr_calc': None, 'time_hr': None
            }
            return True
        except Exception as e:
            print(f"[AI Reference] Error: {e}")
            return False

    def process_reference_ppg(self, invert=False, window_s=10, overlap_pct=50):
        """Zpracování PPG signálu: filtrace, peaky, HR trend."""
        if self.reference_data is None:
            raise ValueError("Reference data not loaded.")
        
        sig_raw = self.reference_data["signal"]
        fs = self.reference_data["fs"]
        
        if len(sig_raw) == 0:
            raise ValueError("Signal is empty.")

        # Odstranění NaN/Inf
        sig = np.nan_to_num(sig_raw.astype(float))
        
        # Invertovat pokud je požadováno
        if invert:
            sig = -sig

        # Odstranění dlouhé počáteční nulové/saturované části pouze pro HR (pro waveform zachovat)
        # Najdeme index, kde začíná skutečná aktivita
        start_idx = 0
        diff_sig = np.abs(np.diff(sig))
        activity_threshold = np.max(diff_sig) * 0.01 if len(diff_sig) > 0 else 0
        for i, val in enumerate(diff_sig):
            if val > activity_threshold:
                start_idx = i
                break
        
        sig_for_hr = sig[start_idx:]
        time_for_hr = self.reference_data["time"][start_idx:]

        # 1. Filtrace pro waveform (0.5 - 8 Hz)
        ppg_clean = self.safe_bandpass(sig, fs, 0.5, 8.0)
        
        # 2. Filtrace pro HR (0.7 - 4 Hz ~ 42 - 240 BPM)
        ppg_filtered = self.safe_bandpass(sig, fs, 0.7, 4.0)
        
        # Normalizace
        ppg_clean = (ppg_clean - np.mean(ppg_clean)) / (np.std(ppg_clean) + 1e-9)
        ppg_filtered = (ppg_filtered - np.mean(ppg_filtered)) / (np.std(ppg_filtered) + 1e-9)

        # 3. Detekce systolických peaků
        # Fyziologické limity: 35-220 BPM -> dist_min = fs / (220/60)
        dist_min = int(fs / (220/60))
        
        # Adaptivní prominence
        prominence = 0.5 * np.std(ppg_filtered)
        
        peaks, _ = signal.find_peaks(ppg_filtered, distance=dist_min, prominence=prominence)
        
        if len(peaks) < 2:
            print("[AI Reference] Warning: Nebyly nalezeny žádné peaky.")
            # Zkusíme snížit prominenci
            peaks, _ = signal.find_peaks(ppg_filtered, distance=dist_min, prominence=prominence*0.2)
            if len(peaks) < 2:
                # Pokud stále nic, zkusíme bez prominence (jen distance)
                peaks, _ = signal.find_peaks(ppg_filtered, distance=dist_min)

        if len(peaks) == 0:
            raise ValueError("PPG HR could not be computed - no peaks were found.")

        peak_times = self.reference_data["time"][peaks]
        
        # 4. Výpočet HR v klouzavých oknech
        duration = len(sig) / fs
        step_s = window_s * (1 - overlap_pct / 100.0)
        n_windows = int(max(0, (duration - window_s) / step_s)) + 1
        
        time_hr = []
        hr_calc = []
        sqi_list = []
        
        for i in range(n_windows):
            t_start = i * step_s
            t_end = t_start + window_s
            mid_t = t_start + window_s / 2
            
            # Peaky v okně
            w_peaks = peaks[(self.reference_data["time"][peaks] >= t_start) & (self.reference_data["time"][peaks] <= t_end)]
            
            if len(w_peaks) >= 2:
                # Beat-to-beat HR
                ibi = np.diff(self.reference_data["time"][w_peaks])
                # Filtr na IBIs (0.27s - 1.7s -> 220 - 35 BPM)
                ibi_valid = ibi[(ibi >= 0.27) & (ibi <= 1.7)]
                
                if len(ibi_valid) >= 1:
                    hr = 60.0 / np.mean(ibi_valid)
                    # SQI založené na variabilitě IBI a síle signálu
                    ibi_std = np.std(ibi_valid) / np.mean(ibi_valid)
                    sqi = max(0, 1 - ibi_std * 2) # Jednoduché SQI
                    
                    time_hr.append(mid_t)
                    hr_calc.append(hr)
                    sqi_list.append(sqi)
                else:
                    time_hr.append(mid_t)
                    hr_calc.append(np.nan)
                    sqi_list.append(0)
            else:
                time_hr.append(mid_t)
                hr_calc.append(np.nan)
                sqi_list.append(0)

        # Uložit do reference_data
        self.reference_data.update({
            "ppg_clean": ppg_clean,
            "ppg_filtered": ppg_filtered,
            "peaks": peaks,
            "peak_times": peak_times,
            "time_hr": np.array(time_hr),
            "hr_calc": np.array(hr_calc),
            "sqi": np.array(sqi_list)
        })
        
        # Kontrolní výpisy
        avg_hr = np.nanmean(hr_calc) if len(hr_calc) > 0 else 0
        avg_sqi = np.nanmean(sqi_list) if len(sqi_list) > 0 else 0
        print(f"[AI Reference] Zpracování PPG dokončeno:")
        print(f"  - Počet vzorků: {len(sig_raw)}")
        print(f"  - fs: {fs} Hz")
        print(f"  - Délka: {duration:.1f} s")
        print(f"  - Počet peaků: {len(peaks)}")
        print(f"  - Počet HR oken: {len(time_hr)}")
        print(f"  - Průměrná PPG HR: {avg_hr:.1f} BPM")
        print(f"  - Průměrné SQI: {avg_sqi:.2f}")

        return True

    def clean_ppg_signal(self, ppg, fs):
        """Preprocess PPG: detrend, normalize, filter."""
        if len(ppg) == 0: return ppg
        ppg = np.nan_to_num(ppg.astype(float))
        ppg = signal.detrend(ppg)
        # Bandpass 0.5 - 5.0 Hz
        nyq = 0.5 * fs
        low, high = 0.5, min(5.0, 0.9*nyq)
        if low < high:
            sos = signal.butter(4, [low, high], btype='bandpass', fs=fs, output='sos')
            ppg = signal.sosfiltfilt(sos, ppg)
        return (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-9)

    def compute_ppg_quality(self, ppg_seg, fs):
        """Calculates PPG Signal Quality Index (SQI) 0-1."""
        if len(ppg_seg) < 2 * fs: return 0.0
        
        # 1. Check for clipping/saturation
        clipped = np.sum((ppg_seg > 0.95 * np.max(ppg_seg)) | (ppg_seg < 0.95 * np.min(ppg_seg))) / len(ppg_seg)
        
        # 2. SNR in cardiac band
        f, p = signal.welch(ppg_seg, fs, nperseg=min(len(ppg_seg), 512))
        card_idx = np.where((f >= 0.8) & (f <= 3.0))[0]
        if len(card_idx) == 0: return 0.0
        
        card_power = np.sum(p[card_idx])
        total_power = np.sum(p) + 1e-9
        snr_factor = card_power / total_power
        
        # 3. Peak regularity
        # Use find_peaks to get IBIs
        peaks, _ = signal.find_peaks(ppg_seg, distance=int(fs*0.3), prominence=0.1)
        if len(peaks) > 2:
            ibis = np.diff(peaks) / fs
            regularity = 1.0 / (1.0 + np.std(ibis))
        else:
            regularity = 0.0
            
        sqi = (0.4 * snr_factor + 0.4 * regularity + 0.2 * (1.0 - clipped))
        return float(np.clip(sqi, 0, 1))

    def extract_ppg_hr_robust(self, window_s=10, overlap_pct=50):
        if not self.reference_data or len(self.reference_data.get('signal', [])) == 0: return False
        
        # Pokud již máme vypočtená data, nemusíme počítat znovu
        # Ale pokud se změnily parametry (window_s, overlap), měli bychom přepočítat.
        # Pro jednoduchost teď v GUI vždy voláme process_reference_ppg.
        if self.reference_data.get('hr_calc') is not None and len(self.reference_data['hr_calc']) > 0:
             return True
             
        sig = self.reference_data.get('signal_processed', self.reference_data['signal'])
        fs = self.reference_data['fs']
        ppg_clean = self.clean_ppg_signal(sig, fs)
        
        win_len = int(window_s * fs)
        step = max(1, int(win_len * (1 - overlap_pct/100)))
        
        hrs, times, sqis = [], [], []
        
        # Global peaks for visualization
        all_peaks, _ = signal.find_peaks(ppg_clean, distance=int(fs*60/220), prominence=0.1)
        self.reference_data['peaks'] = all_peaks
        
        for start in range(0, len(ppg_clean) - win_len + 1, step):
            seg = ppg_clean[start:start+win_len]
            t_center = self.reference_data['time'][start + win_len//2]
            
            # Robust peak detection in window
            min_dist = int(fs * 60 / 220)
            peaks, _ = signal.find_peaks(seg, distance=min_dist, prominence=0.1)
            
            bpm = 0.0
            sqi = self.compute_ppg_quality(seg, fs)
            
            if len(peaks) >= 2:
                ibis = np.diff(peaks) / fs
                # Filter physiological IBIs (35 to 220 BPM)
                valid_ibis = ibis[(ibis >= 60/220) & (ibis <= 60/35)]
                if len(valid_ibis) >= 1:
                    bpm = 60.0 / np.median(valid_ibis)
                else:
                    # Fallback to FFT peak
                    f, p = signal.welch(seg, fs, nperseg=min(len(seg), 512), nfft=1024)
                    h_idx = np.where((f >= 0.7) & (f <= 3.5))[0]
                    if len(h_idx) > 0:
                        bpm = f[h_idx[np.argmax(p[h_idx])]] * 60
            
            hrs.append(bpm)
            times.append(t_center)
            sqis.append(sqi)
            
        self.reference_data['hr_calc'] = np.array(hrs)
        self.reference_data['time_hr'] = np.array(times)
        self.reference_data['sqi'] = np.array(sqis)
        return True

    def synchronize_reference_with_radar(self, radar_start_timestamp, radar_duration_s,
                                         radar_time_s=None, manual_offset=0.0):
        """Align PPG and Pulse using absolute timestamps and crop to one interval.

        ``manual_offset`` follows the existing GUI convention: a positive value
        shifts reference data later on the radar-relative time axis.
        """
        if self.reference_data is None:
            raise ValueError("PPG reference data are not loaded.")
        if radar_start_timestamp is None or radar_duration_s is None:
            raise ValueError("Radar start timestamp and duration are required.")

        radar_start_timestamp = float(radar_start_timestamp)
        radar_duration_s = float(radar_duration_s)
        if not np.isfinite(radar_start_timestamp) or radar_duration_s <= 0:
            raise ValueError("Invalid radar time information.")

        ppg = self.reference_data
        pulse = self.pulse_data
        refs = [("ppg", ppg, "signal")]
        if pulse is not None:
            refs.append(("pulse", pulse, "hr_ref" if len(pulse.get("hr_ref", [])) else "signal"))

        aligned = {}
        for name, data, value_key in refs:
            timestamps = np.asarray(data.get("timestamp", []), dtype=float)
            values = np.asarray(data.get(value_key, []), dtype=float)
            if len(timestamps) == 0 or len(values) == 0:
                raise ValueError(f"{name.upper()} does not contain absolute timestamps and values.")
            n = min(len(timestamps), len(values))
            timestamps, values = timestamps[:n], values[:n]
            valid = np.isfinite(timestamps) & np.isfinite(values)
            timestamps, values = timestamps[valid], values[valid]
            order = np.argsort(timestamps, kind="stable")
            timestamps, values = timestamps[order], values[order]

            # Leading zero Pulse samples are not a measured heart rate and must
            # not shorten the common interval.
            if name == "pulse":
                nonzero = np.flatnonzero(np.abs(values) > 1e-12)
                if len(nonzero) == 0:
                    raise ValueError("Pulse contains no non-zero reference values.")
                first = int(nonzero[0])
                timestamps, values = timestamps[first:], values[first:]

            time_radar = timestamps - radar_start_timestamp + float(manual_offset)
            if len(time_radar) < 1:
                raise ValueError(f"{name.upper()} has no valid samples.")
            aligned[name] = (time_radar, values, timestamps)

        starts = [0.0]
        ends = [radar_duration_s]
        for time_radar, _, _ in aligned.values():
            starts.append(float(time_radar[0]))
            ends.append(float(time_radar[-1]))
        common_start = max(starts)
        common_end = min(ends)
        if common_end <= common_start:
            raise ValueError("Radar, PPG and Pulse do not overlap in absolute time.")

        def crop(data, name, value_key):
            time_radar, values, timestamps = aligned[name]
            mask = (time_radar >= common_start) & (time_radar <= common_end)
            if not np.any(mask):
                raise ValueError(f"No {name.upper()} samples in the common interval.")
            time_crop = time_radar[mask]
            values_crop = values[mask]
            timestamps_crop = timestamps[mask]
            data["timestamp"] = timestamps_crop
            data["time_sync"] = time_crop
            data["time"] = np.arange(len(values_crop), dtype=float) / float(data["fs"])
            data["sync_offset_s"] = float(time_radar[0])
            data["common_start_s"] = float(common_start)
            data["common_end_s"] = float(common_end)
            if name == "ppg":
                data["signal"] = values_crop
            else:
                data["hr_ref"] = values_crop
                data["signal"] = values_crop

        crop(ppg, "ppg", "signal")
        if pulse is not None:
            crop(pulse, "pulse", "hr_ref" if len(pulse.get("hr_ref", [])) else "signal")

        if radar_time_s is None:
            radar_time_s = np.arange(0.0, radar_duration_s + 0.5 / 1000.0, 1.0 / 20.0)
        radar_time_s = np.asarray(radar_time_s, dtype=float)
        radar_mask = (radar_time_s >= common_start) & (radar_time_s <= common_end)
        radar_common = radar_time_s[radar_mask]
        pulse_on_radar = np.array([])
        if pulse is not None and len(radar_common):
            pulse_on_radar = np.interp(radar_common, pulse["time_sync"], pulse["hr_ref"])
            pulse["radar_time_s"] = radar_common
            pulse["radar_values"] = pulse_on_radar

        ppg["time_hr_sync"] = None
        self.reference_sync = {
            "radar_start_timestamp": radar_start_timestamp,
            "manual_offset_s": float(manual_offset),
            "common_start_s": float(common_start),
            "common_end_s": float(common_end),
            "common_duration_s": float(common_end - common_start),
            "radar_time_s": radar_common,
            "pulse_on_radar": pulse_on_radar,
        }
        return self.reference_sync

    def finalize_reference_processing_sync(self):
        """Refresh HR trend timestamps after PPG processing/cropping."""
        if not self.reference_data:
            return
        common_start = (self.reference_sync or {}).get("common_start_s", 0.0)
        time_hr = self.reference_data.get("time_hr")
        if time_hr is not None:
            self.reference_data["time_hr_sync"] = np.asarray(time_hr, dtype=float) + common_start

    def synchronize_with_radar(self, radar_time_s, radar_hr_bpm, method='relative_start', manual_offset=0.0):
        if not self.reference_data: return 0.0
        offset = manual_offset
        if method == 'cross_correlation' and self.reference_data['hr_calc'] is not None and radar_time_s is not None and len(radar_hr_bpm) > 2:
            try:
                # Synchronizace pomocí křížové korelace trendů HR
                t_common = np.arange(0, max(radar_time_s[-1], self.reference_data['time_hr'][-1]), 1.0)
                r_hr = np.interp(t_common, radar_time_s, radar_hr_bpm)
                p_hr = np.interp(t_common, self.reference_data['time_hr'], self.reference_data['hr_calc'])
                
                # Odstranění průměru pro korelaci
                r_hr_c = r_hr - np.mean(r_hr)
                p_hr_c = p_hr - np.mean(p_hr)
                
                corr = signal.correlate(r_hr_c, p_hr_c, mode='full')
                lags = signal.correlation_lags(len(r_hr_c), len(p_hr_c), mode='full')
                best_lag = lags[np.argmax(corr)]
                # offset je v sekundách (protože t_common má krok 1s)
                offset = float(best_lag)
                print(f"[AI Sync] Best cross-correlation lag: {offset} s")
            except Exception as e:
                print(f"[AI Sync] Cross-correlation failed: {e}")
            
        self.reference_data['time_sync'] = self.reference_data['time'] + offset
        if self.reference_data['time_hr'] is not None:
            self.reference_data['time_hr_sync'] = self.reference_data['time_hr'] + offset
        return offset

    def compare_results(self, radar_time, radar_hr, radar_valid):
        if not self.reference_data or self.reference_data.get('time_hr_sync') is None: return None
        ref_hr_func = lambda t: np.interp(t, self.reference_data['time_hr_sync'], self.reference_data['hr_calc'], left=0, right=0)
        ref_hr_interp = ref_hr_func(radar_time)
        
        mask = radar_valid & (ref_hr_interp > 30) & (ref_hr_interp < 200)
        if not np.any(mask): return None
        
        diff = radar_hr[mask] - ref_hr_interp[mask]
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff**2))
        bias = np.mean(diff)
        std_err = np.std(diff)
        corr = np.corrcoef(radar_hr[mask], ref_hr_interp[mask])[0, 1] if len(diff) > 1 else 0
        
        # Bland-Altman
        ba_mean = (radar_hr[mask] + ref_hr_interp[mask]) / 2.0
        ba_diff = diff
        
        return {
            'mae': mae, 'rmse': rmse, 'bias': bias, 'std_err': std_err, 'corr': corr,
            'radar_hr': radar_hr, 'ref_hr': ref_hr_interp, 'time': radar_time, 'mask': mask,
            'diff': diff, 'ba_mean': ba_mean, 'ba_diff': ba_diff
        }

    def get_synthetic_signal(self, fs, duration, hr_bpm=70, rr_bpm=15, noise_lvl=0.02,
                             wavelength=299792458 / 60e9, static=3.0, rng=None):
        """
        The complex signal of a chest's range bin: 2 mm of breathing and 0.2 mm of heart
        (5.1 and 0.5 rad of phase at 60 GHz) on a circle around a static reflection, with
        complex noise of noise_lvl. It was 0.4 rad of breathing around the origin - an arc
        far shorter than a chest gives, which the centring every path applies
        (dc_offset_correction) bends into breathing harmonics: 50 bpm instead of 72.
        """
        rng = rng if rng is not None else np.random.default_rng(0)
        t = np.arange(int(round(duration * fs))) / fs
        disp = (2e-3 * np.sin(2 * np.pi * (rr_bpm / 60) * t)
                + 0.2e-3 * np.sin(2 * np.pi * (hr_bpm / 60) * t))
        iq = np.exp(1j * (4 * np.pi * disp / wavelength + rng.uniform(0, 2 * np.pi)))
        iq = iq + static * np.exp(1j * rng.uniform(0, 2 * np.pi))
        return iq + noise_lvl * (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t)))

    def export_results(self, results, filename, fmt='json'):
        def d_s(obj):
            if isinstance(obj, (np.integer, np.int64)): return int(obj)
            if isinstance(obj, (np.floating, np.float64)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return str(obj)
        if fmt == 'json':
            with open(filename, 'w') as f: json.dump(results, f, indent=4, default=d_s)
        elif fmt == 'csv':
            if 'window_results' in results:
                rows = results['window_results']
                if rows:
                    with open(filename, 'w', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=rows[0].keys())
                        w.writeheader()
                        w.writerows(rows)