import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import theme
import numpy as np
import scipy.signal as signal
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches
from plot_interactivity import create_interactive_canvas, normalize_for_plot, PlotInteractivityManager, add_plot_control_bar
from flow_layout import ScrollFrame
import os
import json
import csv

from ai_vital_analysis import AIVitalAnalysis, SKLEARN_AVAILABLE
import vital_bands
import capture_params

class AIVitalSignsTab:
    CANDIDATE_SPAN_M = 0.15     # candidate bins this close to the breathing person

    def __init__(self, parent, get_data_func):
        self.parent = parent
        self.get_data_func = get_data_func
        self.analysis = AIVitalAnalysis()
        self.last_ai_result = None
        self.last_cand_iq = None        # {bin: (frames x RX x bins around it, its index)}
        
        self.is_running = False
        self.ref_data_path = None
        self.pulse_data_path = None
        
        self.selected_window_indices = set() # Set of (win_idx, candidate_idx)
        self.min_ai_score_var = tk.DoubleVar(value=0.5)
        self.selection_result_var = tk.StringVar(value="No windows selected")
        
        self.init_ui()

    def init_ui(self):
        self.main_paned = ttk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        # The settings column keeps its own width (weight 0); the plots take what the
        # window gains or loses. With a fifth of the width it was cut off at 1024 px.
        self.left_frame = ttk.Frame(self.main_paned)
        self.right_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.left_frame, weight=0)
        self.main_paned.add(self.right_frame, weight=1)

        self._init_controls()
        self._init_plots()

    def _init_controls(self):
        # flow_layout.ScrollFrame: the scrollbar used to be packed after a 340 px canvas
        # and a narrower pane squeezed it out of sight; the mouse wheel did nothing.
        panel = ScrollFrame(self.left_frame)
        panel.pack(fill="both", expand=True)
        self.scrollable_frame = panel.inner

        # --- Section: Actions ---
        act_f = ttk.LabelFrame(self.scrollable_frame, text="Actions")
        act_f.pack(fill="x", padx=5, pady=5)
        ttk.Button(act_f, text="RUN AI ANALYSIS", command=self.run_ai_analysis, style="Accent.TButton").pack(fill="x", padx=5, pady=2)
        ttk.Button(act_f, text="Run Synthetic Test", command=self.run_synthetic_test).pack(fill="x", padx=5, pady=2)

        # --- Section: AI Mode ---
        mode_f = ttk.LabelFrame(self.scrollable_frame, text="AI Mode & Selection")
        mode_f.pack(fill="x", padx=5, pady=5)
        
        self.ai_mode_var = tk.StringVar(value="Rule-based AI scoring")
        modes = ["Rule-based AI scoring", "PCA scoring", "KMeans clustering", "Gaussian Mixture", "Isolation Forest"]
        ttk.Label(mode_f, text="Scoring Method:").pack(anchor="w", padx=5)
        self.mode_cb = ttk.Combobox(mode_f, textvariable=self.ai_mode_var, values=modes, state="readonly")
        self.mode_cb.pack(fill="x", padx=5, pady=2)
        
        self.phase_method_var = tk.StringVar(value="unwrap")
        ttk.Label(mode_f, text="Phase Extraction:").pack(anchor="w", padx=5)
        ttk.Combobox(mode_f, textvariable=self.phase_method_var, values=["unwrap", "dacm", "circlefit"], state="readonly").pack(fill="x", padx=5)

        self.auto_bin_var = tk.BooleanVar(value=True)
        tk.Checkbutton(mode_f, text="Auto Select Best Bin", variable=self.auto_bin_var).pack(anchor="w", padx=5)
        
        
        ttk.Label(mode_f, text="Manual Bin Index:").pack(anchor="w", padx=5)
        self.bin_idx_spin = ttk.Spinbox(mode_f, from_=0, to=511, width=10)
        self.bin_idx_spin.set(10)
        self.bin_idx_spin.pack(anchor="w", padx=5)

        # --- Section: ROI Fusion ---
        roi_f = ttk.LabelFrame(self.scrollable_frame, text="ROI & RX Fusion")
        roi_f.pack(fill="x", padx=5, pady=5)
        
        # every receiver's centred displacement: one receiver's heart track was 6 bpm off a
        # pulse oximeter on six seated recordings, all of them 2.7
        self.use_roi_var = tk.BooleanVar(value=True)
        tk.Checkbutton(roi_f, text="Enable ROI Fusion", variable=self.use_roi_var).pack(anchor="w", padx=5)
        
        ttk.Label(roi_f, text="ROI Source:").pack(anchor="w", padx=5)
        self.roi_source_var = tk.StringVar(value="All RX at selected bin")
        roi_sources = ["Best RX only", "All RX at selected bin", "ROI bins at best RX", "ROI bins x all RX", "Top-K bin/RX candidates"]
        ttk.Combobox(roi_f, textvariable=self.roi_source_var, values=roi_sources, state="readonly").pack(fill="x", padx=5)

        ttk.Label(roi_f, text="ROI Width (bins):").pack(anchor="w", padx=5)
        self.roi_width_var = tk.IntVar(value=1)
        theme.Slider(roi_f, from_=0, to=5, orient="horizontal", variable=self.roi_width_var).pack(fill="x", padx=5)
        
        ttk.Label(roi_f, text="Fusion Method:").pack(anchor="w", padx=5)
        self.fusion_method_var = tk.StringVar(value="weighted displacement")
        fusion_modes = ["best bin", "weighted complex", "weighted displacement", "PCA fusion", "ICA fusion"]
        ttk.Combobox(roi_f, textvariable=self.fusion_method_var, values=fusion_modes, state="readonly").pack(fill="x", padx=5)

        # --- Section: Sliding Window ---
        win_f = ttk.LabelFrame(self.scrollable_frame, text="Sliding Window Settings")
        win_f.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(win_f, text="Window Length [s]:").pack(anchor="w", padx=5)
        self.win_len_var = tk.DoubleVar(value=30.0)     # the heart track needs about 30 s
        ttk.Entry(win_f, textvariable=self.win_len_var).pack(fill="x", padx=5)
        
        ttk.Label(win_f, text="Overlap [%]:").pack(anchor="w", padx=5)
        self.overlap_var = tk.DoubleVar(value=90.0)     # a window every 3 s
        ttk.Entry(win_f, textvariable=self.overlap_var).pack(fill="x", padx=5)

        # --- Section: Window Selection ---
        sel_f = ttk.LabelFrame(self.scrollable_frame, text="Temporal Window Selection")
        sel_f.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(sel_f, text="Min AI Score:").pack(anchor="w", padx=5)
        ttk.Entry(sel_f, textvariable=self.min_ai_score_var).pack(fill="x", padx=5)
        
        ttk.Button(sel_f, text="Use Selected Windows", command=self.process_selected_windows).pack(fill="x", padx=5, pady=2)
        ttk.Button(sel_f, text="Clear Selection", command=self.clear_window_selection).pack(fill="x", padx=5, pady=2)
        
        tk.Label(sel_f, textvariable=self.selection_result_var, wraplength=300, font=theme.font(9, "bold"), fg=theme.BLUE).pack(fill="x", padx=5, pady=5)

        # --- Section: AI Weights ---
        w_f = ttk.LabelFrame(self.scrollable_frame, text="AI Score Weights")
        w_f.pack(fill="x", padx=5, pady=5)
        self.weights = {
            'hr_snr': tk.DoubleVar(value=1.0), 'rr_snr': tk.DoubleVar(value=1.0),
            'phase_stability': tk.DoubleVar(value=0.5), 'rx_coherence': tk.DoubleVar(value=0.5),
            'motion_penalty': tk.DoubleVar(value=2.0), 'entropy_penalty': tk.DoubleVar(value=0.5),
            'harmonic_penalty': tk.DoubleVar(value=1.0)
        }
        for k, var in self.weights.items():
            f = ttk.Frame(w_f); f.pack(fill="x", padx=2)
            ttk.Label(f, text=k.replace('_',' ').title()+":", width=18).pack(side="left")
            ttk.Entry(f, textvariable=var, width=8).pack(side="right")

        # --- Section: Reference Measurement ---
        bands_f = ttk.LabelFrame(self.scrollable_frame, text="Search bands (shared by all vital tabs)")
        bands_f.pack(fill="x", padx=5, pady=5)
        vital_bands.tk_controls(bands_f, vertical=True).pack(fill="x", padx=5, pady=2)

        ref_f = ttk.LabelFrame(self.scrollable_frame, text="Reference PPG/Pleth")
        ref_f.pack(fill="x", padx=5, pady=5)
        ttk.Button(ref_f, text="Load PPG (TXT/CSV)", command=lambda: self.load_reference("ppg")).pack(fill="x", padx=5, pady=2)
        ttk.Button(ref_f, text="Load Pulse (TXT/CSV)", command=lambda: self.load_reference("pulse")).pack(fill="x", padx=5, pady=2)
        self.ref_info_var = tk.StringVar(value="No reference loaded")
        tk.Label(ref_f, textvariable=self.ref_info_var, wraplength=300, font=theme.font(8), justify="left").pack(fill="x", padx=5)
        
        self.ref_invert_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ref_f, text="Invert PPG Signal", variable=self.ref_invert_var).pack(anchor="w", padx=5)
        
        
        f_off = ttk.Frame(ref_f); f_off.pack(fill="x", padx=5)
        ttk.Label(f_off, text="Manual Offset [s]:").pack(side="left")
        self.ref_offset_var = tk.DoubleVar(value=0.0); ttk.Entry(f_off, textvariable=self.ref_offset_var, width=10).pack(side="right")
        
        ttk.Button(ref_f, text="Process PPG & Sync", command=self.sync_and_compare).pack(fill="x", padx=5, pady=2)
        self.sync_info_var = tk.StringVar(value="Not synchronized")
        tk.Label(ref_f, textvariable=self.sync_info_var, wraplength=300, font=theme.font(8), justify="left", fg=theme.GREEN).pack(fill="x", padx=5)

        # --- Section: Export ---
        exp_f = ttk.LabelFrame(self.scrollable_frame, text="Export")
        exp_f.pack(fill="x", padx=5, pady=5)
        ttk.Button(exp_f, text="Export Results (JSON)", command=lambda: self.export('json')).pack(fill="x", padx=5, pady=2)
        ttk.Button(exp_f, text="Export Features (CSV)", command=lambda: self.export('csv')).pack(fill="x", padx=5, pady=2)

    def _init_plots(self):
        self.notebook = ttk.Notebook(self.right_frame)
        self.notebook.pack(fill="both", expand=True)
        tabs = ["Overview", "Signals", "Spectra", "Spatial Map", "Temporal Map", "Features", "Trends", "Comparison", "Reference PPG"]
        self.plots = {}
        for name in tabs:
            f = ttk.Frame(self.notebook); self.notebook.add(f, text=name)
            fig = Figure(figsize=(8, 6))
            # Interactive canvas: navigation toolbar + X span zoom / reset controls
            canvas, toolbar = create_interactive_canvas(f, fig)
            manager = PlotInteractivityManager(canvas)
            zoom_bar, _ = add_plot_control_bar(f, manager)
            zoom_bar.pack(side="top", fill="x", before=canvas.get_tk_widget())
            self.plots[name] = {"fig": fig, "canvas": canvas, "toolbar": toolbar, "manager": manager}
            
            if name == "Temporal Map":
                canvas.mpl_connect("button_press_event", self._on_map_click)

    def _refresh_interactivity(self, name, time_axes=()):
        """Znovu nastaví SpanSelectory a uloží výchozí rozsahy po překreslení grafu."""
        plot = self.plots[name]
        manager = plot.get("manager")
        if manager is not None:
            manager.register_time_axes(list(time_axes))
        plot["canvas"].draw_idle()

    def _on_map_click(self, event):
        if self.last_ai_result is None or event.inaxes is None:
            return
        
        # Check if clicked on the map axes (ax1 in _update_plots)
        fig = self.plots["Temporal Map"]["fig"]
        if not fig.axes or event.inaxes != fig.axes[0]:
            return
            
        res = self.last_ai_result
        t_axis = res['t_axis_temp']
        candidates = res['candidates']
        r_axis = res['r_axis']
        candidates_m = [r_axis[c] for c in candidates]
        
        # Map click coordinates to indices
        t_click = event.xdata
        r_click = event.ydata
        
        win_idx = np.argmin(np.abs(t_axis - t_click))
        cand_idx = np.argmin(np.abs(np.array(candidates_m) - r_click))
        
        idx_tuple = (int(win_idx), int(cand_idx))
        if idx_tuple in self.selected_window_indices:
            self.selected_window_indices.remove(idx_tuple)
        else:
            self.selected_window_indices.add(idx_tuple)
            
        self._update_plots()
        self.selection_result_var.set(f"Selected: {len(self.selected_window_indices)} windows")

    def process_selected_windows(self):
        if not self.last_ai_result:
            return
            
        res = self.last_ai_result
        # the candidates' signals of that analysis: the recording loaded now may be another
        # one (after the synthetic test there was none, and this failed)
        cand_iq = self.last_cand_iq
        fps = res['fps']
        temp_map = res['temp_map']
        min_score = self.min_ai_score_var.get()
        
        # the windows of the temporal map: those of the analysis, whatever the fields say now
        win_len_s = res.get('win_len_s', self.win_len_var.get())
        overlap = res.get('overlap_pct', self.overlap_var.get())

        hr_list, hr_weights, rr_list, weight_list = [], [], [], []
        
        # Determine which windows to use: manual selection OR all above threshold if nothing selected
        working_set = self.selected_window_indices
        if not working_set:
            # Auto-select all windows above threshold
            for w in range(temp_map.shape[0]):
                for c in range(temp_map.shape[1]):
                    if temp_map[w, c] >= min_score:
                        working_set.add((w, c))
        
        if not working_set:
            messagebox.showinfo("Selection", f"No windows selected and no windows above threshold ({min_score}).")
            return
            
        # every selected bin's own trends (its receivers: the heart track of their mean
        # displacement, the breathing of their spectra summed) - the largest peak of one
        # window was the heart rate here before, 8 bpm off a pulse oximeter on average
        bin_trends = {}
        for win_idx, cand_idx in working_set:
            rbin = res['candidates'][cand_idx]
            score = temp_map[win_idx, cand_idx]
            # Skip if below threshold even if manually selected?
            # Requirement says "vybrat pouze kvalitní časové úseky", so let's respect threshold.
            if score < min_score:
                continue
            if rbin not in bin_trends:
                iq_n, c = cand_iq[int(rbin)]
                bin_trends[rbin] = self.analysis.bin_trends(iq_n[:, :, c], fps, win_len_s, overlap,
                                                            res.get('wavelength', 1.0),
                                                            heart_iq=iq_n, centre=c)
            tr = bin_trends[rbin]
            if win_idx >= len(tr):
                continue
            w = max(0.1, score + 2.0)
            if tr[win_idx].get('rr_valid', True):   # the breathing where its peak is clear
                rr_list.append(tr[win_idx]['rr_bpm'])
                weight_list.append(w)
            if tr[win_idx]['valid']:            # the heart only where its track passed
                hr_list.append(tr[win_idx]['hr_bpm'])
                hr_weights.append(w)

        if rr_list or hr_list:
            hr_txt = (f"{self.analysis.compute_weighted_median(np.array(hr_list), np.array(hr_weights)):.1f} BPM"
                      if hr_list else "- (no window of the heart track passed)")
            rr_txt = (f"{self.analysis.compute_weighted_median(np.array(rr_list), np.array(weight_list)):.1f} BPM"
                      if rr_list else "- (no clear breathing peak)")
            self.selection_result_var.set(f"RESULT (HR {len(hr_list)}, RR {len(rr_list)} windows):\n"
                                          f"HR: {hr_txt} | RR: {rr_txt}")
            messagebox.showinfo("Weighted Results", f"Weighted Median Results (Min Score: {min_score}):\n"
                                f"HR: {hr_txt} ({len(hr_list)} windows)\nRR: {rr_txt} ({len(rr_list)} windows)")
            
            # Highlight these windows in the map
            self.selected_window_indices = set(working_set)
            self._update_plots()
        else:
            messagebox.showinfo("Selection", f"No windows satisfied the Min Score threshold ({min_score}).")

    def clear_window_selection(self):
        self.selected_window_indices.clear()
        self.selection_result_var.set("No windows selected")
        self._update_plots()

    def run_ai_analysis(self):
        self.selected_window_indices.clear()
        self.selection_result_var.set("No windows selected")
        
        data = self.get_data_func()
        if data is None or data.get('rfft_cube') is None:
            messagebox.showwarning("Warning", "No complex Range FFT data available.")
            return

        rfft_cube = data['rfft_cube'] # [Frames, Chirps, RX, RangeBins]
        fps = data.get('fps', 20.0)
        r_axis = data.get('range_axis')
        params = data.get('params', {})
        # The same wavelength as the Vital Signs and HR tabs: in the middle of the ADC
        # window (capture_params.wavelength_m, roadmap 1.9).
        wavelength = capture_params.wavelength_m(params, default_ghz=60.0)
        slow_cube = np.mean(rfft_cube, axis=1) if rfft_cube.ndim == 4 else rfft_cube
        n_frames, n_rx, n_bins = slow_cube.shape
        
        # Range Profile & Clutter Removal
        slow_cube_cr = slow_cube - np.mean(slow_cube, axis=0, keepdims=True)
        range_profile = np.mean(np.abs(slow_cube_cr), axis=(0, 1))
        
        # Candidate selection
        anchor = None
        if self.auto_bin_var.get():
            threshold = 0.1 * np.max(range_profile)
            candidates = [r for r in range(n_bins) if range_profile[r] > threshold]
            if r_axis is not None: candidates = [r for r in candidates if 0.3 <= r_axis[r] <= 15.0]
            # Around the person: the range that breathes most (the Vital Signs view's Auto
            # Bin) and its neighbours. The score alone, over every bright bin of the room,
            # took a bin 0.2-0.5 m off the chest on four of six recordings.
            if r_axis is not None and rfft_cube.ndim == 4 and len(r_axis) == n_bins:
                import vitalsigns
                bands_ab = vital_bands.current()
                anchor = vitalsigns.auto_bin(rfft_cube, fps, bands_ab.resp_hz, r_axis, bands=bands_ab)
                near = [r for r in range(n_bins)
                        if abs(r_axis[r] - r_axis[anchor]) <= self.CANDIDATE_SPAN_M and r_axis[r] >= 0.3]
                if near:
                    candidates = near
        else:
            c_bin = int(self.bin_idx_spin.get())
            candidates = [c_bin] if 0 <= c_bin < n_bins else [np.argmax(range_profile)]
        if not candidates: candidates = [np.argmax(range_profile)]

        # --- STAGE 1: Feature Extraction & Scoring ---
        weight_vals = {k: v.get() for k, v in self.weights.items()}
        phase_method = self.phase_method_var.get()
        feature_list = []
        bin_rx_scores = np.full((n_bins, n_rx), -100.0)
        
        for rbin in candidates:
            for rx in range(n_rx):
                iq = slow_cube[:, rx, rbin]
                neighbors = [slow_cube[:, i, rbin] for i in range(n_rx) if i != rx] if n_rx > 1 else []
                f = self.analysis.compute_features(iq, fps, neighboring_iq=neighbors, phase_method=phase_method)
                f.update({'range_bin': rbin, 'range_m': r_axis[rbin] if r_axis is not None else rbin, 'rx': rx, 'amp': range_profile[rbin]})
                f['ai_score'] = self.analysis.compute_ai_score(f, weight_vals) + max(-10, 10*np.log10(range_profile[rbin]/(np.max(range_profile)+1e-9)+1e-9)/10.0)
                feature_list.append(f)
                bin_rx_scores[rbin, rx] = f['ai_score']

        if self.ai_mode_var.get() != "Rule-based AI scoring" and SKLEARN_AVAILABLE and len(feature_list) > 2:
            f_matrix = np.array([[f[name] for name in self.analysis.feature_names] for f in feature_list])
            ml_s = self.analysis.run_unsupervised_scoring(f_matrix, self.ai_mode_var.get())
            for i, s in enumerate(ml_s):
                feature_list[i]['ai_score'] = s
                bin_rx_scores[feature_list[i]['range_bin'], feature_list[i]['rx']] = s

        # --- STAGE 2: Selection & ROI Fusion ---
        # With Auto Bin the bin is the breathing person's and the score picks the receiver
        # there (the neighbours stay in the feature table and the quality map): the best
        # score of the neighbourhood sat one bin off the chest and its heart track 24 bpm
        # off the pulse on one recording.
        pool = [f for f in feature_list if f['range_bin'] == anchor] if anchor is not None else []
        best_f = sorted(pool or feature_list, key=lambda x: x['ai_score'], reverse=True)[0]
        best_bin, best_rx = best_f['range_bin'], best_f['rx']
        
        # "Enable ROI Fusion" off: the single best bin/RX, whatever ROI Source says
        roi_src = self.roi_source_var.get() if self.use_roi_var.get() else "Best RX only"
        hw = self.roi_width_var.get()
        fused_signals, fused_w = [], []
        roi_bins, roi_rxs = [], []
        
        if roi_src == "Best RX only":
            roi_bins = [best_bin]
            roi_rxs = [best_rx]
            fused_signals = slow_cube[:, best_rx, best_bin : best_bin+1]
        elif roi_src == "All RX at selected bin":
            roi_bins = [best_bin]
            roi_rxs = list(range(n_rx))
            fused_signals = slow_cube[:, :, best_bin]
            fused_w = bin_rx_scores[best_bin, :]
        elif roi_src == "ROI bins at best RX":
            idx = np.arange(max(0, best_bin-hw), min(n_bins, best_bin+hw+1))
            roi_bins = idx.tolist()
            roi_rxs = [best_rx]
            fused_signals = slow_cube[:, best_rx, idx]
            fused_w = bin_rx_scores[idx, best_rx]
        elif roi_src == "ROI bins x all RX":
            idx = np.arange(max(0, best_bin-hw), min(n_bins, best_bin+hw+1))
            roi_bins = idx.tolist()
            roi_rxs = list(range(n_rx))
            fused_signals = slow_cube[:, :, idx].reshape(n_frames, -1)
            fused_w = bin_rx_scores[idx, :].flatten()
        elif roi_src == "Top-K bin/RX candidates":
            top_f = sorted(feature_list, key=lambda x: x['ai_score'], reverse=True)[:5]
            for f in top_f:
                roi_bins.append(f['range_bin'])
                roi_rxs.append(f['rx'])
                fused_signals.append(slow_cube[:, f['rx'], f['range_bin']])
                fused_w.append(f['ai_score'])
            fused_signals = np.array(fused_signals).T

        fused_sig, fused_type = self.analysis.fuse_signals(fused_signals, fps, self.fusion_method_var.get(), weights=np.array(fused_w) if len(fused_w)>0 else None)

        # --- STAGE 3: Final Processing ---
        if fused_type == "complex_iq":
            phase = self.analysis.extract_phase(self.analysis.dc_offset_correction(fused_sig), method=phase_method)
            disp = self.analysis.phase_to_displacement(phase, wavelength)
        elif fused_type == "phase":
            phase = fused_sig
            disp = self.analysis.phase_to_displacement(phase, wavelength)
        else:
            disp = fused_sig
            phase = disp * 4 * np.pi / wavelength

        bands = vital_bands.current()
        resp_sig = self.analysis.safe_bandpass(disp, fps, *bands.resp_hz)
        card_sig = self.analysis.safe_bandpass(disp, fps, bands.heart_hz[0], max(4.0, bands.heart_hz[1]))
        
        # Trends & Temporal Quality Map
        win_len_s = self.win_len_var.get()
        overlap = self.overlap_var.get()
        # the breathing from the receivers' spectra summed when they were all fused
        rx_d = (self.analysis.centred_displacements(fused_signals, wavelength)
                if fused_type == "phase" and np.ndim(fused_signals) == 2 and roi_src == "All RX at selected bin"
                else None)
        # and the heart from those receivers and the neighbouring bins' (as in the Vital
        # Signs view: the chest spans more than one bin)
        heart_d = None
        if rx_d is not None and self.fusion_method_var.get() == "weighted displacement":
            import vitalsigns
            hw = vitalsigns.HEART_BINS_HW
            lo, hi = max(0, best_bin - hw), min(n_bins, best_bin + hw + 1)
            heart_d = vitalsigns.heart_channels(slow_cube[:, :, lo:hi], best_bin - lo, wavelength,
                                                half=hw).mean(axis=0)
        trends = self.analysis.estimate_trends(disp, fps, win_len_s, overlap, rx_disps=rx_d,
                                               heart_disp=heart_d)
        
        # Calculate final HR/RR for the fused signal (Spectra/Overview unification)
        # segments of 30 s (or the whole recording): 1024 samples were 10 s at 100 fps,
        # bins of 6 /min for the breathing rate
        seg = int(min(len(disp), max(256, 30 * fps)))
        f_s, p_s = signal.welch(disp, fps, nperseg=seg, nfft=max(8192, 1 << (seg - 1).bit_length()))
        hr_f, hr_bpm, _ = self.analysis.find_best_peak(f_s, p_s, bands.heart_hz)
        rr_f, rr_bpm, _ = self.analysis.find_best_peak(f_s, p_s, bands.resp_hz)
        # the heart rate reported: where the heart track ran (its valid windows), not the
        # largest peak of the whole recording's spectrum
        valid_hr = [w['hr_bpm'] for w in trends if w['valid']]
        if valid_hr:
            hr_bpm = float(np.median(valid_hr))
            hr_f = hr_bpm / 60.0
        # and the breathing rate the median of the windows that count (the receivers' spectra
        # summed): the peak of the whole recording was 9.9 and 9.4 /min on two recordings
        # whose reference went 6.3 and 6.2, these medians 5.9 and 7.4
        if trends:
            rr_ok = [w['rr_bpm'] for w in trends if w.get('rr_valid', True)]
            rr_bpm = float(np.median(rr_ok or [w['rr_bpm'] for w in trends]))
            rr_f = rr_bpm / 60.0
        
        # Temporal Map Logic - calculate AI score per window for each candidate bin
        win_samples = int(win_len_s * fps)
        if win_samples <= 0 or win_samples > n_frames:
            win_samples = n_frames      # one window, as estimate_trends makes it (none here before)
        step = max(1, int(win_samples * (1 - overlap / 100)))
        temp_map = []
        t_axis = []
        
        print(f"[AI] Calculating temporal quality map for {len(candidates)} candidates...")
        for start in range(0, n_frames - win_samples + 1, step):
            win_scores = []
            for rbin in candidates:
                # Use pre-selected best RX for this bin to save time, or max across all RX
                # Let's take the max AI score across RX for this specific window
                max_win_score = -10.0
                for rx in range(n_rx):
                    iq_seg = slow_cube[start:start+win_samples, rx, rbin]
                    # Quick feature extraction for scoring (simplified)
                    f_win = self.analysis.compute_features(iq_seg, fps, phase_method=phase_method)
                    s_win = self.analysis.compute_ai_score(f_win, weight_vals)
                    if s_win > max_win_score:
                        max_win_score = s_win
                win_scores.append(max_win_score)
            temp_map.append(win_scores)
            t_axis.append((start + win_samples/2) / fps)
        
        temp_map = np.array(temp_map)

        # for Use Selected Windows: every candidate with its neighbours, (frames, RX, bins)
        # and the candidate's index in them
        import vitalsigns
        hw = vitalsigns.HEART_BINS_HW
        self.last_cand_iq = {int(c): (slow_cube[:, :, max(0, c - hw):min(n_bins, c + hw + 1)],
                                      int(c - max(0, c - hw))) for c in candidates}
        self.last_ai_result = {
            'feature_list': sorted(feature_list, key=lambda x: x['ai_score'], reverse=True),
            'best_bin': best_bin, 'best_rx': best_rx, 'best_range_m': best_f['range_m'],
            'roi_bins': roi_bins, 'roi_rxs': roi_rxs, 'roi_weights': fused_w,
            'range_profile': range_profile, 'bin_rx_scores': bin_rx_scores,
            'phase': phase, 'disp': disp, 'resp_sig': resp_sig, 'card_sig': card_sig,
            'window_results': trends, 'fps': fps, 'r_axis': r_axis, 'wavelength': wavelength,
            'temp_map': temp_map, 't_axis_temp': np.array(t_axis), 'candidates': candidates,
            'win_len_s': win_len_s, 'overlap_pct': overlap,
            'hr_bpm': hr_bpm, 'rr_bpm': rr_bpm
        }
        self._update_plots()

    def export(self, fmt):
        """
        Export Results (JSON): everything the last analysis produced. Export Features
        (CSV): one row per range bin and RX that was scored. The buttons called this
        method before it existed, so both did nothing.
        """
        if not self.last_ai_result:
            messagebox.showwarning("Export", "Run the AI analysis first.")
            return
        ext = ".json" if fmt == "json" else ".csv"
        path = filedialog.asksaveasfilename(defaultextension=ext,
                                            filetypes=[(ext[1:].upper() + " files", "*" + ext)])
        if not path:
            return
        try:
            if fmt == "json":
                self.analysis.export_results(self.last_ai_result, path, fmt="json")
            else:
                rows = self.last_ai_result.get("feature_list") or []
                if not rows:
                    messagebox.showwarning("Export", "The analysis produced no features.")
                    return
                keys = list(rows[0].keys())
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                    w.writeheader()
                    for r in rows:
                        w.writerow({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                                    for k, v in r.items()})
            messagebox.showinfo("Export", f"Saved to {path}")
        except Exception as e:
            messagebox.showerror("Export", f"Export failed: {e}")

    def load_reference(self, kind="ppg"):
        path = filedialog.askopenfilename(filetypes=[("Data files", "*.csv *.txt"), ("All", "*.*")])
        if not path: return
        
        try:
            previous_ppg = self.analysis.reference_data
            default_fs = 4.0 if kind == "pulse" else 128.0
            if self.analysis.load_reference_txt_csv(path, config={"fs": default_fs}, signal_kind=kind):
                loaded = self.analysis.reference_data
                if kind == "pulse":
                    self.pulse_data_path = path
                    self.analysis.pulse_data = loaded
                    self.analysis.reference_data = previous_ppg
                    if previous_ppg is None:
                        self.ref_info_var.set(
                            f"Pulse: {os.path.basename(path)}\nFS: {loaded['fs']:.1f} Hz\n"
                            f"Samples: {len(loaded['signal'])}\nLoad PPG before processing."
                        )
                else:
                    self.ref_data_path = path
                    self.analysis.reference_data = loaded
                    self.analysis.process_reference_ppg(
                        invert=self.ref_invert_var.get(),
                        window_s=self.win_len_var.get(),
                        overlap_pct=self.overlap_var.get()
                    )

                ref = self.analysis.reference_data
                if ref is not None:
                    pulse_text = "not loaded"
                    if self.analysis.pulse_data is not None:
                        pulse_text = f"{os.path.basename(self.pulse_data_path)} ({len(self.analysis.pulse_data['signal'])} samples)"
                    self.ref_info_var.set(
                        f"PPG: {os.path.basename(self.ref_data_path) if self.ref_data_path else 'not loaded'}\n"
                        f"FS: {ref['fs']:.1f} Hz | Samples: {len(ref['time'])}\n"
                        f"Absolute PPG time: {'Yes' if ref.get('has_absolute_timestamps') else 'No'}\n"
                        f"Pulse: {pulse_text}"
                    )

                self._update_reference_plots()
                for i in range(self.notebook.index("end")):
                    if self.notebook.tab(i, "text") == "Reference PPG":
                        self.notebook.select(i)
                        break
            else:
                messagebox.showerror("Error", f"Could not load the file: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Processing error: {e}")

    def _update_reference_plots(self):
        """Aktualizace speciální záložky pro Reference PPG."""
        if not self.analysis.reference_data: return
        ref = self.analysis.reference_data
        
        fig = self.plots["Reference PPG"]["fig"]
        fig.clear()
        
        ppg_time = np.asarray(ref.get('time_sync', ref['time']))
        # 1. Raw PPG on the radar-relative (or local, before sync) time axis.
        ax1 = fig.add_subplot(511)
        ax1.plot(ppg_time, ref['signal'], 'k', alpha=0.6, label="Raw Pleth/PPG")
        ax1.set_title(f"Reference Signal: {ref.get('signal_type', 'N/A')} ({os.path.basename(ref['path'])})")
        ax1.legend(loc='upper right')
        
        # 2. Filtered PPG + Peaks
        ax2 = fig.add_subplot(512, sharex=ax1)
        if ref.get('ppg_clean') is not None:
            ax2.plot(ppg_time, ref['ppg_clean'], 'b', label="Filtered PPG")
            if ref.get('peaks') is not None:
                peaks = np.asarray(ref['peaks'], dtype=int)
                peaks = peaks[peaks < len(ppg_time)]
                ax2.plot(ppg_time[peaks], ref['ppg_clean'][peaks], 'ro', markersize=3, label="Systolic Peaks")
        ax2.set_title("Processed Waveform & Peak Detection")
        ax2.legend(loc='upper right')
        
        # 3. HR Trend
        ax3 = fig.add_subplot(513, sharex=ax1)
        if ref.get('hr_calc') is not None and len(ref['hr_calc']) > 0:
            time_hr = np.asarray(ref.get('time_hr_sync', ref['time_hr']))
            ax3.plot(time_hr, ref['hr_calc'], 'r-o', markersize=4, label="PPG HR Trend")
            avg_hr = np.nanmean(ref['hr_calc'])
            ax3.axhline(avg_hr, color='r', ls='--', alpha=0.5, label=f"Avg: {avg_hr:.1f}")
        ax3.set_ylabel("HR [BPM]")
        ax3.set_ylim(40, 180)
        ax3.legend(loc='upper right')
        
        # 4. Pulse reference on the same radar-relative time axis.
        ax4 = fig.add_subplot(514, sharex=ax1)
        pulse = self.analysis.pulse_data
        if pulse is not None and len(pulse.get('hr_ref', [])):
            pulse_time = np.asarray(pulse.get('time_sync', pulse.get('time', [])))
            ax4.plot(pulse_time, pulse['hr_ref'], 'm.-', label="Pulse reference HR")
            if len(pulse.get('radar_time_s', [])):
                ax4.plot(pulse['radar_time_s'], pulse['radar_values'], 'c-', alpha=0.7, label="Pulse interpolated to radar")
        ax4.set_ylabel("Pulse [BPM]")
        ax4.legend(loc='upper right')

        # 5. SQI / Quality
        ax5 = fig.add_subplot(515, sharex=ax1)
        if ref.get('sqi') is not None and len(ref['sqi']) > 0:
            time_hr = np.asarray(ref.get('time_hr_sync', ref['time_hr']))
            ax5.fill_between(time_hr, 0, ref['sqi'], color='g', alpha=0.3, label="SQI (Signal Quality)")
            ax5.set_ylim(0, 1.1)
        ax5.set_xlabel("Seconds from radar start")
        ax5.set_ylabel("Quality")
        ax5.legend(loc='upper right')
        
        fig.tight_layout()
        self._refresh_interactivity("Reference PPG", [ax1, ax2, ax3, ax4, ax5])

    def sync_and_compare(self):
        if not self.analysis.reference_data:
            messagebox.showwarning("Warning", "Load PPG reference data first.")
            return
        
        if not self.last_ai_result:
            # Pokud nemáme radarová data, přesto můžeme zkusit přepočítat PPG (pokud uživatel změnil parametry)
            self.analysis.process_reference_ppg(
                invert=self.ref_invert_var.get(),
                window_s=self.win_len_var.get(),
                overlap_pct=self.overlap_var.get()
            )
            self._update_reference_plots()
            messagebox.showinfo("Info", "Reference PPG is processed, but radar analysis is not available yet.")
            return
        
        radar_data = self.get_data_func() or {}
        radar_params = radar_data.get('params', {})
        radar_start = radar_data.get('radar_start_timestamp', radar_params.get('radar_start_timestamp'))
        radar_time = np.arange(len(self.last_ai_result['disp']), dtype=float) / self.last_ai_result['fps']
        radar_duration = float(radar_time[-1]) if len(radar_time) else 0.0
        try:
            sync = self.analysis.synchronize_reference_with_radar(
                radar_start, radar_duration, radar_time_s=radar_time,
                manual_offset=float(self.ref_offset_var.get())
            )
            self.analysis.process_reference_ppg(
                invert=self.ref_invert_var.get(),
                window_s=self.win_len_var.get(),
                overlap_pct=self.overlap_var.get()
            )
            self.analysis.finalize_reference_processing_sync()
            self.ref_offset_var.set(round(float(self.ref_offset_var.get()), 3))
            self.sync_info_var.set(
                f"Radar start: {radar_params.get('radar_start_time', 'timestamp loaded')}\n"
                f"Common interval: {sync['common_start_s']:.3f}–{sync['common_end_s']:.3f} s "
                f"({sync['common_duration_s']:.3f} s)\n"
                f"Manual correction: {sync['manual_offset_s']:+.3f} s"
            )
        except Exception as exc:
            self.sync_info_var.set("Synchronization failed")
            messagebox.showerror("Synchronization", str(exc))
            return
        self._update_plots()

    def run_synthetic_test(self):
        """A chest at 1.5 m (72 bpm, 16 /min) seen by four receivers, the antenna coupling
        at 0 m and noise everywhere, analysed as a recording; the Log says what was found."""
        fps, dur, n_rx, n_bins, chest = 20.0, 60.0, 4, 64, 15
        hr_true, rr_true = 72, 16
        rng = np.random.default_rng(0)
        n = int(round(fps * dur))
        cube = 0.02 * (rng.standard_normal((n, 1, n_rx, n_bins)) + 1j * rng.standard_normal((n, 1, n_rx, n_bins)))
        cube[:, 0, :, 0] += 30.0
        for rx in range(n_rx):
            cube[:, 0, rx, chest] += self.analysis.get_synthetic_signal(fps, dur, hr_bpm=hr_true, rr_bpm=rr_true,
                                                                        wavelength=299792458 / 60e9, rng=rng)
        fake_data = {'rfft_cube': cube, 'fps': fps, 'range_axis': np.arange(n_bins) * 0.1,
                     'params': {'startFreqGHz': 60.0}}
        old_get = self.get_data_func; self.get_data_func = lambda: fake_data
        try:
            self.run_ai_analysis()
        finally:
            self.get_data_func = old_get
        res = self.last_ai_result
        if res:
            print(f"[AI] Synthetic test: {hr_true} bpm and {rr_true} /min generated at {chest * 0.1:.1f} m, "
                  f"found {res['hr_bpm']:.1f} bpm and {res['rr_bpm']:.1f} /min at {res['best_range_m']:.1f} m")

    def _update_plots(self):
        if not self.last_ai_result: 
            # Pokud není AI výsledek, zkusíme alespoň aktualizovat referenční grafy
            self._update_reference_plots()
            return
        
        # Vždy aktualizovat referenční grafy, pokud máme data
        self._update_reference_plots()
        
        res = self.last_ai_result
        t = np.arange(len(res['disp'])) / res['fps']
        sync = getattr(self.analysis, "reference_sync", None) or {}
        common_start = float(sync.get("common_start_s", 0.0))
        common_end = float(sync.get("common_end_s", t[-1] if len(t) else 0.0))
        radar_common_mask = (t >= common_start) & (t <= common_end)
        t_common = t[radar_common_mask]
        
        # Overview (normalizované křivky v jedněch osách -- bez překrytého twinx)
        fig = self.plots["Overview"]["fig"]; fig.clear()
        ax1 = fig.add_subplot(211)
        # both scaled to 0..1, the score only where a bin was scored (the others hold -100,
        # which drew a line along zero over the whole range)
        def _unit(v):
            v = np.asarray(v, float)
            lo, hi = np.nanmin(v), np.nanmax(v)
            return (v - lo) / (hi - lo) if hi > lo else np.ones_like(v)
        score = np.max(res['bin_rx_scores'], axis=1).astype(float)
        scored = score > -100.0
        score_n = np.full(score.shape, np.nan)
        if scored.any():
            score_n[scored] = _unit(score[scored])
        ax1.plot(res['r_axis'], score_n, 'b-o', label="AI Score (norm.)")
        ax1.plot(res['r_axis'], _unit(res['range_profile']), 'k-', alpha=0.5, label="Range Profile (norm.)")
        ax1.axvline(res['best_range_m'], color='r', ls='--', label="Best bin")
        ax1.set_xlabel("Range [m]"); ax1.set_ylabel("Normalized value")
        ax1.set_title("AI Score vs Range Profile (normalized signals)")
        ax1.legend(loc='upper right', fontsize=8)
        ax3 = fig.add_subplot(212); ax3.axis('off')
        # the features of the bin and receiver chosen (with Auto Bin not always the top score)
        f = next((x for x in res['feature_list']
                  if x['range_bin'] == res['best_bin'] and x['rx'] == res['best_rx']), res['feature_list'][0])
        summary = f"Best: {res['best_bin']} ({res['best_range_m']:.2f} m) RX{res['best_rx']}\nScore: {f['ai_score']:.2f} | HR SNR: {f['hr_snr']:.1f} dB\nRR: {res['rr_bpm']:.1f} BPM | HR: {res['hr_bpm']:.1f} BPM"
        ax3.text(0.5, 0.5, summary, ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        self._refresh_interactivity("Overview")

        # Signals (radar cardiac + reference PPG jako dvě normalizované křivky v JEDNOM grafu)
        fig = self.plots["Signals"]["fig"]; fig.clear()
        ax1 = fig.add_subplot(211)
        ax1.plot(t_common, np.asarray(res['resp_sig'])[radar_common_mask], 'g', label="Radar respiration")
        ax1.set_title("Radar Mechanical Respiration waveform")
        ax1.legend(loc='upper right', fontsize=8)
        ax2 = fig.add_subplot(212, sharex=ax1)

        ref_data = self.analysis.reference_data
        has_ref = bool(ref_data) and len(ref_data.get('signal', [])) > 0
        if has_ref:
            ax2.plot(t_common, normalize_for_plot(np.asarray(res['card_sig'])[radar_common_mask]), 'r', label="Radar cardiac (norm.)")
            # Použijeme ppg_clean pokud existuje, jinak raw
            sig_p = ref_data.get('ppg_clean')
            if sig_p is None: sig_p = ref_data['signal']
            # sync_time pokud existuje
            t_ref = np.asarray(ref_data.get('time_sync', ref_data['time']))
            sig_p = normalize_for_plot(sig_p)
            n = min(len(t_ref), len(sig_p))
            ref_mask = (t_ref[:n] >= common_start) & (t_ref[:n] <= common_end)
            ax2.plot(t_ref[:n][ref_mask], sig_p[:n][ref_mask], 'b', alpha=0.7, label="Reference PPG (norm.)")
            if ref_data.get('peaks') is not None:
                p_idx = np.asarray(ref_data['peaks'])
                p_idx = p_idx[p_idx < n]
                ax2.plot(t_ref[p_idx], sig_p[p_idx], 'ro', markersize=3, label="PPG peaks")
            pulse = self.analysis.pulse_data
            if pulse is not None and len(pulse.get('radar_time_s', [])):
                pulse_ax = ax2.twinx()
                pulse_ax.plot(pulse['radar_time_s'], pulse['radar_values'], 'm-', alpha=0.7, label="Pulse reference")
                pulse_ax.set_ylabel("Pulse [BPM]")
                pulse_ax.set_ylim(30, 220)
            ax2.set_title("Radar cardiac vs Reference PPG (normalized signals)")
        else:
            ax2.plot(t_common, np.asarray(res['card_sig'])[radar_common_mask], 'r', label="Radar cardiac")
            ax2.set_title("Radar mechanical cardiac waveform")
        ax2.legend(loc='upper right', fontsize=8)

        fig.tight_layout()
        self._refresh_interactivity("Signals", [ax1, ax2])

        # Spectra
        fig = self.plots["Spectra"]["fig"]; fig.clear(); ax = fig.add_subplot(111)
        seg = int(min(len(res['disp']), max(256, 30 * res['fps'])))     # as run_ai_analysis
        f_s, p_s = signal.welch(res['disp'], res['fps'], nperseg=seg,
                                nfft=max(8192, 1 << (seg - 1).bit_length()))
        bands = vital_bands.current()
        ax.semilogy(f_s, p_s); ax.axvspan(*bands.resp_hz, color='g', alpha=0.1); ax.axvspan(*bands.heart_hz, color='r', alpha=0.1)
        
        # The rates reported (the heart: where its track ran), marked on this spectrum
        hr_bpm, rr_bpm = res['hr_bpm'], res['rr_bpm']
        hr_f, rr_f = hr_bpm / 60.0, rr_bpm / 60.0
        
        if hr_f > 0:
            ax.plot(hr_f, p_s[np.argmin(np.abs(f_s - hr_f))], 'ro')
            ax.annotate(f"HR: {hr_bpm:.1f} BPM\n({hr_f:.2f} Hz)", (hr_f, p_s[np.argmin(np.abs(f_s - hr_f))]), 
                        xytext=(10, 10), textcoords='offset points', color='red', fontweight='bold')
        if rr_f > 0:
            ax.plot(rr_f, p_s[np.argmin(np.abs(f_s - rr_f))], 'go')
            ax.annotate(f"RR: {rr_bpm:.1f} BPM\n({rr_f:.2f} Hz)", (rr_f, p_s[np.argmin(np.abs(f_s - rr_f))]), 
                        xytext=(10, -20), textcoords='offset points', color='green', fontweight='bold')

        ax.set_xlim(0, 5); ax.set_xlabel("Frequency [Hz]")
        vis = p_s[(f_s >= 0) & (f_s <= 5)]                  # the y range of what is shown
        if vis.size and np.max(vis) > 0:
            ax.set_ylim(max(np.min(vis[vis > 0]) * 0.5, np.max(vis) * 1e-6), np.max(vis) * 3)
        self._refresh_interactivity("Spectra")

        # Spatial Map
        fig = self.plots["Spatial Map"]["fig"]; fig.clear(); ax = fig.add_subplot(111)
        # the scored bins only (the others hold -100 and took the whole colour scale), pixels
        # centred on their ranges, the view around them
        r_ax = np.asarray(res['r_axis'], float)
        dr = r_ax[1] - r_ax[0] if len(r_ax) > 1 else 1.0
        scores = np.ma.masked_less_equal(res['bin_rx_scores'], -100.0)
        im = ax.imshow(scores.T, aspect='auto', origin='lower', extent=[r_ax[0] - dr / 2, r_ax[-1] + dr / 2, -0.5, res['bin_rx_scores'].shape[1]-0.5])
        fig.colorbar(im, ax=ax, label="AI Score"); ax.set_xlabel("Range [m]"); ax.set_ylabel("RX Channel")
        cand_r = r_ax[np.asarray(res['candidates'], int)] if len(res['candidates']) else r_ax[:0]
        if cand_r.size:
            ax.set_xlim(cand_r.min() - max(0.3, 2 * dr), cand_r.max() + max(0.3, 2 * dr))
        self._refresh_interactivity("Spatial Map")

        # Temporal Map
        fig = self.plots["Temporal Map"]["fig"]; fig.clear()
        if 'temp_map' in res and res['temp_map'].size > 0:
            ax1 = fig.add_subplot(211)
            candidates_m = [res['r_axis'][c] for c in res['candidates']]
            t_axis = res['t_axis_temp']
            im = ax1.imshow(res['temp_map'].T, aspect='auto', origin='lower', 
                          extent=[t_axis[0], t_axis[-1], candidates_m[0], candidates_m[-1]],
                          cmap='magma')
            fig.colorbar(im, ax=ax1, label="AI Score")
            ax1.set_xlabel("Time [s]"); ax1.set_ylabel("Range [m]"); ax1.set_title("Temporal AI Quality Map")
            
            # Overlay selected windows
            dt = t_axis[1] - t_axis[0] if len(t_axis) > 1 else 1.0
            dr = candidates_m[1] - candidates_m[0] if len(candidates_m) > 1 else 0.1
            for (w_idx, c_idx) in self.selected_window_indices:
                rect = patches.Rectangle((t_axis[w_idx] - dt/2, candidates_m[c_idx] - dr/2), dt, dr, 
                                        linewidth=2, edgecolor='cyan', facecolor='none')
                ax1.add_patch(rect)
                
            # Selected RX trend (simplified: just showing the best RX for the fused segment)
            ax2 = fig.add_subplot(212)
            ax2.axis('off')
            # In a full impl we'd store selected RX per window. For now, we show the ROI RXs.
            # (in the axes' own coordinates: on the shared time axis 0.5 s lay left of it)
            ax2.text(0.5, 0.5, f"Selected ROI Bins: {res['roi_bins']}\nSelected ROI RXs: {res['roi_rxs']}",
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("ROI Selection Info")
            
        fig.tight_layout()
        self._refresh_interactivity("Temporal Map")

        # Features Table
        f_tab = self.plots["Features"]["fig"].canvas.get_tk_widget().master
        for c in f_tab.winfo_children():
            if isinstance(c, ttk.Treeview): c.destroy()
        tree = ttk.Treeview(f_tab, columns=("Bin", "RX", "Score", "HR SNR", "Motion"), show='headings')
        for col in tree["columns"]: tree.heading(col, text=col); tree.column(col, width=60)
        for f in res['feature_list'][:30]: tree.insert("", "end", values=(f['range_bin'], f['rx'], f"{f['ai_score']:.2f}", f"{f['hr_snr']:.1f}", f"{f['motion_score']:.3f}"))
        tree.pack(fill="both", expand=True)

        # Trends (radar HR + reference PPG HR jako dvě křivky v jednom grafu HR [BPM])
        fig = self.plots["Trends"]["fig"]; fig.clear()
        # dots: the windows that count (as in the Vitals view), rings: the others
        wr = res['window_results']
        t_tr = np.array([w['t_s'] for w in wr]); hr_tr = np.array([w['hr_bpm'] for w in wr]); rr_tr = np.array([w['rr_bpm'] for w in wr])
        hr_ok = np.array([bool(w['valid']) for w in wr], dtype=bool)
        rr_ok = np.array([bool(w.get('rr_valid', True)) for w in wr], dtype=bool)
        ax1 = fig.add_subplot(211); ax1.plot(t_tr, rr_tr, 'g-', alpha=0.4)
        ax1.plot(t_tr[rr_ok], rr_tr[rr_ok], 'go', label="Radar RR")
        ax1.plot(t_tr[~rr_ok], rr_tr[~rr_ok], 'o', mfc='none', mec='g', label="Radar RR, not counted")
        ax1.set_ylabel("RR [BPM]")
        ax1.legend(loc='upper right', fontsize=8)
        ax2 = fig.add_subplot(212, sharex=ax1)
        ax2.plot(t_tr, hr_tr, 'r-', alpha=0.4)
        ax2.plot(t_tr[hr_ok], hr_tr[hr_ok], 'ro', label="Radar HR")
        ax2.plot(t_tr[~hr_ok], hr_tr[~hr_ok], 'o', mfc='none', mec='r', label="Radar HR, not counted")
        if ref_data and ref_data.get('hr_calc') is not None and len(ref_data['hr_calc']) > 0:
            t_hr_ref = ref_data.get('time_hr_sync')
            if t_hr_ref is None: t_hr_ref = ref_data.get('time_hr')
            if t_hr_ref is not None:
                ax2.plot(t_hr_ref, ref_data['hr_calc'], 'b--s', markersize=3, alpha=0.7, label="Reference PPG HR")
        ax2.set_ylabel("HR [BPM]"); ax2.set_xlabel("Time [s]")
        ax2.legend(loc='upper right', fontsize=8)
        self._refresh_interactivity("Trends", [ax1, ax2])

        # Comparison
        fig = self.plots["Comparison"]["fig"]; fig.clear()
        ref_data = self.analysis.reference_data
        comparison_time_axes = []
        
        if ref_data and 'time_sync' in ref_data:
            ax1 = fig.add_subplot(211)
            ax1.plot(t_tr, hr_tr, 'r-o', label="Radar-derived HR")
            if ref_data.get('hr_ref') is not None and len(ref_data['hr_ref']) > 0:
                ax1.plot(ref_data['time_sync'], ref_data['hr_ref'], 'k--', label="Reference HR (File)")
            if ref_data.get('hr_calc') is not None and len(ref_data['hr_calc']) > 0:
                ax1.plot(ref_data['time_hr_sync'], ref_data['hr_calc'], 'b--', label="Reference PPG HR (Calculated)")
            ax1.legend(); ax1.set_title("Radar-derived HR vs Reference"); ax1.set_ylabel("HR [BPM]")
            comparison_time_axes = [ax1]
            
            comp = self.analysis.compare_results(np.array(t_tr), np.array(hr_tr), np.array([w['valid'] for w in res['window_results']]))
            if comp:
                # Scatter + Bland-Altman
                ax2 = fig.add_subplot(223)
                ax2.scatter(comp['ref_hr'][comp['mask']], comp['radar_hr'][comp['mask']], alpha=0.5)
                ax2.plot([40, 160], [40, 160], 'k--'); ax2.set_xlabel("Ref HR"); ax2.set_ylabel("Radar HR")
                ax2.set_title(f"Correlation: {comp['corr']:.3f}")
                
                ax3 = fig.add_subplot(224)
                ax3.scatter(comp['ba_mean'], comp['ba_diff'], alpha=0.5, color='purple')
                ax3.axhline(comp['bias'], color='r', linestyle='-', label=f"Bias: {comp['bias']:.1f}")
                ax3.axhline(comp['bias'] + 1.96*comp['std_err'], color='r', linestyle='--', alpha=0.5)
                ax3.axhline(comp['bias'] - 1.96*comp['std_err'], color='r', linestyle='--', alpha=0.5)
                ax3.set_xlabel("Mean HR"); ax3.set_ylabel("Difference HR")
                ax3.legend(fontsize=8)
            
            self._refresh_interactivity("Comparison", comparison_time_axes)
        
        fig.tight_layout()
        self.plots["Overview"]["canvas"].draw_idle()
        self.plots["Signals"]["canvas"].draw_idle()
        self.plots["Spectra"]["canvas"].draw_idle()
        self.plots["Spatial Map"]["canvas"].draw_idle()
        self.plots["Temporal Map"]["canvas"].draw_idle()
        self.plots["Trends"]["canvas"].draw_idle()
        self.plots["Comparison"]["canvas"].draw_idle()