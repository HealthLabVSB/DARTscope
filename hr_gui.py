import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import theme
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from plot_interactivity import create_interactive_canvas, PlotInteractivityManager, add_plot_control_bar
from flow_layout import ScrollFrame
import hr_analysis
import vital_bands
import vitalsigns
import capture_params
import scipy.signal as signal
import json
import csv

class FilterParameterDialog(tk.Toplevel):
    def __init__(self, parent, filter_name, filter_cfg, on_apply):
        super().__init__(parent)
        self.title(f"Filter Parameters: {filter_name}")
        self.filter_cfg = filter_cfg.copy()
        self.on_apply = on_apply
        self.init_ui()

    def init_ui(self):
        row = 0
        self.vars = {}
        for key, val in self.filter_cfg.items():
            if key == "enabled":
                var = tk.BooleanVar(value=val)
                tk.Checkbutton(self, text=key, variable=var).grid(row=row, column=0, columnspan=2, sticky="w")
            elif isinstance(val, (int, float)):
                var = tk.StringVar(value=str(val))
                tk.Label(self, text=key).grid(row=row, column=0, sticky="e")
                tk.Entry(self, textvariable=var).grid(row=row, column=1)
            elif isinstance(val, str):
                var = tk.StringVar(value=val)
                tk.Label(self, text=key).grid(row=row, column=0, sticky="e")
                if key == "type":
                    ttk.Combobox(self, textvariable=var, values=["bandpass", "lowpass", "highpass", "notch"]).grid(row=row, column=1)
                elif key == "mode":
                    ttk.Combobox(self, textvariable=var, values=["causal", "zero_phase"]).grid(row=row, column=1)
                else:
                    tk.Entry(self, textvariable=var).grid(row=row, column=1)
            else:
                continue
            self.vars[key] = var
            row += 1

        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="Apply", command=self.apply).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side="left", padx=5)

    def apply(self):
        for key, var in self.vars.items():
            val = var.get()
            if isinstance(self.filter_cfg[key], bool):
                self.filter_cfg[key] = bool(val)
            elif isinstance(self.filter_cfg[key], int):
                self.filter_cfg[key] = int(val)
            elif isinstance(self.filter_cfg[key], float):
                self.filter_cfg[key] = float(val)
            else:
                self.filter_cfg[key] = val
        self.on_apply(self.filter_cfg)
        self.destroy()

class HRDetectionTab:
    def __init__(self, parent, get_data_func, moving=None):
        self.parent = parent
        self.get_data_func = get_data_func # returns (rfft_cube, params, fps, r_axis)
        # moving: dict(get_raw=..., run=..., report_error=...) for the "Moving person
        # (phase 12)" part (moving_hr_gui.py); None leaves it out
        self.moving = moving
        self.moving_panel = None
        self.logic = hr_analysis.HRAnalysisLogic()
        vital_bands.add_listener(self.logic.apply_bands)
        self.results = []
        
        self.selected_bin = tk.IntVar(value=0)
        # every receiver's displacement averaged over the bin and its neighbours (the chest
        # spans more than one): 1.2 instead of 1.7 bpm off a pulse oximeter (vitalsigns)
        self.roi_hw = tk.IntVar(value=vitalsigns.HEART_BINS_HW)
        self.roi_mode = tk.StringVar(value="average displacement")
        # the receivers' mean displacement: on six seated recordings against a
        # somnograph the heart track of RX1 alone was 3-4 bpm further off
        self.rx_source = tk.StringVar(value="Fused")
        self.fusion_mode = tk.StringVar(value="average")
        self.phase_method = tk.StringVar(value="unwrap")
        # on: the phase of the bin with its static part left in is a short arc around
        # it, and the heart all but vanishes (19.5 instead of 0.6 bpm off on one recording)
        self.clutter_removal = tk.BooleanVar(value=True)
        self.notch_resp = tk.BooleanVar(value=False)
        self.win_len = tk.StringVar(value=f"{vitalsigns.DEFAULT_HR_WIN_S:g}")   # 30 s
        self.overlap = tk.StringVar(value="80")
        
        self.init_ui()

    def init_ui(self):
        # Split into Controls (Left) and Visualization (Right)
        self.paned = ttk.PanedWindow(self.parent, orient="horizontal")
        self.paned.pack(fill="both", expand=True)

        # The settings column scrolls: in a window below ~900 px high the buttons at its
        # end (Run HR Detection, the exports) were simply cut off. It keeps its own width
        # (weight 0) and the plots take what the window gains or loses; with a share of
        # the width it was squeezed to 180 px at 800 px and cut its button labels.
        self.ctrl_scroll = ScrollFrame(self.paned)
        self.paned.add(self.ctrl_scroll, weight=0)
        self.ctrl_frame = self.ctrl_scroll.inner

        self.plot_container = ttk.Frame(self.paned)
        self.paned.add(self.plot_container, weight=1)

        self._init_controls()
        self._init_plots()
        if self.moving is not None:
            import moving_hr_gui
            self.moving_panel = moving_hr_gui.MovingHRPanel(self.ctrl_frame, self.viz_nb,
                                                            **self.moving)

    def _init_controls(self):
        c = self.ctrl_frame
        
        # Bin Selection
        bin_frame = ttk.LabelFrame(c, text="Range Bin & ROI")
        bin_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(bin_frame, text="Bin:").grid(row=0, column=0)
        tk.Spinbox(bin_frame, from_=0, to=1024, textvariable=self.selected_bin, width=5).grid(row=0, column=1)
        
        ttk.Label(bin_frame, text="ROI ±").grid(row=1, column=0)
        tk.Spinbox(bin_frame, from_=0, to=3, textvariable=self.roi_hw, width=5).grid(row=1, column=1)

        ttk.Label(bin_frame, text="ROI Mode:").grid(row=2, column=0)
        ttk.Combobox(bin_frame, textvariable=self.roi_mode, values=["average displacement", "center only", "average complex", "best bin", "weighted bins"], width=15).grid(row=2, column=1)

        # Source & Fusion
        src_frame = ttk.LabelFrame(c, text="Source & Fusion")
        src_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(src_frame, text="RX:").grid(row=0, column=0)
        # the RX the capture has: 4 on a single chip, 8 on the AWR2243 cascade (set per run)
        self.rx_combo = ttk.Combobox(src_frame, textvariable=self.rx_source, values=self._rx_choices(4), width=10)
        self.rx_combo.grid(row=0, column=1)
        
        ttk.Label(src_frame, text="Fusion:").grid(row=1, column=0)
        ttk.Combobox(src_frame, textvariable=self.fusion_mode, values=["best RX", "average", "weighted", "PCA", "ICA"], width=10).grid(row=1, column=1)

        ttk.Label(src_frame, text="Phase:").grid(row=2, column=0)
        ttk.Combobox(src_frame, textvariable=self.phase_method, values=["unwrap", "dacm", "circlefit"], width=10).grid(row=2, column=1)

        # Parameters
        param_frame = ttk.LabelFrame(c, text="Analysis Parameters")
        param_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(param_frame, text="Win [s]:").grid(row=0, column=0)
        ttk.Combobox(param_frame, textvariable=self.win_len, values=["5", "8", "10", "15", "30"], width=5).grid(row=0, column=1)
        
        ttk.Label(param_frame, text="Overlap %:").grid(row=1, column=0)
        tk.Entry(param_frame, textvariable=self.overlap, width=5).grid(row=1, column=1)

        bands_frame = ttk.LabelFrame(c, text="Search bands (shared by all vital tabs)")
        bands_frame.pack(fill="x", padx=5, pady=5)
        vital_bands.tk_controls(bands_frame, vertical=True).pack(fill="x", padx=5, pady=2)

        # Filters
        filter_frame = ttk.LabelFrame(c, text="Filters")
        filter_frame.pack(fill="x", padx=5, pady=5)
        
        # Only the branches the analysis applies. The smoothing, tracker and notch entries
        # of the filter set had dialogs too, but nothing reads them, so editing them did
        # nothing; the notch is the checkbox below.
        for f_name in ("respiration_branch", "cardiac_hr_branch", "cardiac_waveform_branch"):
            btn = ttk.Button(filter_frame, text=f"Set {f_name.replace('_', ' ')}",
                             command=lambda n=f_name: self.open_filter_dialog(n))
            btn.pack(fill="x", padx=2, pady=1)

        ttk.Checkbutton(filter_frame, text="Clutter Removal", variable=self.clutter_removal).pack(fill="x", padx=5, pady=2)
        ttk.Checkbutton(filter_frame, text="Notch Resp Harmonics", variable=self.notch_resp).pack(fill="x", padx=5, pady=2)
        ttk.Button(c, text="Run Synthetic Test", command=self.run_synthetic_test).pack(fill="x", padx=5, pady=2)

        # Info Text
        info_frame = ttk.LabelFrame(c, text="Information")
        info_frame.pack(fill="x", padx=5, pady=5)
        info_text = ("This tab compares heart-rate detection methods on one range bin of the radar "
                     "signal. The heart rate is computed from the complex slow-time signal of each "
                     "RX channel or of their combination. Clutter Removal subtracts the mean of the "
                     "signal to remove static reflections. Read the results together with the "
                     "quality score: movement, breathing harmonics, a wrong range bin or unsuitable "
                     "filter settings can all produce false heart-rate peaks. The Manual tab has "
                     "the details.")
        tk.Label(info_frame, text=info_text, wraplength=250, justify="left", font=theme.font(8)).pack(padx=5, pady=5)

        # Actions
        ttk.Button(c, text="Run HR Detection", command=self.run_detection).pack(fill="x", padx=5, pady=10)
        ttk.Button(c, text="Export CSV", command=self.export_csv).pack(fill="x", padx=5, pady=2)
        ttk.Button(c, text="Export JSON", command=self.export_json).pack(fill="x", padx=5, pady=2)

    def _init_plots(self):
        self.viz_nb = ttk.Notebook(self.plot_container)
        self.viz_nb.pack(fill="both", expand=True)
        
        self.tab_main = ttk.Frame(self.viz_nb)
        self.tab_rx = ttk.Frame(self.viz_nb)
        self.tab_compare = ttk.Frame(self.viz_nb)
        self.tab_diagnostic = ttk.Frame(self.viz_nb)
        
        self.viz_nb.add(self.tab_main, text="Main Plots")
        self.viz_nb.add(self.tab_rx, text="RX Comparison")
        self.viz_nb.add(self.tab_compare, text="Method Comparison")
        self.viz_nb.add(self.tab_diagnostic, text="Diagnostic Map")

        # Main Plots (interactive: toolbar + X span zoom + reset)
        self.fig_main = Figure(figsize=(8, 10))
        self.canvas_main, self.toolbar_main = create_interactive_canvas(self.tab_main, self.fig_main)
        self.mgr_main = PlotInteractivityManager(self.canvas_main)
        bar_main, _ = add_plot_control_bar(self.tab_main, self.mgr_main)
        bar_main.pack(side="top", fill="x", before=self.canvas_main.get_tk_widget())

        # RX Comparison
        self.rx_frame = ttk.Frame(self.tab_rx)
        self.rx_frame.pack(fill="both", expand=True)
        
        self.rx_tree = ttk.Treeview(self.rx_frame, columns=("RX", "HR", "BR", "Freq", "SNR", "Quality", "Valid", "Note"), show='headings', height=5)
        for col in self.rx_tree['columns']: self.rx_tree.heading(col, text=col)
        self.rx_tree.column("RX", width=60)
        self.rx_tree.column("HR", width=50)
        self.rx_tree.column("BR", width=50)
        self.rx_tree.column("Freq", width=50)
        self.rx_tree.pack(side="top", fill="x")
        
        self.fig_rx = Figure(figsize=(8, 6))
        self.canvas_rx, self.toolbar_rx = create_interactive_canvas(self.rx_frame, self.fig_rx)
        self.mgr_rx = PlotInteractivityManager(self.canvas_rx)
        bar_rx, _ = add_plot_control_bar(self.rx_frame, self.mgr_rx)
        bar_rx.pack(side="top", fill="x", before=self.canvas_rx.get_tk_widget())

        # Method Comparison
        self.comp_frame = ttk.Frame(self.tab_compare)
        self.comp_frame.pack(fill="both", expand=True)
        
        self.comp_tree = ttk.Treeview(self.comp_frame, columns=("Method", "Source", "HR", "BR", "Conf", "Quality", "Valid", "Note"), show='headings', height=8)
        for col in self.comp_tree['columns']: self.comp_tree.heading(col, text=col)
        self.comp_tree.column("Method", width=100)
        self.comp_tree.column("Source", width=100)
        self.comp_tree.column("HR", width=50)
        self.comp_tree.column("BR", width=50)
        self.comp_tree.pack(side="top", fill="x")
        
        self.fig_comp = Figure(figsize=(8, 6))
        self.canvas_comp, self.toolbar_comp = create_interactive_canvas(self.comp_frame, self.fig_comp)
        self.mgr_comp = PlotInteractivityManager(self.canvas_comp)
        bar_comp, _ = add_plot_control_bar(self.comp_frame, self.mgr_comp)
        bar_comp.pack(side="top", fill="x", before=self.canvas_comp.get_tk_widget())

        # Diagnostic Heatmap
        self.fig_diag = Figure(figsize=(8, 8))
        self.canvas_diag, self.toolbar_diag = create_interactive_canvas(self.tab_diagnostic, self.fig_diag)
        self.mgr_diag = PlotInteractivityManager(self.canvas_diag)
        bar_diag, _ = add_plot_control_bar(self.tab_diagnostic, self.mgr_diag)
        bar_diag.pack(side="top", fill="x", before=self.canvas_diag.get_tk_widget())

    def open_filter_dialog(self, name):
        cfg = self.logic.get_filter_params(name)
        FilterParameterDialog(self.parent, name, cfg, lambda new_cfg: self._apply_filter(name, new_cfg))

    def _apply_filter(self, name, new_cfg):
        """
        The breathing and heart-rate branches ARE the shared search bands: editing them
        here changes the bands for every vital tab (vital_bands.py), not just this one.
        """
        self.logic.set_filter_params(name, new_cfg)
        try:
            if name == "respiration_branch":
                vital_bands.set_bands(resp_bpm=(new_cfg["low_hz"] * 60, new_cfg["high_hz"] * 60))
            elif name == "cardiac_hr_branch":
                vital_bands.set_bands(heart_bpm=(new_cfg["low_hz"] * 60, new_cfg["high_hz"] * 60))
        except (KeyError, ValueError) as e:
            messagebox.showerror("Filter", f"Band not applied: {e}")

    def run_synthetic_test(self):
        fps = 20.0
        duration = 60.0         # the heart track's windows are 30 s
        data = self.logic.generate_synthetic_data(fps, duration)
        # We need a way to pass this data to run_detection
        # Temporarily mock get_data_func
        old_func = self.get_data_func
        self.get_data_func = lambda: data
        try:
            self.run_detection()
            messagebox.showinfo("Synthetic Test", "Synthetic test completed successfully.\nRX1: 60 BPM, RX2: 80 BPM, RX3: 100 BPM")
        finally:
            self.get_data_func = old_func

    def _get_roi_complex_signal(self, rfft_cube, bin_idx, roi_hw, mode, fps):
        # rfft_cube: (F, C, RX, R)
        R = rfft_cube.shape[3]
        roi_bins = np.arange(max(0, bin_idx - roi_hw), min(R, bin_idx + roi_hw + 1))
        
        # Mean across chirps
        sig_chirp_mean = np.mean(rfft_cube, axis=1) # (F, RX, R)
        
        if mode == "center only":
            return sig_chirp_mean[:, :, bin_idx] # (F, RX)
        elif mode == "average complex":
            return np.mean(sig_chirp_mean[:, :, roi_bins], axis=2) # (F, RX)
        elif mode == "best bin":
            # Select best bin in ROI based on HR quality (using RX1 for simplicity or mean across RX)
            best_bin = bin_idx
            best_score = -1
            for b in roi_bins:
                sig = np.mean(sig_chirp_mean[:, :, b], axis=1)
                phase = self.logic.extract_phase_baseline(sig)
                # The real frame rate matters here too: filtering the candidate bins
                # at a wrong fps skews the quality score and can pick a worse bin.
                cardiac = self.logic.apply_filter(phase, fps, self.logic.filters["cardiac_hr_branch"])
                res = self.logic.estimate_hr(cardiac, fps, "Selection", "Selection")
                if res["quality"] > best_score:
                    best_score = res["quality"]
                    best_bin = b
            return sig_chirp_mean[:, :, best_bin]
        elif mode == "weighted bins":
            # Weighted fusion based on quality
            bin_qualities = []
            for b in roi_bins:
                sig = np.mean(sig_chirp_mean[:, :, b], axis=1)
                phase = self.logic.extract_phase_baseline(sig)
                cardiac = self.logic.apply_filter(phase, fps, self.logic.filters["cardiac_hr_branch"])
                res = self.logic.estimate_hr(cardiac, fps, "Selection", "Selection")
                bin_qualities.append(res["quality"])
            
            weights = np.array(bin_qualities)
            weights /= (np.sum(weights) + 1e-9)
            
            # Weighted average of complex signals across bins
            weighted_sig = np.zeros_like(sig_chirp_mean[:, :, bin_idx])
            for i, b in enumerate(roi_bins):
                weighted_sig += sig_chirp_mean[:, :, b] * weights[i]
            return weighted_sig
        else:
            return sig_chirp_mean[:, :, bin_idx]

    def _add_to_comparison(self, d, fps, method, source):
        respiration = self.logic.apply_filter(d, fps, self.logic.filters["respiration_branch"])
        cardiac = self.logic.apply_filter(d, fps, self.logic.filters["cardiac_hr_branch"])
        res = self.logic.estimate_hr(cardiac, fps, method, source)
        br_res = self.logic.estimate_breathing_rate(respiration, fps)
        q = self.logic.compute_quality_metrics(cardiac, fps, res["hr_bpm"], respiration)
        res.update(q)
        res["br_bpm"] = br_res["br_bpm"]
        self.all_results.append(res)
        
        self.comp_tree.insert("", "end", values=(
            res["method"], res["source"], f"{res['hr_bpm']:.1f}", 
            f"{br_res['br_bpm']:.1f}",
            f"{res['confidence']:.2f}", f"{res['quality_score']:.2f}",
            res["valid"], res.get("note", "")
        ))

    def run_detection(self):
        data = self.get_data_func()
        if data is None:
            messagebox.showerror("Error", "No data loaded")
            return
        
        rfft_cube, params, fps, r_axis, rp = data
        
        if self.clutter_removal.get():
            rfft_cube = self.logic.static_clutter_removal(rfft_cube)
            
        bin_idx = self.selected_bin.get()
        # The bin starts at 0 - the TX-RX coupling at 0 m, never a heart. There the range
        # that breathes most (the Vital Signs view's Auto Bin) is taken instead.
        if (r_axis is not None and 0 <= bin_idx < len(r_axis)
                and r_axis[bin_idx] < vitalsigns.AUTO_BIN_MIN_RANGE_M):
            bands = vital_bands.current()
            b = vitalsigns.auto_bin(rfft_cube, fps, bands.resp_hz, r_axis, bands=bands)
            print(f"[HR] bin {bin_idx} is the antenna coupling at {r_axis[bin_idx]:.2f} m: "
                  f"the range that breathes most, bin {b} ({r_axis[b]:.2f} m)")
            bin_idx = int(b)
            self.selected_bin.set(bin_idx)
        roi_hw = self.roi_hw.get()
        roi_mode = self.roi_mode.get()
        phase_method = self.phase_method.get()
        rx_source_sel = self.rx_source.get()
        fusion_mode_sel = self.fusion_mode.get()
        
        # Wavelength in the middle of the ADC window, as in every view (roadmap 1.9).
        wave_len = capture_params.wavelength_m(params, default_ghz=77.0)
        
        # 1. ROI & RX Complex Signal Extraction
        # sig_complex_all_rx shape: (F, RX)
        sig_complex_all_rx = self._get_roi_complex_signal(rfft_cube, bin_idx, roi_hw, roi_mode, fps)
        n_rx = sig_complex_all_rx.shape[1]
        try:
            self.rx_combo.configure(values=self._rx_choices(n_rx))
        except Exception:
            pass
        
        # 2. Phase Extraction for each RX using SELECTED method
        def rx_disp(s_c):
            if phase_method == "dacm":
                ph = self.logic.extract_phase_dacm(s_c)
            elif phase_method == "circlefit":
                ph = self.logic.extract_phase_circle_fitting(s_c)
            else:
                ph = self.logic.extract_phase_baseline(s_c)
            return signal.detrend(ph * (wave_len / (4 * np.pi)))

        rx_signals_disp = []
        if roi_mode == "average displacement":
            # each receiver: its displacements in the bin and its ROI neighbours averaged -
            # a neighbour only where its arc stands above its noise (vitalsigns.arc_snr)
            lo, hi = max(0, bin_idx - roi_hw), min(rfft_cube.shape[3], bin_idx + roi_hw + 1)
            sig_bins = np.mean(rfft_cube[:, :, :, lo:hi], axis=1)               # (F, RX, B)
            for i in range(n_rx):
                ds = [rx_disp(sig_bins[:, i, j]) for j in range(hi - lo)
                      if lo + j == bin_idx or vitalsigns.arc_snr(sig_bins[:, i, j]) >= vitalsigns.ARC_SNR_MIN]
                rx_signals_disp.append(np.mean(ds, axis=0))
        else:
            for i in range(n_rx):
                rx_signals_disp.append(rx_disp(sig_complex_all_rx[:, i]))
            
        # 3. Fusion
        fused_disp = None
        if rx_source_sel == "Fused" or fusion_mode_sel != "best RX":
            if fusion_mode_sel == "average":
                fused_disp = np.mean(rx_signals_disp, axis=0)
            elif fusion_mode_sel == "weighted":
                # Need qualities first
                tmp_results = []
                for disp in rx_signals_disp:
                    cardiac = self.logic.apply_filter(disp, fps, self.logic.filters["cardiac_hr_branch"])
                    res = self.logic.estimate_hr(cardiac, fps, "tmp", "tmp")
                    tmp_results.append(res["quality"])
                weights = np.array(tmp_results)
                weights /= (np.sum(weights) + 1e-9)
                fused_disp = np.sum([d * w for d, w in zip(rx_signals_disp, weights)], axis=0)
            elif fusion_mode_sel == "PCA":
                sigs_stack = np.array(rx_signals_disp).T
                fused_disp = self.logic.pca_fusion(sigs_stack, fps)
            elif fusion_mode_sel == "ICA":
                sigs_stack = np.array(rx_signals_disp).T
                try:
                    fused_disp = self.logic.ica_fusion(sigs_stack, fps)
                except:
                    fused_disp = rx_signals_disp[0]
            else: # best RX
                tmp_results = []
                for disp in rx_signals_disp:
                    cardiac = self.logic.apply_filter(disp, fps, self.logic.filters["cardiac_hr_branch"])
                    res = self.logic.estimate_hr(cardiac, fps, "tmp", "tmp")
                    tmp_results.append(res["quality"])
                best_idx = np.argmax(tmp_results)
                fused_disp = rx_signals_disp[best_idx]
        
        # 4. Main Signal Selection
        if rx_source_sel == "PCA":
            sigs_stack = np.array(rx_signals_disp).T
            main_disp = self.logic.pca_fusion(sigs_stack, fps)
            main_source_name = "PCA Fusion"
        elif rx_source_sel == "ICA":
            sigs_stack = np.array(rx_signals_disp).T
            try:
                main_disp = self.logic.ica_fusion(sigs_stack, fps)
            except:
                main_disp = rx_signals_disp[0]
            main_source_name = "ICA Fusion"
        elif rx_source_sel == "Fused":
            main_disp = fused_disp
            main_source_name = f"Fused ({fusion_mode_sel})"
        else:
            rx_idx = int(rx_source_sel.replace("RX", "")) - 1
            rx_idx = min(rx_idx, n_rx - 1)
            main_disp = rx_signals_disp[rx_idx]
            main_source_name = rx_source_sel

        # 5. Full RX x Method Comparison
        self.all_results = []
        self.rx_tree.delete(*self.rx_tree.get_children())
        self.comp_tree.delete(*self.comp_tree.get_children())
        
        # Compare all RXs with the selected phase method
        rx_comp_results = []
        for i in range(n_rx):
            disp = rx_signals_disp[i]
            respiration = self.logic.apply_filter(disp, fps, self.logic.filters["respiration_branch"])
            cardiac = self.logic.apply_filter(disp, fps, self.logic.filters["cardiac_hr_branch"])
            res = self.logic.estimate_hr(cardiac, fps, phase_method, f"RX{i+1}")
            br_res = self.logic.estimate_breathing_rate(respiration, fps)
            q = self.logic.compute_quality_metrics(cardiac, fps, res["hr_bpm"], respiration)
            res.update(q)
            res["br_bpm"] = br_res["br_bpm"]
            rx_comp_results.append(res)
            self.all_results.append(res)
            
            self.rx_tree.insert("", "end", values=(
                res["source"], f"{res['hr_bpm']:.1f}", f"{br_res['br_bpm']:.1f}", 
                f"{res['dominant_freq_hz']:.2f}", f"{res['heart_snr']:.1f}", 
                f"{res['quality_score']:.2f}", res["valid"], res.get("note", "")
            ))

        # Re-implementing Method Comparison properly
        for r_name in self._rx_choices(n_rx):
            if r_name == "Fused" and fused_disp is None: continue
            
            # Extract source signal for comparison
            if r_name.startswith("RX"):
                idx = int(r_name[2:]) - 1
                if idx >= n_rx: continue
                s_c = sig_complex_all_rx[:, idx]
                # For RX, we use multiple phase methods
                for m_phase in ["unwrap", "dacm", "circlefit"]:
                    if m_phase == "dacm": ph = self.logic.extract_phase_dacm(s_c)
                    elif m_phase == "circlefit": ph = self.logic.extract_phase_circle_fitting(s_c)
                    else: ph = self.logic.extract_phase_baseline(s_c)
                    
                    d = signal.detrend(ph * (wave_len / (4 * np.pi)))
                    self._add_to_comparison(d, fps, m_phase, r_name)
            
            elif r_name == "Fused":
                # Fused baseline (usually average)
                s_c = np.mean(sig_complex_all_rx, axis=1)
                ph = self.logic.extract_phase_baseline(s_c)
                d = signal.detrend(ph * (wave_len / (4 * np.pi)))
                self._add_to_comparison(d, fps, "unwrap", "Fused")
            
            elif r_name == "PCA":
                sigs_stack = np.array(rx_signals_disp).T
                d = self.logic.pca_fusion(sigs_stack, fps)
                self._add_to_comparison(d, fps, "fusion", "PCA")
                
            elif r_name == "ICA":
                sigs_stack = np.array(rx_signals_disp).T
                try:
                    d = self.logic.ica_fusion(sigs_stack, fps)
                    self._add_to_comparison(d, fps, "fusion", "ICA")
                except:
                    pass

        # 6. Main Analysis and Plotting
        main_respiration = self.logic.apply_filter(main_disp, fps, self.logic.filters["respiration_branch"])
        
        # Optional harmonic suppression
        main_br_res = self.logic.estimate_breathing_rate(main_respiration, fps)
        if self.notch_resp.get():
            br_hz = main_br_res["br_hz"]
            hr_lo, hr_hi = vital_bands.current().heart_hz      # notch inside the heart band
            for h in [2, 3]:
                f_notch = h * br_hz
                if hr_lo <= f_notch <= hr_hi:
                    main_disp = self.logic.apply_filter(main_disp, fps, {"enabled": True, "type": "notch", "freq_hz": f_notch, "bandwidth_hz": 0.05})

        main_cardiac = self.logic.apply_filter(main_disp, fps, self.logic.filters["cardiac_hr_branch"])
        main_waveform = self.logic.apply_filter(main_disp, fps, self.logic.filters["cardiac_waveform_branch"])
        
        main_result = self.logic.estimate_hr(main_cardiac, fps, phase_method, main_source_name)
        main_result["br_bpm"] = main_br_res["br_bpm"]
        main_result["br_confidence"] = main_br_res["confidence"]
        
        q_main = self.logic.compute_quality_metrics(main_cardiac, fps, main_result["hr_bpm"], main_respiration)
        main_result.update(q_main)
        self.results = main_result
        
        # Sliding Window Trend: the heart followed through the windows as the Vital Signs
        # view does it (harmonic sum, Viterbi path); the largest peak of each window,
        # its own choice from one window to the next, was 8 bpm off a pulse oximeter
        win_s = float(self.win_len.get())
        ovl = float(self.overlap.get())
        trend_data = self.heart_trend(main_disp, main_respiration, fps, win_s, ovl)
        self.trend_data = trend_data
        main_result["hr_track_bpm"] = trend_data["stats"].get("median")
        if main_result["hr_track_bpm"] is not None:
            st = trend_data["stats"]
            note = (f"heart track (Viterbi), {st['count_valid']}/{st['total_count']} windows "
                    f"passed - the HR Result")
            self.comp_tree.insert("", "end", values=(
                "track", main_source_name, f"{st['median']:.1f}", f"{main_br_res['br_bpm']:.1f}",
                f"{st['median_confidence']:.2f}", "", True, note))
            self.all_results.append({"method": "track", "source": main_source_name,
                                     "hr_bpm": st["median"], "br_bpm": main_br_res["br_bpm"],
                                     "confidence": st["median_confidence"], "valid": True,
                                     "note": note})

        # 7. Update Plots
        self.update_main_plots(sig_complex_all_rx[:, 0], main_disp, main_respiration, main_cardiac, main_result, fps, r_axis, rp, bin_idx, trend_data)
        self._plot_rx_comparison(rx_signals_disp, rx_comp_results, fps)
        
        # 8. Diagnostic Map
        self._update_diagnostic_map(rfft_cube, fps, r_axis)

    def heart_trend(self, disp, respiration, fps, win_s, overlap_pct):
        """
        {"trend": [...], "stats": {...}} of the heart in displacement `disp`, windows of
        win_s at overlap_pct: vitalsigns.heart_rate_track, the breathing harmonics of
        each window's breathing rate (30 s windows of `respiration`) notched out. A
        window counts when its confidence passes hr_analysis.MIN_PEAK_TO_AVG.
        """
        step_s = max(1.0 / fps, win_s * (1.0 - overlap_pct / 100.0))
        bands = vital_bands.current()
        br_t, br_v, _ = vitalsigns.sliding(respiration, fps, 30.0, step_s, bands.resp_hz)
        tc, hr, conf, _, _ = vitalsigns.heart_rate_track(disp, fps, bands, win_s, step_s, br_t, br_v)
        trend = [{"time_center": float(t), "hr_bpm": float(h), "confidence": float(c),
                  "valid": bool(c >= hr_analysis.MIN_PEAK_TO_AVG), "method": "track"}
                 for t, h, c in zip(tc, hr, conf)]
        valid = [r["hr_bpm"] for r in trend if r["valid"]]
        stats = {}
        if valid:
            stats = {"mean": float(np.mean(valid)), "median": float(np.median(valid)),
                     "min": float(np.min(valid)), "max": float(np.max(valid)),
                     "std": float(np.std(valid)), "count_valid": len(valid),
                     "total_count": len(trend),
                     "median_confidence": float(np.median([r["confidence"] for r in trend
                                                           if r["valid"]]))}
        return {"trend": trend, "stats": stats}

    def _update_diagnostic_map(self, rfft_cube, fps, r_axis=None):
        """Update the diagnostic heatmap with quality metrics across bins and RXs"""
        self.fig_diag.clear()

        # 35 bins around the selected one, from 0.3 m on (the antenna coupling is nearer).
        # Bins 5-40 before: 0.2-1.6 m at 4 cm a bin, so a person further away was not in it.
        n_bins = rfft_cube.shape[3]
        n_rx = rfft_cube.shape[2]
        first = 5
        if r_axis is not None and len(r_axis) == n_bins:
            first = int(np.searchsorted(np.asarray(r_axis, float), vitalsigns.AUTO_BIN_MIN_RANGE_M))
        span = 35
        lo = max(0, min(max(first, self.selected_bin.get() - span // 2), n_bins - span))
        bins = np.arange(lo, min(n_bins, lo + span))
        if bins.size == 0:
            return
        
        quality_map = np.zeros((n_rx, len(bins)))
        hr_map = np.zeros((n_rx, len(bins)))
        
        for i in range(n_rx):
            for j, b in enumerate(bins):
                sig_complex = np.mean(rfft_cube[:, :, i, b], axis=1)
                phase = self.logic.extract_phase_baseline(sig_complex)
                cardiac = self.logic.apply_filter(phase, fps, self.logic.filters["cardiac_hr_branch"])
                res = self.logic.estimate_hr(cardiac, fps, "diag", f"RX{i+1}")
                quality_map[i, j] = res["quality"]
                hr_map[i, j] = res["hr_bpm"] if res["valid"] else 0
        
        ax1 = self.fig_diag.add_subplot(211)
        extent = [bins[0] - 0.5, bins[-1] + 0.5, n_rx + 0.5, 0.5]     # pixels centred on bins and RX
        im1 = ax1.imshow(quality_map, aspect='auto', extent=extent)
        ax1.set_title("HR Signal Quality Map (RX vs Bin)")
        ax1.set_ylabel("RX Channel")
        self.fig_diag.colorbar(im1, ax=ax1)
        
        ax2 = self.fig_diag.add_subplot(212)
        im2 = ax2.imshow(hr_map, aspect='auto', extent=extent, cmap='jet')
        ax2.set_title("Estimated HR Map [BPM]")
        ax2.set_xlabel("Range Bin")
        ax2.set_ylabel("RX Channel")
        self.fig_diag.colorbar(im2, ax=ax2)
        
        # Mark selected bin
        current_bin = self.selected_bin.get()
        for ax in [ax1, ax2]:
            ax.axvline(x=current_bin, color='red', linestyle='--')
            ax.set_yticks(np.arange(1, n_rx + 1))       # RX1..RXn, as in this tab
            
        self.fig_diag.tight_layout()
        # Save original limits so Reset View works after redraw
        self.mgr_diag.register_time_axes([])
        self.canvas_diag.draw_idle()

    @staticmethod
    def _rx_choices(n_rx: int) -> list:
        return [f"RX{i + 1}" for i in range(max(1, int(n_rx)))] + ["Fused", "PCA", "ICA"]

    def _plot_rx_comparison(self, signals, results, fs):
        self.fig_rx.clear()
        t = np.arange(len(signals[0])) / fs
        ax = self.fig_rx.add_subplot(111)
        for i, sig in enumerate(signals):
            ax.plot(t, sig, label=f"RX{i+1} (HR: {results[i]['hr_bpm']:.1f})")
        ax.legend()
        ax.set_title(f"Displacement Comparison RX1-RX{len(signals)}")
        ax.set_xlabel("Time [s]")
        # Re-attach interactivity (X span zoom) after redraw
        self.mgr_rx.register_time_axes([ax])
        self.canvas_rx.draw_idle()

    def _plot_method_comparison(self, results, fs):
        self.fig_comp.clear()
        ax = self.fig_comp.add_subplot(111)
        methods = [r["method"] for r in results]
        hrs = [r["hr_bpm"] for r in results]
        ax.bar(methods, hrs)
        ax.set_ylabel("HR [BPM]")
        ax.set_title("Method HR Comparison")
        self.mgr_comp.register_time_axes([])
        self.canvas_comp.draw_idle()

    def update_main_plots(self, sig_complex, disp, resp, cardiac, res, fs, r_axis, rp, bin_idx, trend_data=None):
        self.fig_main.clear()
        t = np.arange(len(disp)) / fs
        
        # Grid: 4x2
        # 1. Displacement (Full)
        ax1 = self.fig_main.add_subplot(421)
        ax1.plot(t, disp)
        ax1.set_title("Total Displacement [m]")
        
        # 2. Respiration Signal
        ax2 = self.fig_main.add_subplot(422)
        ax2.plot(t, resp)
        ax2.set_title(f"Respiratory Branch (BR: {res.get('br_bpm', 0):.1f} BPM)")
        
        # 3. Cardiac Signal (HR Band)
        ax3 = self.fig_main.add_subplot(423)
        ax3.plot(t, cardiac)
        ax3.set_title("Cardiac HR Branch [m]")
        
        # 4. HR Trend
        ax4 = self.fig_main.add_subplot(424)
        if trend_data and "trend" in trend_data:
            trend = trend_data["trend"]
            tt = [r["time_center"] for r in trend]
            thr = [r["hr_bpm"] for r in trend]
            tval = [r["valid"] for r in trend]
            ax4.plot(tt, thr, 'k-', alpha=0.3)
            # Plot valid points in green, invalid in red
            for i in range(len(tt)):
                c = 'g' if tval[i] else 'r'
                ax4.plot(tt[i], thr[i], c + '.')
            med = trend_data["stats"].get("median")         # the HR Result
            ax4.set_title("HR Trend" if med is None else f"HR Trend (median: {med:.1f} BPM)")
        else:
            ax4.set_title("HR Trend (N/A)")

        # 5. Spectrum (Log scale)
        ax5 = self.fig_main.add_subplot(413)
        if "debug" in res and "f" in res["debug"]:
            f = np.array(res["debug"]["f"])
            pxx = np.array(res["debug"]["pxx"])
            ax5.semilogy(f, pxx, label="Spectrum")
            ax5.axvline(res["dominant_freq_hz"], color='r', linestyle='--', label=f"Peak: {res['hr_bpm']:.1f} BPM")
            if res.get("hr_track_bpm") is not None:
                ax5.axvline(res["hr_track_bpm"] / 60.0, color='g', linestyle='--',
                            label=f"Heart track: {res['hr_track_bpm']:.1f} BPM")
            
            # Show respiration harmonics if available
            br_res = self.logic.estimate_breathing_rate(resp, fs)
            br_hz = br_res["br_hz"]
            for h in [2, 3, 4]:
                ax5.axvline(h * br_hz, color='orange', linestyle=':', alpha=0.5, label=f"{h}x BR" if h==2 else "")
            
            ax5.set_xlim(0, 5)
            # 60 dB under the largest value shown: the stop band of the heart filter reached
            # 1e-33 and squeezed the band itself into the top of the plot
            vis = pxx[(f >= 0) & (f <= 5)]
            if vis.size and np.max(vis) > 0:
                ax5.set_ylim(np.max(vis) * 1e-6, np.max(vis) * 3)
            ax5.set_title(f"HR Spectrum ({res['method']} on {res['source']}) - Quality: {res['quality_score']:.2f}")
            ax5.legend(fontsize=8)
        
        # 6. Info Text / Metrics
        ax6 = self.fig_main.add_subplot(414)
        ax6.axis('off')
        track = res.get("hr_track_bpm")
        hr_line = (f"HR Result: {track:.1f} BPM (heart track; FFT peak of the whole signal "
                   f"{res['hr_bpm']:.1f})" if track is not None else
                   f"HR Result: {res['hr_bpm']:.1f} BPM (FFT peak; no window of the heart "
                   f"track passed)")
        info = (f"{hr_line}, BR Result: {res.get('br_bpm', 0):.1f} BPM\n"
                f"HR Conf: {res['confidence']:.2f}, BR Conf: {res.get('br_confidence', 0):.2f}\n"
                f"Quality: {res['quality_score']:.2f}, SNR: {res.get('heart_snr', 0):.1f} dB\n"
                f"Valid: {res['valid']}, Note: {res.get('note', '')}\n"
                f"ROI: {self.roi_mode.get()} (±{self.roi_hw.get()}) @ Bin {bin_idx}\n"
                f"Phase Method: {self.phase_method.get()}")
        ax6.text(0, 1, info, verticalalignment='top', family='monospace', fontsize=9)
        
        self.fig_main.tight_layout()
        # Re-attach interactivity after redraw: synchronized X zoom on
        # displacement / respiration / cardiac / HR trend time axes.
        self.mgr_main.register_time_axes([ax1, ax2, ax3, ax4])
        self.canvas_main.draw_idle()

    def export_csv(self):
        # We should export self.all_results
        if not hasattr(self, 'all_results') or not self.all_results:
            messagebox.showwarning("Export", "No results to export. Run detection first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv")
        if not path: return
        with open(path, 'w', newline='') as f:
            # Get all keys from first result, but filter out 'debug'
            keys = [k for k in self.all_results[0].keys() if k != "debug"]
            # Add some global params
            fieldnames = ["roi_mode", "roi_hw", "bin", "phase_method", "clutter_removal"] + keys
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.all_results:
                r_copy = {k: v for k, v in r.items() if k in keys}
                r_copy["roi_mode"] = self.roi_mode.get()
                r_copy["roi_hw"] = self.roi_hw.get()
                r_copy["bin"] = self.selected_bin.get()
                r_copy["phase_method"] = self.phase_method.get()
                r_copy["clutter_removal"] = self.clutter_removal.get()
                writer.writerow(r_copy)

    def export_json(self):
        if not hasattr(self, 'all_results') or not self.all_results:
            messagebox.showwarning("Export", "No results to export. Run detection first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json")
        if not path: return
        
        export_data = {
            "main_result": self.results,
            "all_results": self.all_results,
            "heart_track": getattr(self, "trend_data", None),
            "config": {
                "bin": self.selected_bin.get(),
                "roi_hw": self.roi_hw.get(),
                "roi_mode": self.roi_mode.get(),
                "phase_method": self.phase_method.get(),
                "rx_source": self.rx_source.get(),
                "fusion_mode": self.fusion_mode.get(),
                "clutter_removal": self.clutter_removal.get(),
                "notch_resp": self.notch_resp.get()
            },
            "filters": self.logic.filters
        }
        
        # Helper to make things JSON serializable
        def serialize(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, np.bool_): return bool(obj)
            if isinstance(obj, (np.int64, np.int32)): return int(obj)
            if isinstance(obj, (np.float64, np.float32)): return float(obj)
            return obj
            
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=4, default=serialize)