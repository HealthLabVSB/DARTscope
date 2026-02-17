"""
DCA_processing.py

GUI application for inspecting, processing, and visualizing TI mmWave radar data captured via DCA1000.
Provides tabbed views (Decode, Range FFT, RTI, Range Profile, Range‑Doppler, Helper), orchestrates parsing of
mmWave Studio logs, auto LVDS decoding and reshaping of raw captures, and interactive plotting with playback.

Includes:
- Decode tab: BIN selection, LOG auto-discovery, grouped parameter view.
- Range FFT, RTI, Range Profile, Range‑Doppler: fast/slow‑time processing and visualization.
- Helper: calculator for radar configuration (bandwidth, slopes, sampling, duty cycles) and command generation.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import math
import re
import webbrowser
import io
try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None
from AOA import compute_aoa_spectrum, compute_range_azimuth_map, compute_azimuth_elevation_map  # NEW
from dca1000_decode import (
    parse_logfile_full,
    find_log_file,
)
from recents import (
    RECENTS,
    tk_pick_recent_and_open,
)
from radar_processing import (
    derive_capture_dimensions,
    load_raw_cube,
    compute_range_doppler_map,  # FIX: add missing import
)
from rti import compute_rti                 # NEW: RTI from module
from range_profile import range_profile_dbfs  # NEW: range profile function
from gui_helper import init_helper_tab
from detection import cfar_2d_ca, extract_peaks, detections_to_range_velocity  # NEW
from range_fft import compute_range_fft_cube
import matplotlib.animation as animation
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

# --- GLOBAL AOA STATE ---
aoa_mimo_mode_var = None
lane_swap_var = None
texas_aoa_shift_var = None
aoa_2d_shift_var = None
aoa_2d_method_var = None
texas_aoa_method_var = None
aoa_min_range_var = None # NEW: Global range crop
aoa_angle_limit_var = None # NEW: Global angle limit
aoa_range_var = None
aoa_range_2d_var = None
texas_aoa_range_var = None
aoa_invert_phase_var = None

# NEW: Movement Map state
last_mov_map = None
last_mov_meta = None
mov_im = None
mov_cbar = None
mov_band_var = None
mov_accum_var = None
mov_win_var = None

class Tooltip:
    """
    It creates a tooltip for a given widget as the mouse goes over it.
    """
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.tipwindow = None

    def enter(self, event=None):
        self.showtip()

    def leave(self, event=None):
        self.hidetip()

    def showtip(self):
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 27
        y = y + cy + self.widget.winfo_rooty() + 27
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                      background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                      font=("Segoe UI", 9, "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

def export_to_png(fig, title="Export"):
    file_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                             filetypes=[("PNG files", "*.png")],
                                             title=f"Save {title} as PNG")
    if file_path:
        fig.savefig(file_path)
        messagebox.showinfo("Export", f"Plot saved to {file_path}")

def export_to_csv(data, title="Export"):
    if data is None:
        messagebox.showwarning("Export", "No data to export.")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", 
                                             filetypes=[("CSV files", "*.csv")],
                                             title=f"Save {title} data as CSV")
    if file_path:
        try:
            if data.ndim <= 2:
                np.savetxt(file_path, data, delimiter=",")
            else:
                # For 3D and more we save current frame as flattened matrix
                np.savetxt(file_path, data.reshape(data.shape[0], -1), delimiter=",")
            messagebox.showinfo("Export", f"Data saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Export", f"Error saving CSV: {e}")

def export_to_video(fig, update_func, title="Export"):
    if global_frames_total <= 0:
        messagebox.showwarning("Export", "No data to export.")
        return
        
    try:
        from matplotlib.animation import FFMpegWriter
        if not FFMpegWriter.isAvailable():
            messagebox.showerror("Export", "ffmpeg is not available. Video export requires ffmpeg installed (in PATH).")
            return
    except ImportError:
        messagebox.showerror("Export", "matplotlib.animation or FFMpegWriter module is not available.")
        return

    file_path = filedialog.asksaveasfilename(defaultextension=".mp4", 
                                             filetypes=[("MP4 files", "*.mp4")],
                                             title=f"Save {title} as Video")
    if not file_path:
        return

    messagebox.showinfo("Export", "Starting video export. The application may be unresponsive for a moment.")
    
    try:
        # Try to determine FPS from parameters, otherwise 10
        try:
            ms = max(1, int(global_play_interval_ms))
            fps = 1000.0 / float(ms)
        except:
            fps = 10
            
        writer = FFMpegWriter(fps=fps)
        with writer.saving(fig, file_path, dpi=100):
            for i in range(global_frames_total):
                update_func(i)
                writer.grab_frame()
        messagebox.showinfo("Export", f"Video saved to {file_path}")
    except Exception as e:
        messagebox.showerror("Export", f"Video export failed: {e}")

def add_card_header_extras(parent_frame, card_name, info_text, fig, data_func, update_func=None):
    """
    Adds ⓘ icon with tooltip and Export button to the card control bar.
    """
    extras_frame = ttk.Frame(parent_frame)
    extras_frame.pack(side="right", padx=10)
    
    # Info icon
    info_lbl = tk.Label(extras_frame, text="ⓘ", font=("Segoe UI", 12, "bold"), fg="blue", cursor="hand2")
    info_lbl.pack(side="left", padx=5)
    Tooltip(info_lbl, info_text)
    
    # Export button with menu
    export_btn = ttk.Menubutton(extras_frame, text="Export")
    export_btn.pack(side="left", padx=5)
    
    export_menu = tk.Menu(export_btn, tearoff=0)
    export_btn["menu"] = export_menu
    
    export_menu.add_command(label="Current frame (PNG)", command=lambda: export_to_png(fig, card_name))
    export_menu.add_command(label="Data (CSV)", command=lambda: export_to_csv(data_func(), card_name))
    if update_func:
        export_menu.add_command(label="All frames (Video)", command=lambda: export_to_video(fig, update_func, card_name))

# --- Processing info strings (for tooltips) ---
INFO_RANGE_FFT = "Processing Chain:\n1. DC Removal (per chirp)\n2. Hann Window\n3. 1D FFT (fast-time)\n4. Magnitude calculation\n5. Non-coherent average (across RX and Chirps)\n6. Logarithmic scale (dB)"
INFO_RTI = "Processing Chain:\n1. Range FFT (Hann window, DC removal)\n2. Magnitude calculation\n3. Non-coherent average (across RX)\n4. Time Unfolding (stacking chirps/frames)\n5. DC & Edge Notch filtering\n6. Logarithmic scale (dB)"
INFO_RP = "Processing Chain:\n1. DC Removal\n2. Hann Window\n3. 1D FFT\n4. Coherent Averaging (across chirps)\n5. Magnitude calculation\n6. Normalization to Full Scale (dBFS)\n7. Selected RX channel only"
INFO_RD = "Processing Chain:\n1. Range FFT (Hann window, DC removal)\n2. TX Separation (TDM demux)\n3. Clutter Removal (DC subtraction in slow-time)\n4. Doppler Windowing (Hann/Hamming)\n5. Doppler FFT (Slow-time, shifted)\n6. RX Combining (MRC/SUM/RXk)\n7. Logarithmic scale (dB)"
INFO_AOA = "Processing Chain:\n1. Complex Range FFT\n2. TX Separation & MIMO array formation\n3. Phase/Amplitude Calibration\n4. Clutter Removal (optional)\n5. Hamming Window (on antenna array)\n6. Bartlett Beamforming\n7. Non-coherent averaging across loops\n8. Peak detection"
INFO_AOA_2D = "Processing Chain:\n1. Complex Range FFT\n2. TX Separation & MIMO array formation\n3. Phase/Amplitude Calibration\n4. Clutter Removal (optional)\n5. Hamming Window (antenna array)\n6. Bartlett Beamforming (all range bins)\n7. Cartesian/Polar Projection"
INFO_MOV_MAP = "Processing Chain:\n1. Compute 2D Range-Azimuth map for each frame\n2. Perform Temporal FFT per (Range, Azimuth) pixel\n3. Sum energy in selected band (Movement, Breathing, HR)\n4. Accumulate over time (sliding window or full)\n5. Normalize and display as 2D heatmap"
INFO_MD = "Processing Chain:\n1. Range-Doppler Map calculation\n2. Range Bin range selection\n3. Summation across selected range bins\n4. Time-Doppler Map formation\n5. Logarithmic scale (dB)"
INFO_VIEW1D = "Processing Chain:\n1. Range-Doppler Map calculation\n2. Slice at selected Range Bin\n3. Doppler Spectrum extraction\n4. Temporal FFT (across frames)\n5. Biometric peak detection (BR/HR)"
INFO_CFAR = "Processing Chain:\n1. Range-Doppler Map calculation\n2. 2D CA-CFAR or OS-CFAR detector\n3. Guard & Training cells noise estimation\n4. Thresholding based on Pfa factor\n5. Connected Components Clustering\n6. Peak Extraction\n7. Physical unit conversion"

last_params = {}
last_cube = None
last_range_fft = None
last_range_fft_meta = None

# NEW: Global frame sync/playback state (top bar)
global_frames_total = 0
global_playing = False
global_play_interval_ms = 150
global_frame_slider = None
global_play_btn = None
global_stop_btn = None
_in_global_frame_set = False  # re-entrancy guard for apply_global_frame

# NEW: Global antenna (RX combine) selector refs (initialized later in UI)
global_rx_mode_var = None
global_rx_mode_combo = None

# NEW: RTI state variables
last_rti = None
last_rti_meta = None
rti_im = None
rti_cbar = None

# NEW: Range Profile state variables
last_range_profile = None
last_range_profile_meta = None

# NEW: Range-Doppler state variables
last_rd_map = None
last_rd_meta = None
rd_im = None
rd_cbar = None
rd_zero_line = None  # NEW: zero-velocity guide line

# NEW: Range-Doppler playback state
rd_playing = False
rd_play_interval_ms = 150
rd_full_spectrum_var = None  # NEW: checkbox for full spectrum in RD
# NEW: RD zoom variables
rd_vmin_var = None
rd_vmax_var = None
# NEW: RD Doppler window toggle
rd_hann_var = None  # <-- add state var
# NEW: RD auto velocity range
rd_auto_vrange_var = None
rd_vmin_entry = None
rd_vmax_entry = None

# NEW: AoA state variables
last_aoa_spec = None
last_aoa_meta = None
aoa_invert_phase_var = None # NEW

# NEW: Texas AoA state variables
last_texas_aoa_spec = None
last_texas_aoa_meta = None
texas_aoa_fig = None
texas_aoa_ax = None
texas_aoa_canvas = None
texas_aoa_im = None
texas_aoa_cbar = None
texas_aoa_remove_static_var = None
texas_aoa_mimo_var = None
texas_aoa_range_var = None
texas_aoa_view_var = None # NEW: View type (1D, RA, AE)

# NEW: 2D AoA state variables
last_aoa_2d_map = None
last_aoa_2d_meta = None
aoa_2d_im = None
aoa_2d_cbar = None
aoa_2d_remove_static_var = None
aoa_2d_mimo_var = None  # NEW
aoa_2d_info_var = None  # NEW

# NEW: AoA bin selection UI state
aoa_bin_slider = None
aoa_bin_var = None
aoa_bin_label_var = None
aoa_remove_static_var = None
aoa_mimo_var = None  # NEW

# NEW: Range FFT playback state
range_fft_playing = False
range_fft_play_interval_ms = 150
range_fft_full_spectrum_var = None  # NEW: checkbox for full spectrum

# NEW: Range FFT stationary-removal
last_range_fft_bg = None
range_fft_rm_stationary_var = None
# NEW: Range FFT stationary-removal intensity (0.0 .. 1.0)
range_fft_rm_strength_var = None  # 0.0 .. 1.0

# NEW: Micro-Doppler state variables
last_md_map = None
last_md_meta = None
md_im = None
md_cbar = None
md_r0_var = None
md_r1_var = None
md_indicator_line = None

# NEW: 1D View (Doppler Spectrum View) state variables
view1d_rbin_var = None
view1d_rbin_label_var = None
view1d_fig = None
view1d_ax_prof = None
view1d_ax_spec = None
view1d_ax_bars = None
view1d_canvas = None
view1d_slider = None # NEW
view1d_band_src_var = None
view1d_band_w_var = None
view1d_just_frame_var = None

# RTI widgets
rti_im = None               # NEW
rti_cbar = None             # NEW
# NEW: RTI chirp selector
rti_chirp_mode_var = None

# REMOVED: Helper tab variables are now in gui_helper.py

params_canvas = None
params_scroll = None
params_inner = None
PARAMS_GRID_COLS = 3  # pairs of "name : value" per row distributed across columns

# ---------------- Param groups (NEW) ----------------
def _build_param_groups(params: dict):
    """
    Build grouped (title -> [(name, value)]) list.
    Only final values (no candidates).
    """
    general_keys = [
        "timestamp", "profilesFound", "chip_hint", "chipHintBand", "band",
        "startFreqGHz", "effectiveStartFreqGHz", "stopFreqGHz",
        "freqSlopeConst", "freqSlopeMHz_us", "freqSlopeScale_used",
        "idleTime_us", "adcStartTime_us", "txStartTime_us", "rampEndTime_us",
        "numAdcSamples", "digOutSampleRate_MHz",
    ]
    # NOTE: removed effectiveBandwidthGHz from GUI
    bw_keys = [
        "bandwidthGHz", "effectiveBandwidthMHz",
        "rangeRes_cm", "effectiveRangeRes_cm", "rangeResolution_gui_cm",
        "adcSampleTime_us", "rampWindow_us", "effCaptureWindow_us",
    ]
    datafmt_keys = [
        "dataFmt_adcFmt", "dataFmt_adcBits", "dataFmt_iqOrder",
        "dataFmt_rxChanEnMask", "dataFmt_rxChannels", "dataFmt_isInterleaved",
        "numLanes", "numLanes_source", "lanesActive",
    ]
    rx_tx_keys = ["rxGain", "txOutPowerBackoffCode", "txPhaseShifter"]
    frame_keys = ["framePeriodicity_ms_selected", "framePeriodicity_selected_source",
                  "framePeriodicity_ms_derived", "framePeriodicity_ms_log",
                  # NEW: show effective and logged uniqueChirps
                  "uniqueChirps_effective", "uniqueChirps_log", "uniqueChirps_mismatch"]
    # FrameConfig compact
    fc = params.get("frameConfig") or {}
    fc_items = []
    if fc:
        fc_items = [
            ("frameCfgIdx", fc.get("frameCfgIdx")),
            ("chirpStartIdx", fc.get("chirpStartIdx")),
            ("chirpEndIdx", fc.get("chirpEndIdx")),
            ("uniqueChirps", fc.get("uniqueChirps")),
            ("numLoops", fc.get("numLoops")),
            ("chirpsPerFrame", fc.get("chirpsPerFrame")),
            ("numFrames", fc.get("numFrames")),
            ("framePeriodicity_ms", fc.get("framePeriodicity_ms")),
            ("triggerSelect", fc.get("triggerSelect")),
            ("triggerDelay", fc.get("triggerDelay")),
        ]
    # AdvancedFrameConfig compact is now hidden
    afc = params.get("advancedFrameConfig") or {}
    afc_items = []
    # if afc:
    #     afc_items = [
    #         ("adv_frameHeader", afc.get("adv_frameHeader")),
    #         ("adv_numSubFrames", afc.get("adv_numSubFrames")),
    #         ("adv_periodicity_1_us", afc.get("adv_periodicity_1_us")),
    #         ("adv_periodicity_2_us", afc.get("adv_periodicity_2_us")),
    #         ("adv_raw_len", len(afc.get("adv_raw") or [])),
    #     ]

    def pick(keys):
        return [(k, params.get(k)) for k in keys if k in params]

    groups = []
    groups.append(("General", pick(general_keys)))
    groups.append(("Bandwidth & Resolution", pick(bw_keys)))
    groups.append(("Data Format", pick(datafmt_keys)))
    groups.append(("RX/TX & Gains", pick(rx_tx_keys)))
    groups.append(("Frame (derived/log)", pick(frame_keys)))
    if fc_items:
        groups.append(("FrameConfig", fc_items))
    # NEW: Chirp TX Pattern summary
    try:
        order = params.get("tx_order_names")
        if order:
            items = []
            items.append(("order", " | ".join(order)))
            act = params.get("tdm_active_masks")
            if act:
                items.append(("active", ", ".join(act)))
            if params.get("tdm_pattern_len"):
                items.append(("pattern_len", params.get("tdm_pattern_len")))
            # Optional: positions by TX for clarity
            pos = params.get("tdm_positions_by_name")
            if pos:
                items.append(("positions", str(pos)))
            groups.append(("Chirp TX Pattern", items))
    except Exception:
        pass
    # Chirps as sub-groups (will be placed to a dedicated column in populate_params_view)
    chirps = params.get("chirps") or []
    for ch in chirps:
        title = f"Chirp {ch.get('chirpStartIdx')}-{ch.get('chirpEndIdx')} (prof {ch.get('profileId')})"
        items = []
        for ck in ["txEnable", "txActive", "startFreqVar", "freqSlopeVar", "idleTimeVar", "adcStartTimeVar", "bpfSel"]:
            if ck in ch:
                items.append((ck, ch.get(ck)))
        groups.append((title, items))
    return groups

def _clear_params_grid():
    if params_inner is None:
        return
    for w in list(params_inner.children.values()):
        w.destroy()

def populate_params_view(params: dict):
    """
    Grouped, multi-column layout with section headers and separators.
    Chirp groups are rendered in a dedicated last column.
    """
    if params_inner is None:
        return
    _clear_params_grid()

    groups = _build_param_groups(params)

    # Split base groups vs. chirp groups
    base_groups = [(t, items) for (t, items) in groups if not t.startswith("Chirp ")]
    chirp_groups = [(t, items) for (t, items) in groups if t.startswith("Chirp ")]

    # Column allocation: base groups spread across first (PARAMS_GRID_COLS-1) columns,
    # chirps occupy the last column exclusively.
    total_cols = max(1, PARAMS_GRID_COLS)
    base_cols = max(1, total_cols - 1)
    chirp_col_exists = (total_cols >= 2)

    # Styles for headers
    style = ttk.Style()
    style.configure("ParamHeader.TLabel", font=("Segoe UI", 10, "bold"))

    # ----- Render base groups across base_cols -----
    # Compute rows per base column
    base_lengths = []
    base_total_rows = 0
    for _, items in base_groups:
        length = 1 + len(items) + 1  # header + items + separator
        base_lengths.append(length)
        base_total_rows += length
    base_rows_per_col = max(1, math.ceil(base_total_rows / base_cols))

    cur_col = 0
    cur_row = 0
    for (title, items), length in zip(base_groups, base_lengths):
        if cur_row + length > base_rows_per_col and cur_col < base_cols - 1:
            cur_col += 1
            cur_row = 0
        base_col = cur_col * 2
        # Header
        hdr = ttk.Label(params_inner, text=title, style="ParamHeader.TLabel", anchor="w")
        hdr.grid(row=cur_row, column=base_col, columnspan=2, sticky="ew", padx=(6, 6), pady=(4, 2))
        cur_row += 1
        # Items
        for name, value in items:
            ttk.Label(params_inner, text=str(name), anchor="w").grid(row=cur_row, column=base_col, sticky="w", padx=(12, 4), pady=1)
            ttk.Label(params_inner, text=_fmt_val(value), anchor="w").grid(row=cur_row, column=base_col + 1, sticky="w", padx=(4, 12), pady=1)
            cur_row += 1
        # Separator
        ttk.Separator(params_inner, orient="horizontal").grid(row=cur_row, column=base_col, columnspan=2, sticky="ew", padx=(6, 6), pady=(4, 6))
        cur_row += 1

    # ----- Render chirp groups in the last column -----
    if chirp_groups and chirp_col_exists:
        chirp_base_col = (total_cols - 1) * 2
        chirp_row = 0
        # Compute total rows for chirps to allow future balancing if needed
        for (title, items) in chirp_groups:
            # Header
            ttk.Label(params_inner, text=title, style="ParamHeader.TLabel", anchor="w").grid(
                row=chirp_row, column=chirp_base_col, columnspan=2, sticky="ew", padx=(6, 6), pady=(4, 2)
            )
            chirp_row += 1
            # Items
            for name, value in items:
                ttk.Label(params_inner, text=str(name), anchor="w").grid(
                    row=chirp_row, column=chirp_base_col, sticky="w", padx=(12, 4), pady=1
                )
                ttk.Label(params_inner, text=_fmt_val(value), anchor="w").grid(
                    row=chirp_row, column=chirp_base_col + 1, sticky="w", padx=(4, 12), pady=1
                )
                chirp_row += 1
            # Separator
            ttk.Separator(params_inner, orient="horizontal").grid(
                row=chirp_row, column=chirp_base_col, columnspan=2, sticky="ew", padx=(6, 6), pady=(4, 6)
            )
            chirp_row += 1

    # Stretch columns
    for c in range(total_cols * 2):
        params_inner.grid_columnconfigure(c, weight=1)

    params_inner.update_idletasks()
    params_canvas.configure(scrollregion=params_canvas.bbox("all"))

# ---- helpers for chirp selection (NEW) ----
def _tx_selection_options(params: dict):
    """
    Returns combobox items according to tx_order_names from LOG.
    Output: (['All', 'TX:TX0', 'TX:TX1', ...], default_index)
    """
    try:
        names = params.get("tx_order_names") or []
        # Fallback: create names from 'txActive' from chirps[]
        if not names and isinstance(params.get("chirps"), list):
            seq = []
            for ch in params["chirps"]:
                nm = ch.get("txActive")
                seq.append(nm if nm else "None")
            names = seq
        if not names:
            return (["All"], 0)
        seen = set()
        ordered_unique = []
        for n in names:
            if n and n != "None" and n not in seen:
                seen.add(n)
                ordered_unique.append(n)
        if not ordered_unique:
            return (["All"], 0)
        opts = ["All"] + [f"TX:{n}" for n in ordered_unique]
        return (opts, 0)
    except Exception:
        return (["All"], 0)

def _selected_chirp_positions(params: dict, selection: str):
    """
    Converts GUI selection to chirp positions within one unique set.
    Input: selection = "All" or "TX:TX0"/"TX:TX1"/...
    Returns: None (for All) or {'positions':[...], 'pattern_len': eff_unique}
    """
    try:
        if not selection or selection == "All":
            return None
        if not selection.startswith("TX:"):
            return None
        txname = selection.split("TX:", 1)[1].strip()
        names = params.get("tx_order_names") or []
        # Fallback to chirps[].txActive if tx_order_names is missing
        if not names and isinstance(params.get("chirps"), list):
            seq = []
            for ch in params["chirps"]:
                nm = ch.get("txActive")
                seq.append(nm if nm else "None")
            names = seq
        if not names:
            return None
        positions = [i for i, n in enumerate(names) if n == txname]
        if not positions:
            return None
        return {"positions": positions, "pattern_len": len(names)}
    except Exception:
        return None

def select_bin_file():
    file_path = filedialog.askopenfilename(filetypes=[("Binary files", "*.bin")])
    if file_path:
        bin_var.set(file_path)
        RECENTS.add(file_path)
        try:
            populate_recent_list()
        except Exception:
            pass
        decode_log()

def _fmt_val(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)

def _reset_cached_data():
    """Clear all previously loaded and computed data."""
    global last_cube, last_range_fft, last_range_fft_meta
    global last_rti, last_rti_meta, last_range_profile, last_range_profile_meta
    global last_detections, last_detection_peaks, detection_im, detection_cbar, detection_scatter  # NEW
    global last_aoa_spec, last_aoa_meta, last_aoa_2d_map, last_aoa_2d_meta, aoa_2d_im, aoa_2d_cbar  # NEW AoA
    global last_md_map, last_md_meta, md_im, md_cbar, md_indicator_line  # NEW Micro-Doppler
    global view1d_fig, view1d_canvas # NEW 1D View
    global last_mov_map, last_mov_meta, mov_im, mov_cbar # NEW Movement Map
    global global_frames_total

    global_frames_total = 0
    last_cube = None
    last_range_fft = None
    last_range_fft_meta = None
    last_rti = None
    last_rti_meta = None
    last_range_profile = None
    last_range_profile_meta = None
    last_detections = None          # NEW
    last_detection_peaks = None     # NEW
    detection_im = None             # NEW
    detection_cbar = None           # NEW
    detection_scatter = None        # NEW
    last_aoa_spec = None            # NEW
    last_aoa_meta = None            # NEW
    last_aoa_2d_map = None          # NEW
    last_aoa_2d_meta = None         # NEW
    aoa_2d_im = None                # NEW
    aoa_2d_cbar = None              # NEW
    last_md_map = None              # NEW
    last_md_meta = None             # NEW
    md_im = None                    # NEW
    md_cbar = None                  # NEW
    md_indicator_line = None        # NEW
    last_mov_map = None             # NEW
    last_mov_meta = None            # NEW
    mov_im = None                   # NEW
    mov_cbar = None                 # NEW
    # Clear 1D View plots
    if view1d_ax_prof is not None:
        try: view1d_ax_prof.clear()
        except: pass
    if view1d_ax_spec is not None:
        try: view1d_ax_spec.clear()
        except: pass
    if view1d_ax_bars is not None:
        try: view1d_ax_bars.clear()
        except: pass
    if view1d_canvas is not None:
        try: view1d_canvas.draw_idle()
        except: pass
    
    if 'rd_play_btn' in globals() and rd_play_btn is not None and rd_play_btn.winfo_exists():
        stop_rd()
        rd_play_btn.config(state="disabled")
        rd_stop_btn.config(state="disabled")
    if 'detection_frame_slider' in globals() and detection_frame_slider is not None and detection_frame_slider.winfo_exists():  # NEW
        detection_frame_slider.set(0)
        detection_frame_slider.config(to=0, state="disabled")
    # NEW: reset global frame controls
    if 'global_play_btn' in globals() and global_play_btn is not None and global_play_btn.winfo_exists():
        stop_global()
        global_play_btn.config(state="disabled")
    if 'global_stop_btn' in globals() and global_stop_btn is not None and global_stop_btn.winfo_exists():
        global_stop_btn.config(state="disabled")
    if 'global_frame_slider' in globals() and global_frame_slider is not None and global_frame_slider.winfo_exists():
        global_frame_slider.set(0)
        global_frame_slider.config(to=0, state="disabled")
    # NEW: reset 1D View and AoA bin sliders
    if 'view1d_slider' in globals() and view1d_slider is not None and view1d_slider.winfo_exists():
        view1d_slider.set(0)
        view1d_slider.config(to=0, state="disabled")
    if 'aoa_bin_slider' in globals() and aoa_bin_slider is not None and aoa_bin_slider.winfo_exists():
        aoa_bin_slider.set(0)
        aoa_bin_slider.config(to=0, state="disabled")
    # Reset antenna selector to default if present
    if 'global_rx_mode_var' in globals() and isinstance(global_rx_mode_var, tk.StringVar):
        try:
            global_rx_mode_var.set("MRC")
        except Exception:
            pass
    # Reset FPS label
    if 'global_fps_var' in globals() and isinstance(global_fps_var, tk.StringVar):
        try:
            global_fps_var.set("-- fps")
        except Exception:
            pass

# NEW: helper to derive frame period from params (ms)
def _derive_period_ms(params: dict) -> int | None:
    try:
        # Preferred: already selected value
        val = params.get('framePeriodicity_ms_selected')
        if isinstance(val, (int, float)) and val > 0:
            return int(round(val))
        # Fallback: frameConfig.framePeriodicity_ms (if exposed)
        fc = params.get('frameConfig') or {}
        val = fc.get('framePeriodicity_ms')
        if isinstance(val, (int, float)) and val > 0:
            return int(round(val))
        # Fallback: raw ticks in 5 ns units if present
        ticks = fc.get('framePeriodicity_ticks')
        if isinstance(ticks, (int, float)) and ticks > 0:
            return int(round((ticks * 5.0) / 1000.0))
    except Exception:
        pass
    return None

# NEW: FPS label updater
def _update_global_fps_label():
    try:
        if 'global_fps_var' in globals() and isinstance(global_fps_var, tk.StringVar):
            ms = max(1, int(global_play_interval_ms))
            fps = 1000.0 / float(ms)
            global_fps_var.set(f"{fps:.2f} fps ({ms} ms)")
    except Exception:
        pass

def decode_log():
    bin_path = bin_var.get()
    if not os.path.isfile(bin_path):
        messagebox.showerror("Error", "Please select a valid BIN file.")
        return
    try:
        RECENTS.add(bin_path)
        populate_recent_list()
    except Exception:
        pass
    log_path = find_log_file(bin_path)
    if not log_path:
        messagebox.showerror("Error", "Corresponding LOG file not found.")
        return
    try:
        global last_params
        # Reset all data from previous file
        _reset_cached_data()
        params, summary_lines = parse_logfile_full(log_path, print_summary=True)
        if not params:
            messagebox.showerror("Error", "ProfileConfig nenalezen v log souboru.")
            return
        last_params = params
        # Derive dims (also validates config)
        _ = derive_capture_dimensions(params)
        populate_params_view(last_params)

        # NEW: populate Chirp selection combobox
        try:
            if global_chirp_sel_combo is not None:
                opts, def_idx = _tx_selection_options(last_params)
                global_chirp_sel_combo.config(values=opts, state="readonly")
                if isinstance(global_chirp_sel_var, tk.StringVar):
                    global_chirp_sel_var.set(opts[def_idx] if opts else "All")
        except Exception:
            pass

        # NEW: publish real-time playback interval from LOG immediately
        try:
            per_ms = _derive_period_ms(last_params)
            if per_ms and per_ms > 0:
                set_global_interval_ms(int(per_ms))
        except Exception:
            pass

    except Exception as e:
        messagebox.showerror("Error", str(e))

def ensure_cube_loaded():
    global last_cube
    if not last_params:
        messagebox.showerror("Error", "Please decode the log file first.")
        return None
    if last_cube is None:
        try:
            dims = derive_capture_dimensions(last_params)
            # Build chirp selection from GUI
            try:
                sel = global_chirp_sel_var.get() if isinstance(global_chirp_sel_var, tk.StringVar) else "All"
            except Exception:
                sel = "All"
            chirp_sel = _selected_chirp_positions(last_params, sel)

            # Check for lane swap (1<->2)
            l_swap = False
            try:
                if (lane_swap_var is not None):
                    l_swap = bool(lane_swap_var.get())
            except Exception:
                pass
            lane_order = [0, 2, 1, 3] if l_swap else None

            cube = load_raw_cube(bin_var.get(), last_params, dims, chirp_selection=chirp_sel, lane_order=lane_order)
            last_cube = cube
        except Exception as e:
            messagebox.showerror("Error", f"Data loading error: {e}")
            return None
    return last_cube

def init_range_fft_plot(parent):
    global range_fft_fig, range_fft_ax, range_fft_canvas, range_fft_frame_slider
    range_fft_fig = Figure(figsize=(5,3), dpi=100)
    range_fft_ax = range_fft_fig.add_subplot(111)
    range_fft_ax.set_title("Range FFT (Frame 0)")
    range_fft_ax.set_xlabel("Range [m]")
    range_fft_ax.set_ylabel("Magnitude")
    range_fft_canvas = FigureCanvasTkAgg(range_fft_fig, master=parent)
    range_fft_canvas.get_tk_widget().grid(row=1, column=0, columnspan=5, sticky="nsew", padx=4, pady=4)
    parent.rowconfigure(1, weight=1)
    parent.columnconfigure(0, weight=1)

def update_range_fft_plot(frame_idx: int):
    if last_range_fft is None or last_range_fft_meta is None:
        return
    f = min(max(frame_idx, 0), last_range_fft.shape[0]-1)
    fft_frame = last_range_fft[f]  # (C, RX, R)
    mag_1d = fft_frame.mean(axis=(0,1))

    # Optionally remove stationary background (motion-only)
    try:
        if (
            isinstance(range_fft_rm_stationary_var, tk.BooleanVar)
            and range_fft_rm_stationary_var.get()
            and (last_range_fft_bg is not None)
        ):
            bg_1d = last_range_fft_bg.mean(axis=(0,1))
            try:
                alpha = float(range_fft_rm_strength_var.get())
            except Exception:
                alpha = 1.0
            alpha = max(0.0, min(alpha, 1.0))
            mag_1d = np.clip(mag_1d - alpha * bg_1d, 1e-9, None)
    except Exception:
        pass

    # Convert to dB
    mag_db = 20 * np.log10(mag_1d)

    range_fft_ax.clear()

    # Use range axis if available
    x = last_range_fft_meta.get('range_m')
    if isinstance(x, np.ndarray) and x.shape[0] == mag_db.shape[0]:
        range_fft_ax.plot(x, mag_db)
        range_fft_ax.set_xlabel("Range [m]")
    else:
        range_fft_ax.plot(mag_db)
        range_fft_ax.set_xlabel("Range Bin")

    range_fft_ax.set_title(f"Range FFT (Frame {f})")
    range_fft_ax.set_ylabel("Magnitude [dB]")
    range_fft_ax.grid(True, alpha=0.3)

    # Set fixed Y-axis limits if available, using motion-only if checkbox is enabled
    use_motion = False
    try:
        use_motion = isinstance(range_fft_rm_stationary_var, tk.BooleanVar) and range_fft_rm_stationary_var.get()
    except Exception:
        pass
    if use_motion:
        vmin = last_range_fft_meta.get('display_vmin_moving', last_range_fft_meta.get('display_vmin'))
        vmax = last_range_fft_meta.get('display_vmax_moving', last_range_fft_meta.get('display_vmax'))
    else:
        vmin = last_range_fft_meta.get('display_vmin')
        vmax = last_range_fft_meta.get('display_vmax')
    if vmin is not None and vmax is not None:
        range_fft_ax.set_ylim(vmin, vmax)

    range_fft_canvas.draw_idle()

def on_range_fft_frame_change(val):
    try:
        idx = int(float(val))
    except ValueError:
        idx = 0
    # NEW: route via global synchronizer
    apply_global_frame(idx)

def run_range_fft():
    cube = ensure_cube_loaded()
    if cube is None:
        return
    try:
        global last_range_fft, last_range_fft_meta, range_fft_play_interval_ms

        # Decide spectrum size based on input type: full for Complex1X, half for Real, PseudoReal, Complex2X.
        is_complex_obj = bool(np.iscomplexobj(cube))
        fmt_str = str(last_params.get('dataFmt_adcFmt', '')).upper()
        
        # We only use full spectrum for Complex1X (or if unknown). 
        # For Real, PseudoReal, and Complex2X we want half-spectrum to avoid mirroring.
        use_full = is_complex_obj
        if is_complex_obj and any(x in fmt_str for x in ('COMPLEX2X', 'PSEUDOREAL', 'REAL')):
            use_full = False

        last_range_fft, last_range_fft_meta = compute_range_fft_cube(
            cube, last_params, window='hann', pad_pow2=False, remove_dc=True,
            n_fft_range=256, use_full_spectrum=use_full
        )

        # Compute stationary background (mean across frames) for removal
        global last_range_fft_bg
        try:
            last_range_fft_bg = last_range_fft.mean(axis=0)
        except Exception:
            last_range_fft_bg = None

        frames = last_range_fft.shape[0]
        set_global_frames(frames)

        # Set playback speed from frame periodicity
        period_ms = last_params.get('framePeriodicity_ms_selected')
        if period_ms and period_ms > 0:
            range_fft_play_interval_ms = int(period_ms)
            set_global_interval_ms(range_fft_play_interval_ms)

        # Calculate and store global min/max for fixed scale (normal & motion-only)
        mag_db_all = 20 * np.log10(last_range_fft + 1e-9)
        vmin, vmax = np.percentile(mag_db_all, [5, 99])
        if np.isfinite(vmin) and np.isfinite(vmax):
            last_range_fft_meta['display_vmin'] = vmin
            last_range_fft_meta['display_vmax'] = vmax + 5
        # Motion-only percentiles
        try:
            moving = np.clip(last_range_fft - last_range_fft_bg, 1e-9, None)
            moving_db = 20 * np.log10(moving)
            mvmin, mvmax = np.percentile(moving_db, [5, 99])
            if np.isfinite(mvmin) and np.isfinite(mvmax):
                last_range_fft_meta['display_vmin_moving'] = mvmin
                last_range_fft_meta['display_vmax_moving'] = mvmax + 5
        except Exception:
            pass

        update_range_fft_plot(0)
    except Exception as e:
        messagebox.showerror("Error", f"Range FFT chyba: {e}")

# --- NEW: Range FFT Playback Functions ---
def _update_range_fft_frame():
    if not range_fft_playing:
        return
    current_val = range_fft_frame_slider.get()
    max_val = range_fft_frame_slider.cget("to")
    next_val = (current_val + 1) % (max_val + 1)
    range_fft_frame_slider.set(next_val)
    root.after(range_fft_play_interval_ms, _update_range_fft_frame)

def init_params_view(parent):
    global params_canvas, params_scroll, params_inner
    container = ttk.Frame(parent)
    container.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=4, pady=6)

    params_canvas = tk.Canvas(container, highlightthickness=0)
    params_scroll = ttk.Scrollbar(container, orient="vertical", command=params_canvas.yview)
    params_inner = ttk.Frame(params_canvas)

    params_inner.bind(
        "<Configure>",
        lambda e: params_canvas.configure(scrollregion=params_canvas.bbox("all"))
    )
    params_canvas.create_window((0, 0), window=params_inner, anchor="nw")

    def _on_mousewheel(event):
        params_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    params_canvas.bind_all("<MouseWheel>", _on_mousewheel)

    params_canvas.configure(yscrollcommand=params_scroll.set)
    params_canvas.grid(row=0, column=0, sticky="nsew")
    params_scroll.grid(row=0, column=1, sticky="ns")

    container.rowconfigure(0, weight=1)
    container.columnconfigure(0, weight=1)

# --- NEW: RTI Functions ---
def init_rti_plot(parent):
    global rti_fig, rti_ax, rti_canvas
    rti_fig = Figure(figsize=(5, 3), dpi=100, constrained_layout=True)
    rti_ax = rti_fig.add_subplot(111)
    rti_ax.set_title("Range-Time Intensity (RTI)")
    rti_ax.set_xlabel("Chirp Index")
    rti_ax.set_ylabel("Range [m]")
    rti_canvas = FigureCanvasTkAgg(rti_fig, master=parent)
    rti_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
    try:
        # Ensure the RTI row keeps a reasonable height and doesn't collapse on redraw
        parent.rowconfigure(1, weight=1, minsize=320)
        rti_canvas.get_tk_widget().configure(height=320)
    except Exception:
        pass
    parent.rowconfigure(1, weight=1)
    parent.columnconfigure(0, weight=1)

def run_rti():
    cube = ensure_cube_loaded()
    if cube is None:
        return
    try:
        global last_rti, last_rti_meta, rti_im, rti_cbar
        # Full reset of RTI axes to avoid cumulative shrinking from repeated colorbar insertions
        global rti_ax
        try:
            rti_fig.clf()
            rti_ax = rti_fig.add_subplot(111)
            # Reset artist handles bound to the previous figure/axes
            rti_im = None
            rti_cbar = None
        except Exception:
            pass

        # Decide path based on input type:
        #  - Complex1X/Complex2X: keep original compute_rti() pipeline.
        #  - REAL (Texas Instruments adcFmt=Real): build RTI from range-FFT cube
        #    using half-spectrum range bins so that RTI matches Range FFT / RD.
        is_complex_input = bool(np.iscomplexobj(cube))

        if is_complex_input:
            # ORIGINAL PIPELINE for Complex1X (unchanged)
            last_rti, last_rti_meta = compute_rti(cube, last_params)
        else:
            # REAL path: reuse range FFT cube to ensure consistent half-spectrum handling.
            #
            # Shape of cube: (F, C, RX, Ns)
            # We compute range FFT with use_full_spectrum=False so that only
            # 0..Rmax (Nyquist) is kept. Then we form RTI by collapsing RX and
            # stacking frames*chirps along time.
            try:
                range_fft_cube, rf_meta = compute_range_fft_cube(
                    cube,
                    last_params,
                    window='hann',
                    pad_pow2=False,
                    remove_dc=True,
                    n_fft_range=256,
                    use_full_spectrum=False
                )
                # range_fft_cube shape: (F, C, RX, R)
                F, C, RX, R = range_fft_cube.shape
                # Magnitude and average over RX
                mag = np.abs(range_fft_cube).reshape(F * C, RX, R).mean(axis=1)  # (T, R)
                # Build chirp index axis (time)
                chirp_axis = np.arange(F * C, dtype=int)
                # Range axis in meters from range FFT meta (already half-spectrum, 0..Rmax)
                range_m = rf_meta.get('range_m')
                last_rti = mag
                last_rti_meta = {
                    "chirp_index": chirp_axis,
                    "range_m": range_m,
                    "n_fft_range": rf_meta.get("n_fft_range"),
                    "rx_mode": rf_meta.get("rx_mode", "MRC"),
                    "source": "real_rti_from_range_fft",
                }
            except Exception as e:
                # Fallback: if anything fails, revert to original compute_rti()
                try:
                    last_rti, last_rti_meta = compute_rti(cube, last_params)
                except Exception:
                    raise e

        # Transpose for y=range, x=chirp and convert to dB (add epsilon to avoid -inf)
        img = 20 * np.log10(last_rti.T + 1e-9)  # shape: (range_bins, time)

        # Range axis in meters
        chirp_axis = last_rti_meta.get('chirp_index')
        y_m = last_rti_meta.get('range_m')  # physical range axis in meters

        # Build extent for imshow in meters; origin lower to have 0 m at the bottom.
        if isinstance(y_m, np.ndarray) and y_m.size == img.shape[0]:
            # Use full chirp index span if available, otherwise fall back to 0..N-1
            if isinstance(chirp_axis, np.ndarray) and chirp_axis.size == img.shape[1]:
                x0, x1 = float(chirp_axis[0]), float(chirp_axis[-1])
            else:
                x0, x1 = 0, img.shape[1] - 1
            extent = [x0, x1, float(y_m[0]), float(y_m[-1])]
        else:
            # Fallback to bins if axis missing
            x0, x1 = 0, img.shape[1] - 1
            extent = [x0, x1, 0, img.shape[0] - 1]

        vmin, vmax = np.percentile(img, [5, 99])

        rti_im = rti_ax.imshow(img, aspect='auto', origin='lower', cmap='viridis',
                               extent=extent, vmin=vmin, vmax=vmax)

        try:
            recreate_cbar = (
                (rti_cbar is None) or
                (getattr(rti_cbar, 'ax', None) is None) or
                (getattr(rti_cbar.ax, 'figure', None) is not rti_fig)
            )
            if recreate_cbar:
                rti_cbar = rti_fig.colorbar(rti_im, ax=rti_ax)
            else:
                rti_cbar.update_normal(rti_im)
            rti_cbar.set_label("Magnitude [dB]")
        except Exception:
            # Fallback: force recreate if update failed (e.g., stale artist)
            try:
                rti_cbar = rti_fig.colorbar(rti_im, ax=rti_ax)
                rti_cbar.set_label("Magnitude [dB]")
            except Exception:
                pass
        rti_ax.set_title("Range-Time Intensity (RTI)")
        rti_ax.set_xlabel("Chirp Index")
        # Label in meters if available
        rti_ax.set_ylabel("Range [m]" if isinstance(y_m, np.ndarray) else "Range Bin")
        rti_canvas.draw()
    except Exception as e:
        messagebox.showerror("Error", f"RTI chyba: {e}")


# --- NEW: Range Profile (dBFS) Functions ---
def init_range_profile_plot(parent):
    global rp_fig, rp_ax, rp_canvas
    rp_fig = Figure(figsize=(5, 3), dpi=100)
    rp_ax = rp_fig.add_subplot(111)
    rp_ax.set_title("Range Profile (dBFS)")
    rp_ax.set_xlabel("Range [m]")
    rp_ax.set_ylabel("Magnitude [dBFS]")
    rp_canvas = FigureCanvasTkAgg(rp_fig, master=parent)
    rp_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
    parent.rowconfigure(1, weight=1)
    parent.columnconfigure(0, weight=1)

def run_range_profile():
    cube = ensure_cube_loaded()
    if cube is None:
        return
    try:
        global last_range_profile, last_range_profile_meta

        # Extract necessary parameters
        fs = (last_params.get('digOutSampleRate_MHz') or 0) * 1e6
        slope = (last_params.get('freqSlopeMHz_us') or 0) * 1e12
        adc_bits = last_params.get('dataFmt_adcBits') or 16
        nfft = 256

        if fs == 0 or slope == 0:
            messagebox.showerror("Error", "Sample Rate or Slope is zero. Cannot compute profile.")
            return

        # Use all chirps from the first frame and first RX channel
        chirps_for_profile = cube[0, :, 0, :]

        # Compute the profile
        r_axis, mag_dbfs = range_profile_dbfs(
            x=chirps_for_profile,
            Fs=fs,
            slope=slope,
            Nfft=nfft,
            adcBits=adc_bits
        )
        last_range_profile = mag_dbfs
        last_range_profile_meta = {'range_m': r_axis}

        # Plotting
        rp_ax.clear()
        if isinstance(r_axis, np.ndarray) and r_axis.shape[0] == mag_dbfs.shape[0]:
            rp_ax.plot(r_axis, mag_dbfs)
            rp_ax.set_xlabel("Range [m]")
        else:
            rp_ax.plot(mag_dbfs)
            rp_ax.set_xlabel("Range Bin")
        rp_ax.set_title("Range Profile (averaged)")

        rp_ax.set_ylabel("Magnitude [dBFS]")
        rp_ax.grid(True, alpha=0.3)
        rp_ax.set_ylim(-120, 0) # Typical dBFS range
        rp_canvas.draw()

    except Exception as e:
        messagebox.showerror("Error", f"Range Profile chyba: {e}")


# --- NEW: Range-Doppler Functions ---
def init_rd_plot(parent):
    global rd_fig, rd_ax, rd_canvas
    rd_fig = Figure(figsize=(5, 3), dpi=100)
    rd_ax = rd_fig.add_subplot(111)
    rd_ax.set_title("Range-Doppler Map")
    rd_ax.set_xlabel("Velocity [m/s]")
    rd_ax.set_ylabel("Range [m]")
    rd_canvas = FigureCanvasTkAgg(rd_fig, master=parent)
    rd_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
    parent.rowconfigure(1, weight=1)
    parent.columnconfigure(0, weight=1)

def run_range_doppler():
    cube = ensure_cube_loaded()
    if cube is None:
        return
    try:
        # NEW: ensure RD axes/figure are initialized (in case init wasn't called yet)
        try:
            _ = rd_ax
        except Exception:
            init_rd_plot(frame_rd)

        global last_rd_map, last_rd_meta, rd_im, rd_cbar, rd_play_interval_ms
        # Also reset CFAR plot artifacts when recomputing RD
        global detection_im, detection_cbar, detection_scatter, last_detections, last_detection_peaks  # NEW
        last_detections = None          # NEW
        last_detection_peaks = None     # NEW
        detection_im = None             # NEW
        detection_cbar = None           # NEW
        detection_scatter = None        # NEW

        # Reset plot artist to force full redraw with new data/extents
        rd_im = None

        # NEW: pick RX combine mode from global selector
        try:
            rx_mode = (global_rx_mode_var.get() or "MRC").strip().upper()
        except Exception:
            rx_mode = "MRC"

        # NEW: doppler window selection from GUI (default Hann)
        try:
            use_hann = bool(rd_hann_var.get())
        except Exception:
            use_hann = True
        dop_win = 'hann' if use_hann else None

        rd_map, meta = compute_range_doppler_map(
            cube, last_params,
            n_fft_range=256,
            rx_mode=rx_mode,
            doppler_window=dop_win,
        )
        last_rd_map = rd_map
        last_rd_meta = meta

        # Set playback speed from frame periodicity
        period_ms = last_params.get('framePeriodicity_ms_selected')
        if period_ms and period_ms > 0:
            rd_play_interval_ms = int(period_ms)
            set_global_interval_ms(rd_play_interval_ms)

        # Global color scale
        img_db_all = 20 * np.log10(rd_map + 1e-9)
        vmin, vmax = np.percentile(img_db_all, [10, 99.5])
        if np.isfinite(vmin) and np.isfinite(vmax):
            meta['display_vmin'] = vmin
            meta['display_vmax'] = vmax

        frames = rd_map.shape[0]
        # RD lokální slider odstraněn; publikujeme pouze do globálního
        set_global_frames(frames)

        # Sync CFAR frame slider range & enable (ponecháno)
        if 'detection_frame_slider' in globals():
            detection_frame_slider.configure(to=frames - 1, state="normal")

        update_rd_plot(0)
        # Enable playback buttons
        rd_play_btn.config(state="normal")
        rd_stop_btn.config(state="disabled")

        # NEW: set global playback interval from periodicity
        period_ms = last_params.get('framePeriodicity_ms_selected')
        if period_ms and period_ms > 0:
            rd_play_interval_ms = int(period_ms)
            set_global_interval_ms(rd_play_interval_ms)

    except Exception as e:
        messagebox.showerror("Error", f"Range-Doppler chyba: {e}")

def update_rd_plot(frame_idx: int):
    global rd_im, rd_cbar, rd_zero_line
    # NEW: ensure RD axes/figure are initialized before plotting
    try:
        _ = rd_ax
    except Exception:
        init_rd_plot(frame_rd)

    if last_rd_map is None or last_rd_meta is None:
        return
    f = min(max(frame_idx, 0), last_rd_map.shape[0] - 1)

    img_db = 20 * np.log10(last_rd_map[f] + 1e-9)

    range_axis = last_rd_meta.get('range_axis')
    velocity_axis = last_rd_meta.get('velocity_axis')
    extent = [velocity_axis[0], velocity_axis[-1], range_axis[0], range_axis[-1]]
    vmin = last_rd_meta.get('display_vmin')
    vmax = last_rd_meta.get('display_vmax')

    if rd_im is None:  # First time plotting or after re-running
        rd_ax.clear()
        rd_im = rd_ax.imshow(img_db.T, aspect='auto', origin='lower', cmap='viridis',
                             extent=extent, vmin=vmin, vmax=vmax)
        # Create or update colorbar
        if rd_cbar is None:
            rd_cbar = rd_fig.colorbar(rd_im, ax=rd_ax)
        else:
            rd_cbar.update_normal(rd_im)
        rd_cbar.set_label("Magnitude [dB]")

        rd_ax.set_xlabel("Velocity [m/s]")
        rd_ax.set_ylabel("Range [m]")

        # Apply zoom (auto/manual)
        try:
            auto_v = bool(rd_auto_vrange_var.get())
        except Exception:
            auto_v = True
        if not auto_v:
            try:
                v_min_zoom = float(rd_vmin_var.get())
                v_max_zoom = float(rd_vmax_var.get())
                if v_min_zoom < v_max_zoom:
                    rd_ax.set_xlim(v_min_zoom, v_max_zoom)
            except (ValueError, tk.TclError):
                pass
        else:
            rd_ax.set_xlim(extent[0], extent[1])
    else:
        rd_im.set_data(img_db.T)
        rd_im.set_extent(extent)
        if vmin is not None and vmax is not None:
            rd_im.set_clim(vmin=vmin, vmax=vmax)
        # Re-apply zoom on update
        try:
            auto_v = bool(rd_auto_vrange_var.get())
        except Exception:
            auto_v = True
        if not auto_v:
            try:
                v_min_zoom = float(rd_vmin_var.get())
                v_max_zoom = float(rd_vmax_var.get())
                if v_min_zoom < v_max_zoom:
                    rd_ax.set_xlim(v_min_zoom, v_max_zoom)
            except (ValueError, tk.TclError):
                pass
        else:
            rd_ax.set_xlim(extent[0], extent[1])

    # Draw or remove zero-velocity guide depending on view
    try:
        is_full = (last_rd_meta.get('doppler_view') == 'full')
        # Remove old line if present
        if rd_zero_line is not None:
            try:
                rd_zero_line.remove()
            except Exception:
                pass
            rd_zero_line = None
        if is_full:
            rd_zero_line = rd_ax.axvline(0.0, color='w', alpha=0.6, linewidth=0.8, linestyle='--')
    except Exception:
        pass

    rd_ax.set_title(f"Range-Doppler Map (Frame {f})" + (f" — {last_rd_meta.get('rx_mode')}" if isinstance(last_rd_meta, dict) and last_rd_meta.get('rx_mode') else ""))
    rd_canvas.draw()

def on_rd_frame_change(val):
    try:
        idx = int(float(val))
    except ValueError:
        idx = 0
    # NEW: route via global synchronizer
    apply_global_frame(idx)

# --- NEW: Range-Doppler Playback Functions ---
def _update_rd_frame():
    if not rd_playing:
        return
    # RD lokální slider odstraněn → posouváme globální slider
    try:
        current_val = int(global_frame_slider.get())
        max_val = int(global_frame_slider.cget("to"))
    except Exception:
        current_val, max_val = 0, 0
    if max_val < 0:
        return
    next_val = (current_val + 1) % (max_val + 1)
    apply_global_frame(next_val)
    root.after(rd_play_interval_ms, _update_rd_frame)

def play_rd():
    global rd_playing
    if last_rd_map is None:
        return
    rd_playing = True
    rd_play_btn.config(state="disabled")
    rd_stop_btn.config(state="normal")
    _update_rd_frame()

def stop_rd():
    global rd_playing
    rd_playing = False
    rd_play_btn.config(state="normal")
    rd_stop_btn.config(state="disabled")

# --- NEW: AoA Functions ---

aoa_fig = None
aoa_ax = None
aoa_canvas = None

# --- NEW: Texas AoA Functions ---

def init_texas_aoa_plot(parent):
    global texas_aoa_fig, texas_aoa_ax, texas_aoa_canvas
    texas_aoa_fig = Figure(figsize=(5, 3), dpi=100, constrained_layout=True)
    texas_aoa_ax = texas_aoa_fig.add_subplot(111)
    texas_aoa_ax.set_title("Texas AoA (TI)")
    texas_aoa_canvas = FigureCanvasTkAgg(texas_aoa_fig, master=parent)
    texas_aoa_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
    parent.rowconfigure(1, weight=1)
    parent.columnconfigure(0, weight=1)

def run_texas_aoa():
    global last_texas_aoa_spec, last_texas_aoa_meta, aoa_mimo_mode_var
    cube = ensure_cube_loaded()
    if cube is None:
        return
    try:
        global last_texas_aoa_spec, last_texas_aoa_meta, texas_aoa_im, texas_aoa_cbar, aoa_mimo_mode_var
        
        # Reset colorbar if exists
        if texas_aoa_cbar is not None:
            try:
                texas_aoa_cbar.remove()
            except: pass
            texas_aoa_cbar = None
        texas_aoa_ax.clear()
        texas_aoa_im = None

        r_txt = ""
        try:
            if texas_aoa_range_var is not None:
                r_txt = texas_aoa_range_var.get().strip()
        except Exception:
            r_txt = ""
        r_m = float(r_txt) if r_txt else None
        sel_bin = None
        try:
            if (not r_txt) and (aoa_bin_slider is not None) and (aoa_bin_slider.cget("state") != "disabled"):
                sel_bin = int(aoa_bin_slider.get())
        except Exception:
            sel_bin = None

        range_cube, rf_meta = compute_range_fft_cube(
            cube, last_params, window='hann', pad_pow2=False, remove_dc=True, n_fft_range=256, return_complex=True
        )
        rm_static = False
        try:
            if texas_aoa_remove_static_var is not None:
                rm_static = bool(texas_aoa_remove_static_var.get())
        except Exception: pass
        use_mimo = True
        try:
            if texas_aoa_mimo_var is not None:
                use_mimo = bool(texas_aoa_mimo_var.get())
        except Exception: pass

        view_type = "1D Spectrum"
        try:
            if texas_aoa_view_var is not None:
                view_type = texas_aoa_view_var.get()
        except: pass

        params_aoa = last_params.copy()
        params_aoa['is_frequency_domain'] = True
        params_aoa['range_m'] = rf_meta.get('range_m')
        params_aoa['platform'] = 'TEXAS'
        
        # Pass Shift Spectrum setting to AOA module
        try:
            if texas_aoa_shift_var is not None:
                params_aoa['aoa_shift_spectrum'] = bool(texas_aoa_shift_var.get())
            else:
                params_aoa['aoa_shift_spectrum'] = True # Force default
        except: pass

        m_mode = str(aoa_mimo_mode_var.get()).strip() if (aoa_mimo_mode_var is not None) else "MIMO"
        print(f"[GUI] run_texas_aoa: m_mode='{m_mode}'")
        rx_override = 'auto' if m_mode.upper() == 'FIRST TX' else None
        
        # Method selection
        method = "bartlett"
        try:
            if texas_aoa_method_var is not None:
                method = str(texas_aoa_method_var.get()).lower()
        except: pass

        # Angular range
        ang_limit = 90.0
        try:
            if aoa_angle_limit_var is not None:
                ang_limit = float(aoa_angle_limit_var.get())
        except: pass
        angles = np.arange(-ang_limit, ang_limit + 1.0, 1.0)

        if view_type == "1D Spectrum":
            spec, meta = compute_aoa_spectrum(
                range_cube, params_aoa, range_m=r_m, range_bin=sel_bin, remove_static=rm_static, mimo=use_mimo,
                invert_tx_phase=False, mimo_mode=m_mode, rx_order_override=rx_override, method=method,
                angles_deg=angles
            )
            meta['view_type'] = "1D Spectrum"
        elif view_type == "Range-Azimuth":
            spec, meta = compute_range_azimuth_map(
                range_cube, params_aoa, remove_static=rm_static, mimo=use_mimo,
                invert_tx_phase=False, mimo_mode=m_mode,
                rx_order_override=rx_override, method=method,
                angles_deg=angles
            )
            meta['view_type'] = "Range-Azimuth"
        elif view_type == "Azimuth-Elevation":
            # For Az-El we use a slightly coarser grid if it's large, but let's respect limits
            angles_az = np.arange(-ang_limit, ang_limit + 2.0, 2.0)
            spec, meta = compute_azimuth_elevation_map(
                range_cube, params_aoa, range_bin=sel_bin, remove_static=rm_static, mimo=use_mimo,
                invert_tx_phase=False, rx_order_override=rx_override, method=method,
                angles_az_deg=angles_az
            )
            meta['view_type'] = "Azimuth-Elevation"
        else:
            raise ValueError(f"Unknown view type: {view_type}")

        last_texas_aoa_spec, last_texas_aoa_meta = spec, meta
        set_global_frames(spec.shape[0])
        update_texas_aoa_plot(0)
    except Exception as e:
        messagebox.showerror("Error", f"Texas AoA chyba: {e}")

def update_texas_aoa_plot(frame_idx: int):
    if last_texas_aoa_spec is None or last_texas_aoa_meta is None:
        return
    global texas_aoa_im, texas_aoa_cbar
    f = min(max(frame_idx, 0), last_texas_aoa_spec.shape[0]-1)
    
    view_type = last_texas_aoa_meta.get('view_type')
    if view_type is None:
        if last_texas_aoa_spec.ndim == 3:
            if 'angles_az_deg' in last_texas_aoa_meta:
                view_type = "Azimuth-Elevation"
            else:
                view_type = "Range-Azimuth"
        else:
            view_type = "1D Spectrum"

    texas_aoa_ax.clear()
    if texas_aoa_cbar is not None:
        try:
            texas_aoa_cbar.remove()
        except: pass
        texas_aoa_cbar = None

    # Apply Range Crop if available
    min_r = 0.0
    try:
        if aoa_min_range_var is not None and aoa_min_range_var.get().strip():
            min_r = float(aoa_min_range_var.get())
    except Exception:
        pass
    
    if view_type == "1D Spectrum":
        ang = last_texas_aoa_meta.get('angles_deg')
        y = last_texas_aoa_spec[f]
        texas_aoa_ax.plot(ang, y)
        texas_aoa_ax.set_xlabel("Angle [deg]")
        texas_aoa_ax.set_ylabel("Relative Power [dB]")
        texas_aoa_ax.set_title(f"Texas AoA Spectrum (Frame {f})")
        texas_aoa_ax.grid(True, alpha=0.3)
        if ang is not None and len(ang) > 0:
            texas_aoa_ax.set_xlim(ang[0], ang[-1])
        texas_aoa_ax.set_ylim(-40, 0)
    elif view_type == "Range-Azimuth":
        data = last_texas_aoa_spec[f]
        ang = last_texas_aoa_meta.get('angles_deg')
        rng = last_texas_aoa_meta.get('range_axis')
        
        # Apply range crop
        if min_r > 0 and rng is not None:
            mask = rng >= min_r
            if np.any(mask):
                data = data[mask, :]
                rng = rng[mask]

        # Dynamic color scaling based on visible data
        vmax_dyn = np.max(data)
        vmin_dyn = vmax_dyn - 35 # Use a fixed 35dB dynamic range from the new peak

        texas_aoa_im = texas_aoa_ax.imshow(
            data, aspect='auto', origin='lower',
            extent=[ang[0], ang[-1], rng[0], rng[-1]],
            vmin=vmin_dyn, vmax=vmax_dyn, cmap='jet'
        )
        texas_aoa_cbar = texas_aoa_fig.colorbar(texas_aoa_im, ax=texas_aoa_ax, label='dB')
        texas_aoa_ax.set_xlabel("Angle [deg]")
        texas_aoa_ax.set_ylabel("Range [m]")
        texas_aoa_ax.set_title(f"Texas Range-Azimuth (Frame {f})")
    elif view_type == "Azimuth-Elevation":
        data = last_texas_aoa_spec[f]
        az = last_texas_aoa_meta.get('angles_az_deg')
        el = last_texas_aoa_meta.get('angles_el_deg')
        texas_aoa_im = texas_aoa_ax.imshow(
            data, aspect='equal', origin='lower',
            extent=[az[0], az[-1], el[0], el[-1]],
            vmin=-20, vmax=0, cmap='jet'
        )
        texas_aoa_cbar = texas_aoa_fig.colorbar(texas_aoa_im, ax=texas_aoa_ax, label='dB')
        texas_aoa_ax.set_xlabel("Azimuth [deg]")
        texas_aoa_ax.set_ylabel("Elevation [deg]")
        texas_aoa_ax.set_title(f"Texas Azimuth-Elevation (Frame {f})")

    try:
        texas_aoa_canvas.draw_idle()
    except Exception: pass

def init_aoa_plot(parent):
    global aoa_fig, aoa_ax, aoa_canvas
    aoa_fig = Figure(figsize=(5, 3), dpi=100)
    aoa_ax = aoa_fig.add_subplot(111)
    aoa_ax.set_title("Angle-of-Arrival (AoA)")
    aoa_ax.set_xlabel("Angle [deg]")
    aoa_ax.set_ylabel("Relative Power [dB]")
    aoa_canvas = FigureCanvasTkAgg(aoa_fig, master=parent)
    aoa_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
    parent.rowconfigure(1, weight=1)
    parent.columnconfigure(0, weight=1)

def run_aoa():
    global last_aoa_spec, last_aoa_meta, aoa_mimo_mode_var
    cube = ensure_cube_loaded()
    if cube is None:
        return
    try:
        global last_aoa_spec, last_aoa_meta, aoa_mimo_mode_var
        # Priority: if user entered Range[m], use it; otherwise use slider-selected range bin
        r_txt = aoa_range_var.get().strip()
        r_m = float(r_txt) if r_txt else None
        sel_bin = None
        try:
            if (not r_txt) and (aoa_bin_slider is not None) and (aoa_bin_slider.cget("state") != "disabled"):
                sel_bin = int(aoa_bin_slider.get())
        except Exception:
            sel_bin = None

        # Compute Range FFT cube (complex) for AoA to ensure we have frequency data
        range_cube, rf_meta = compute_range_fft_cube(
            cube, last_params, window='hann', pad_pow2=False, remove_dc=True, n_fft_range=256, return_complex=True
        )
        # Use remove_static if the checkbox exists
        rm_static = True
        try:
            if aoa_remove_static_var is not None:
                rm_static = bool(aoa_remove_static_var.get())
        except Exception:
            pass

        use_mimo = True
        try:
            if aoa_mimo_var is not None:
                use_mimo = bool(aoa_mimo_var.get())
        except Exception:
            pass

        # NEW
        invert_phase = False
        try:
            if aoa_invert_phase_var is not None:
                invert_phase = bool(aoa_invert_phase_var.get())
        except Exception:
            pass

        params_aoa = last_params.copy()
        params_aoa['is_frequency_domain'] = True
        params_aoa['range_m'] = rf_meta.get('range_m')
        params_aoa['platform'] = 'TEXAS' # NEW: Trigger Texas antenna logic
        
        m_mode = str(aoa_mimo_mode_var.get()).strip() if (aoa_mimo_mode_var is not None) else "MIMO"
        print(f"[GUI] run_aoa: m_mode='{m_mode}'")
        rx_override = 'auto' if m_mode.upper() == 'FIRST TX' else None

        # Angular range
        ang_limit = 90.0
        try:
            if aoa_angle_limit_var is not None:
                ang_limit = float(aoa_angle_limit_var.get())
        except: pass
        angles = np.arange(-ang_limit, ang_limit + 1.0, 1.0)

        spec, meta = compute_aoa_spectrum(
            range_cube, params_aoa, range_m=r_m, range_bin=sel_bin, remove_static=rm_static, mimo=use_mimo,
            invert_tx_phase=invert_phase, mimo_mode=m_mode, rx_order_override=rx_override,
            angles_deg=angles
        )
        last_aoa_spec, last_aoa_meta = spec, meta

        # Publish frames to global slider
        set_global_frames(spec.shape[0])

        # After first run, enable and size the bin slider to number of range bins in the cube
        try:
            R = range_cube.shape[-1]
            if aoa_bin_slider is not None:
                aoa_bin_slider.configure(to=max(0, R-1), state="normal")
                # Sync slider to the actual bin used by compute_aoa_spectrum
                used_bin = int(meta.get('range_bin', 0))
                aoa_bin_slider.set(used_bin)
        except Exception:
            pass

        # Update label with bin and meters (if available)
        try:
            used_bin = int(meta.get('range_bin', 0))
            virt_rx = meta.get('virt_rx_count', 4)
            if 'range_m' in meta:
                aoa_bin_label_var.set(f"bin {used_bin} (≈ {meta['range_m']:.2f} m) | {virt_rx} virt. antennas")
            else:
                aoa_bin_label_var.set(f"bin {used_bin} | {virt_rx} virt. antennas")
        except Exception:
            pass

        update_aoa_plot(0)
    except Exception as e:
        messagebox.showerror("Error", f"AoA chyba: {e}")

def update_aoa_plot(frame_idx: int):
    if last_aoa_spec is None or last_aoa_meta is None:
        return
    f = min(max(frame_idx, 0), last_aoa_spec.shape[0]-1)
    ang = last_aoa_meta.get('angles_deg')
    y = last_aoa_spec[f]
    aoa_ax.clear()
    aoa_ax.plot(ang, y)
    aoa_ax.set_xlabel("Angle [deg]")
    aoa_ax.set_ylabel("Relative Power [dB]")
    title = f"AoA Spectrum (Frame {f})"
    try:
        if 'range_m' in last_aoa_meta:
            title += f" — r≈{last_aoa_meta['range_m']:.2f} m"
            try:
                used_bin = int(last_aoa_meta.get('range_bin', 0))
                aoa_bin_label_var.set(f"bin {used_bin} (≈ {last_aoa_meta['range_m']:.2f} m)")
            except Exception:
                pass
        else:
            used_bin = int(last_aoa_meta.get('range_bin', 0))
            title += f" — bin {used_bin}"
            try:
                aoa_bin_label_var.set(f"bin {used_bin}")
            except Exception:
                pass
    except Exception:
        pass
    aoa_ax.set_title(title)
    aoa_ax.grid(True, alpha=0.3)
    # Respect angular limits from meta if available
    if ang is not None and len(ang) > 0:
        aoa_ax.set_xlim(ang[0], ang[-1])
    else:
        aoa_ax.set_xlim(-90, 90)
    aoa_ax.set_ylim(-40, 0)
    aoa_canvas.draw_idle()
# --- NEW: 2D AoA Functions ---
aoa_2d_fig = None
aoa_2d_ax = None
aoa_2d_canvas = None
def init_aoa_2d_plot(parent):
    global aoa_2d_fig, aoa_2d_ax, aoa_2d_canvas
    aoa_2d_fig = Figure(figsize=(5, 3), dpi=100, constrained_layout=True)
    aoa_2d_ax = aoa_2d_fig.add_subplot(111)
    aoa_2d_ax.set_title("2D Angle-of-Arrival (Range-Azimuth)")
    aoa_2d_ax.set_xlabel("Angle [deg]")
    aoa_2d_ax.set_ylabel("Range [m]")
    aoa_2d_canvas = FigureCanvasTkAgg(aoa_2d_fig, master=parent)
    aoa_2d_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
    parent.rowconfigure(1, weight=1)
    parent.columnconfigure(0, weight=1)
def run_aoa_2d():
    global last_aoa_2d_map, last_aoa_2d_meta, aoa_mimo_mode_var
    cube = ensure_cube_loaded()
    if cube is None:
        return
    try:
        global last_aoa_2d_map, last_aoa_2d_meta, aoa_2d_im, aoa_2d_cbar, aoa_mimo_mode_var
        
        # Reset plot if needed
        try:
            if aoa_2d_cbar is not None:
                aoa_2d_cbar.remove()
                aoa_2d_cbar = None
            aoa_2d_ax.clear()
            aoa_2d_im = None
        except Exception:
            pass
        rm_static = True
        try:
            if aoa_2d_remove_static_var is not None:
                rm_static = bool(aoa_2d_remove_static_var.get())
        except Exception:
            pass

        use_mimo = True
        try:
            if aoa_2d_mimo_var is not None:
                use_mimo = bool(aoa_2d_mimo_var.get())
        except Exception:
            pass

        # NEW
        invert_phase = False
        try:
            if aoa_invert_phase_var is not None: # Use the shared variable
                invert_phase = bool(aoa_invert_phase_var.get())
        except Exception:
            pass

        # Compute Range FFT cube (complex) for AoA to ensure we have frequency data
        range_cube, rf_meta = compute_range_fft_cube(
            cube, last_params, window='hann', pad_pow2=False, remove_dc=True, n_fft_range=256, return_complex=True
        )

        # Spočítáme 2D mapu
        params_aoa = last_params.copy()
        params_aoa['is_frequency_domain'] = True
        params_aoa['range_m'] = rf_meta.get('range_m')
        params_aoa['platform'] = 'TEXAS' # NEW: Trigger Texas antenna logic
        
        # Pass Shift Spectrum setting to AOA module
        try:
            if aoa_2d_shift_var is not None:
                params_aoa['aoa_shift_spectrum'] = bool(aoa_2d_shift_var.get())
            else:
                params_aoa['aoa_shift_spectrum'] = True # Force default
        except: pass

        m_mode = str(aoa_mimo_mode_var.get()).strip() if (aoa_mimo_mode_var is not None) else "MIMO"
        print(f"[GUI] run_aoa_2d: m_mode='{m_mode}'")
        rx_override = 'auto' if m_mode.upper() == 'FIRST TX' else None

        # Method selection
        method = "bartlett"
        try:
            if aoa_2d_method_var is not None:
                method = str(aoa_2d_method_var.get()).lower()
        except: pass

        # Angular range
        ang_limit = 90.0
        try:
            if aoa_angle_limit_var is not None:
                ang_limit = float(aoa_angle_limit_var.get())
        except: pass
        angles = np.arange(-ang_limit, ang_limit + 1.0, 1.0)

        ra_map, meta = compute_range_azimuth_map(
            range_cube, params_aoa, remove_static=rm_static, mimo=use_mimo,
            invert_tx_phase=invert_phase, mimo_mode=m_mode,
            rx_order_override=rx_override, method=method,
            angles_deg=angles
        )
        last_aoa_2d_map, last_aoa_2d_meta = ra_map, meta
        
        # Update UI label if exists
        try:
            virt_rx = meta.get('virt_rx_count', 4)
            if 'aoa_2d_info_var' in globals():
                aoa_2d_info_var.set(f"{virt_rx} virt. antennas")
        except Exception:
            pass

        # Publish frames to global slider
        set_global_frames(ra_map.shape[0])
        update_aoa_2d_plot(0)
    except Exception as e:
        messagebox.showerror("Error", f"2D AoA chyba: {e}")
def update_aoa_2d_plot(frame_idx: int):
    if last_aoa_2d_map is None or last_aoa_2d_meta is None:
        return
    
    global aoa_2d_im, aoa_2d_cbar
    f = min(max(frame_idx, 0), last_aoa_2d_map.shape[0]-1)
    data = last_aoa_2d_map[f] # (R, A)
    
    ang = last_aoa_2d_meta.get('angles_deg')
    rng = last_aoa_2d_meta.get('range_axis')

    # Apply Range Crop if available
    min_r = 0.0
    try:
        if aoa_min_range_var is not None and aoa_min_range_var.get().strip():
            min_r = float(aoa_min_range_var.get())
    except Exception:
        pass
    
    if min_r > 0 and rng is not None:
        mask = rng >= min_r
        if np.any(mask):
            data = data[mask, :]
            rng = rng[mask]
    
    # Dynamic color scaling based on visible data
    vmax_dyn = np.max(data)
    vmin_dyn = vmax_dyn - 35 # Use a fixed 35dB dynamic range from the new peak

    use_cartesian = False
    try:
        if aoa_2d_cartesian_var is not None:
            use_cartesian = bool(aoa_2d_cartesian_var.get())
    except Exception:
        pass

    # Clear and redraw everything
    aoa_2d_ax.clear()
    if aoa_2d_cbar is not None:
        try:
            aoa_2d_cbar.remove()
        except Exception:
            pass
        aoa_2d_cbar = None

    aoa_2d_im = None
    if use_cartesian:
        # Cartesian (X-Y) Projection
        # Calculate edges for pcolormesh to avoid non-monotonic warning
        d_ang = ang[1] - ang[0] if len(ang) > 1 else 1.0
        d_rng = rng[1] - rng[0] if len(rng) > 1 else 0.1
        
        ang_edges = np.deg2rad(np.concatenate([[ang[0] - d_ang/2], ang + d_ang/2]))
        rng_edges = np.concatenate([[rng[0] - d_rng/2], rng + d_rng/2])
        
        tt_e, rr_e = np.meshgrid(ang_edges, rng_edges)
        xx_e = rr_e * np.sin(tt_e)
        yy_e = rr_e * np.cos(tt_e)
        
        aoa_2d_im = aoa_2d_ax.pcolormesh(
            xx_e, yy_e, data, cmap='jet', vmin=vmin_dyn, vmax=vmax_dyn, shading='flat'
        )
        aoa_2d_ax.set_aspect('equal')
        aoa_2d_ax.grid(True, alpha=0.2)
        aoa_2d_ax.set_xlabel("X [m]")
        aoa_2d_ax.set_ylabel("Y (Distance) [m]")
        aoa_2d_ax.set_title(f"2D AoA (Cartesian) - Frame {f}")
    else:
        # Range-Angle (Polar-style) projection
        # Use pcolormesh with edges for consistency and to avoid imshow artifacts
        d_ang = ang[1] - ang[0] if len(ang) > 1 else 1.0
        d_rng = rng[1] - rng[0] if len(rng) > 1 else 0.1
        ang_edges = np.concatenate([[ang[0] - d_ang/2], ang + d_ang/2])
        rng_edges = np.concatenate([[rng[0] - d_rng/2], rng + d_rng/2])
        
        aa, rr = np.meshgrid(ang_edges, rng_edges)
        aoa_2d_im = aoa_2d_ax.pcolormesh(
            aa, rr, data, cmap='jet', vmin=vmin_dyn, vmax=vmax_dyn, shading='flat'
        )
        aoa_2d_ax.set_aspect('auto')
        aoa_2d_ax.grid(True, alpha=0.2)
        aoa_2d_ax.set_xlabel("Angle [deg]")
        aoa_2d_ax.set_ylabel("Range [m]")
        aoa_2d_ax.set_title(f"2D AoA (Range-Angle) - Frame {f}")

    # Add colorbar back
    try:
        aoa_2d_cbar = aoa_2d_fig.colorbar(aoa_2d_im, ax=aoa_2d_ax)
        aoa_2d_cbar.set_label("Relative Power [dB]")
    except Exception:
        pass

    aoa_2d_canvas.draw_idle()

# --- NEW: Micro-Doppler Functions ---
md_fig = None
md_ax = None
md_canvas = None

def init_md_plot(parent):
    global md_fig, md_ax, md_canvas
    md_fig = Figure(figsize=(5, 3), dpi=100, constrained_layout=True)
    md_ax = md_fig.add_subplot(111)
    md_ax.set_title("Micro-Doppler (Time-Doppler)")
    md_ax.set_xlabel("Frame")
    md_ax.set_ylabel("Velocity [m/s]")
    md_canvas = FigureCanvasTkAgg(md_fig, master=parent)
    md_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
    parent.rowconfigure(1, weight=1)
    parent.columnconfigure(0, weight=1)

def run_micro_doppler():
    cube = ensure_cube_loaded()
    if cube is None:
        return
    try:
        global last_md_map, last_md_meta, md_im, md_cbar
        
        # Ensure Range-Doppler map is computed or compute it now
        # MD is derived from RD map by summing across range bins
        rx_mode = (global_rx_mode_var.get() or "MRC").strip().upper()
        use_hann = bool(rd_hann_var.get())
        dop_win = 'hann' if use_hann else None

        # Compute RD map (full cube F, D, R)
        rd_map, meta = compute_range_doppler_map(
            cube, last_params,
            n_fft_range=256,
            rx_mode=rx_mode,
            doppler_window=dop_win,
        )
        
        # Get range bins from UI
        try:
            r0 = int(md_r0_var.get())
            r1 = int(md_r1_var.get())
        except Exception:
            r0, r1 = 0, rd_map.shape[2] - 1
        
        r0 = max(0, min(rd_map.shape[2] - 1, r0))
        r1 = max(0, min(rd_map.shape[2] - 1, r1))
        if r1 < r0: r0, r1 = r1, r0
        
        # Sum across selected range bins: (F, D, R) -> (F, D)
        md_map = np.sum(rd_map[:, :, r0:r1+1], axis=2)
        
        # Convert to dB
        md_db = 20 * np.log10(md_map + 1e-12)
        
        last_md_map = md_db
        last_md_meta = meta.copy()
        last_md_meta['md_r0'] = r0
        last_md_meta['md_r1'] = r1
        
        # Global color scale
        vmin, vmax = np.percentile(md_db, [5, 99.5])
        if np.isfinite(vmin) and np.isfinite(vmax):
            last_md_meta['display_vmin'] = vmin
            last_md_meta['display_vmax'] = vmax
            
        update_md_plot(int(global_frame_slider.get()))
    except Exception as e:
        messagebox.showerror("Error", f"Micro-Doppler chyba: {e}")

# --- NEW: Movement Map Functions ---
mov_fig = None
mov_ax = None
mov_canvas = None

def init_mov_plot(parent):
    global mov_fig, mov_ax, mov_canvas
    mov_fig = Figure(figsize=(5, 3), dpi=100, constrained_layout=True)
    mov_ax = mov_fig.add_subplot(111)
    mov_ax.set_title("Movement Map (2D)")
    mov_ax.set_xlabel("Angle [deg]")
    mov_ax.set_ylabel("Range [m]")
    mov_canvas = FigureCanvasTkAgg(mov_fig, master=parent)
    mov_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
    parent.rowconfigure(1, weight=1)
    parent.columnconfigure(0, weight=1)

def run_movement_map():
    global last_mov_map, last_mov_meta, mov_im, mov_cbar, last_aoa_2d_map, last_aoa_2d_meta
    cube = ensure_cube_loaded()
    if cube is None: return
    
    try:
        # We need Range-Azimuth maps for all frames
        # This can be slow, so we'll try to use cached last_aoa_2d_map if possible and relevant
        ra_cube = last_aoa_2d_map
        if ra_cube is None or ra_cube.shape[0] < 4:
            # Recompute Range-Azimuth map
            range_cube, rf_meta = compute_range_fft_cube(
                cube, last_params, window='hann', pad_pow2=False, remove_dc=True, n_fft_range=256, return_complex=True
            )
            params_aoa = last_params.copy()
            params_aoa['is_frequency_domain'] = True
            params_aoa['range_m'] = rf_meta.get('range_m')
            params_aoa['platform'] = 'TEXAS'
            
            # Use current UI settings for AoA 2D
            m_mode = str(aoa_mimo_mode_var.get()).strip() if (aoa_mimo_mode_var is not None) else "MIMO"
            rx_override = 'auto' if m_mode.upper() == 'FIRST TX' else None
            method = str(aoa_2d_method_var.get()).lower() if (aoa_2d_method_var is not None) else "bartlett"
            
            ang_limit = 90.0
            try:
                if aoa_angle_limit_var is not None: ang_limit = float(aoa_angle_limit_var.get())
            except: pass
            angles = np.arange(-ang_limit, ang_limit + 1.0, 1.0)
            
            print("[Movement Map] Computing base Range-Azimuth maps for all frames...")
            ra_cube, meta_2d = compute_range_azimuth_map(
                range_cube, params_aoa, remove_static=True, mimo=True, 
                mimo_mode=m_mode, rx_order_override=rx_override, method=method,
                angles_deg=angles
            )
            # Cache it so we don't recompute if we just change movement map settings
            last_aoa_2d_map, last_aoa_2d_meta = ra_cube, meta_2d
        else:
            meta_2d = last_aoa_2d_meta
            print("[Movement Map] Using cached Range-Azimuth maps.")

        F, R, A = ra_cube.shape
        
        # Get Temporal FFT params
        band = mov_band_var.get() # "Movement", "Breathing", "HR"
        accum = mov_accum_var.get() # "Full", "Sliding Window"
        win_size = int(mov_win_var.get())
        
        # Frequency analysis
        fps = 15.0
        try:
            period_ms = last_params.get('framePeriodicity_ms_selected')
            if period_ms: fps = 1000.0 / float(period_ms)
        except Exception: pass
        
        # Define bands (same as in 1D View)
        # Movement: 0.60-0.80 and >3.50 Hz
        # Breathing: 0.10-0.60 Hz
        # HR: 0.80-3.50 Hz
        
        # To speed up, we can process frame by frame or in chunks
        # But for movement map, we usually want to see where things happened *over time*
        
        print(f"[Movement Map] Processing temporal energy for band: {band}...")
        
        # Compute power spectrum per pixel
        # ra_cube is in dB? No, compute_range_azimuth_map returns dB by default.
        # We need linear scale for temporal FFT to be meaningful.
        # Let's check if we can get linear scale or invert dB.
        # Actually, let's just use the ra_cube as is (dB) and look at variations, 
        # or better, we should have modified compute_range_azimuth_map to return linear.
        # For now, let's invert dB:
        ra_linear = 10**(ra_cube / 10.0)
        
        # Temporal FFT along Frame axis
        # sig shape: (F, R, A)
        # We want to know how much energy is in a certain band at each (R, A)
        
        # DC removal and windowing
        sig = ra_linear - np.mean(ra_linear, axis=0, keepdims=True)
        # Apply window along F
        win = np.hanning(F).reshape(-1, 1, 1)
        sig = sig * win
        
        Nfft = int(2**np.ceil(np.log2(max(32, F))))
        X = np.fft.rfft(sig, n=Nfft, axis=0)
        psd = np.abs(X) # (Nfft//2 + 1, R, A)
        freqs = np.fft.rfftfreq(Nfft, d=1.0/fps)
        
        # Extract band energy
        f_nyq = freqs[-1]
        if band == "Breathing":
            mask = (freqs >= 0.10) & (freqs <= 0.60)
        elif band == "HR":
            mask = (freqs >= 0.80) & (freqs <= min(3.50, f_nyq))
        else: # "Movement"
            mask = ((freqs >= 0.60) & (freqs < 0.80)) | (freqs > 3.50)
            
        band_energy = np.sum(psd[mask, :, :], axis=0) # (R, A)
        
        # Normalize for display
        band_energy = 10 * np.log10(band_energy + 1e-12)
        band_energy -= np.max(band_energy)
        
        last_mov_map = band_energy
        last_mov_meta = {
            'angles_deg': meta_2d.get('angles_deg'),
            'range_axis': meta_2d.get('range_axis'),
            'band': band,
            'fps': fps
        }
        
        update_movement_map_plot()
    except Exception as e:
        import traceback
        traceback.print_exc()
        messagebox.showerror("Error", f"Movement Map chyba: {e}")

def update_movement_map_plot(frame_idx=None):
    global mov_im, mov_cbar
    if last_mov_map is None or last_mov_meta is None: return
    
    mov_ax.clear()
    if mov_cbar is not None:
        try: mov_cbar.remove()
        except: pass
        mov_cbar = None
        
    data = last_mov_map
    ang = last_mov_meta.get('angles_deg')
    rng = last_mov_meta.get('range_axis')
    
    # Apply Range Crop
    min_r = 0.3
    try:
        if aoa_min_range_var is not None and aoa_min_range_var.get().strip():
            min_r = float(aoa_min_range_var.get())
    except: pass
    
    if min_r > 0 and rng is not None:
        mask = rng >= min_r
        if np.any(mask):
            data = data[mask, :]
            rng = rng[mask]

    vmax = np.max(data)
    vmin = vmax - 30
    
    mov_im = mov_ax.imshow(
        data, aspect='auto', origin='lower',
        extent=[ang[0], ang[-1], rng[0], rng[-1]],
        vmin=vmin, vmax=vmax, cmap='hot'
    )
    mov_cbar = mov_fig.colorbar(mov_im, ax=mov_ax, label='Relative Energy [dB]')
    mov_ax.set_xlabel("Angle [deg]")
    mov_ax.set_ylabel("Range [m]")
    band = last_mov_meta.get('band', "")
    mov_ax.set_title(f"Accumulated Movement Map ({band})")
    
    mov_canvas.draw_idle()

def update_md_plot(frame_idx: int):
    global md_im, md_cbar, md_indicator_line
    if last_md_map is None or last_md_meta is None:
        return
    
    f = min(max(frame_idx, 0), last_md_map.shape[0] - 1)
    
    # MD data is (F, D)
    img = last_md_map.T # (D, F)
    
    vel = last_md_meta.get('velocity_axis')
    frames = np.arange(last_md_map.shape[0])
    extent = [frames[0], frames[-1], vel[0], vel[-1]]
    
    vmin = last_md_meta.get('display_vmin')
    vmax = last_md_meta.get('display_vmax')
    
    if md_im is None:
        md_ax.clear()
        md_im = md_ax.imshow(img, aspect='auto', origin='lower', cmap='viridis',
                             extent=extent, vmin=vmin, vmax=vmax)
        if md_cbar is None:
            md_cbar = md_fig.colorbar(md_im, ax=md_ax)
        else:
            md_cbar.update_normal(md_im)
        md_cbar.set_label("Magnitude [dB]")
        md_ax.set_xlabel("Frame")
        md_ax.set_ylabel("Velocity [m/s]")
    else:
        md_im.set_data(img)
        md_im.set_extent(extent)
        if vmin is not None and vmax is not None:
            md_im.set_clim(vmin=vmin, vmax=vmax)
            
    # Update indicator line for current frame
    if md_indicator_line is not None:
        try:
            md_indicator_line.remove()
        except Exception:
            pass
    md_indicator_line = md_ax.axvline(f, color='red', alpha=0.7, linestyle='--', linewidth=1)
    
    r0, r1 = last_md_meta.get('md_r0', 0), last_md_meta.get('md_r1', 0)
    md_ax.set_title(f"Micro-Doppler (Range Bins {r0}-{r1})")
    md_canvas.draw_idle()

# --- NEW: 1D View Functions ---
def init_view1d_plot(parent):
    global view1d_fig, view1d_ax_prof, view1d_ax_spec, view1d_ax_bars, view1d_canvas
    view1d_fig = Figure(figsize=(7, 5), constrained_layout=True)
    gs = view1d_fig.add_gridspec(2, 2, width_ratios=[3, 1])
    view1d_ax_prof = view1d_fig.add_subplot(gs[0, 0])
    view1d_ax_spec = view1d_fig.add_subplot(gs[1, 0])
    view1d_ax_bars = view1d_fig.add_subplot(gs[:, 1])
    
    view1d_ax_prof.set_title("Profile (Current Frame)")
    view1d_ax_spec.set_title("Doppler Spectrum @ Range Bin")
    view1d_ax_bars.set_title("Temporal FFT")
    
    view1d_canvas = FigureCanvasTkAgg(view1d_fig, master=parent)
    view1d_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
    parent.rowconfigure(1, weight=1)
    parent.columnconfigure(0, weight=1)

def run_view1d():
    if last_rd_map is None:
        run_range_doppler()
    if last_rd_map is None:
        return
    
    # Initialize bin slider if not done
    Kr = last_rd_map.shape[2]
    if view1d_slider is not None:
        view1d_slider.configure(to=Kr - 1, state="normal")
        
    update_view1d_plot(int(global_frame_slider.get()))

def on_view1d_rbin_change(val):
    try:
        rbin = int(float(val))
        view1d_rbin_label_var.set(f"{rbin}")
        # Sync with AoA bin slider if it exists and we are in sync mode
        # (optional, but consistent)
        update_view1d_plot(int(global_frame_slider.get()))
    except Exception:
        pass

def _update_view1d_temporal(rbin, current_frame):
    global view1d_ax_bars
    if view1d_ax_bars is None:
        return
    view1d_ax_bars.clear()
    if last_rd_map is None: return
    
    F, D, R = last_rd_map.shape
    
    # 1. Get source signal across frames (F,)
    try:
        src = view1d_band_src_var.get()
        # Magnitude average across Doppler per frame and range
        prof_all = np.mean(last_rd_map, axis=1) # (F, R)
        
        if src == "Whole Range":
            sig = np.mean(prof_all, axis=1)
        elif src == "Neighborhood ±N":
            try:
                w = int(view1d_band_w_var.get())
            except Exception:
                w = 6
            rlo = max(0, rbin - w)
            rhi = min(R - 1, rbin + w)
            sig = np.mean(prof_all[:, rlo:rhi+1], axis=1)
        else: # "Range Bin"
            sig = prof_all[:, rbin]
            
        # Optional: restrict to local window
        if view1d_just_frame_var.get():
            # Approx 5 second window or at least 16 frames
            # Try to get FPS from metadata or defaults
            fps = 15.0
            try:
                period_ms = last_params.get('framePeriodicity_ms_selected')
                if period_ms: fps = 1000.0 / float(period_ms)
            except Exception: pass
            
            W = max(16, int(round(5.0 * fps)))
            f0 = max(0, current_frame - W + 1)
            f1 = current_frame + 1
            sig = sig[f0:f1]
            
        if len(sig) < 4:
            view1d_ax_bars.text(0.5, 0.5, "Too few frames", ha='center', va='center')
            return
            
        # 2. Compute FFT
        sig = sig - np.mean(sig)
        win = np.hanning(len(sig))
        xw = sig * win
        Nfft = int(2**np.ceil(np.log2(max(32, len(xw)))))
        X = np.fft.rfft(xw, n=Nfft)
        
        # Frequency axis
        fps = 15.0
        try:
            period_ms = last_params.get('framePeriodicity_ms_selected')
            if period_ms: fps = 1000.0 / float(period_ms)
        except Exception: pass
        
        freqs = np.fft.rfftfreq(Nfft, d=1.0/fps)
        psd = np.abs(X)
        
        # 3. Najdi BR/HR peaky a připrav pásma pro bar graf
        def find_bpm(f_min, f_max):
            mask = (freqs >= f_min) & (freqs <= f_max)
            if not np.any(mask):
                return 0.0, 0.0, 0.0
            idx = int(np.argmax(psd[mask]))
            f_peak = float(freqs[mask][idx])
            peak_val = float(psd[mask][idx])
            return f_peak, f_peak * 60.0, peak_val

        # Typická pásma (disjunktní):
        #  - Noise/DC:   0.00–0.10 Hz
        #  - Breathing:  0.10–0.60 Hz
        #  - HR:         0.80–3.50 Hz
        #  - Movement:   (0.60–0.80) ULF + >3.50 Hz (rychlé pohyby)
        f_nyq = float(freqs[-1]) if freqs.size > 0 else 0.0
        masks = {
            "Noise/DC": (freqs >= 0.00) & (freqs < 0.10),
            "Breathing": (freqs >= 0.10) & (freqs <= 0.60),
            "HR": (freqs >= 0.80) & (freqs <= min(3.50, f_nyq)),
            "Movement": ((freqs >= 0.60) & (freqs < 0.80)) | (freqs > 3.50)
        }

        band_energy = {}
        total = float(np.sum(psd) + 1e-12)
        for k, m in masks.items():
            band_energy[k] = float(np.sum(psd[m])) / total if total > 0 else 0.0

        # Najdi peaky pro BR/HR
        br_f, br_bpm, br_peak = find_bpm(0.10, 0.60)
        hr_f, hr_bpm, hr_peak = find_bpm(0.80, min(3.50, f_nyq if f_nyq > 0 else 3.50))

        # 4. Plot BAR graph (horizontal) with percentages
        cats = ["Noise/DC", "Breathing", "HR", "Movement"]
        vals = [band_energy.get(c, 0.0) * 100.0 for c in cats]
        colors = ['#888888', '#1f77b4', '#d62728', '#2ca02c']
        
        y_pos = np.arange(len(cats))
        view1d_ax_bars.barh(y_pos, vals, color=colors, alpha=0.85)
        for yi, v in zip(y_pos, vals):
            view1d_ax_bars.text(v + 0.5, yi, f"{v:.1f}%", va='center', fontsize=9)
            
        view1d_ax_bars.set_yticks(y_pos)
        view1d_ax_bars.set_yticklabels(cats)
        view1d_ax_bars.set_xlim(0, max(100.0, max(vals) + 5.0 if vals else 100.0))
        view1d_ax_bars.set_xlabel("Energy [%]")
        view1d_ax_bars.set_title("Frequency separation (temporal)")
        view1d_ax_bars.grid(True, axis='x', alpha=0.2)
        
        # 5. Add BR/HR (peak) info as annotations under the graph
        lines = []
        if br_bpm > 0:
            lines.append(f"BR≈{br_bpm:.1f} bpm ({br_f:.2f} Hz)")
        if hr_bpm > 0:
            lines.append(f"HR≈{hr_bpm:.1f} bpm ({hr_f:.2f} Hz)")
        if lines:
            txt = " | ".join(lines)
            view1d_ax_bars.text(0.01, -0.35, txt, transform=view1d_ax_bars.transAxes,
                                 fontsize=9, color='black', ha='left', va='top')

    except Exception as e:
        view1d_ax_bars.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')

def update_view1d_plot(frame_idx: int):
    global view1d_ax_prof, view1d_ax_spec, view1d_canvas
    if view1d_ax_prof is None or view1d_ax_spec is None or view1d_canvas is None:
        return
        
    if last_rd_map is None or last_rd_meta is None:
        return
    
    f = min(max(frame_idx, 0), last_rd_map.shape[0] - 1)
    
    try:
        rbin = int(view1d_rbin_var.get())
    except Exception:
        rbin = 0
        
    Kr = last_rd_map.shape[2]
    rbin = max(0, min(Kr - 1, rbin))
    
    # --- Profile ---
    view1d_ax_prof.clear()
    prof = np.mean(last_rd_map[f], axis=0)
    prof_db = 20 * np.log10(prof + 1e-9)
    
    r_axis = last_rd_meta.get('range_axis')
    if r_axis is not None and r_axis.size == Kr:
        view1d_ax_prof.plot(r_axis, prof_db)
        view1d_ax_prof.set_xlabel("Range [m]")
        view1d_ax_prof.axvline(r_axis[rbin], color='red', linestyle='--', alpha=0.5)
    else:
        view1d_ax_prof.plot(prof_db)
        view1d_ax_prof.set_xlabel("Range Bin")
        view1d_ax_prof.axvline(rbin, color='red', linestyle='--', alpha=0.5)
        
    view1d_ax_prof.set_ylabel("dB")
    view1d_ax_prof.set_title(f"Range Profile (Frame {f})")
    view1d_ax_prof.grid(True, alpha=0.3)
    
    # --- Doppler Spectrum ---
    view1d_ax_spec.clear()
    spec = last_rd_map[f, :, rbin]
    spec_db = 20 * np.log10(spec + 1e-9)
    
    v_axis = last_rd_meta.get('velocity_axis')
    if v_axis is not None and v_axis.size == spec.size:
        view1d_ax_spec.plot(v_axis, spec_db, color='C3')
        view1d_ax_spec.set_xlabel("Velocity [m/s]")
    else:
        view1d_ax_spec.plot(spec_db, color='C3')
        view1d_ax_spec.set_xlabel("Doppler Bin")
        
    view1d_ax_spec.set_ylabel("dB")
    view1d_ax_spec.set_title(f"Doppler Spectrum @ Bin {rbin}")
    view1d_ax_spec.grid(True, alpha=0.3)
    
    # --- Temporal ---
    _update_view1d_temporal(rbin, f)
    
    view1d_canvas.draw_idle()

# --- NEW: Detection Functions ---
def init_detection_plot(parent):
    global detection_fig, detection_ax, detection_canvas
    detection_fig = Figure(figsize=(5, 3), dpi=100)
    detection_ax = detection_fig.add_subplot(111)
    detection_ax.set_title("CFAR Detections (Range-Doppler)")  # UPDATED title
    detection_ax.set_xlabel("Velocity [m/s]")
    detection_ax.set_ylabel("Range [m]")
    detection_canvas = FigureCanvasTkAgg(detection_fig, master=parent)
    detection_canvas.get_tk_widget().grid(row=2, column=0, columnspan=5, sticky="nsew", padx=4, pady=4)
    parent.rowconfigure(2, weight=1)
    parent.columnconfigure(0, weight=1)

def run_cfar_detection():
    global last_detections, last_detection_peaks, detection_im, detection_cbar, detection_scatter

    if last_rd_map is None or last_rd_meta is None:
        messagebox.showerror("Error", "Please run Range-Doppler first.")
        return

    try:
        # Get CFAR parameters
        guard = int(cfar_guard_var.get())
        training = int(cfar_training_var.get())
        threshold = float(cfar_threshold_var.get())
        method = cfar_method_var.get()

        # Get current frame from slider
        frame_idx = detection_frame_slider.get() if 'detection_frame_slider' in globals() else 0
        frame_idx = min(max(frame_idx, 0), last_rd_map.shape[0] - 1)

        # Get RD map for current frame
        rd_frame = last_rd_map[frame_idx]  # (Doppler, Range)

        # Run CFAR detection
        detections = cfar_2d_ca(
            rd_frame,
            guard_cells=guard,
            training_cells=training,
            threshold_factor=threshold,
            method=method
        )

        # Extract peaks
        peaks = extract_peaks(detections, rd_frame, min_distance=3)

        # Convert to physical units
        range_axis = last_rd_meta.get('range_axis')
        velocity_axis = last_rd_meta.get('velocity_axis')
        peaks_physical = detections_to_range_velocity(peaks, range_axis, velocity_axis)

        last_detections = detections
        last_detection_peaks = peaks_physical
        # Ensure slider enabled (if RD already computed)  # NEW
        if 'detection_frame_slider' in globals() and detection_frame_slider.cget("state") == "disabled":  # NEW
            detection_frame_slider.config(state="normal")                                                # NEW
        # Update plot
        update_detection_plot(frame_idx)

        # Print detections to console
        print(f"\n[CFAR Detection] Frame {frame_idx}: {len(peaks_physical)} targets detected")
        for i, peak in enumerate(peaks_physical):
            print(f"  Target {i+1}: Range={peak.get('range_m', 'N/A'):.2f} m, "
                  f"Velocity={peak.get('velocity_ms', 'N/A'):.2f} m/s, "
                  f"Magnitude={peak['magnitude']:.2f}")

    except Exception as e:
        messagebox.showerror("Error", f"CFAR Detection error: {e}")

def update_detection_plot(frame_idx: int):
    global detection_im, detection_cbar, detection_scatter

    # RELAXED: allow RD background update even if no detections computed yet
    if last_rd_map is None:
        return

    f = min(max(frame_idx, 0), last_rd_map.shape[0] - 1)
    rd_frame = last_rd_map[f]
    img_db = 20 * np.log10(rd_frame + 1e-9)

    range_axis = last_rd_meta.get('range_axis')
    velocity_axis = last_rd_meta.get('velocity_axis')
    extent = [velocity_axis[0], velocity_axis[-1], range_axis[0], range_axis[-1]]
    vmin = last_rd_meta.get('display_vmin')
    vmax = last_rd_meta.get('display_vmax')

    if detection_im is None:
        detection_ax.clear()
        detection_im = detection_ax.imshow(img_db.T, aspect='auto', origin='lower',
                                           cmap='viridis', extent=extent, vmin=vmin, vmax=vmax)
        if detection_cbar is None:
            detection_cbar = detection_fig.colorbar(detection_im, ax=detection_ax)
        detection_cbar.set_label("Magnitude [dB]")
    else:
        detection_im.set_data(img_db.T)
        detection_im.set_extent(extent)
        if vmin is not None and vmax is not None:
            detection_im.set_clim(vmin=vmin, vmax=vmax)

    # Apply auto/manual velocity range to CFAR background too
    try:
        auto_v = bool(rd_auto_vrange_var.get())
    except Exception:
        auto_v = True
    if not auto_v:
        try:
            v_min_zoom = float(rd_vmin_var.get())
            v_max_zoom = float(rd_vmax_var.get())
            if v_min_zoom < v_max_zoom:
                detection_ax.set_xlim(v_min_zoom, v_max_zoom)
        except (ValueError, tk.TclError):
            pass
    else:
        detection_ax.set_xlim(extent[0], extent[1])

    # Overlay detections
    if detection_scatter is not None:
        detection_scatter.remove()
        detection_scatter = None

    if last_detection_peaks:
        vel_coords = [p['velocity_ms'] for p in last_detection_peaks if p.get('velocity_ms') is not None]
        range_coords = [p['range_m'] for p in last_detection_peaks if p.get('range_m') is not None]

        if vel_coords and range_coords:
            detection_scatter = detection_ax.scatter(
                vel_coords, range_coords,
                c='red', s=100, marker='x', linewidths=2,
                label=f'{len(vel_coords)} detections'
            )
            detection_ax.legend(loc='upper right')

    detection_ax.set_xlabel("Velocity [m/s]")
    detection_ax.set_ylabel("Range [m]")
    detection_ax.set_title(f"CFAR Detections (Frame {f})")
    detection_canvas.draw()

def on_detection_frame_change(val):
    try:
        idx = int(float(val))
    except ValueError:
        idx = 0
    # NEW: route via global synchronizer
    apply_global_frame(idx)

# ---- Recent files UI (Decode tab) ----
recent_listbox = None

def populate_recent_list():
    global recent_listbox
    if recent_listbox is None:
        return
    try:
        items = RECENTS.list()
    except Exception:
        items = []
    recent_listbox.delete(0, "end")
    for p in items:
        recent_listbox.insert("end", p)

def on_recent_open(evt=None):
    if recent_listbox is None:
        return
    sel = recent_listbox.curselection()
    if not sel:
        return
    path = recent_listbox.get(sel[0])
    if not os.path.isfile(path):
        messagebox.showerror("Error", "File does not exist. Removing from history.")
        try:
            RECENTS.remove(path)
            populate_recent_list()
        except Exception:
            pass
        return
    bin_var.set(path)
    decode_log()

def on_recent_dialog():
    sel = tk_pick_recent_and_open(parent=root)
    if sel:
        bin_var.set(sel)
        try:
            RECENTS.add(sel)
        except Exception:
            pass
        decode_log()

# ---------------- GUI layout ----------------
root = tk.Tk()
root.title("DARTscope")

# Pokus o nastavení ikony aplikace
try:
    icon_loaded = False
    # Zkusíme nejdříve PNG (lepší kompatibilita), pak SVG
    for icon_file in ["icon.png", "icon.svg"]:
        if icon_loaded:
            break
        icon_path = resource_path(icon_file)
        if os.path.exists(icon_path):
            try:
                # 1. Pokus: Použití Pillow (pokud je k dispozici)
                from PIL import Image, ImageTk
                img = Image.open(icon_path)
                photo = ImageTk.PhotoImage(img)
                root.iconphoto(True, photo)
                root._icon_img = photo # Nutná reference proti Garbage Collectoru
                icon_loaded = True
            except ImportError:
                # 2. Pokus: Standardní Tkinter PhotoImage
                try:
                    photo = tk.PhotoImage(file=icon_path)
                    root.iconphoto(True, photo)
                    root._icon_img = photo
                    icon_loaded = True
                except Exception:
                    continue
except Exception:
    pass

root.geometry("1400x900")

bin_var = tk.StringVar()

# NEW: Global frame controls (callbacks stay as-is)
def on_global_frame_change(val):
    try:
        idx = int(float(val))
    except ValueError:
        idx = 0
    apply_global_frame(idx)
def _update_global_frame():
    if not global_playing:
        return
    try:
        current_val = int(global_frame_slider.get())
        max_val = int(global_frame_slider.cget("to"))
    except Exception:
        current_val, max_val = 0, 0
    if max_val < 0:
        return
    next_val = (current_val + 1) % (max_val + 1)
    apply_global_frame(next_val)
    root.after(global_play_interval_ms, _update_global_frame)

def play_global():
    global global_playing
    if int(global_frame_slider.cget("to")) <= 0:
        return
    global_playing = True
    global_play_btn.config(state="disabled")
    global_stop_btn.config(state="normal")
    _update_global_frame()

def stop_global():
    global global_playing
    global_playing = False
    if 'global_play_btn' in globals() and global_play_btn.winfo_exists():
        global_play_btn.config(state="normal")
    if 'global_stop_btn' in globals() and global_stop_btn.winfo_exists():
        global_stop_btn.config(state="disabled")

def set_global_frames(n_frames: int):
    """Publish available frames to global slider (keeps the max across producers)."""
    global global_frames_total
    try:
        n_frames = int(n_frames)
    except Exception:
        return
    if n_frames <= 0:
        return

    # Ensure slider is initialized
    if global_frame_slider is None or not global_frame_slider.winfo_exists():
        return

    # ALWAYS ensure it's enabled if we have valid frames
    if global_frame_slider.cget("state") == "disabled":
        global_frame_slider.config(state="normal")
        if global_play_btn is not None and global_play_btn.winfo_exists():
            global_play_btn.config(state="normal")
        if global_stop_btn is not None and global_stop_btn.winfo_exists():
            global_stop_btn.config(state="disabled")

    # Update range if new producer has more frames than current total
    if n_frames > global_frames_total:
        global_frames_total = n_frames
        global_frame_slider.config(to=n_frames - 1)

    # Keep FPS label in sync even if frames appear before interval is set
    _update_global_fps_label()

def set_global_interval_ms(ms: int):
    """Update global playback interval (ms)."""
    global global_play_interval_ms, rd_play_interval_ms, range_fft_play_interval_ms
    try:
        msi = int(ms)
        if msi > 0:
            global_play_interval_ms = msi
            # Mirror to per-tab players for consistent timing
            try:
                rd_play_interval_ms = msi
            except Exception:
                pass
            try:
                range_fft_play_interval_ms = msi
            except Exception:
                pass
            _update_global_fps_label()  # NEW
    except Exception:
        pass

def apply_global_frame(idx: int):
    """Central frame change propagation + view updates (re‑entrant safe)."""
    global _in_global_frame_set
    if _in_global_frame_set:
        return
    _in_global_frame_set = True
    try:
        # Clamp to global range
        try:
            max_idx = int(global_frame_slider.cget("to"))
        except Exception:
            max_idx = 0
        i = max(0, min(int(idx), max_idx))

        # 1) Update all sliders to the same index (mirroring)
        if 'global_frame_slider' in globals() and global_frame_slider is not None and global_frame_slider.winfo_exists():
            if int(global_frame_slider.get()) != i:
                global_frame_slider.set(i)
        if 'range_fft_frame_slider' in globals() and range_fft_frame_slider is not None and range_fft_frame_slider.winfo_exists():
            try:
                rf_max = int(range_fft_frame_slider.cget("to"))
                range_fft_frame_slider.set(min(i, rf_max))
            except Exception:
                pass
        if 'detection_frame_slider' in globals() and detection_frame_slider is not None and detection_frame_slider.winfo_exists():
            try:
                det_max = int(detection_frame_slider.cget("to"))
                detection_frame_slider.set(min(i, det_max))
            except Exception:
                pass
    finally:
        _in_global_frame_set = False

    # 2) Trigger plot updates explicitly (callbacks were suppressed by guard)
    try:
        if last_range_fft is not None:
            rf_max = last_range_fft.shape[0] - 1
            update_range_fft_plot(min(i, rf_max))
    except Exception:
        pass
    try:
        if last_rd_map is not None:
            rd_max = last_rd_map.shape[0] - 1
            update_rd_plot(min(i, rd_max))
    except Exception:
        pass
    try:
        # Update CFAR tab background; detections overlay appears if available for that frame
        if last_rd_map is not None and detection_ax is not None:
            det_max = last_rd_map.shape[0] - 1
            update_detection_plot(min(i, det_max))
    except Exception:
        pass

    try:
        if last_aoa_spec is not None:
            aoa_max = last_aoa_spec.shape[0] - 1
            update_aoa_plot(min(i, aoa_max))
    except Exception:
        pass

    try:
        if last_texas_aoa_spec is not None:
            tx_aoa_max = last_texas_aoa_spec.shape[0] - 1
            update_texas_aoa_plot(min(i, tx_aoa_max))
    except Exception:
        pass

    try:
        if last_aoa_2d_map is not None:
            aoa_2d_max = last_aoa_2d_map.shape[0] - 1
            update_aoa_2d_plot(min(i, aoa_2d_max))
    except Exception:
        pass

    try:
        if last_md_map is not None:
            md_max = last_md_map.shape[0] - 1
            update_md_plot(min(i, md_max))
    except Exception:
        pass

    try:
        if last_rd_map is not None and view1d_fig is not None:
            update_view1d_plot(i)
    except Exception:
        pass

# NEW: Main notebook with top-level tabs
main_notebook = ttk.Notebook(root)
frame_decode_main = ttk.Frame(main_notebook)
frame_analysis_main = ttk.Frame(main_notebook)
frame_helper_main = ttk.Frame(main_notebook)
frame_manual_main = ttk.Frame(main_notebook)

main_notebook.add(frame_decode_main, text="Decode")
main_notebook.add(frame_analysis_main, text="Analysis")
main_notebook.add(frame_helper_main, text="Helper")
main_notebook.add(frame_manual_main, text="Manual")
main_notebook.pack(fill="both", expand=True, padx=6, pady=6)

# Decode tab (moved into frame_decode_main)
tk.Label(frame_decode_main, text="Select BIN file:").grid(row=0, column=0, sticky="e")
tk.Entry(frame_decode_main, textvariable=bin_var, width=60).grid(row=0, column=1, padx=4, pady=4)
tk.Button(frame_decode_main, text="Browse...", command=select_bin_file).grid(row=0, column=2, padx=4)
tk.Button(frame_decode_main, text="Decode Log", command=decode_log).grid(row=1, column=1, pady=8)

# Recent files box
recent_frame = ttk.LabelFrame(frame_decode_main, text="Recent history (max 5)")
recent_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=2, pady=2)

# Listbox with double-click to open
recent_listbox = tk.Listbox(recent_frame, height=5)
recent_listbox.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=6)
recent_listbox.bind("<Double-Button-1>", on_recent_open)
recent_listbox.bind("<Return>", on_recent_open)

# Right-side buttons
rf_btns = ttk.Frame(recent_frame)
rf_btns.pack(side="right", fill="y", padx=6, pady=6)

ttk.Button(rf_btns, text="Open", command=on_recent_open).pack(fill="x")
ttk.Button(rf_btns, text="Dialog...", command=on_recent_dialog).pack(fill="x", pady=(4,0))

def _rf_refresh():
    try:
        populate_recent_list()
    except Exception:
        pass

ttk.Button(rf_btns, text="Refresh", command=_rf_refresh).pack(fill="x", pady=(8,0))
ttk.Button(rf_btns, text="Clear", command=lambda: (RECENTS.clear(), populate_recent_list())).pack(fill="x", pady=(4,0))

# Initial fill
try:
    populate_recent_list()
except Exception:
    pass

frame_decode_main.rowconfigure(3, weight=1)
frame_decode_main.columnconfigure(0, weight=0)
frame_decode_main.columnconfigure(1, weight=1)
frame_decode_main.columnconfigure(2, weight=0)

frame_decode_main.rowconfigure(2, weight=0)

init_params_view(frame_decode_main)

# NEW: Global controls bar INSIDE Analysis tab
global_bar = ttk.Frame(frame_analysis_main)
ttk.Label(global_bar, text="Global Frame:", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(4, 8))
global_play_btn = tk.Button(global_bar, text="Play All", command=play_global, state="disabled")
global_play_btn.pack(side="left")
global_stop_btn = tk.Button(global_bar, text="Stop", command=stop_global, state="disabled")
global_stop_btn.pack(side="left", padx=(6, 12))

# Antenna (RX combine) selector – uses on_rx_mode_change as before
def on_rx_mode_change(_evt=None):
    """
    Recompute RD with the newly selected RX combine mode.
    Uses existing loaded cube/params; updates CFAR background automatically.
    """
    if last_params and bin_var.get():
        try:
            run_range_doppler()
        except Exception:
            pass

ttk.Label(global_bar, text="Antenna:").pack(side="left", padx=(6, 2))
global_rx_mode_var = tk.StringVar(value="MRC")
global_rx_mode_combo = ttk.Combobox(
    global_bar, textvariable=global_rx_mode_var,
    values=["MRC", "SUM", "RX0", "RX1"], state="readonly", width=6
)
global_rx_mode_combo.pack(side="left", padx=(0, 10))
global_rx_mode_combo.bind("<<ComboboxSelected>>", on_rx_mode_change)

# NEW: Chirp selection change handler
def on_chirp_sel_change(_evt=None):
    global last_cube, last_rd_map, last_md_map, last_range_fft
    # Reset only cube and derived maps; necháme parametry a GUI zůstat
    last_cube = None
    last_rd_map = None
    last_md_map = None
    last_range_fft = None
    try:
        # re-run dependent views lazily (načte se při prvním runu)
        pass
    except Exception:
        pass

# NEW: Chirp selection UI
ttk.Label(global_bar, text="Chirp:").pack(side="left", padx=(10, 2))
global_chirp_sel_var = tk.StringVar(value="All")
global_chirp_sel_combo = ttk.Combobox(
    global_bar, textvariable=global_chirp_sel_var, values=["All"], state="disabled", width=10
)
global_chirp_sel_combo.pack(side="left", padx=(0, 10))
global_chirp_sel_combo.bind("<<ComboboxSelected>>", on_chirp_sel_change)

# NEW: Lane Swap control (Global)
def on_lane_swap_change():
    global last_cube, last_rd_map, last_md_map, last_range_fft
    last_cube = None
    last_rd_map = None
    last_md_map = None
    last_range_fft = None
    print("[GUI] Lane swap changed. Reloading data.")

lane_swap_var = tk.BooleanVar(value=False)
tk.Checkbutton(global_bar, text="Lane Swap (1<->2)", variable=lane_swap_var,
               command=on_lane_swap_change).pack(side="left", padx=(10, 0))

# Real-time FPS label
ttk.Label(global_bar, text=" | ").pack(side="left", padx=(2, 2))
global_fps_var = tk.StringVar(value="-- fps")
ttk.Label(global_bar, textvariable=global_fps_var).pack(side="left", padx=(0, 8))

# NEW: Global Range Crop
ttk.Label(global_bar, text="Range Cut [m]:").pack(side="left", padx=(10, 2))
aoa_min_range_var = tk.StringVar(value="0.3")
tk.Entry(global_bar, textvariable=aoa_min_range_var, width=5).pack(side="left")

# NEW: Global Angle Limit
ttk.Label(global_bar, text="Angle Lim [±deg]:").pack(side="left", padx=(10, 2))
aoa_angle_limit_var = tk.StringVar(value="90")
tk.Entry(global_bar, textvariable=aoa_angle_limit_var, width=5).pack(side="left")

global_frame_slider = tk.Scale(global_bar, from_=0, to=0, orient="horizontal",
                               label="", command=on_global_frame_change, state="disabled", length=800)
global_frame_slider.pack(side="left", fill="x", expand=True, padx=6, pady=6)
global_bar.pack(fill="x", padx=6, pady=(6, 0))

# NEW: Analysis sub-notebook and sub-tabs
analysis_nb = ttk.Notebook(frame_analysis_main)
frame_range = ttk.Frame(analysis_nb)
frame_rti = ttk.Frame(analysis_nb)
frame_rp = ttk.Frame(analysis_nb)
frame_rd = ttk.Frame(analysis_nb)
frame_mov = ttk.Frame(analysis_nb) # NEW: Movement Map
frame_aoa = ttk.Frame(analysis_nb)  # NEW
frame_aoa_texas = ttk.Frame(analysis_nb)  # NEW: Texas AoA
frame_aoa_2d = ttk.Frame(analysis_nb)  # NEW: 2D AoA
frame_md = ttk.Frame(analysis_nb)  # NEW: Micro-Doppler
frame_view1d = ttk.Frame(analysis_nb)  # NEW: 1D View
frame_detection = ttk.Frame(analysis_nb)

analysis_nb.add(frame_range, text="Range FFT")
analysis_nb.add(frame_rti, text="RTI")
analysis_nb.add(frame_rp, text="Range Profile (dBFS)")
analysis_nb.add(frame_rd, text="Range-Doppler")
analysis_nb.add(frame_mov, text="Movement Map") # NEW
analysis_nb.add(frame_aoa, text="AoA")  # NEW
analysis_nb.add(frame_aoa_texas, text="Texas AoA")  # NEW
analysis_nb.add(frame_aoa_2d, text="2D AoA")  # NEW
analysis_nb.add(frame_md, text="Micro-Doppler")  # NEW
analysis_nb.add(frame_view1d, text="1D View (Doppler)")  # NEW
analysis_nb.add(frame_detection, text="CFAR")

analysis_nb.pack(fill="both", expand=True, padx=6, pady=6)


# Range FFT tab (under Analysis)
controls_frame = ttk.Frame(frame_range)
controls_frame.grid(row=0, column=0, padx=8, pady=4, sticky="w")
tk.Button(controls_frame, text="Run Range FFT", command=run_range_fft).pack(side="left")
# NEW: Remove stationary checkbox (motion-only view)
range_fft_rm_stationary_var = tk.BooleanVar(value=False)
tk.Checkbutton(controls_frame, text="Remove stationary (motion only)", variable=range_fft_rm_stationary_var,
               command=lambda: update_range_fft_plot(0)).pack(side="left", padx=(12, 0))

# NEW: intensity slider (0..100%) mapped to 0.0..1.0
range_fft_rm_strength_var = tk.DoubleVar(value=1.0)

def _on_rm_strength_change(_val=None):
    try:
        update_range_fft_plot(int(global_frame_slider.get()))
    except Exception:
        update_range_fft_plot(0)

strength_frame = ttk.Frame(controls_frame)
strength_frame.pack(side="left", padx=(12, 0))

ttk.Label(strength_frame, text="Intensity:").pack(side="left")
rm_scale = tk.Scale(strength_frame, from_=0, to=100, orient="horizontal", showvalue=True, length=150,
                    command=lambda _v: (_on_rm_strength_change(_v)), resolution=1)
# bind var so that get() returns 0.0..1.0 when used in plotting
rm_scale.set(100)

# Numeric entry (percent) for exact intensity
range_fft_rm_strength_pct_var = tk.IntVar(value=100)

def _on_pct_change(*_):
    try:
        pct = int(range_fft_rm_strength_pct_var.get())
    except Exception:
        pct = 100
    pct = max(0, min(pct, 100))
    range_fft_rm_strength_pct_var.set(pct)
    try:
        rm_scale.set(pct)
    except Exception:
        pass
    _sync_rm_var()
    _on_rm_strength_change()

pct_entry = tk.Spinbox(
    strength_frame,
    from_=0,
    to=100,
    textvariable=range_fft_rm_strength_pct_var,
    width=4,
    command=_on_pct_change,
)
pct_entry.pack(side="left", padx=(6, 0))
# add percent label
ttk.Label(strength_frame, text="%").pack(side="left", padx=(2, 0))

# keep slider -> entry sync too
range_fft_rm_strength_pct_var.trace_add("write", lambda *_: None)

def _sync_rm_var(*_):
    try:
        pct = rm_scale.get() / 100.0
    except Exception:
        pct = 1.0
    range_fft_rm_strength_var.set(pct)

# Helper to push slider updates into the Spinbox
def _slider_to_entry_sync():
    try:
        range_fft_rm_strength_pct_var.set(int(rm_scale.get()))
    except Exception:
        pass

rm_scale.bind("<ButtonRelease-1>", lambda e: (_sync_rm_var(), _slider_to_entry_sync()))
rm_scale.bind("<B1-Motion>", lambda e: (_sync_rm_var(), _slider_to_entry_sync()))
# Initial sync
_sync_rm_var()

init_range_fft_plot(frame_range)
add_card_header_extras(controls_frame, "Range FFT", INFO_RANGE_FFT, range_fft_fig, 
                       lambda: last_range_fft[int(global_frame_slider.get())].mean(axis=(0,1)) if last_range_fft is not None else None,
                       update_range_fft_plot)

# AoA tab
aoa_controls = ttk.Frame(frame_aoa)
aoa_controls.grid(row=0, column=0, padx=8, pady=4, sticky="w")
tk.Button(aoa_controls, text="Run AoA", command=run_aoa).pack(side="left")

init_aoa_plot(frame_aoa)
add_card_header_extras(aoa_controls, "AoA", INFO_AOA, aoa_fig, 
                       lambda: last_aoa_spec[int(global_frame_slider.get())] if last_aoa_spec is not None else None,
                       update_aoa_plot)

# Texas AoA tab
texas_aoa_controls = ttk.Frame(frame_aoa_texas)
texas_aoa_controls.grid(row=0, column=0, padx=8, pady=4, sticky="w")
tk.Button(texas_aoa_controls, text="Run Texas AoA", command=run_texas_aoa).pack(side="left")

init_texas_aoa_plot(frame_aoa_texas)
add_card_header_extras(texas_aoa_controls, "Texas AoA", INFO_AOA, texas_aoa_fig,
                       lambda: last_texas_aoa_spec[int(global_frame_slider.get())] if last_texas_aoa_spec is not None else None,
                       update_texas_aoa_plot)

tk.Label(texas_aoa_controls, text="Range [m] (optional):").pack(side="left", padx=(12, 2))
texas_aoa_range_var = tk.StringVar(value="")
tk.Entry(texas_aoa_controls, textvariable=texas_aoa_range_var, width=8).pack(side="left")

if texas_aoa_remove_static_var is None:
    texas_aoa_remove_static_var = tk.BooleanVar(value=False)
tk.Checkbutton(texas_aoa_controls, text="Remove Static", variable=texas_aoa_remove_static_var).pack(side="left", padx=(12, 0))

if texas_aoa_mimo_var is None:
    texas_aoa_mimo_var = tk.BooleanVar(value=True)
tk.Checkbutton(texas_aoa_controls, text="MIMO", variable=texas_aoa_mimo_var).pack(side="left", padx=(12, 0))

if texas_aoa_shift_var is None:
    texas_aoa_shift_var = tk.BooleanVar(value=True) # Default ON
tk.Checkbutton(texas_aoa_controls, text="Shift Spectrum", variable=texas_aoa_shift_var,
               command=lambda: run_texas_aoa()).pack(side="left", padx=(12, 0))

ttk.Label(texas_aoa_controls, text="View:").pack(side="left", padx=(12, 2))
texas_aoa_view_var = tk.StringVar(value="1D Spectrum")
texas_aoa_view_cb = ttk.Combobox(
    texas_aoa_controls, textvariable=texas_aoa_view_var, state="readonly", width=15,
    values=["1D Spectrum", "Range-Azimuth", "Azimuth-Elevation"]
)
texas_aoa_view_cb.pack(side="left", padx=2)

tk.Label(aoa_controls, text="Range [m] (optional):").pack(side="left", padx=(12, 2))
aoa_range_var = tk.StringVar(value="")
tk.Entry(aoa_controls, textvariable=aoa_range_var, width=8).pack(side="left")

def on_mimo_mode_change(e=None):
    if last_aoa_spec is not None: run_aoa()
    if last_aoa_2d_map is not None: run_aoa_2d()
    if last_texas_aoa_spec is not None: run_texas_aoa()

def create_mimo_mode_cb(parent):
    global aoa_mimo_mode_var
    ttk.Label(parent, text="Mode:").pack(side="left", padx=(12, 2))
    if aoa_mimo_mode_var is None:
        aoa_mimo_mode_var = tk.StringVar(value="MIMO")
    cb = ttk.Combobox(
        parent, state="readonly", width=10,
        values=["MIMO", "First TX", "All Chirps"],
        textvariable=aoa_mimo_mode_var
    )
    cb.pack(side="left")
    cb.bind("<<ComboboxSelected>>", on_mimo_mode_change)
    return cb

def create_method_cb(parent, var_name):
    ttk.Label(parent, text="Method:").pack(side="left", padx=(12, 2))
    v = tk.StringVar(value="Bartlett")
    globals()[var_name] = v
    cb = ttk.Combobox(
        parent, state="readonly", width=10,
        values=["Bartlett", "Capon"],
        textvariable=v
    )
    cb.pack(side="left")
    cb.bind("<<ComboboxSelected>>", on_mimo_mode_change)
    return cb

create_mimo_mode_cb(texas_aoa_controls)
create_method_cb(texas_aoa_controls, "texas_aoa_method_var")

create_mimo_mode_cb(aoa_controls)

if aoa_remove_static_var is None:
    aoa_remove_static_var = tk.BooleanVar(value=False)
tk.Checkbutton(aoa_controls, text="Remove Static", variable=aoa_remove_static_var).pack(side="left", padx=(12, 0))

if aoa_mimo_var is None:
    aoa_mimo_var = tk.BooleanVar(value=True)
tk.Checkbutton(aoa_controls, text="MIMO", variable=aoa_mimo_var).pack(side="left", padx=(12, 0))

if aoa_invert_phase_var is None:
    aoa_invert_phase_var = tk.BooleanVar(value=False)
tk.Checkbutton(aoa_controls, text="Invert TX Phase", variable=aoa_invert_phase_var).pack(side="left", padx=(12, 0))

# 2D AoA tab
aoa_2d_controls = ttk.Frame(frame_aoa_2d)
aoa_2d_controls.grid(row=0, column=0, padx=8, pady=4, sticky="w")
tk.Button(aoa_2d_controls, text="Run 2D AoA", command=run_aoa_2d).pack(side="left")

init_aoa_2d_plot(frame_aoa_2d)
add_card_header_extras(aoa_2d_controls, "2D AoA", INFO_AOA_2D, aoa_2d_fig, 
                       lambda: last_aoa_2d_map[int(global_frame_slider.get())] if last_aoa_2d_map is not None else None,
                       update_aoa_2d_plot)

if aoa_2d_info_var is None:
    aoa_2d_info_var = tk.StringVar(value="")
ttk.Label(aoa_2d_controls, textvariable=aoa_2d_info_var).pack(side="left", padx=(12, 0))
create_mimo_mode_cb(aoa_2d_controls)
create_method_cb(aoa_2d_controls, "aoa_2d_method_var")

if aoa_2d_remove_static_var is None:
    aoa_2d_remove_static_var = tk.BooleanVar(value=False)
tk.Checkbutton(aoa_2d_controls, text="Remove Static", variable=aoa_2d_remove_static_var).pack(side="left", padx=(12, 0))

if aoa_2d_mimo_var is None:
    aoa_2d_mimo_var = tk.BooleanVar(value=True)
tk.Checkbutton(aoa_2d_controls, text="MIMO", variable=aoa_2d_mimo_var).pack(side="left", padx=(12, 0))

if aoa_2d_shift_var is None:
    aoa_2d_shift_var = tk.BooleanVar(value=True) # Default ON
tk.Checkbutton(aoa_2d_controls, text="Shift Spectrum", variable=aoa_2d_shift_var,
               command=lambda: run_aoa_2d()).pack(side="left", padx=(12, 0))

if aoa_invert_phase_var is None:
    aoa_invert_phase_var = tk.BooleanVar(value=False)
tk.Checkbutton(aoa_2d_controls, text="Invert TX Phase", variable=aoa_invert_phase_var).pack(side="left", padx=(12, 0))

# NEW: Cartesian projection toggle
aoa_2d_cartesian_var = tk.BooleanVar(value=False)
tk.Checkbutton(aoa_2d_controls, text="Cartesian (X-Y)", variable=aoa_2d_cartesian_var, 
               command=lambda: update_aoa_2d_plot(int(global_frame_slider.get())) if last_aoa_2d_map is not None else None).pack(side="left", padx=(12, 0))

# Movement Map tab
mov_controls = ttk.Frame(frame_mov)
mov_controls.grid(row=0, column=0, padx=8, pady=4, sticky="w")
tk.Button(mov_controls, text="Run Movement Map", command=run_movement_map).pack(side="left")

init_mov_plot(frame_mov)
add_card_header_extras(mov_controls, "Movement Map", INFO_MOV_MAP, mov_fig, 
                       lambda: last_mov_map if last_mov_map is not None else None,
                       update_movement_map_plot)

ttk.Label(mov_controls, text="Band:").pack(side="left", padx=(12, 2))
if mov_band_var is None:
    mov_band_var = tk.StringVar(value="Movement")
mov_band_cb = ttk.Combobox(
    mov_controls, state="readonly", width=12,
    values=["Movement", "Breathing", "HR"],
    textvariable=mov_band_var
)
mov_band_cb.pack(side="left")

ttk.Label(mov_controls, text="Accumulation:").pack(side="left", padx=(12, 2))
if mov_accum_var is None:
    mov_accum_var = tk.StringVar(value="Full")
mov_accum_cb = ttk.Combobox(
    mov_controls, state="readonly", width=12,
    values=["Full", "Sliding Window"],
    textvariable=mov_accum_var
)
mov_accum_cb.pack(side="left")

ttk.Label(mov_controls, text="Win [frames]:").pack(side="left", padx=(12, 2))
if mov_win_var is None:
    mov_win_var = tk.StringVar(value="32")
tk.Entry(mov_controls, textvariable=mov_win_var, width=5).pack(side="left")

# Micro-Doppler tab
md_controls = ttk.Frame(frame_md)
md_controls.grid(row=0, column=0, padx=8, pady=4, sticky="w")
tk.Button(md_controls, text="Run Micro-Doppler", command=run_micro_doppler).pack(side="left")

init_md_plot(frame_md)
add_card_header_extras(md_controls, "Micro-Doppler", INFO_MD, md_fig, 
                       lambda: last_md_map if last_md_map is not None else None,
                       update_md_plot)

tk.Label(md_controls, text="Range bin od:").pack(side="left", padx=(12, 2))
md_r0_var = tk.StringVar(value="5")
tk.Entry(md_controls, textvariable=md_r0_var, width=6).pack(side="left")

tk.Label(md_controls, text="do:").pack(side="left", padx=(6, 2))
md_r1_var = tk.StringVar(value="100")
tk.Entry(md_controls, textvariable=md_r1_var, width=6).pack(side="left")

# 1D View tab
view1d_controls = ttk.Frame(frame_view1d)
view1d_controls.grid(row=0, column=0, padx=8, pady=4, sticky="w")
tk.Button(view1d_controls, text="Run 1D View", command=run_view1d).pack(side="left")

init_view1d_plot(frame_view1d)
add_card_header_extras(view1d_controls, "1D View", INFO_VIEW1D, view1d_fig, 
                       lambda: last_rd_map[int(global_frame_slider.get()), :, int(view1d_rbin_var.get())] if last_rd_map is not None else None,
                       update_view1d_plot)

ttk.Label(view1d_controls, text="Range bin:").pack(side="left", padx=(12, 2))
if view1d_rbin_var is None:
    view1d_rbin_var = tk.IntVar(value=0)
view1d_slider = tk.Scale(
    view1d_controls, from_=0, to=0, orient="horizontal",
    command=on_view1d_rbin_change, state="disabled", length=400,
    variable=view1d_rbin_var, showvalue=False
)
view1d_slider.pack(side="left", padx=2)
if view1d_rbin_label_var is None:
    view1d_rbin_label_var = tk.StringVar(value="0")
ttk.Label(view1d_controls, textvariable=view1d_rbin_label_var).pack(side="left")

ttk.Label(view1d_controls, text="Source:").pack(side="left", padx=(12, 2))
if view1d_band_src_var is None:
    view1d_band_src_var = tk.StringVar(value="Range Bin")
view1d_band_src_cb = ttk.Combobox(
    view1d_controls, state="readonly", width=15,
    values=["Range Bin", "Neighborhood ±N", "Whole Range"],
    textvariable=view1d_band_src_var
)
view1d_band_src_cb.pack(side="left")
view1d_band_src_cb.bind("<<ComboboxSelected>>", lambda e: update_view1d_plot(int(global_frame_slider.get())))

ttk.Label(view1d_controls, text="±N:").pack(side="left", padx=(8, 2))
if view1d_band_w_var is None:
    view1d_band_w_var = tk.StringVar(value="6")
ttk.Entry(view1d_controls, textvariable=view1d_band_w_var, width=4).pack(side="left")

if view1d_just_frame_var is None:
    view1d_just_frame_var = tk.BooleanVar(value=False)
tk.Checkbutton(view1d_controls, text="Current Frame Only", variable=view1d_just_frame_var,
               command=lambda: update_view1d_plot(int(global_frame_slider.get()))).pack(side="left", padx=(12, 0))

init_view1d_plot(frame_view1d)

# --- NEW: AoA range-bin slider ---
# Container row under the plot controls (kept separate from global frame slider)
aoa_bin_controls = ttk.Frame(frame_aoa)
aoa_bin_controls.grid(row=2, column=0, padx=8, pady=(2, 6), sticky="ew")

# Label showing current selected bin and (if available) meters
if aoa_bin_label_var is None:
    aoa_bin_label_var = tk.StringVar(value="bin —")
ttk.Label(aoa_bin_controls, textvariable=aoa_bin_label_var).pack(side="right", padx=(8, 4))

def on_aoa_bin_change(val=None):
    """Recompute AoA spectrum for the selected bin (unless range[m] entry is filled)."""
    try:
        # If range [m] is provided, prioritise that and let run_aoa resolve bin.
        if aoa_range_var.get().strip():
            run_aoa()
            return
    except Exception:
        pass
    run_aoa()

# Slider itself (disabled until first run when we know number of bins)
if aoa_bin_var is None:
    aoa_bin_var = tk.IntVar(value=0)

if aoa_bin_slider is None:
    aoa_bin_slider = tk.Scale(
        aoa_bin_controls,
        from_=0,
        to=0,
        orient="horizontal",
        label="Range Bin",
        command=lambda _v: on_aoa_bin_change(_v),
        state="disabled",
        length=600,
        variable=aoa_bin_var,
    )
    aoa_bin_slider.pack(side="left", fill="x", expand=True)

# RTI tab
rti_controls_frame = ttk.Frame(frame_rti)
rti_controls_frame.grid(row=0, column=0, padx=8, pady=4, sticky="w")
tk.Button(rti_controls_frame, text="Run RTI", command=run_rti).pack(side="left")

init_rti_plot(frame_rti)
add_card_header_extras(rti_controls_frame, "RTI", INFO_RTI, rti_fig, 
                       lambda: last_rti if last_rti is not None else None)

# Range Profile tab
rp_controls_frame = ttk.Frame(frame_rp)
rp_controls_frame.grid(row=0, column=0, padx=8, pady=4, sticky="w")
tk.Button(rp_controls_frame, text="Run Profile", command=run_range_profile).pack(side="left")

init_range_profile_plot(frame_rp)
add_card_header_extras(rp_controls_frame, "Range Profile", INFO_RP, rp_fig, 
                       lambda: last_range_profile if last_range_profile is not None else None)

# Range-Doppler tab
rd_controls_frame = ttk.Frame(frame_rd)
rd_controls_frame.grid(row=0, column=0, padx=8, pady=4, sticky="w")
tk.Button(rd_controls_frame, text="Run Range-Doppler", command=run_range_doppler).pack(side="left")

init_rd_plot(frame_rd)
add_card_header_extras(rd_controls_frame, "Range-Doppler", INFO_RD, rd_fig, 
                       lambda: last_rd_map[int(global_frame_slider.get())] if last_rd_map is not None else None,
                       update_rd_plot)

rd_play_btn = tk.Button(rd_controls_frame, text="Play", command=play_rd, state="disabled")
rd_play_btn.pack(side="left", padx=(10, 0))
rd_stop_btn = tk.Button(rd_controls_frame, text="Stop", command=stop_rd, state="disabled")
rd_stop_btn.pack(side="left")
# NEW: Doppler window toggle (Hann)
rd_hann_var = tk.BooleanVar(value=True)
tk.Checkbutton(rd_controls_frame, text="Hann window (Doppler)", variable=rd_hann_var).pack(side="left", padx=(12, 0))
# NEW: Auto velocity range toggle
def _on_rd_vrange_toggle():
    try:
        auto = bool(rd_auto_vrange_var.get())
        state = "disabled" if auto else "normal"
        if rd_vmin_entry is not None:
            rd_vmin_entry.config(state=state)
        if rd_vmax_entry is not None:
            rd_vmax_entry.config(state=state)
        # Redraw current frame with new limits
        if last_rd_map is not None:
            try:
                idx = int(global_frame_slider.get())
            except Exception:
                idx = 0
            update_rd_plot(idx)
            update_detection_plot(idx)
    except Exception:
        pass

rd_auto_vrange_var = tk.BooleanVar(value=True)
tk.Checkbutton(rd_controls_frame, text="Auto velocity range", variable=rd_auto_vrange_var,
               command=_on_rd_vrange_toggle).pack(side="left", padx=(12, 0))
# Zoom controls (manual only)
tk.Label(rd_controls_frame, text="V_min [m/s]:").pack(side="left", padx=(20, 0))
rd_vmin_var = tk.StringVar(value="-5")
rd_vmin_entry = tk.Entry(rd_controls_frame, textvariable=rd_vmin_var, width=5)  # NEW: keep ref for enable/disable
rd_vmin_entry.pack(side="left")
tk.Label(rd_controls_frame, text="V_max [m/s]:").pack(side="left", padx=(10, 0))
rd_vmax_var = tk.StringVar(value="5")
rd_vmax_entry = tk.Entry(rd_controls_frame, textvariable=rd_vmax_var, width=5)  # NEW: keep ref for enable/disable
rd_vmax_entry.pack(side="left")
# Initialize fields state for Auto
_on_rd_vrange_toggle()

# ODSTRANĚNO: lokální RD “Frame” slider (redundantní, nahrazen globálním)
# rd_frame_slider = tk.Scale(frame_rd, from_=0, to=0, orient="horizontal",
#                            label="Frame", command=on_rd_frame_change, state="disabled")
# rd_frame_slider.grid(row=2, column=0, sticky="ew", padx=4, pady=4)

# CFAR tab
detection_controls_frame = ttk.Frame(frame_detection)
detection_controls_frame.grid(row=0, column=0, columnspan=5, padx=8, pady=4, sticky="w")

init_detection_plot(frame_detection)
add_card_header_extras(detection_controls_frame, "CFAR", INFO_CFAR, detection_fig, 
                       lambda: last_detections if last_detections is not None else None,
                       update_detection_plot)

tk.Label(detection_controls_frame, text="CFAR Parameters:", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 10))
tk.Label(detection_controls_frame, text="Guard Cells:").pack(side="left", padx=(10, 2))
cfar_guard_var = tk.StringVar(value="2")
tk.Entry(detection_controls_frame, textvariable=cfar_guard_var, width=5).pack(side="left", padx=(0, 10))
tk.Label(detection_controls_frame, text="Training Cells:").pack(side="left", padx=(0, 2))
cfar_training_var = tk.StringVar(value="8")
tk.Entry(detection_controls_frame, textvariable=cfar_training_var, width=5).pack(side="left", padx=(0, 10))
tk.Label(detection_controls_frame, text="Threshold Factor:").pack(side="left", padx=(0, 2))
cfar_threshold_var = tk.StringVar(value="3.0")
tk.Entry(detection_controls_frame, textvariable=cfar_threshold_var, width=5).pack(side="left", padx=(0, 10))
tk.Label(detection_controls_frame, text="Method:").pack(side="left", padx=(0, 2))
cfar_method_var = tk.StringVar(value="average")
cfar_method_combo = ttk.Combobox(detection_controls_frame, textvariable=cfar_method_var,
                                 values=["average", "ordered"], state="readonly", width=10)
cfar_method_combo.pack(side="left", padx=(0, 10))
tk.Button(detection_controls_frame, text="Run CFAR Detection", command=run_cfar_detection).pack(side="left", padx=(10, 0))
detection_frame_slider = tk.Scale(frame_detection, from_=0, to=0, orient="horizontal",
                                  label="Frame", command=on_detection_frame_change, state="disabled")
detection_frame_slider.grid(row=1, column=0, columnspan=5, sticky="ew", padx=4, pady=4)

# Helper tab (reparented)
init_helper_tab(frame_helper_main)

#
# NEW: Manual tab – render readme.md
_math_image_cache = {}

def latex_to_photoimage(latex_str, fontsize=10, color='black'):
    """Converts a LaTeX string to a tk.PhotoImage using matplotlib."""
    global _math_image_cache
    if Image is None or ImageTk is None:
        return None
    
    # Strip $ if they are at the ends
    latex_trimmed = latex_str.strip()
    if latex_trimmed.startswith('$$') and latex_trimmed.endswith('$$'):
        latex_trimmed = latex_trimmed[2:-2].strip()
    elif latex_trimmed.startswith('$') and latex_trimmed.endswith('$'):
        latex_trimmed = latex_trimmed[1:-1].strip()

    if not latex_trimmed:
        return None

    cache_key = (latex_trimmed, fontsize, color)
    if cache_key in _math_image_cache:
        return _math_image_cache[cache_key]

    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        # Ensure it has $ marks for matplotlib mathtext
        l_text = f"${latex_trimmed}$"

        # Create a figure with transparent background
        fig = Figure(figsize=(0.1, 0.1), dpi=120)
        fig.patch.set_alpha(0)
        FigureCanvasAgg(fig)
        
        # Add text at (0,0)
        fig.text(0, 0, l_text, fontsize=fontsize, color=color)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05, transparent=True)
        buf.seek(0)
        
        img = Image.open(buf)
        photo = ImageTk.PhotoImage(img)
        _math_image_cache[cache_key] = photo
        return photo
    except Exception:
        return None

def render_mermaid_to_photoimage(mermaid_text):
    """
    Renders a simple Mermaid graph to a PhotoImage using matplotlib.
    Currently supports 'graph TD' and 'graph LR' with basic node/edge parsing.
    """
    if Image is None or ImageTk is None:
        return None
        
    try:
        # Basic parsing
        lines = [l.strip() for l in mermaid_text.split("\n") if l.strip()]
        nodes = {} # id -> label
        edges = [] # (from, to)
        
        direction = "TD"
        for line in lines:
            if line.startswith("graph"):
                if "LR" in line: direction = "LR"
                continue
            
            # Simple edge parsing: A --> B
            if "-->" in line:
                parts = re.split(r"-->", line)
                prev_id = None
                for part in parts:
                    part = part.strip()
                    # Match id[label] or id(label) or id
                    m = re.match(r"([A-Za-z0-9_.]+)(?:\[(.*?)\]|\((.*?)\))?", part)
                    if m:
                        node_id = m.group(1)
                        label = m.group(2) or m.group(3)
                        if label:
                            nodes[node_id] = label
                        elif node_id not in nodes:
                            nodes[node_id] = node_id

                        if prev_id:
                            edges.append((prev_id, node_id))
                        prev_id = node_id
            else:
                # Node definition: A[Label]
                m = re.match(r"([A-Za-z0-9_.]+)(?:\[(.*?)\]|\((.*?)\))?", line)
                if m:
                    node_id = m.group(1)
                    label = m.group(2) or m.group(3)
                    if label:
                        nodes[node_id] = label
                    elif node_id not in nodes:
                        nodes[node_id] = node_id

        if not nodes:
            return None

        # Create Figure
        fig = Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.axis("off")

        # Simple hierarchical layout
        layers = {}
        all_to = {e[1] for e in edges}
        roots = [n for n in nodes if n not in all_to]
        if not roots and nodes: roots = [list(nodes.keys())[0]]
        
        curr_layer = roots
        layer_idx = 0
        visited = set()
        while curr_layer:
            for n in curr_layer:
                if n not in layers:
                    layers[n] = layer_idx
                visited.add(n)
            
            next_layer = []
            for n in curr_layer:
                for f, t in edges:
                    if f == n and t not in visited:
                        next_layer.append(t)
            curr_layer = list(dict.fromkeys(next_layer)) # unique
            layer_idx += 1
            if layer_idx > 20: break 

        for n in nodes:
            if n not in layers:
                layers[n] = layer_idx

        # Positions
        pos = {}
        layer_counts = {}
        for n, l in layers.items():
            layer_counts[l] = layer_counts.get(l, 0) + 1
        
        curr_y_offset = {l: 0 for l in layer_counts}
        max_l = max(layers.values()) if layers else 0
        
        for n, l in layers.items():
            count = layer_counts[l]
            if direction == "LR":
                x = 1 + (l / max_l * 8) if max_l > 0 else 5
                y = 5 + (curr_y_offset[l] - (count-1)/2) * 1.2
            else: # TD
                y = 9 - (l / max_l * 8) if max_l > 0 else 5
                x = 5 + (curr_y_offset[l] - (count-1)/2) * 2.5
            pos[n] = (x, y)
            curr_y_offset[l] += 1

        # Draw edges
        for f, t in edges:
            fx, fy = pos[f]
            tx, ty = pos[t]
            ax.annotate("", xy=(tx, ty), xytext=(fx, fy),
                        arrowprops=dict(arrowstyle="-|>", color="#34495e", lw=1.5, 
                                      shrinkA=12, shrinkB=12, patchA=None, patchB=None,
                                      connectionstyle="arc3,rad=0.0"))

        # Draw nodes
        for nid, (x, y) in pos.items():
            label = nodes[nid]
            ax.text(x, y, label, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.5", fc="#ecf0f1", ec="#2c3e50", lw=1.5),
                    fontsize=9, fontfamily="Segoe UI")

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        # Save to PhotoImage
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, transparent=True)
        buf.seek(0)
        img = Image.open(buf)
        return ImageTk.PhotoImage(img)
    except Exception as e:
        print(f"Error rendering mermaid: {e}")
        return None

def render_markdown(text_widget, markdown_text):
    """Simple Markdown renderer for tk.Text using tags and image insertion for math."""
    text_widget.config(state="normal")
    text_widget.delete("1.0", tk.END)

    # Initialize image list to prevent GC
    if not hasattr(text_widget, "_math_images"):
        text_widget._math_images = []
    text_widget._math_images.clear()

    # Define fonts
    font_normal = ("Segoe UI", 10)
    font_h1 = ("Segoe UI", 16, "bold")
    font_h2 = ("Segoe UI", 14, "bold")
    font_h3 = ("Segoe UI", 12, "bold")
    font_h4 = ("Segoe UI", 11, "bold")
    font_code = ("Consolas", 10)
    font_bold = ("Segoe UI", 10, "bold")
    font_italic = ("Segoe UI", 10, "italic")

    # Configure tags
    text_widget.tag_configure("h1", font=font_h1, foreground="#2c3e50", spacing1=15, spacing3=10)
    text_widget.tag_configure("h2", font=font_h2, foreground="#34495e", spacing1=12, spacing3=8)
    text_widget.tag_configure("h3", font=font_h3, foreground="#455a64", spacing1=10, spacing3=6)
    text_widget.tag_configure("h4", font=font_h4, foreground="#546e7a", spacing1=8, spacing3=4)
    text_widget.tag_configure("code", font=font_code, background="#f0f0f0", foreground="#c0392b")
    text_widget.tag_configure("code_block", font=font_code, background="#f8f9fa", lmargin1=20, rmargin=20, spacing1=5, spacing3=5)
    text_widget.tag_configure("bold", font=font_bold)
    text_widget.tag_configure("italic", font=font_italic)
    text_widget.tag_configure("bullet", lmargin1=20, lmargin2=35)
    text_widget.tag_configure("hr", font=("Segoe UI", 2), background="#cccccc", spacing1=10, spacing3=10)
    text_widget.tag_configure("link", foreground="#0000ee", underline=True)
    text_widget.tag_configure("math", font=("Consolas", 10, "italic"), foreground="black")

    def insert_with_links(text, tags=()):
        if not isinstance(tags, tuple):
            tags = (tags,) if tags else ()
        
        # Pattern for [text](url) or plain http/https URL
        # We use non-greedy matching for [text] and (url)
        link_pattern = r'(\[.*?\]\(https?://.*?\)|https?://[^\s)\]]+)'
        parts = re.split(link_pattern, text)
        for i, part in enumerate(parts):
            if i % 2 == 1:
                # Potential link
                if part.startswith("[") and "](" in part:
                    match = re.match(r'\[(.*?)\]\((.*?)\)', part)
                    if match:
                        display_text, url = match.groups()
                        # Create a unique tag for this URL to handle clicks
                        url_tag = f"url-{url}"
                        text_widget.tag_configure(url_tag, foreground="#0000ee", underline=True)
                        text_widget.tag_bind(url_tag, "<Button-1>", lambda e, u=url: webbrowser.open(u))
                        text_widget.tag_bind(url_tag, "<Enter>", lambda e: text_widget.config(cursor="hand2"))
                        text_widget.tag_bind(url_tag, "<Leave>", lambda e: text_widget.config(cursor=""))
                        text_widget.insert(tk.END, display_text, tags + (url_tag, "link"))
                        continue
                
                # Plain URL
                url = part
                url_tag = f"url-{url}"
                text_widget.tag_configure(url_tag, foreground="#0000ee", underline=True)
                text_widget.tag_bind(url_tag, "<Button-1>", lambda e, u=url: webbrowser.open(u))
                text_widget.tag_bind(url_tag, "<Enter>", lambda e: text_widget.config(cursor="hand2"))
                text_widget.tag_bind(url_tag, "<Leave>", lambda e: text_widget.config(cursor=""))
                text_widget.insert(tk.END, part, tags + (url_tag, "link"))
            else:
                if part:
                    text_widget.insert(tk.END, part, tags)

    lines = markdown_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Code blocks
        if line.strip().startswith("```"):
            if line.strip().startswith("```mermaid"):
                mermaid_code = ""
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    mermaid_code += lines[i] + "\n"
                    i += 1
                
                photo = render_mermaid_to_photoimage(mermaid_code)
                if photo:
                    text_widget.image_create(tk.END, image=photo)
                    text_widget._math_images.append(photo)
                    text_widget.insert(tk.END, "\n")
                else:
                    text_widget.insert(tk.END, mermaid_code, "code_block")
                
                if i < len(lines): i += 1 # skip closing ```
                continue
            else:
                # Standard code block
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    text_widget.insert(tk.END, lines[i] + "\n", "code_block")
                    i += 1
                text_widget.insert(tk.END, "\n")
                if i < len(lines): i += 1 # skip closing ```
                continue

        # Horizontal rule
        if line.strip() == "---" or line.strip() == "***":
            text_widget.insert(tk.END, " " * 200 + "\n", "hr")
            i += 1
            continue

        # Headers
        stripped = line.lstrip()
        if stripped.startswith("####"):
            insert_with_links(stripped[4:].lstrip() + "\n", "h4")
            i += 1
            continue
        elif stripped.startswith("###"):
            insert_with_links(stripped[3:].lstrip() + "\n", "h3")
            i += 1
            continue
        elif stripped.startswith("##"):
            insert_with_links(stripped[2:].lstrip() + "\n", "h2")
            i += 1
            continue
        elif stripped.startswith("#"):
            insert_with_links(stripped[1:].lstrip() + "\n", "h1")
            i += 1
            continue

        # List items
        is_bullet = False
        if line.lstrip().startswith("- ") or line.lstrip().startswith("* "):
            text_widget.insert(tk.END, "  • ", "bullet")
            line = line.lstrip()[2:]
            is_bullet = True
        elif line.lstrip().split(".")[0].isdigit() and ". " in line:
            # Numbered list
            prefix = line.lstrip().split(".")[0] + ". "
            text_widget.insert(tk.END, "  " + prefix, "bullet")
            line = line.lstrip()[len(prefix):]
            is_bullet = True

        # Inline formatting (math, bold, italic, code)
        # Process block math ($$text$$)
        if line.strip().startswith("$$") and line.strip().endswith("$$"):
            photo = latex_to_photoimage(line.strip(), fontsize=10)
            if photo:
                text_widget.image_create(tk.END, image=photo)
                text_widget._math_images.append(photo)
                text_widget.insert(tk.END, "\n")
            else:
                text_widget.insert(tk.END, line.strip()[2:-2], "math")
                text_widget.insert(tk.END, "\n")
            i += 1
            continue

        # Process bold (**text**)
        parts = line.split("**")
        for j, part in enumerate(parts):
            tag = "bold" if j % 2 == 1 else None
            
            # Process italic (*text*) within non-bold parts
            if tag is None:
                subparts = part.split("*")
                for k, subpart in enumerate(subparts):
                    subtag = "italic" if k % 2 == 1 else None
                    
                    # Process inline math ($text$)
                    if subtag is None:
                        mathparts = subpart.split("$")
                        for m, mathpart in enumerate(mathparts):
                            mathtag = "math" if m % 2 == 1 else None
                            
                            if mathtag:
                                photo = latex_to_photoimage(mathpart, fontsize=10)
                                if photo:
                                    text_widget.image_create(tk.END, image=photo)
                                    text_widget._math_images.append(photo)
                                else:
                                    text_widget.insert(tk.END, mathpart, mathtag)
                            else:
                                # Process inline code (`text`)
                                codeparts = mathpart.split("`")
                                for n_idx, codepart in enumerate(codeparts):
                                    codetag = "code" if n_idx % 2 == 1 else None
                                    if codetag:
                                        text_widget.insert(tk.END, codepart, codetag)
                                    else:
                                        insert_with_links(codepart)
                    else:
                        insert_with_links(subpart, subtag)
            else:
                insert_with_links(part, tag)
        
        text_widget.insert(tk.END, "\n")
        if is_bullet:
            text_widget.tag_add("bullet", "insert linestart", "insert lineend")
        
        i += 1

    text_widget.config(state="disabled")

def _load_manual_into(text_widget: tk.Text):
    try:
        md_path = resource_path("readme.md")
        with open(md_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        content = f"Failed to load readme.md: {e}"
    
    render_markdown(text_widget, content)

manual_top = ttk.Frame(frame_manual_main)
manual_top.pack(side="top", fill="x", padx=6, pady=(6, 0))
ttk.Button(manual_top, text="Refresh", command=lambda: _load_manual_into(manual_text)).pack(side="left")

manual_container = ttk.Frame(frame_manual_main)
manual_container.pack(side="top", fill="both", expand=True, padx=6, pady=6)
manual_text = tk.Text(manual_container, wrap="word", font=("Segoe UI", 10), padx=10, pady=10)
manual_scroll = ttk.Scrollbar(manual_container, orient="vertical", command=manual_text.yview)
manual_text.configure(yscrollcommand=manual_scroll.set)
manual_scroll.pack(side="right", fill="y")
manual_text.pack(side="left", fill="both", expand=True)
_load_manual_into(manual_text)


# Lock current window size as minimum to avoid accidental shrinking during figure redraws (e.g., RTI)
try:
    root.update_idletasks()
    root.minsize(root.winfo_width(), root.winfo_height())
except Exception:
    pass

if __name__ == "__main__":
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("GUI stopped by user (KeyboardInterrupt).")