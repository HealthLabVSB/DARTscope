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
import _tkinter
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import theme
import os
import log_view
log_view.install()          # the Log tab shows what is printed (the .exe has no console)
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import frame_cache
frame_cache.install()               # a canvas drawn outside the frames' own drawing forgets their pictures
import plot_interactivity
plot_interactivity.install_settled_resize()   # a plot is redrawn when a window drag pauses, not at every pixel
from matplotlib.figure import Figure
import frame_marker

frame_marker.install()      # the frame line in every plot over time (frame_marker.py)
import math
import re
import webbrowser
import io
try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None
from AOA import compute_aoa_spectrum, compute_range_azimuth_map, compute_azimuth_elevation_map, compute_range_azimuth_elevation_map  # NEW
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
    get_all_bin_parts,
    load_raw_cube,
    compute_range_doppler_map,  # FIX: add missing import
    MIN_TARGET_RANGE_M,
    motion_energy,
    strongest_range_bin,
)
import radar_processing
from rti import compute_rti                 # NEW: RTI from module
from range_profile import range_profile_dbfs, range_profile_rx
from gui_helper import init_helper_tab
from detection import cfar_2d_ca, extract_peaks, detections_to_range_velocity, detect_targets_rae  # NEW
from detection import range_only as cfar_range_only
from range_fft import compute_range_fft_cube
import antenna_config
import capture_params
import clutter as clutter_filters
import calibration
import iq_balance
import spectral
import velocity
import limb_motion
import vital_bands
import fastcheck
import fastcheck_gui
from manual_view import ManualView
from flow_layout import FlowFrame, ScrollFrame, cap_plot_heights
from capture_params import (DEFAULT_FPS_FALLBACK, _derive_period_ms,
                            _tx_selection_options, _selected_chirp_positions)
import vitalsigns
import hr_gui
import ai_vital_gui  # NEW
import multi_vitals_gui
import radar_imaging_gui
from plot_interactivity import create_interactive_canvas, PlotInteractivityManager, add_plot_control_bar, bin_extent, few_bins_interpolation, ColumnPooledImage, image_percentiles  # NEW
import matplotlib.animation as animation
import scipy.signal as signal
from scipy import ndimage
import sys
import time
from playback import Meter, frame_step, next_play_frame
import threading
from contextlib import contextmanager

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

# ---------------- Progress / status bar ----------------
# Widgets are created later in the GUI layout section; these helpers look them up
# lazily so they can be used by any run_* function defined above that point.
root = None            # assigned in the GUI layout section below
progress_bar = None
progress_text_var = None
_progress_depth = 0
_progress_handle = None     # the innermost progress() shown now: run_in_background repaints it


def _duration(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s} s" if s < 60 else f"{s // 60} min {s % 60:02d} s"


class _ProgressHandle:
    """Handle yielded by progress(); call step() inside long loops."""

    UI_INTERVAL_S = 0.05  # repaint at most 20x per second

    def __init__(self, text: str, total: int = 0):
        self.text = text
        self.total = int(total or 0)
        self.current = 0
        self._last_ui = 0.0
        self.t0 = time.monotonic()

    def status(self) -> str:
        """'Imaging ...  (37 %, 1 min 05 s, about 1 min 50 s left)': the time running
        shows it has not stopped, the estimate how long it will still take."""
        elapsed = time.monotonic() - self.t0
        parts = []
        if self.total:
            frac = min(self.current, self.total) / max(1, self.total)
            parts.append(f"{int(100 * frac)} %")
        if elapsed >= 1.0:
            parts.append(_duration(elapsed))
            if self.total and 0.03 <= frac < 1.0 and elapsed >= 3.0:
                parts.append(f"about {_duration(elapsed * (1 - frac) / frac)} left")
        return f"{self.text}  ({', '.join(parts)})" if parts else self.text

    def _paint(self, force: bool = False):
        now = time.monotonic()
        if not force and (now - self._last_ui) < self.UI_INTERVAL_S:
            return
        self._last_ui = now
        try:
            if progress_text_var is not None:
                progress_text_var.set(self.status())
            if progress_bar is not None and self.total:
                progress_bar['value'] = 100.0 * min(self.current, self.total) / max(1, self.total)
            if root is not None:
                root.update_idletasks()
        except Exception:
            pass

    def step(self, n: int = 1, text: str | None = None):
        """Advance and repaint. Main thread only."""
        self.current += n
        if text:
            self.text = text
        self._paint()

    def set_text(self, text: str):
        self.text = text
        self._paint(force=True)

    def advance(self, n: int = 1):
        """
        Advance the counter without touching any widget, so it is safe to call from a
        worker thread. run_in_background() repaints from the main thread.
        """
        self.current += n

    def callback(self):
        """
        Adapter for processing functions taking progress_cb(done, total).

        Deliberately does not paint: these callbacks run inside the worker thread, and
        Tk may only be touched from the thread that owns the widgets.
        """
        def _cb(done, total):
            self.total = int(total or 0)
            self.current = int(done)
            if _cancel_requested:
                raise ComputationCancelled()
        return _cb

# ---------------- Background computation ----------------
class ComputationCancelled(Exception):
    """Raised inside a worker when the user presses Cancel."""

_busy = False
_cancel_requested = False
_cancel_button = None

def _set_controls_busy(busy: bool):
    """Disable the Run buttons while a computation is running; enable Cancel."""
    for btn in _run_buttons.values():
        try:
            btn.config(state="disabled" if busy else "normal")
        except Exception:
            pass
    if _cancel_button is not None:
        try:
            _cancel_button.config(state="normal" if busy else "disabled")
        except Exception:
            pass

def request_cancel():
    global _cancel_requested
    if _busy:
        _cancel_requested = True
        try:
            if progress_text_var is not None:
                progress_text_var.set("Cancelling ...")
        except Exception:
            pass

def _pump_events(budget_s: float = 0.05):
    """
    Handle the window's pending events for at most budget_s. Not root.update(): that returns
    only once no event is left, and on macOS, where redrawing the window with its plots
    takes longer than the progress bar's animation step, events never run out - update()
    never returned and a finished computation was never picked up (2026-10-06).
    """
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        if not root.tk.dooneevent(_tkinter.DONT_WAIT):
            break


def run_in_background(fn, *args, handle=None, **kwargs):
    """
    Run a computation off the Tk thread and return its result.

    The heavy numpy work used to run directly in the button callback, which blocks the
    event loop: the window stopped redrawing and the OS marked it "Not Responding".
    Here the work happens in a worker thread while this (main) thread keeps pumping
    events, so the window stays responsive and the progress bar actually animates.

    The worker must not touch widgets - it only computes and returns. Everything the
    caller does with the result afterwards is back on the main thread, unchanged.
    Exceptions are re-raised in the caller so existing error handling still applies.
    """
    global _busy, _cancel_requested
    if root is None:
        return fn(*args, **kwargs)   # headless (tests, import) - just run it

    outcome = {}

    def _worker():
        try:
            outcome['value'] = fn(*args, **kwargs)
        except BaseException as exc:      # re-raised on the main thread below
            outcome['error'] = exc

    was_busy = _busy
    _busy = True
    if not was_busy:
        _cancel_requested = False
        _set_controls_busy(True)
    thread = threading.Thread(target=_worker, daemon=True, name="dartscope-compute")
    try:
        thread.start()
        while thread.is_alive():
            shown = handle if handle is not None else _progress_handle
            if shown is not None:
                shown._paint()
            try:
                _pump_events()
            except tk.TclError:
                break                     # window closed while computing
            thread.join(0.02)
        thread.join()
    finally:
        _busy = was_busy
        if not _busy:
            _set_controls_busy(False)
            _cancel_requested = False

    if 'error' in outcome:
        raise outcome['error']
    return outcome.get('value')

def report_error(title: str, message: str, exc: BaseException | None = None):
    """
    Show an error dialog, except when the user simply cancelled the computation -
    that is a normal outcome, not a failure.
    """
    if isinstance(exc, ComputationCancelled):
        try:
            if progress_text_var is not None:
                progress_text_var.set("Computation cancelled")
        except Exception:
            pass
        print("[GUI] Computation cancelled by the user.")
        return
    messagebox.showerror(title, message)

# ---------------- Deferred recompute ----------------
# Settings widgets only flag that a recompute is needed instead of recomputing
# immediately, so several checkboxes can be changed in one go. The actual (expensive)
# recompute runs when the user presses that view's Run button, which is highlighted
# while changes are pending (the blue default button on macOS, yellow elsewhere).
_run_buttons = {}        # view label -> Run button widget
_pending_views = set()

def register_run_button(label: str, button):
    """Associate a Run button with a view so pending changes can highlight it."""
    _run_buttons[label] = button
    return button

def mark_settings_changed(label: str, _evt=None):
    """Flag that <label> needs recomputing; does NOT start the computation."""
    _pending_views.add(label)
    btn = _run_buttons.get(label)
    if btn is not None:
        theme.set_prominent(btn, True)
    try:
        if progress_text_var is not None:
            progress_text_var.set(f"{label}: settings changed - press Run to recompute")
    except Exception:
        pass

def clear_settings_changed(label: str):
    """Called by the run_* functions once the view has been recomputed."""
    _pending_views.discard(label)
    btn = _run_buttons.get(label)
    if btn is not None:
        theme.set_prominent(btn, False)

@contextmanager
def progress(text: str, total: int = 0):
    """
    Show a status message (and a progress bar) while a long operation runs.

    Determinate when `total` is given, otherwise a pulsing bar. Repaints are
    throttled, so calling step() in a tight loop is cheap.
    """
    global _progress_depth, _progress_handle
    handle = _ProgressHandle(text, total)
    outer = _progress_handle
    _progress_handle = handle
    _progress_depth += 1
    try:
        try:
            if progress_bar is not None:
                if total:
                    progress_bar.configure(mode='determinate', maximum=100.0)
                    progress_bar['value'] = 0.0
                else:
                    progress_bar.configure(mode='indeterminate')
                    progress_bar.start(50)
            if root is not None:
                root.config(cursor="watch")
        except Exception:
            pass
        handle._paint(force=True)
        yield handle
    finally:
        _progress_depth = max(0, _progress_depth - 1)
        _progress_handle = outer
        elapsed = time.monotonic() - handle.t0
        if elapsed >= 1.0:
            # the Log tab keeps what each step took: where a long run spent its time
            print(f"[time] {text.rstrip(' .')}: {elapsed:.1f} s")
        if outer is not None:
            outer._paint(force=True)
        if _progress_depth == 0:
            try:
                if progress_bar is not None:
                    progress_bar.stop()
                    progress_bar.configure(mode='determinate')
                    progress_bar['value'] = 0.0
                if progress_text_var is not None:
                    progress_text_var.set("Ready")
                if root is not None:
                    root.config(cursor="")
                    root.update_idletasks()
            except Exception:
                pass

# --- GLOBAL VITAL SIGNS STATE ---
vital_fig = None
vital_canvas = None
vital_toolbar = None # NEW: interactive navigation toolbar
vital_plot_manager = None # NEW: SpanSelector / reset view manager
vital_ax_br = None
vital_ax_hr = None
vital_ax_phase = None
vital_ax_spec_br = None
vital_ax_spec_hr = None
vital_br_val_var = None
vital_hr_val_var = None
vital_rbin_var = None
vital_bin_idx_var = None # NEW
vital_clutter_var = None # NEW
vital_dwt_var = None # NEW
vital_kalman_var = None # NEW
vital_dc_method_var = None # NEW
vital_phase_method_var = None # NEW
vital_dist_text_var = None # NEW
vital_bin_slider = None # NEW
vital_auto_bin_var = None
vital_rx_var = None
vital_auto_rx_var = None
vital_win_br_var = None
vital_win_hr_var = None
last_vital_data = None
last_vital_rfft_cube = None
last_vital_rf_meta = None
last_vital_cube_ptr = None
last_vital_clutter_cube = None
last_vital_clutter_val = None
last_ai_rfft_complex = None
last_ai_rfft_meta = None
in_run_vital_signs = False # NEW
vital_ax_rp = None # NEW

# NEW for sliding window and ROI
vital_roi_width_var = None
vital_roi_mode_var = None
vital_win_step_var = None
vital_conf_thresh_var = None
vital_hr_stats_var = None
vital_rr_stats_var = None
vital_roi_info_var = None
vital_ax_hr_time = None
vital_ax_rr_time = None

# --- GLOBAL AOA STATE ---
texas_aoa_mimo_mode_var = None    # Angle of Arrival tab (also drives the Movement Map)
lane_swap_var = None
texas_aoa_shift_var = None

# --- GLOBAL TIME-TRACKING STATE ---
texas_aoa_ax2 = None
empty_scene_cube = None
texas_aoa_sub_clutter_var = None
texas_aoa_doppler_comp_var = None
texas_aoa_track_moving_var = None

# --- GLOBAL RANGE-ANGLE STATE ---
ra_fig = None
ra_ax = None
ra_canvas = None
ra_im = None
ra_cbar = None
ra_peak_scatter = None   # peak overlay artist for the Range-Angle tab
ra_nang_var = None
ra_clutter_var = None
ra_peaks_var = None
ra_remove_dc_var = None
ra_r_start_var = None
ra_r_end_var = None
last_ra_map = None
last_ra_meta = None
texas_aoa_method_var = None
aoa_min_range_var = None # NEW: Global range crop
aoa_angle_limit_var = None # NEW: Global angle limit
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
                      background=theme.TOOLTIP_BG, foreground=theme.LABEL,
                      relief=tk.FLAT, borderwidth=0, highlightthickness=1,
                      highlightbackground=theme.SEPARATOR, padx=6, pady=3,
                      font=theme.font(9))
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
    """data: an array, or (array, "col1,col2,...") to write a header line first."""
    header = None
    if isinstance(data, tuple) and len(data) == 2:
        data, header = data
    if data is None:
        messagebox.showwarning("Export", "No data to export.")
        return
    data = np.asarray(data)
    file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV files", "*.csv")],
                                             title=f"Save {title} data as CSV")
    if file_path:
        try:
            if header is not None:
                np.savetxt(file_path, data, delimiter=",", header=header, comments="")
            elif data.ndim <= 2:
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

def add_card_header_extras(parent_frame, card_name, info_text, fig, data_func, update_func=None,
                           extra_items=None, manual_section=None):
    """
    Adds ⓘ icon with tooltip and Export button to the card control bar.
    extra_items: further Export menu entries, [(label, callable)].
    manual_section: the Manual section a click on ⓘ opens (default: card_name).
    """
    extras_frame = ttk.Frame(parent_frame)
    extras_frame.pack(side="right", padx=10)
    
    # Info icon
    info_lbl = tk.Label(extras_frame, text="ⓘ", font=theme.font(12), fg=theme.ACCENT, cursor="hand2")
    info_lbl.pack(side="left", padx=5)
    Tooltip(info_lbl, info_text + "\n\n(click to open this view in the Manual)")
    info_lbl.bind("<Button-1>", lambda _e: open_manual(manual_section or card_name))
    
    # Export button with menu
    export_btn = ttk.Menubutton(extras_frame, text="Export")
    export_btn.pack(side="left", padx=5)
    
    export_menu = tk.Menu(export_btn, tearoff=0)
    export_btn["menu"] = export_menu
    
    export_menu.add_command(label="Current frame (PNG)", command=lambda: export_to_png(fig, card_name))
    export_menu.add_command(label="Data (CSV)", command=lambda: export_to_csv(data_func(), card_name))
    if update_func:
        export_menu.add_command(label="All frames (Video)", command=lambda: export_to_video(fig, update_func, card_name))
    for label, func in (extra_items or []):
        export_menu.add_command(label=label, command=func)

# --- Processing info strings (for tooltips) ---
INFO_RANGE_FFT = "Processing Chain:\n1. DC Removal (per chirp)\n2. Hann Window\n3. 1D FFT (fast-time)\n4. Magnitude calculation\n5. Non-coherent average (across RX and Chirps)\n6. Logarithmic scale (dB)"

INFO_VITAL_SIGNS = """Vital Signs Pipeline:
1. Range FFT of the complex ADC data.
2. Clutter removal (optional): subtract the time-average of each range bin.
3. Range bin: the slider, or Auto Bin (most breathing-band energy).
4. ROI mode: single bin, best bin in ROI, or weighted ROI average.
5. Average the chirps of each frame; RX from the RX box or Auto.
6. IQ DC correction, phase extraction (unwrap or DACM).
7. Phase -> displacement [m], drift removal.
8. Kalman filter / Wiener denoising (optional).
9. Band-pass filters in the shared search bands (Bands box).
10. Sliding windows (BR Win, HR Win, Step): spectral peak in each,
    breathing harmonics suppressed in the heart band.
11. Statistics from the windows whose peak/average >= Min peak/avg.
"""
INFO_RTI = "Processing Chain:\n1. Range FFT (Hann window, DC removal)\n2. Magnitude calculation\n3. Non-coherent average (across RX)\n4. Time Unfolding (stacking chirps/frames)\n5. DC & Edge Notch filtering\n6. Logarithmic scale (dB)"
INFO_RP = "Processing Chain:\n1. DC Removal\n2. Hann Window\n3. 1D FFT (Range FFT size from the Analysis bar)\n4. Coherent Averaging (across the chirps of the frame on the frame slider)\n5. RX combining as the Antenna selector says (MRC / SUM / RXk)\n6. Normalization to Full Scale (dBFS)\n7. Optional Zoom FFT over a narrow range band"
INFO_RD = "Processing Chain:\n1. Range FFT (Hann window, DC removal)\n2. TX Separation (TDM demux)\n3. Clutter suppression (Clutter selector on the Analysis bar: mean, ema, mti2, mti3, svd, none)\n4. Doppler Windowing (Hann, optional)\n5. Doppler FFT (Slow-time, shifted)\n6. RX Combining (MRC/SUM/RXk)\n7. Logarithmic scale (dB)"
INFO_AOA = "Processing Chain:\n1. Complex Range FFT\n2. TX Separation & MIMO array formation (antenna configuration)\n3. TDM Doppler compensation (optional)\n4. Channel calibration (optional)\n5. Clutter suppression (Remove Static, method from the Analysis bar)\n6. Bartlett or Capon beamforming\n7. Averaging across loops\n8. Tracking / 3D CFAR detection (optional)"
INFO_MOV_MAP = "Processing Chain:\n1. Compute 2D Range-Azimuth map for each frame\n2. Perform Temporal FFT per (Range, Azimuth) pixel\n3. Sum energy in selected band (Breathing and HR from the shared vital bands; Movement = everything between and above them)\n4. Accumulate over time (sliding window or full)\n5. Normalize and display as 2D heatmap"
INFO_MD = "Processing Chain:\n1. Range FFT; the chirps of one TX, as in the Range-Doppler map\n2. Clutter suppression (Clutter selector), along slow time\n3. Range gate: following the person (Auto) or the fixed bins\n4. STFT along slow time (Window [ms], Blackman-Harris), power over gate and RX\n5. Noise floor, torso line, upper/lower envelope on the circular velocity axis\n   (a swing faster than v_max is unfolded)\n6. Swings: envelope above its usual distance from the torso by Min swing\n7. Height of the fastest point from elevation (2D arrays), leg or arm"
INFO_VIEW1D = "Processing Chain:\n1. Range-Doppler Map calculation (run automatically if missing)\n2. Slice at selected Range Bin\n3. Doppler Spectrum extraction\n4. Temporal FFT (across frames)\n5. Energy per band and BR/HR peaks (shared vital bands)"
INFO_MULTI_VITALS = ("Processing Chain:\n1. Chirps of every TX averaged per frame; virtual array, element phase "
                     "correction and calibration as in the AoA views\n2. Every 0.5 s a map of what moves in "
                     "range and azimuth (Motion: chirps and 0.5 s of frames; Breathing: 10 s of frames; "
                     "Range only: no angles)\n3. Peaks above the noise, angle sidelobes of stronger ones "
                     "dropped\n4. Tracker: Kalman filter, Hungarian assignment, broken tracks joined\n"
                     "5. Every stay of a person (slower than 0.2 m/s, 8 s or more): the range bins around "
                     "them through beams steered at them (Bartlett or Capon)\n6. Breathing and heart "
                     "rate of the cell with Fastcheck's chain; echoes found by their breathing")
INFO_IMAGING = ("Processing Chain:\n1. Range FFT; chirps of every TX slot as slow time; the still scene (mean "
                "of the loops) apart from what moves; Doppler FFT\n2. Detection: range-Doppler cells above "
                "the ordered-statistics noise (Moving), range bins above the thermal noise (Still); window "
                "sidelobes of stronger cells dropped\n3. TDM-MIMO slot timing corrected; velocities beyond "
                "one slot's limit unfolded where two directions explain the echo clearly better\n4. Angles: "
                "Bartlett over azimuth and elevation, up to 2 sources a cell (4 in the still scene), near "
                "echoes with the exact paths, refined against each other (RELAX)\n5. A source kept where its "
                "own echo peaks in range and Doppler; range and velocity refined there\n6. Points x, y, z, "
                "velocity (radar height and tilt applied); DBSCAN over 3 frames into objects; Kalman "
                "tracker")
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
rp_zoom_var = rp_zoom_lo_var = rp_zoom_hi_var = None
clutter_method_var = clutter_tau_var = clutter_rank_var = None
cfar_unfold_var = cfar_vlimit_var = None
range_fft_var = None
range_fft_combo = None
active_calibration = None          # calibration.Calibration applied to all AoA views
active_calibration_path = None
# Time segment (roadmap 9.6): last_full_cube is the whole decoded capture, last_cube the
# chosen segment of it (a view, no copy) - re-cutting needs no new decode.
last_full_cube = None
last_frame_damaged_full = None
segment_frame0 = 0
segment_from_var = segment_to_var = segment_info_var = None
iq_correct_var = None
calib_status_var = None

# NEW: Range-Doppler state variables
last_rd_map = None
last_rd_buffer = None  # NEW: Precomputed dB map for playback
last_rd_meta = None
rd_im = None
rd_cbar = None
rd_zero_line = None  # NEW: zero-velocity guide line

# NEW: Range-Doppler playback state
rd_playing = False
_rd_play_after_id = None   # pending root.after() handle of the RD playback loop
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

# --- Angle of Arrival state ---
aoa_invert_phase_var = None
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
texas_aoa_view_var = None # NEW: View type (1D, RA, AE, RAE)
texas_aoa_detect_var = None # NEW: CFAR detection toggle (Range-Az-El view)
texas_aoa_cfar_factor_var = None # NEW: CFAR threshold factor
texas_rae_proj_var = None # NEW: RAE projection mode (Top / Slice)

# Range-Azimuth cube the Movement Map is built on; cached so changing a Movement Map
# setting does not recompute it.
last_mov_ra_cube = None
last_mov_ra_meta = None

# NEW: AoA bin selection UI state
aoa_bin_slider = None
aoa_bin_var = None
aoa_bin_label_var = None

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
md_gate_var = md_width_var = md_window_var = md_thr_var = None   # limb_motion settings
md_swing_var = md_height_var = md_env_var = md_readout_var = None

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
        "rangeRes_cm", "effectiveRangeRes_cm", "rangeBinSpacing_gui_cm",
        "maxRange_m", "maxRange_note",
        "adcSampleTime_us", "rampWindow_us", "effCaptureWindow_us",
    ]
    doppler_keys = [
        "wavelength_mm", "chirpRepetitionTime_us", "dopplerBins",
        "maxVelocity_m_s", "velocityRes_m_s",
    ]
    datafmt_keys = [
        "dataFmt_adcFmt", "dataFmt_adcBits", "dataFmt_iqOrder",
        "dataFmt_rxChanEnMask", "dataFmt_rxChannels", "dataFmt_isInterleaved",
        "numLanes", "numLanes_source", "lanesActive",
    ]
    rx_tx_keys = ["rxGain", "txOutPowerBackoffCode", "txPhaseShifter"]
    # What the DCA1000 itself recorded, from the capture's _Raw_LogFile.csv
    recorder_keys = [
        "dca_log_mode", "dca_stop_mode", "dca_lane_mode", "lane_mode_mismatch",
        "dca_duration_s", "dca_packets_received", "dca_packets_zero_filled",
        "dca_packet_loss_pct", "dca_out_of_sequence",
    ]
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
    groups.append(("Velocity (Doppler)", pick(doppler_keys)))
    groups.append(("Data Format", pick(datafmt_keys)))
    groups.append(("RX/TX & Gains", pick(rx_tx_keys)))
    groups.append(("Frame (derived/log)", pick(frame_keys)))
    rec = pick(recorder_keys)
    if rec:
        groups.append(("Recorder (DCA1000 log)", rec))
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
    style.configure("ParamHeader.TLabel", font=theme.font(10, "bold"))

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

def _drop_colorbar(name: str):
    """
    Take the colorbar held in the global `name` off its figure before forgetting it.
    Setting the handle to None alone left the old bar in place, and the next plot added
    another one beside it (one more after every recording opened) while the plot itself
    got narrower. By name, because some of them do not exist before their view is built.
    """
    cbar = globals().get(name)
    if cbar is not None:
        try:
            cbar.remove()
        except Exception:
            pass
    return None


def _reset_cached_data():
    """Clear all previously loaded and computed data."""
    global last_cube, last_frame_damaged, last_range_fft, last_range_fft_meta
    global last_rti, last_rti_meta, last_range_profile, last_range_profile_meta
    global last_detections, last_detection_peaks, detection_im, detection_cbar, detection_scatter  # NEW
    global last_mov_ra_cube, last_mov_ra_meta
    global last_md_map, last_md_meta, md_im, md_cbar, md_indicator_line  # NEW Micro-Doppler
    global view1d_fig, view1d_canvas # NEW 1D View
    global last_mov_map, last_mov_meta, mov_im, mov_cbar # NEW Movement Map
    global last_rd_map, last_rd_meta, last_rd_buffer, rd_im, rd_cbar, rd_zero_line  # Range-Doppler
    global last_texas_aoa_spec, last_texas_aoa_meta, texas_aoa_im, texas_aoa_cbar  # Angle of Arrival
    global last_ra_map, last_ra_meta, ra_im, ra_cbar, ra_peak_scatter  # Range-Angle
    global global_frames_total, last_vital_data, last_vital_rfft_cube, last_vital_rf_meta, last_vital_cube_ptr, last_vital_clutter_cube, last_vital_clutter_val # NEW Vital Signs
    global last_ai_rfft_complex, last_ai_rfft_meta # NEW AI Vital Signs

    global_frames_total = 0
    frame_cache.forget()                      # another recording: no picture of the last one stays
    last_cube = None
    last_frame_damaged = None
    if globals().get("fastcheck_tab") is not None:
        fastcheck_tab.clear("New recording - press Run Fastcheck")
    if globals().get("multi_vitals_tab") is not None:
        multi_vitals_tab.clear("New recording - press Run Multi Vitals.")
    if globals().get("imaging_tab") is not None:
        imaging_tab.clear("New recording - press Run Imaging.")
    global last_full_cube, last_frame_damaged_full, segment_frame0
    last_full_cube = last_frame_damaged_full = None
    segment_frame0 = 0
    last_range_fft = None
    last_range_fft_meta = None
    last_rti = None
    last_rti_meta = None
    last_range_profile = None
    last_range_profile_meta = None
    last_detections = None          # NEW
    last_detection_peaks = None     # NEW
    detection_im = None             # NEW
    detection_cbar = _drop_colorbar("detection_cbar")
    detection_scatter = None        # NEW
    last_mov_ra_cube = None
    last_mov_ra_meta = None
    last_ra_map = None
    last_md_map = None              # NEW
    last_md_meta = None             # NEW
    md_im = None                    # NEW
    md_cbar = _drop_colorbar("md_cbar")
    md_indicator_line = None        # NEW
    last_mov_map = None             # NEW
    last_mov_meta = None            # NEW
    mov_im = None                   # NEW
    mov_cbar = _drop_colorbar("mov_cbar")
    # Range-Doppler / Angle of Arrival / Range-Angle were previously left in place, so after
    # loading a new file those tabs still showed the PREVIOUS capture until the user
    # pressed Run again.
    last_rd_map = None
    last_rd_meta = None
    last_rd_buffer = None
    rd_im = None
    rd_cbar = _drop_colorbar("rd_cbar")
    rd_zero_line = None
    last_texas_aoa_spec = None
    last_texas_aoa_meta = None
    texas_aoa_im = None
    texas_aoa_cbar = _drop_colorbar("texas_aoa_cbar")
    last_ra_meta = None
    ra_im = None
    ra_cbar = _drop_colorbar("ra_cbar")
    ra_peak_scatter = None
    last_vital_data = None          # NEW
    last_vital_rfft_cube = None
    last_vital_rf_meta = None
    last_vital_cube_ptr = None
    last_vital_clutter_cube = None
    last_vital_clutter_val = None
    last_ai_rfft_complex = None
    last_ai_rfft_meta = None
    if 'vital_bin_slider' in globals() and vital_bin_slider is not None and vital_bin_slider.winfo_exists():
        vital_bin_slider.set(0)
        vital_bin_slider.config(to=100, state="disabled")
        # the slider is back at bin 0 (0 m): a new recording starts with Auto Bin again
        if vital_auto_bin_var is not None:
            vital_auto_bin_var.set(True)
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
        if aoa_bin_label_var is not None:
            aoa_bin_label_var.set("bin -")
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
def _show_fps_fallback(msg: str) -> None:
    """Put a frame-rate fallback in the status bar; the only GUI part of derive_fps."""
    if progress_text_var is not None:
        progress_text_var.set(msg)


def derive_fps(params: dict, context: str = "") -> float:
    """Frame rate of the capture - see capture_params.derive_fps, which owns the rule."""
    return capture_params.derive_fps(params, context, on_fallback=_show_fps_fallback)

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
    global last_ai_rfft_complex, last_ai_rfft_meta
    last_ai_rfft_complex = None
    last_ai_rfft_meta = None
    bin_path = bin_var.get()
    if not os.path.isfile(bin_path):
        messagebox.showerror("Error", "Please select a valid BIN file.")
        return
    try:
        RECENTS.add(bin_path)
        populate_recent_list()
    except Exception:
        pass
    # an AWR2243 cascade take (cascade/) is described by <name>_cascade.json instead of
    # a mmWave Studio log
    from cascade.decode import cascade_params, sidecar_for
    log_path = sidecar_for(bin_path) or find_log_file(bin_path)
    if not log_path:
        messagebox.showerror("Error", "Corresponding LOG file not found.")
        return
    log_path = str(log_path)
    try:
        global last_params
        # Reset all data from previous file
        _reset_cached_data()
        # last_params is not part of _reset_cached_data(); clear it explicitly so a
        # failed parse can never leave the previous file's config paired with the
        # newly selected BIN file.
        last_params = None
        with progress(f"Parsing log {os.path.basename(log_path)} ..."):
            if log_path.endswith("_cascade.json"):
                params, summary_lines = run_in_background(cascade_params, bin_path)
                print("\n".join(summary_lines))
            else:
                params, summary_lines = run_in_background(
                    parse_logfile_full, log_path, print_summary=True)
        if not params:
            messagebox.showerror("Error", "ProfileConfig not found in the log file.")
            return
        last_params = params
        _apply_range_bias()          # an active calibration's range bias (roadmap 9.5)
        try:                         # a segment belongs to one capture (roadmap 9.6)
            segment_from_var.set("")
            segment_to_var.set("")
            _update_segment_info()
        except Exception:
            pass
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
        try:
            _refresh_range_fft_choices()
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

def range_fft_size() -> int:
    """
    Range FFT length for the loaded capture (roadmap 9.2): the Analysis bar's choice,
    "auto" being the next power of two of the samples per chirp but at least 256 - what
    every view used before, so existing captures look the same. Never shorter than a
    chirp: radar_processing.range_fft_size refuses to drop samples.
    """
    from radar_processing import range_fft_size as _size
    n = int((last_params or {}).get('numAdcSamples') or 256)
    choice = ""
    try:
        choice = str(range_fft_var.get()) if range_fft_var is not None else ""
    except Exception:
        pass
    if choice.isdigit():
        return _size(n, int(choice))
    return max(256, _size(n))


def _refresh_range_fft_choices():
    """Offer only FFT lengths that hold a whole chirp of the loaded capture."""
    if range_fft_combo is None:
        return
    from radar_processing import range_fft_size as _size
    n = int((last_params or {}).get('numAdcSamples') or 256)
    base = _size(n)
    sizes = [s for s in (256, 512, 1024, 2048, 4096) if s >= base]
    range_fft_var.set("auto")
    range_fft_combo.config(values=["auto"] + [str(s) for s in sizes])


def on_range_fft_change(_evt=None):
    _invalidate_cube_derived_data(f"Range FFT set to {range_fft_size()} points", keep_cube=True)


# ---------------- Display: dynamic range and colour map (roadmap 9.4) ----------------
# Every view used to fix its own: Range Profile -120..0 dBFS, the AoA spectrum -40..0 dB,
# Azimuth-Elevation -20..0 dB, and 'jet' in six maps - a colour map whose bright bands
# draw edges into smooth data. Each view still computes its own limits ("auto" keeps
# them); a chosen range shows the top N dB of it instead, and the colour map is shared.
# Views register what they draw, so a change re-colours them without recomputing.
DISPLAY_CMAPS = ["auto", "viridis", "magma", "inferno", "plasma", "cividis", "turbo",
                 "gray", "jet"]
DISPLAY_RANGES = ["auto", "20", "30", "40", "50", "60", "80", "100"]
display_cmap_var = None
display_range_var = None
_display_items = {}


def display_cmap(default: str = "viridis") -> str:
    """The chosen colour map, or the view's own default for "auto"."""
    try:
        c = display_cmap_var.get() if display_cmap_var is not None else "auto"
    except Exception:
        c = "auto"
    return default if c in ("", "auto") else c


def display_range_db():
    """The chosen dynamic range in dB, or None for "auto"."""
    try:
        v = str(display_range_var.get()) if display_range_var is not None else "auto"
        return float(v) if v not in ("", "auto") and float(v) > 0 else None
    except Exception:
        return None


def display_limits(vmin, vmax, data_max=None):
    """A view's (vmin, vmax), or the top N dB below its peak (data_max, else vmax)."""
    n = display_range_db()
    top = data_max if data_max is not None else vmax
    if n is None or top is None:
        return vmin, vmax
    return top - n, top


def show_image(name, im, vmin, vmax, canvas=None, default_cmap="viridis", data_max=None):
    """Apply the display settings to an image and remember its own limits."""
    _display_items[name] = dict(kind="image", artist=im, limits=(vmin, vmax), canvas=canvas,
                                cmap=default_cmap, data_max=data_max)
    lo, hi = display_limits(vmin, vmax, data_max)
    try:
        im.set_cmap(display_cmap(default_cmap))
        if lo is not None and hi is not None:
            im.set_clim(lo, hi)
    except Exception:
        pass


def show_axis(name, ax, vmin, vmax, canvas=None, data_max=None):
    """The same for the y range of a 1D plot."""
    _display_items[name] = dict(kind="axis", artist=ax, limits=(vmin, vmax), canvas=canvas,
                                data_max=data_max)
    lo, hi = display_limits(vmin, vmax, data_max)
    if lo is not None and hi is not None:
        ax.set_ylim(lo, hi)


def on_display_change(_evt=None):
    if display_range_var is not None and display_range_db() is None:
        display_range_var.set("auto")             # anything unreadable means auto
    for name, it in list(_display_items.items()):
        try:
            if it["kind"] == "image":
                show_image(name, it["artist"], *it["limits"], canvas=it["canvas"],
                           default_cmap=it["cmap"], data_max=it["data_max"])
            else:
                show_axis(name, it["artist"], *it["limits"], canvas=it["canvas"],
                          data_max=it["data_max"])
            if it["canvas"] is not None:
                it["canvas"].draw_idle()
        except Exception:
            _display_items.pop(name, None)


def clutter_settings() -> dict:
    """
    The clutter method chosen on the Analysis bar, as the keys clutter.from_params reads.

    The EMA weight comes from a time constant in seconds and the capture's frame period,
    so the same setting means the same thing at 10 fps and at 100 fps.
    """
    method = "mean"
    tau_s, rank = 5.0, clutter_filters.DEFAULT_RANK
    try:
        method = (clutter_method_var.get() or "mean").strip().lower()
        tau_s = float(clutter_tau_var.get())
        rank = max(0, int(float(clutter_rank_var.get())))
    except Exception:
        pass
    period_s = (_derive_period_ms(last_params) or 0) / 1000.0 if last_params else 0.0
    alpha = clutter_filters.alpha_from_time_constant(tau_s, period_s) if period_s > 0         else clutter_filters.DEFAULT_ALPHA
    return {"clutter_method": method, "clutter_alpha": alpha, "clutter_rank": rank}


def ensure_cube_loaded():
    global last_cube, last_full_cube, last_frame_damaged_full
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

            with progress(f"Loading {os.path.basename(bin_var.get())} (LVDS decode) ..."):
                cube = run_in_background(
                    load_raw_cube, bin_var.get(), last_params, dims,
                    chirp_selection=chirp_sel, lane_order=lane_order)
            _report_dropped_frames(cube)
            _report_capture_health(cube)       # on the whole capture, as recorded
            last_full_cube = cube
            last_frame_damaged_full = last_frame_damaged
            _select_segment()                  # last_cube = the chosen time segment
        except Exception as e:
            report_error("Error", f"Data loading error: {e}", e)
            return None
    return last_cube


def segment_t0_s() -> float:
    """Capture time of the first frame of the current segment, in seconds."""
    if not segment_frame0 or not last_params:
        return 0.0
    fps = derive_fps(last_params, "Segment")
    return segment_frame0 / fps if fps else 0.0


def _segment_request():
    """(t_from, t_to) typed in the segment fields; None for an empty field."""
    def _val(var):
        s = str(var.get()).strip() if var is not None else ""
        if not s:
            return None
        try:
            return float(s.replace(",", "."))
        except ValueError:
            raise ValueError(f"segment bound {s!r} is not a number of seconds")
    return _val(segment_from_var), _val(segment_to_var)


def _update_segment_info(f0=None, f1=None):
    """Say which part of the capture the frame numbers now count from."""
    if segment_info_var is None:
        return
    try:
        fps = derive_fps(last_params, "Segment") if last_params else 0.0
        n = last_full_cube.shape[0] if last_full_cube is not None else \
            int((last_params or {}).get('frameConfig', {}).get('numFrames') or 0)
        total = n / fps if fps else 0.0
        if f0 is None or (f0, f1) == (0, n):
            segment_info_var.set(f"whole capture, {total:.1f} s" if total else "")
        else:
            segment_info_var.set(f"{f0 / fps:.1f}-{f1 / fps:.1f} s of {total:.1f} s "
                                 f"(frame 0 = {f0 / fps:.1f} s)")
    except Exception:
        pass


def _select_segment():
    """
    last_cube = the chosen time segment of the loaded capture (roadmap 9.6), with the
    lost-packet mask cut the same way and the IQ correction applied to it.
    """
    global last_cube, last_frame_damaged, segment_frame0
    full = last_full_cube
    if full is None:
        return None
    fps = derive_fps(last_params, "Segment")
    f0, f1 = capture_params.segment_frames(full.shape[0], fps, *_segment_request())
    segment_frame0 = f0
    last_frame_damaged = None if last_frame_damaged_full is None else last_frame_damaged_full[f0:f1]
    cube = _maybe_correct_iq(full[f0:f1])
    last_cube = cube
    _refresh_rx_choices(cube.shape[2])
    _update_segment_info(f0, f1)
    if (f0, f1) != (0, full.shape[0]):
        print(f"[Segment] frames {f0}-{f1} of {full.shape[0]} "
              f"({f0 / fps:.2f}-{f1 / fps:.2f} s of {full.shape[0] / fps:.2f} s)")
    return cube


def on_segment_apply(_evt=None):
    """Cut the loaded capture anew - no new decode - and drop what the old cut gave."""
    global global_frames_total
    if last_full_cube is None:
        try:
            _segment_request()
        except ValueError as e:
            messagebox.showerror("Segment", str(e))
        return                                  # applies when the capture is loaded
    try:
        capture_params.segment_frames(last_full_cube.shape[0],
                                      derive_fps(last_params, "Segment"), *_segment_request())
    except ValueError as e:
        messagebox.showerror("Segment", str(e))
        return
    # the Fastcheck always covers the whole recording, so a segment leaves it valid
    _invalidate_cube_derived_data("Time segment changed", keep_cube=True,
                                  outdates_fastcheck=False)
    global_frames_total = 0                     # the frame slider starts over
    frame_cache.forget()
    try:
        global_frame_slider.set(0)
        global_frame_slider.config(to=0)
    except Exception:
        pass
    _select_segment()


def _maybe_correct_iq(cube):
    """
    IQ imbalance correction (iq_balance.py), when switched on. Each RX is corrected only
    if the image rejection of frames not used for the estimate does not get worse.
    """
    try:
        if iq_correct_var is None or not iq_correct_var.get() or not np.iscomplexobj(cube):
            return cube
    except Exception:
        return cube
    out, imb, before, after = iq_balance.correct(cube, "image")
    print("[IQ] image-pair estimate, judged on held-out frames:\n" + imb.summary())
    for r in range(before.size):
        print(f"[IQ]   RX{r}: image rejection {before[r]:.1f} -> {after[r]:.1f} dB "
              f"({'corrected' if imb.use[r] else 'left as recorded'})")
    return out


def _report_capture_health(cube):
    """
    Print the capture health report after a load; put an error in the status bar.

    Warnings stay in the console so they do not overwrite the dropped-frames message,
    which _report_dropped_frames has just put there. Nothing here may stop a load.
    """
    try:
        import capture_health
        bin_path = bin_var.get()
        report = capture_health.assess(
            os.path.basename(bin_path), last_params or {},
            bin_paths=get_all_bin_parts(bin_path), cube=cube,
            check_zero_fill=False)          # _report_dropped_frames has just done it
        print("[health] " + report.format().replace("\n", "\n[health] "))
        if report.worst == "error" and progress_text_var is not None:
            progress_text_var.set(report.summary())
    except Exception as e:
        print(f"[health] report could not run: {type(e).__name__}: {e}")


last_frame_damaged = None   # per-frame "recorder filled this with zeros" mask


def _report_dropped_frames(cube):
    """
    Say how much of the capture the DCA1000 filled with zeros.

    When packets are lost the recorder substitutes zeros, and nothing downstream can
    tell those apart from a real reading. It matters most for vital signs, where the
    whole measurement is sub-millimetre phase: a zero block is a phase discontinuity
    in the middle of the breathing waveform. The capture's own _Raw_LogFile.csv
    reports the count ("Number of zero filled packets"); this is the same figure
    measured on the data that was actually decoded.
    """
    global last_frame_damaged
    last_frame_damaged = None
    try:
        damaged = vitalsigns.detect_damaged_frames(cube)
        last_frame_damaged = damaged
        n_bad = int(damaged.sum())
        if n_bad == 0:
            return
        pct = 100.0 * n_bad / len(damaged)
        # Longest uninterrupted stretch, which is what limits a vital-signs analysis.
        run = best = 0
        for d in damaged:
            run = 0 if d else run + 1
            best = max(best, run)
        fps = derive_fps(last_params, "capture check")
        msg = (f"{n_bad} of {len(damaged)} frames ({pct:.1f} %) contain zero-filled "
               f"samples - packets were lost during recording. Longest clean stretch: "
               f"{best / fps:.1f} s.")
        print(f"[GUI] WARNING: {msg}")
        if progress_text_var is not None:
            progress_text_var.set(f"Loaded with packet loss: {pct:.1f} % of frames")
    except Exception as exc:
        last_frame_damaged = None
        print(f"[GUI] Could not check the capture for dropped frames: {exc}")

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
        show_axis("range_fft", range_fft_ax, vmin, vmax, range_fft_canvas,
                  data_max=float(np.max(mag_db)))

    range_fft_canvas.draw_idle()

def on_range_fft_frame_change(val):
    try:
        idx = int(float(val))
    except ValueError:
        idx = 0
    # NEW: route via global synchronizer
    apply_global_frame(idx)

def run_range_fft():
    clear_settings_changed("Range FFT")
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

        with progress("Computing Range FFT ..."):
            last_range_fft, last_range_fft_meta = run_in_background(
                compute_range_fft_cube,
                cube, last_params, window='hann', pad_pow2=False, remove_dc=True,
                n_fft_range=range_fft_size(), use_full_spectrum=use_full
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
        vmin, vmax = image_percentiles(mag_db_all, [5, 99])
        if np.isfinite(vmin) and np.isfinite(vmax):
            last_range_fft_meta['display_vmin'] = vmin
            last_range_fft_meta['display_vmax'] = vmax + 5
        # Motion-only percentiles
        try:
            moving = np.clip(last_range_fft - last_range_fft_bg, 1e-9, None)
            moving_db = 20 * np.log10(moving)
            mvmin, mvmax = image_percentiles(moving_db, [5, 99])
            if np.isfinite(mvmin) and np.isfinite(mvmax):
                last_range_fft_meta['display_vmin_moving'] = mvmin
                last_range_fft_meta['display_vmax_moving'] = mvmax + 5
        except Exception:
            pass

        update_range_fft_plot(0)
    except Exception as e:
        report_error("Error", f"Range FFT error: {e}", e)

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
    """
    The decoded parameters, scrolling both ways: three columns of name/value pairs are
    wider than a small window. The mouse wheel scrolls it only while the pointer is
    over it - it used to be bound to the whole application, so every turn of the wheel
    anywhere also scrolled this panel.
    """
    global params_canvas, params_scroll, params_inner
    panel = ScrollFrame(parent, xscroll=True)
    panel.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=4, pady=6)
    params_canvas, params_scroll, params_inner = panel.canvas, panel.vbar, panel.inner

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
    clear_settings_changed("RTI")
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
            with progress("Computing RTI ..."):
                last_rti, last_rti_meta = run_in_background(compute_rti, cube, last_params,
                                                            n_fft_range=range_fft_size())
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
                    n_fft_range=range_fft_size(),
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
                    last_rti, last_rti_meta = compute_rti(cube, last_params,
                                                          n_fft_range=range_fft_size())
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

        vmin, vmax = image_percentiles(img, [5, 99])

        # 180 000 chirps of a 3-min take drawn whole took 3.4 s at every redraw (resize,
        # frame line, zoom): shown averaged to the screen's width, finer when zoomed in
        rti_im = ColumnPooledImage(rti_ax, img, extent, aspect='auto', origin='lower',
                                   cmap=display_cmap(), vmin=vmin, vmax=vmax).im
        show_image("rti", rti_im, vmin, vmax, rti_canvas, data_max=float(np.max(img)))

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
        report_error("Error", f"RTI error: {e}", e)


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

def _current_frame_index() -> int:
    try:
        return int(global_frame_slider.get()) if global_frame_slider is not None else 0
    except Exception:
        return 0


def _current_rx_mode() -> str:
    try:
        return (global_rx_mode_var.get() or "MRC").strip().upper()
    except Exception:
        return "MRC"


def _refresh_rx_choices(n_rx: int):
    """The antenna selector lists the RX the capture has - it was fixed to RX0 and RX1."""
    try:
        values = ["MRC", "SUM"] + [f"RX{k}" for k in range(max(1, int(n_rx)))]
        global_rx_mode_combo.config(values=values)
        if global_rx_mode_var.get() not in values:
            global_rx_mode_var.set("MRC")
    except Exception:
        pass


def run_range_profile():
    cube = ensure_cube_loaded()
    if cube is None:
        return
    set_global_frames(cube.shape[0])       # so the frame slider can move through it
    update_range_profile_plot(_current_frame_index(), interactive=True)


def update_range_profile_plot(frame_idx: int, interactive: bool = False):
    """
    Range profile of one frame, combined over RX as the Analysis bar's antenna
    selector says (roadmap 9.3). It used to be frame 0 and RX0 whatever was selected.
    Called by Run and by the global frame slider once the profile has been computed.
    """
    global last_range_profile, last_range_profile_meta
    cube = last_cube
    if cube is None or not last_params:
        return
    try:
        fs = (last_params.get('digOutSampleRate_MHz') or 0) * 1e6
        slope = (last_params.get('freqSlopeMHz_us') or 0) * 1e12
        adc_bits = last_params.get('dataFmt_adcBits') or 16
        nfft = range_fft_size()

        if fs == 0 or slope == 0:
            if interactive:
                messagebox.showerror("Error", "Sample Rate or Slope is zero. Cannot compute profile.")
            return

        f = max(0, min(int(frame_idx), cube.shape[0] - 1))
        mode = _current_rx_mode()
        frame = cube[f]                                    # (chirps, rx, samples)
        r_axis, mag_dbfs = range_profile_rx(frame, fs, slope, nfft, adc_bits, mode=mode,
                                            range_bias_m=last_params.get('range_bias_m', 0.0))
        last_range_profile = mag_dbfs
        last_range_profile_meta = {'range_m': r_axis, 'frame': f, 'rx_mode': mode}

        # Zoom FFT (spectral.py): the same profile on a fine grid over a narrow band,
        # to place a target between the range bins. Placement, not resolution - two
        # targets closer than c/2B still merge.
        zoom = None
        if rp_zoom_var is not None and rp_zoom_var.get():
            r_lo, r_hi = float(rp_zoom_lo_var.get()), float(rp_zoom_hi_var.get())
            if not 0 <= r_lo < r_hi:
                raise ValueError("Zoom FFT needs 0 <= from < to")
            x = frame - frame.mean(axis=-1, keepdims=True)
            r_zoom, z = spectral.zoom_range_profile(x, last_params, r_lo, r_hi, n=2048)
            from scipy.signal import get_window
            gain = get_window("hann", x.shape[-1], fftbins=True).sum()
            zc = z.mean(axis=0) / gain * (1.0 if np.iscomplexobj(x) else 2.0)   # (rx, n)
            if mode.startswith("RX"):
                k = int(mode[2:]) if mode[2:].isdigit() else 0
                z_mag = np.abs(zc[max(0, min(zc.shape[0] - 1, k))])
            elif mode == "SUM":
                z_mag = np.abs(zc.mean(axis=0))
            else:
                z_mag = np.sqrt(np.mean(np.abs(zc) ** 2, axis=0))
            z_db = 20 * np.log10(z_mag + 1e-9)
            k = int(np.argmax(z_db))
            r_peak, _ = spectral.refine_peak(r_zoom, z_db, k, scale="db")
            zoom = (r_zoom, z_db, r_peak, float(z_db[k]))
            last_range_profile_meta.update(zoom_range_m=r_zoom, zoom_dbfs=z_db,
                                           zoom_peak_m=r_peak)

        # Plotting
        rp_ax.clear()
        if isinstance(r_axis, np.ndarray) and r_axis.shape[0] == mag_dbfs.shape[0]:
            rp_ax.plot(r_axis, mag_dbfs, label="FFT")
            rp_ax.set_xlabel("Range [m]")
        else:
            rp_ax.plot(mag_dbfs)
            rp_ax.set_xlabel("Range Bin")
        rp_ax.set_title(f"Range Profile - frame {f}, {mode}")
        if zoom is not None:
            r_zoom, z_db, r_peak, p_peak = zoom
            rp_ax.plot(r_zoom, z_db, label="Zoom FFT")
            rp_ax.axvline(r_peak, color="red", lw=0.8, ls="--")
            rp_ax.set_xlim(r_zoom[0], r_zoom[-1])
            rp_ax.set_title(f"Range Profile - frame {f}, {mode} - zoom peak {r_peak:.4f} m "
                            f"({p_peak:.1f} dBFS)")
            rp_ax.legend(loc="upper right", fontsize=8)

        rp_ax.set_ylabel("Magnitude [dBFS]")
        rp_ax.grid(True, alpha=0.3)
        # -120..0 dBFS unless a dynamic range is chosen: then that many dB under the peak
        top = float(np.ceil(np.max(mag_dbfs) / 5.0) * 5.0 + 5.0)
        show_axis("range_profile", rp_ax, -120, 0, rp_canvas, data_max=top)
        rp_canvas.draw_idle()

    except Exception as e:
        if interactive:
            report_error("Error", f"Range Profile error: {e}", e)
        else:
            print(f"[Range Profile] frame {frame_idx}: {e}")


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

RD_MIN_SPAN_DB = 20.0       # the narrowest colour range the Range-Doppler map auto-scales to


def run_range_doppler():
    clear_settings_changed("Range-Doppler")
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
        detection_cbar = _drop_colorbar("detection_cbar")
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

        with progress("Computing Range-Doppler map ..."):
            rd_map, meta = run_in_background(
                compute_range_doppler_map,
                cube, {**last_params, **clutter_settings()},
                n_fft_range=range_fft_size(),
                rx_mode=rx_mode,
                doppler_window=dop_win,
            )
        last_rd_map = rd_map
        last_rd_meta = meta
        if meta.get('few_loops_note'):
            print(f"[Range-Doppler] {meta['few_loops_note']}")

        # NEW: Precompute dB buffer for real-time playback
        global last_rd_buffer
        last_rd_buffer = 20 * np.log10(rd_map + 1e-9)

        # Set playback speed from frame periodicity
        period_ms = last_params.get('framePeriodicity_ms_selected')
        if period_ms and period_ms > 0:
            rd_play_interval_ms = int(period_ms)
            set_global_interval_ms(rd_play_interval_ms)

        # Global color scale (reuse the dB buffer computed above instead of
        # recomputing log10 over the whole cube a second time), from just below the noise
        # floor (the median: most cells hold no target) over at least RD_MIN_SPAN_DB. Once
        # the clutter filter has taken the still scene away, a map without motion is
        # noise; the 10th-99.5th percentiles spread it over a few dB and painted it as
        # structure (an AWR2243 cascade take of a seated person: 9 dB, with streaks that
        # looked like targets moving at 15 m/s).
        floor, top = image_percentiles(last_rd_buffer, [50, 99.5])
        if np.isfinite(floor) and np.isfinite(top):
            vmin = floor - 3.0
            meta['display_vmin'] = vmin
            meta['display_vmax'] = max(top, vmin + RD_MIN_SPAN_DB)

        frames = rd_map.shape[0]
        # RD lokální slider odstraněn; publikujeme pouze do globálního
        set_global_frames(frames)


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
        report_error("Error", f"Range-Doppler error: {e}", e)

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

    # Use precomputed buffer if available
    if last_rd_buffer is not None:
        img_db = last_rd_buffer[f]
    else:
        img_db = 20 * np.log10(last_rd_map[f] + 1e-9)

    range_axis = last_rd_meta.get('range_axis')
    velocity_axis = last_rd_meta.get('velocity_axis')
    extent = bin_extent(velocity_axis, range_axis)
    vmin = last_rd_meta.get('display_vmin')
    vmax = last_rd_meta.get('display_vmax')

    if rd_im is None:  # First time plotting or after re-running
        rd_ax.clear()
        rd_im = rd_ax.imshow(img_db.T, aspect='auto', origin='lower', cmap=display_cmap(),
                             extent=extent, vmin=vmin, vmax=vmax,
                             interpolation=few_bins_interpolation(len(velocity_axis)))
        show_image("rd", rd_im, vmin, vmax, rd_canvas)
        # Create or update colorbar
        if rd_cbar is None:
            rd_cbar = rd_fig.colorbar(rd_im, ax=rd_ax)
        else:
            rd_cbar.update_normal(rd_im)
        rd_cbar.set_label("Magnitude [dB]")

        rd_ax.set_xlabel("Velocity [m/s]")
        rd_ax.set_ylabel("Range [m]")
    else:
        rd_im.set_data(img_db.T)
        rd_im.set_extent(extent)
        if vmin is not None and vmax is not None:
            show_image("rd", rd_im, vmin, vmax, rd_canvas)
    
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

    title = f"Range-Doppler Map (Frame {f})" + (f" — {last_rd_meta.get('rx_mode')}" if isinstance(last_rd_meta, dict) and last_rd_meta.get('rx_mode') else "")
    # what the map can resolve: a handful of Doppler bins puts slow motion into 0 m/s,
    # where the clutter filter removes it
    try:
        if len(velocity_axis) > 1:
            dv = abs(float(velocity_axis[1]) - float(velocity_axis[0]))
            title += f"\n{len(velocity_axis)} Doppler bins, Δv {dv:.2f} m/s"
    except Exception:
        pass
    rd_ax.set_title(title)
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
    global _rd_play_after_id
    _rd_play_after_id = None
    if not rd_playing:
        return
    
    start_time = time.time()
    
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
    
    # Timing correction for real-time playback
    elapsed_ms = (time.time() - start_time) * 1000
    delay = max(1, int(rd_play_interval_ms - elapsed_ms))
    _rd_play_after_id = root.after(delay, _update_rd_frame)

def play_rd():
    global rd_playing
    if last_rd_map is None:
        return
    if rd_playing:
        return
    # The RD tab and the global bar drive the same frame slider; running both at
    # once advanced the frame twice per tick, so starting one stops the other.
    if global_playing:
        stop_global()
    rd_playing = True
    rd_play_btn.config(state="disabled")
    rd_stop_btn.config(state="normal")
    _update_rd_frame()

def stop_rd():
    global rd_playing, _rd_play_after_id
    rd_playing = False
    if _rd_play_after_id is not None:
        try:
            root.after_cancel(_rd_play_after_id)
        except Exception:
            pass
        _rd_play_after_id = None
    rd_play_btn.config(state="normal")
    rd_stop_btn.config(state="disabled")

# --- NEW: AoA Functions ---


# --- Angle of Arrival ---

def init_texas_aoa_plot(parent):
    global texas_aoa_fig, texas_aoa_ax, texas_aoa_ax2, texas_aoa_canvas
    texas_aoa_fig = Figure(figsize=(9, 4), dpi=100, constrained_layout=True)
    texas_aoa_ax = texas_aoa_fig.add_subplot(121)
    texas_aoa_ax2 = texas_aoa_fig.add_subplot(122)
    texas_aoa_ax.set_title("Angle of Arrival")
    texas_aoa_ax2.set_title("Target tracking over time")
    texas_aoa_canvas = FigureCanvasTkAgg(texas_aoa_fig, master=parent)
    texas_aoa_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
    parent.rowconfigure(1, weight=1)
    parent.columnconfigure(0, weight=1)

def record_clutter():
    global empty_scene_cube
    cube = ensure_cube_loaded()
    if cube is None:
        return
    try:
        range_cube, _ = compute_range_fft_cube(
            cube, last_params, window='hann', pad_pow2=False, remove_dc=True,
            n_fft_range=range_fft_size(), return_complex=True
        )
        empty_scene_cube = range_cube
        messagebox.showinfo("Clutter", "Empty scene (clutter) recorded successfully.")
    except Exception as e:
        report_error("Error", f"Failed to record the empty scene: {e}", e)

def run_texas_aoa():
    clear_settings_changed("Angle of Arrival")
    global last_texas_aoa_spec, last_texas_aoa_meta, texas_aoa_mimo_mode_var, last_aoa_range_m
    cube = ensure_cube_loaded()
    if cube is None:
        return
    try:
        global last_texas_aoa_spec, last_texas_aoa_meta, texas_aoa_im, texas_aoa_cbar, texas_aoa_mimo_mode_var
        
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
        # "1.5" selects one range; "0.3-2" limits the analysis to a window. The window
        # form went straight into float() here and raised, so the whole run ended in an
        # error dialog - the ROI parser further down never got to see it.
        r_m = None
        if r_txt and '-' not in r_txt:
            try:
                r_m = float(r_txt)
            except ValueError:
                report_error("Angle of Arrival",
                             f"Range '{r_txt}' is not a number or a range like 0.3-2.")
                return
        sel_bin = None
        try:
            if (not r_txt) and (aoa_bin_slider is not None) and (aoa_bin_slider.cget("state") != "disabled"):
                sel_bin = int(aoa_bin_slider.get())
        except Exception:
            sel_bin = None

        range_cube, rf_meta = compute_range_fft_cube(
            cube, last_params, window='hann', pad_pow2=False, remove_dc=True, n_fft_range=range_fft_size(), return_complex=True
        )

        # Coherent clutter subtraction if enabled
        sub_clutter = False
        try:
            if texas_aoa_sub_clutter_var is not None:
                sub_clutter = bool(texas_aoa_sub_clutter_var.get())
        except Exception: pass
        if sub_clutter and empty_scene_cube is not None:
            if range_cube.shape[1:] == empty_scene_cube.shape[1:]:
                clutter_mean = empty_scene_cube.mean(axis=0, keepdims=True)
                range_cube = range_cube - clutter_mean
                print("[GUI] Coherent empty scene subtraction applied.")

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
        params_aoa['antenna_config'] = analysis_antenna_config()
        params_aoa.update(clutter_settings())
        params_aoa.update(calibration_params())
        # NOTE: platform/chip_hint are passed through as-is from the parsed log
        # (last_params). Do NOT force them to IWR6843AOP here - a non-AOP board (e.g.
        # ISK/ODS eval kit, AWR1843) would then silently get the wrong 2D patch-antenna
        # geometry and produce wrong angle estimates. AOA._get_antenna_geometry()
        # already has its own (logged) fallback for when platform/chip_hint are unknown.

        invert_phase = False
        try:
            if aoa_invert_phase_var is not None:
                invert_phase = bool(aoa_invert_phase_var.get())
        except Exception: pass

        # Pass Shift Spectrum setting to AOA module
        try:
            if texas_aoa_shift_var is not None:
                params_aoa['aoa_shift_spectrum'] = bool(texas_aoa_shift_var.get())
            else:
                params_aoa['aoa_shift_spectrum'] = False # Force default
        except: pass

        # Pass TDM Doppler compensation variable
        doppler_comp = False
        try:
            if texas_aoa_doppler_comp_var is not None:
                doppler_comp = bool(texas_aoa_doppler_comp_var.get())
        except Exception: pass
        params_aoa['tdm_doppler_compensation'] = doppler_comp

        m_mode = str(texas_aoa_mimo_mode_var.get()).strip() if (texas_aoa_mimo_mode_var is not None) else "MIMO"
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

        # Support range ROI input (e.g. "0.5-3.0")
        r_start_bin = 0
        r_end_bin = None
        r_start_m = None
        r_end_m = None
        if r_txt:
            if '-' in r_txt:
                try:
                    parts = r_txt.split('-')
                    r_start_m = float(parts[0])
                    r_end_m = float(parts[1])
                except Exception: pass
            else:
                try:
                    r_start_m = float(r_txt)
                except Exception: pass
        
        r_axis = rf_meta.get('range_m')
        if r_axis is not None:
            if r_start_m is not None:
                r_start_bin = int(np.searchsorted(r_axis, r_start_m))
            if r_end_m is not None:
                r_end_bin = int(np.searchsorted(r_axis, r_end_m)) + 1

        # One range, none given: the range that moves most. Without it the 1D spectrum took
        # the middle bin (2 m of a 4 m range, whatever was there) and the slider bin 0 the
        # antenna coupling at 0 m.
        one_range = view_type in ("1D Spectrum", "Azimuth-Elevation")
        if one_range and r_m is None and r_axis is not None and len(r_axis) == range_cube.shape[-1] \
                and (sel_bin is None or r_axis[min(sel_bin, len(r_axis) - 1)] < MIN_TARGET_RANGE_M):
            sel_bin = strongest_range_bin(motion_energy(range_cube), r_axis)
            print(f"[AoA] no range given: the range that moves most, {r_axis[sel_bin]:.2f} m "
                  f"(bin {sel_bin}) - type a range or move the Range Bin slider for another")

        if view_type == "1D Spectrum":
            spec, meta = compute_aoa_spectrum(
                range_cube, params_aoa, range_m=r_m, range_bin=sel_bin, remove_static=rm_static, mimo=use_mimo,
                invert_alternate_tx_phase=invert_phase, mimo_mode=m_mode, rx_order_override=rx_override, method=method,
                angles_deg=angles
            )
            meta['view_type'] = "1D Spectrum"
        elif view_type == "Range-Azimuth":
            with progress("Computing Range-Azimuth map ...", total=range_cube.shape[0]) as p:
                spec, meta = run_in_background(
                    compute_range_azimuth_map,
                    range_cube, params_aoa, remove_static=rm_static, mimo=use_mimo,
                    invert_alternate_tx_phase=invert_phase, mimo_mode=m_mode,
                    rx_order_override=rx_override, method=method,
                    angles_deg=angles, progress_cb=p.callback(), handle=p
                )
            meta['view_type'] = "Range-Azimuth"
            # Crop to ROI if set
            if r_end_bin is None:
                r_end_bin = spec.shape[1]
            r_start_bin = max(0, min(spec.shape[1] - 1, r_start_bin))
            r_end_bin = max(r_start_bin + 1, min(spec.shape[1], r_end_bin))
            spec = spec[:, r_start_bin:r_end_bin, :]
            if 'range_axis' in meta and meta['range_axis'] is not None:
                meta['range_axis'] = meta['range_axis'][r_start_bin:r_end_bin]
        elif view_type == "Azimuth-Elevation":
            # For Az-El we use a slightly coarser grid if it's large, but let's respect limits
            angles_az = np.arange(-ang_limit, ang_limit + 2.0, 2.0)
            spec, meta = compute_azimuth_elevation_map(
                range_cube, params_aoa, range_bin=sel_bin, remove_static=rm_static, mimo=use_mimo,
                invert_alternate_tx_phase=invert_phase, rx_order_override=rx_override, method=method,
                angles_az_deg=angles_az
            )
            meta['view_type'] = "Azimuth-Elevation"
        elif view_type == "Range-Az-El":
            # 3D Range-Azimuth-Elevation map (Capon/Bartlett), 2 deg angular step
            angles_az = np.arange(-ang_limit, ang_limit + 2.0, 2.0)
            with progress("Computing Range-Azimuth-Elevation map ...", total=range_cube.shape[0]) as p:
                spec, meta = run_in_background(
                    compute_range_azimuth_elevation_map,
                    range_cube, params_aoa, angles_az_deg=angles_az,
                    remove_static=rm_static, mimo=use_mimo,
                    invert_alternate_tx_phase=invert_phase, mimo_mode=m_mode,
                    rx_order_override=rx_override, method=method,
                    r_start=r_start_bin, r_end=r_end_bin, progress_cb=p.callback(), handle=p
                )
            meta['view_type'] = "Range-Az-El"

            # Fit the 3D axes to where the returns are, once, so playback shows movement
            # instead of a blob parked in the middle of an empty 9 m box.
            try:
                meta['view_extent'] = rae_view_extent(
                    spec, meta['range_axis'], meta['angles_az_deg'], meta['angles_el_deg'],
                    min_r=float(aoa_min_range_var.get() or 0.0) if aoa_min_range_var is not None else 0.0)
            except Exception as exc:
                print(f"[AoA] Could not derive the 3D view extent: {exc}")

            # Optional CFAR detection per frame
            do_detect = False
            try:
                if texas_aoa_detect_var is not None:
                    do_detect = bool(texas_aoa_detect_var.get())
            except Exception: pass
            if do_detect:
                thr = 6.0
                try:
                    if texas_aoa_cfar_factor_var is not None and texas_aoa_cfar_factor_var.get().strip():
                        thr = float(texas_aoa_cfar_factor_var.get())
                except Exception: pass
                def _detect_all(handle):
                    out = []
                    for fi in range(spec.shape[0]):
                        out.append(detect_targets_rae(
                            spec[fi], meta['range_axis'],
                            meta['angles_az_deg'], meta['angles_el_deg'],
                            threshold_factor=thr
                        ))
                        handle.advance()
                        if _cancel_requested:
                            raise ComputationCancelled()
                    return out

                with progress("CFAR detection ...", total=spec.shape[0]) as p:
                    detections = run_in_background(_detect_all, p, handle=p)
                meta['detections'] = detections
                print(f"[RAE] CFAR detection done (factor={thr}): "
                      f"{[len(d) for d in detections]} targets per frame")
        else:
            raise ValueError(f"Unknown view type: {view_type}")

        # --- TARGET TRACKING PIPELINE ---
        track_moving = False
        try:
            if texas_aoa_track_moving_var is not None:
                track_moving = bool(texas_aoa_track_moving_var.get())
        except Exception: pass

        F_count, C_total, RX, R_total = range_cube.shape
        eff_unique = int(params_aoa.get('uniqueChirps_effective') or 1)
        unique_stride = params_aoa.get('unique_chirps_frame') or eff_unique
        unique_stride = max(1, int(unique_stride))

        tracked_targets = []
        # Tracking used a plain argmax over the whole map, so the TX leakage at ~0 m won
        # every frame (the readout was stuck at r=0.1 m) and the reported angle was noise.
        # Honour the same Range Cut the display uses, and prefer a peak near the previous
        # position so the track does not jump between grid edges.
        track_min_r = 0.0
        try:
            if aoa_min_range_var is not None:
                track_min_r = float(aoa_min_range_var.get() or 0.0)
        except Exception:
            pass
        prev_r_idx = None
        GATE_BINS = 12          # how far the target may move between frames
        if view_type in ("Range-Azimuth", "Range-Az-El") and spec.shape[0] > 0:
            for fi in range(spec.shape[0]):
                rng = meta.get('range_axis')
                az = meta.get('angles_deg') if 'angles_deg' in meta else meta.get('angles_az_deg')
                el = meta.get('angles_el_deg') # None for Range-Azimuth
                
                # 1. Determine best range bin index
                if track_moving:
                    f_cube = range_cube[fi]
                    if unique_stride > 1 and C_total >= unique_stride:
                        loops = C_total // unique_stride
                        try:
                            resh = f_cube[:loops*unique_stride].reshape(loops, unique_stride, RX, R_total)
                            d_fft = np.fft.fft(resh[:, 0, 0, :], axis=0)
                            power_d = np.abs(d_fft)**2
                            power_d[0, :] = 0
                            r_moving_energy = np.sum(power_d, axis=0)
                        except Exception:
                            r_moving_energy = np.zeros(R_total)
                    else:
                        d_fft = np.fft.fft(f_cube[:, 0, :], axis=0)
                        power_d = np.abs(d_fft)**2
                        power_d[0, :] = 0
                        r_moving_energy = np.sum(power_d, axis=0)
                    
                    # Constrain to full range ROI
                    r_moving_energy[:r_start_bin] = -1.0
                    if r_end_bin is not None:
                        r_moving_energy[r_end_bin:] = -1.0
                    best_r_orig = np.argmax(r_moving_energy)
                    
                    r_idx_spec = best_r_orig - r_start_bin
                    r_idx_spec = max(0, min(spec.shape[1] - 1, r_idx_spec))
                else:
                    if view_type in ("Range-Azimuth", "Range-Az-El"):
                        # Strength per range bin, with the near-range cut applied.
                        axes = tuple(range(1, spec[fi].ndim))
                        per_range = spec[fi].max(axis=axes) if axes else spec[fi]
                        per_range = np.array(per_range, dtype=float, copy=True)
                        if rng is not None and track_min_r > 0:
                            per_range[np.asarray(rng) < track_min_r] = -np.inf
                        if prev_r_idx is not None:
                            gated = np.full_like(per_range, -np.inf)
                            lo = max(0, prev_r_idx - GATE_BINS)
                            hi = min(per_range.size, prev_r_idx + GATE_BINS + 1)
                            gated[lo:hi] = per_range[lo:hi]
                            # Stay with the gated peak while it is still a real return,
                            # otherwise re-acquire from the whole (cut) profile.
                            if np.isfinite(gated.max()) and gated.max() > per_range.max() - 6.0:
                                per_range = gated
                        r_idx_spec = int(np.argmax(per_range))
                        if not np.isfinite(per_range[r_idx_spec]):
                            r_idx_spec = 0
                    else:
                        r_idx_spec = 0
                prev_r_idx = r_idx_spec
                
                # 2. Determine best angles at that range bin
                if view_type == "Range-Azimuth":
                    az_idx = np.argmax(spec[fi, r_idx_spec, :])
                    el_idx = 0
                    el_val = 0.0
                elif view_type == "Range-Az-El":
                    el_idx, az_idx = np.unravel_index(np.argmax(spec[fi, r_idx_spec, :, :]), spec[fi, r_idx_spec, :, :].shape)
                    el_val = float(el[el_idx]) if el is not None else 0.0
                else:
                    az_idx = 0
                    el_idx = 0
                    el_val = 0.0
                
                r_val = float(rng[r_idx_spec]) if rng is not None else 0.0
                az_val = float(az[az_idx]) if az is not None else 0.0
                
                # 3. Cartesian positions
                az_rad = np.deg2rad(az_val)
                el_rad = np.deg2rad(el_val)
                x_m = r_val * np.cos(el_rad) * np.sin(az_rad)
                y_m = r_val * np.cos(el_rad) * np.cos(az_rad)
                z_m = r_val * np.sin(el_rad)
                
                pwr_db = float(spec[fi, r_idx_spec, el_idx, az_idx]) if view_type == "Range-Az-El" else float(spec[fi, r_idx_spec, az_idx])
                
                tracked_targets.append({
                    'range': r_val,
                    'azimuth': az_val,
                    'elevation': el_val,
                    'x': x_m,
                    'y': y_m,
                    'z': z_m,
                    'power_db': pwr_db,
                    'r_idx': r_idx_spec,
                    'az_idx': az_idx,
                    'el_idx': el_idx
                })
            meta['tracked_targets'] = tracked_targets

        last_texas_aoa_spec, last_texas_aoa_meta = spec, meta
        # The Range Bin slider was never switched on, so it could not choose anything:
        # now it shows the range analysed and moves over the whole range axis.
        last_aoa_range_m = r_axis if r_axis is not None and len(r_axis) == range_cube.shape[-1] else None
        if one_range and aoa_bin_slider is not None and meta.get('range_bin') is not None:
            aoa_bin_slider.config(to=range_cube.shape[-1] - 1, state="normal")
            aoa_bin_var.set(int(meta['range_bin']))
            _aoa_bin_label(int(meta['range_bin']))
        set_global_frames(spec.shape[0])
        update_texas_aoa_plot(0)
    except Exception as e:
        report_error("Error", f"Angle of Arrival error: {e}", e)


last_aoa_range_m = None    # range axis of the last AoA run, for the Range Bin label


def _aoa_bin_label(k: int):
    if aoa_bin_label_var is None:
        return
    r = last_aoa_range_m
    aoa_bin_label_var.set(f"bin {k} / {r[k]:.2f} m" if r is not None and 0 <= k < len(r) else f"bin {k}")

_vmax_cache = {}   # (id(array), crop signature) -> (array reference, vmax)

def _global_vmax(arr, crop_mask=None):
    """
    np.max() over a whole cached result cube, memoized.

    The plot update functions run on every frame-slider tick, but this value
    depends only on the cached array and the crop mask - recomputing it per frame
    rescanned the entire (potentially tens of MB) cube on every redraw.
    """
    if arr is None:
        return 0.0
    key = (id(arr), None if crop_mask is None else crop_mask.tobytes())
    hit = _vmax_cache.get(key)
    if hit is not None and hit[0] is arr:
        return hit[1]
    value = float(np.max(arr if crop_mask is None else arr[:, crop_mask]))
    _vmax_cache.clear()   # only the currently displayed view matters
    _vmax_cache[key] = (arr, value)   # keep a reference so id() stays valid
    return value

TRACK_TRAIL_FRAMES = 40   # how much of the tracked path the 3D view keeps visible
MIN_EXTENT_POINTS = 20    # never fit the 3D box to fewer returns than this


def rae_strong_voxels(data, vmax_g, floor_db=15.0):
    """
    Voxels of one Range-Az-El frame worth drawing as a point cloud, as a boolean mask.

    Thresholding on power alone does not give a point cloud, it gives a shell: a strong
    reflector spreads over the whole angular grid at its range, so the top 0.5 % of the
    volume came out as ~1000 voxels all sitting on one curved surface, and everything
    else - a person at 1 m among a wall at 4.6 m - disappeared inside it. That is why a
    walking target looked like a stationary blob.

    Keeping only local maxima instead reduces the same frame to a few dozen points, one
    per actual reflector, which is what the view is supposed to show. The plain
    threshold stays as a fallback for a map with no structure at all.

    Kept in one place because the plot and the automatic axis scaling have to agree on
    what "the returns" are - otherwise the axes frame something other than the picture.
    """
    peak = ndimage.maximum_filter(data, size=3, mode='nearest')
    strong = (data >= peak) & (data >= vmax_g - floor_db)
    if not strong.any():
        strong = data >= (vmax_g - floor_db)
        if strong.sum() > 0.02 * data.size:
            strong = data >= float(np.percentile(data, 99.5))
    return strong


def rae_view_extent(spec, range_axis, angles_az_deg, angles_el_deg, min_r=0.0):
    """
    Cartesian bounding box of the returns in a Range-Az-El cube, as
    (x0, x1, y0, y1, z0, z1), or None when nothing stands out.

    The 3D axes have to stay fixed across frames - a box that rescales per frame turns
    real movement into a still image - but spanning the whole angular grid is just as
    bad: at +-60 deg over 5 m that is a 9 m wide box, in which a person walking 0.6 m
    sideways is a sub-pixel wobble. So the box is fitted once, to where the returns
    actually are. Outliers are trimmed with a percentile because a single sidelobe
    voxel at the edge of the grid would otherwise stretch the box right back out.
    """
    r_ax = np.asarray(range_axis, dtype=float)
    az_ax = np.deg2rad(np.asarray(angles_az_deg, dtype=float))
    el_ax = np.deg2rad(np.asarray(angles_el_deg, dtype=float))
    keep = r_ax >= min_r if min_r > 0 else np.ones_like(r_ax, dtype=bool)
    if not np.any(keep):
        keep = np.ones_like(r_ax, dtype=bool)

    sub = spec[::max(1, spec.shape[0] // 40)][:, keep]   # sample frames, stay cheap
    if sub.size == 0:
        return None
    vmax_g = float(sub.max())
    r_keep = r_ax[keep]

    xs, ys, zs, vs = [], [], [], []
    for frame in sub:
        strong = rae_strong_voxels(frame, vmax_g)
        vox = np.argwhere(strong)
        if not vox.size:
            continue
        r_v = r_keep[vox[:, 0]]
        el_v = el_ax[vox[:, 1]]
        az_v = az_ax[vox[:, 2]]
        xs.append(r_v * np.cos(el_v) * np.sin(az_v))
        ys.append(r_v * np.cos(el_v) * np.cos(az_v))
        zs.append(r_v * np.sin(el_v))
        vs.append(frame[strong])
    if not xs:
        return None
    xs, ys, zs = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    vs = np.concatenate(vs)

    # Frame the prominent returns only. Weak local maxima are noise peaks scattered over
    # the whole grid - with no range window the grid reaches 16 m - and letting them set
    # the box puts it straight back to the full extent, which is the problem this is
    # here to solve.
    if vs.size > MIN_EXTENT_POINTS:
        cut = float(np.percentile(vs, 90.0))
        strongest = vs >= cut
        if strongest.sum() < MIN_EXTENT_POINTS:      # keep enough points to be meaningful
            strongest = vs >= np.sort(vs)[-MIN_EXTENT_POINTS]
        xs, ys, zs = xs[strongest], ys[strongest], zs[strongest]

    def span(v):
        lo, hi = float(np.percentile(v, 2.0)), float(np.percentile(v, 98.0))
        pad = max(0.15, 0.1 * (hi - lo))
        return lo - pad, hi + pad

    x0, x1 = span(xs)
    y0, y1 = span(ys)
    z0, z1 = span(zs)
    return (x0, x1, max(0.0, y0), y1, z0, z1)


# ---------------- Channel calibration (calibration.py) ----------------
def calibration_params() -> dict:
    """The active channel calibration as params keys, or nothing."""
    return active_calibration.to_params() if active_calibration is not None else {}


def _apply_range_bias():
    """
    Put the active calibration's range bias into the capture parameters (roadmap 9.5),
    so every range axis - they all come from radar_processing.range_bias_m - uses it.
    Results computed with the old axes are dropped; the cube is kept.
    """
    if not last_params:
        return
    b = getattr(active_calibration, 'range_bias_m', None) if active_calibration is not None else None
    old = float(last_params.get('range_bias_m') or 0.0)
    if b:
        last_params['range_bias_m'] = float(b)
    else:
        last_params.pop('range_bias_m', None)
    new = float(last_params.get('range_bias_m') or 0.0)
    if new != old:
        _invalidate_cube_derived_data(f"Range bias {new * 1000:+.1f} mm applied" if new
                                      else "Range bias removed", keep_cube=True)


def _set_active_calibration(cal, path=None):
    global active_calibration, active_calibration_path
    active_calibration, active_calibration_path = cal, path
    try:
        if calib_status_var is not None:
            if cal is None:
                calib_status_var.set("none")
            else:
                name = os.path.basename(path) if path else "estimated, not saved"
                bias = getattr(cal, 'range_bias_m', None)
                bias_txt = f", range {bias * 1000:+.0f} mm" if bias else ""
                calib_status_var.set(f"{cal.num_tx}x{cal.num_rx} ch{bias_txt} - {name}")
    except Exception:
        pass
    _apply_range_bias()
    for view in ("Angle of Arrival", "Range-Angle"):
        mark_settings_changed(view)


def estimate_calibration(range_m: float, az_deg: float = 0.0, el_deg: float = 0.0,
                         window_m: float = 0.15, frames: int = 200,
                         true_range_m: float | None = None):
    """
    Calibration from the loaded capture, a corner reflector at a known angle.
    true_range_m, the measured distance to it, adds the range bias (roadmap 9.5).
    """
    cube = ensure_cube_loaded()
    if cube is None:
        return None
    # measured on the raw range axis, whatever bias an earlier calibration applied
    params = {k: v for k, v in last_params.items() if k != 'range_bias_m'}
    params['antenna_config'] = analysis_antenna_config()
    rc, meta = compute_range_fft_cube(cube[:frames], params, window='hann', pad_pow2=False,
                                      remove_dc=True, n_fft_range=range_fft_size(), return_complex=True)
    cal = calibration.estimate(rc, params, range_m, az_deg, el_deg, window_m,
                               true_range_m=true_range_m, range_axis=meta.get('range_m'))
    cal.meta['capture'] = os.path.basename(bin_var.get())
    print("[Calibration]\n" + cal.summary())
    return cal


def on_estimate_calibration():
    from tkinter import simpledialog
    r = simpledialog.askfloat("Calibration", "Corner reflector range [m]:", minvalue=0.1)
    if r is None:
        return
    az = simpledialog.askfloat("Calibration", "Reflector azimuth [deg] (0 = boresight):",
                               initialvalue=0.0)
    el = simpledialog.askfloat("Calibration", "Reflector elevation [deg]:", initialvalue=0.0)
    if az is None or el is None:
        return
    true_r = simpledialog.askfloat(
        "Calibration", "Measured distance to the reflector [m] - for the range bias.\n"
        "Cancel to calibrate the angles only.", minvalue=0.05)
    try:
        with progress("Estimating channel calibration ..."):
            cal = estimate_calibration(r, az, el, true_range_m=true_r)
    except Exception as e:
        report_error("Calibration", f"Calibration failed: {e}", e)
        return
    if cal is None:
        return
    warn = ""
    if cal.reflector_margin_db < 20:
        warn = (f"\n\nWARNING: the reflector is only {cal.reflector_margin_db:.1f} dB above the "
                f"range cells around it - something else is probably in its cell.")
    messagebox.showinfo("Calibration", cal.summary() + warn +
                        "\n\nThis fit is always good on its own recording; check it on a "
                        "second reflector position (tools/calibrate.py --verify).")
    path = filedialog.asksaveasfilename(defaultextension=".json",
                                        filetypes=[("Calibration", "*.json")],
                                        title="Save calibration (Cancel = use without saving)")
    if path:
        calibration.save(path, cal)
    _set_active_calibration(cal, path or None)


def on_load_calibration():
    path = filedialog.askopenfilename(filetypes=[("Calibration", "*.json")])
    if not path:
        return
    try:
        cal = calibration.load(path)
    except Exception as e:
        report_error("Calibration", f"Cannot load {os.path.basename(path)}: {e}", e)
        return
    print(f"[Calibration] loaded {path}\n" + cal.summary())
    _set_active_calibration(cal, path)


# ---------------- Antenna configuration ----------------
aoa_antenna_cfg_var = None      # selected configuration name (shared by all AoA views)
aoa_antenna_cfg_combo = None

def analysis_antenna_config():
    """
    The antenna configuration for the angle views: the one chosen in the GUI, else the
    one the decoded take brought (a calibrated cascade take: cascade/decode.py attaches
    its configuration and calibration), else None for auto-detection.
    """
    return selected_antenna_config() or (last_params or {}).get('antenna_config')


def selected_antenna_config():
    """The antenna configuration chosen in the GUI, or None for auto-detection."""
    try:
        if aoa_antenna_cfg_var is None:
            return None
        cfg = antenna_config.get_config(aoa_antenna_cfg_var.get())
        if cfg is not None:
            _warn_antenna_config_mismatch(cfg)
        return cfg
    except Exception as e:
        print(f"[GUI] Antenna configuration lookup failed: {e}")
        return None

def _warn_antenna_config_mismatch(cfg):
    """Tell the user when the chosen array does not match the loaded capture."""
    problems = []
    n_tx_cfg = len(cfg.get('tx_offsets') or [])
    n_rx_cfg = len(cfg.get('rx_positions') or [])
    n_tx_data = int((last_params or {}).get('uniqueChirps_effective') or 0)
    n_rx_data = int(last_cube.shape[2]) if last_cube is not None else 0
    if n_tx_data and n_tx_cfg != n_tx_data:
        problems.append(f"{n_tx_cfg} TX in config vs {n_tx_data} in capture")
    if n_rx_data and n_rx_cfg != n_rx_data:
        problems.append(f"{n_rx_cfg} RX in config vs {n_rx_data} in capture")
    if not problems:
        return
    msg = f"Antenna config '{cfg.get('name')}' does not match the capture: " + ", ".join(problems)
    print(f"[GUI] WARNING: {msg}")
    try:
        if progress_text_var is not None:
            progress_text_var.set(msg)
    except Exception:
        pass

def _refresh_antenna_cfg_combo(select: str | None = None):
    if aoa_antenna_cfg_combo is None or aoa_antenna_cfg_var is None:
        return
    names = antenna_config.config_names()
    aoa_antenna_cfg_combo.config(values=names)
    if select and select in names:
        aoa_antenna_cfg_var.set(select)
    elif aoa_antenna_cfg_var.get() not in names:
        aoa_antenna_cfg_var.set(names[0])

def _parse_pairs(text: str):
    """Parse "x,y; x,y; ..." into a list of [x, y] pairs."""
    pairs = []
    for chunk in str(text).replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [c for c in chunk.replace("\t", " ").replace(",", " ").split(" ") if c]
        if len(parts) != 2:
            raise ValueError(f"'{chunk}' is not an x, y pair")
        pairs.append([float(parts[0]), float(parts[1])])
    return pairs

def _format_pairs(pairs):
    return "; ".join(f"{float(p[0]):g},{float(p[1]):g}" for p in pairs)

def _parse_numbers(text: str):
    out = []
    for chunk in str(text).replace(";", " ").replace(",", " ").split():
        chunk = chunk.strip()
        if chunk:
            out.append(float(chunk))
    return out

def compare_antenna_configs(known_angle_deg=None, range_bin=None, max_frames=8):
    """
    Estimate the angle of the strongest target with every antenna configuration.

    Which array layout a board really has cannot be decided from the recording alone -
    two configurations can both be internally consistent and still differ by tens of
    degrees. Measuring settles it: put a strong reflector at a known angle, record, and
    see which configuration puts the peak where the target actually is.

    Returns a list of dicts sorted by how well each configuration matches, or raises
    with a message when there is nothing to compare.
    """
    cube = ensure_cube_loaded()
    if cube is None:
        raise RuntimeError("Load a capture first (Decode tab).")

    frames = min(int(max_frames), cube.shape[0])
    cube = cube[:frames]

    with progress("Preparing capture ...") as p:
        range_cube, rf_meta = run_in_background(
            compute_range_fft_cube, cube, last_params,
            window='hann', pad_pow2=False, remove_dc=True,
            n_fft_range=range_fft_size(), return_complex=True, handle=p)

    r_axis = rf_meta.get('range_m')
    # Ignore the first bins: TX leakage near zero range would win every time.
    min_range_m = 0.3
    try:
        if aoa_min_range_var is not None:
            min_range_m = float(aoa_min_range_var.get())
    except Exception:
        pass
    profile = np.mean(np.abs(range_cube), axis=(0, 1, 2))
    first_bin = 0
    if r_axis is not None and len(r_axis) == len(profile):
        first_bin = int(np.searchsorted(np.asarray(r_axis), min_range_m))
    first_bin = max(0, min(len(profile) - 1, first_bin))
    if range_bin is None:
        range_bin = first_bin + int(np.argmax(profile[first_bin:]))
    range_bin = int(max(0, min(len(profile) - 1, range_bin)))
    range_m = float(r_axis[range_bin]) if r_axis is not None and range_bin < len(r_axis) else None

    configs = antenna_config.all_configs()
    angles = np.arange(-90.0, 91.0, 1.0)

    def _sweep(handle):
        rows = []
        for cfg in configs:
            params = dict(last_params)
            params['is_frequency_domain'] = True
            params['range_m'] = r_axis
            params['antenna_config'] = cfg
            try:
                spec, meta = compute_aoa_spectrum(
                    range_cube, params, range_bin=range_bin,
                    angles_deg=angles, remove_static=False)
                # Average the per-frame spectra, then read off the peak.
                mean_spec = np.mean(spec, axis=0)
                peak_az = float(meta['angles_deg'][int(np.argmax(mean_spec))])
                lin = 10 ** (mean_spec / 10.0)
                sharpness = float(np.max(lin) / (np.median(lin) + 1e-12))
                rows.append({'name': cfg['name'], 'peak_deg': peak_az,
                             'sharpness_db': 10 * np.log10(sharpness + 1e-12),
                             'error': None if known_angle_deg is None
                                      else abs(peak_az - float(known_angle_deg))})
            except Exception as exc:
                rows.append({'name': cfg['name'], 'peak_deg': None,
                             'sharpness_db': None, 'error': None, 'failed': str(exc)})
            handle.advance()
            if _cancel_requested:
                raise ComputationCancelled()
        return rows

    with progress("Comparing antenna configurations ...", total=len(configs)) as p:
        rows = run_in_background(_sweep, p, handle=p)

    if known_angle_deg is None:
        rows.sort(key=lambda r: (-(r['sharpness_db'] if r['sharpness_db'] is not None else -1e9)))
    else:
        rows.sort(key=lambda r: (r['error'] if r['error'] is not None else 1e9))
    return {'rows': rows, 'range_bin': range_bin, 'range_m': range_m, 'frames': frames}


def open_antenna_config_dialog(parent=None):
    """
    Create / edit / select the antenna (virtual array) configuration.

    The layout cannot be reliably guessed from the channel counts alone - an
    IWR6843AOP and a non-AOP 3TX board both report 4 RX and 12 virtual channels - and a
    wrong layout silently produces wrong angles, so it is chosen explicitly here.
    """
    win = tk.Toplevel(parent or root)
    win.title("Antenna configuration")
    win.geometry(f"{theme.px(1080)}x{theme.px(780)}")
    win.minsize(theme.px(900), theme.px(640))
    win.transient(parent or root)

    left = ttk.Frame(win); left.pack(side="left", fill="y", padx=8, pady=8)
    right = ttk.Frame(win); right.pack(side="right", fill="both", expand=True, padx=8, pady=8)

    ttk.Label(left, text="Configurations", font=theme.font(10, "bold")).pack(anchor="w")
    listbox = tk.Listbox(left, width=38, height=8, exportselection=False)
    listbox.pack(fill="x", expand=False, pady=(2, 6))

    form = ttk.Frame(left); form.pack(fill="x")
    fields = {}
    for key, label in [("name", "Name"),
                       ("rx", "RX positions [lambda]  x,y; x,y; ..."),
                       ("tx", "TX offsets [lambda]  x,y; x,y; ..."),
                       ("masks", "TX masks (e.g. 1 2 4)"),
                       ("phase", "Element phase [deg] (optional)")]:
        ttk.Label(form, text=label).pack(anchor="w")
        var = tk.StringVar()
        ttk.Entry(form, textvariable=var, width=44).pack(fill="x", pady=(0, 4))
        fields[key] = var

    ttk.Label(right, text="Virtual array preview", font=theme.font(10, "bold")).pack(anchor="w")
    fig = Figure(figsize=(4.6, 3.4), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    info_frame = ttk.Frame(right)
    info_frame.pack(fill="both", expand=True, pady=(6, 0))
    info = tk.Text(info_frame, height=14, wrap="none", font=theme.mono(9))
    info_vs = ttk.Scrollbar(info_frame, orient="vertical", command=info.yview)
    info_hs = ttk.Scrollbar(info_frame, orient="horizontal", command=info.xview)
    info.configure(yscrollcommand=info_vs.set, xscrollcommand=info_hs.set)
    info.grid(row=0, column=0, sticky="nsew")
    info_vs.grid(row=0, column=1, sticky="ns")
    info_hs.grid(row=1, column=0, sticky="ew")
    info_frame.rowconfigure(0, weight=1)
    info_frame.columnconfigure(0, weight=1)

    state = {"configs": antenna_config.all_configs()}

    def current_cfg_from_form():
        return {
            "name": fields["name"].get().strip(),
            "rx_positions": _parse_pairs(fields["rx"].get()),
            "tx_offsets": _parse_pairs(fields["tx"].get()),
            "tx_masks": [int(v) for v in _parse_numbers(fields["masks"].get())],
            "element_phase_deg": _parse_numbers(fields["phase"].get()),
            "notes": state.get("notes", ""),
            "builtin": False,
        }

    def show_preview(cfg=None, message=None):
        ax.clear()
        info.config(state="normal")
        info.delete("1.0", "end")
        if message and cfg is None:
            info.insert("end", message)
        elif cfg is not None:
            masks = (last_params or {}).get('tx_order_masks')
            try:
                x, y = antenna_config.build_geometry(cfg, masks=masks)
                ax.scatter(x, y, s=70, c=np.arange(len(x)), cmap="viridis")
                for i, (xi, yi) in enumerate(zip(x, y)):
                    ax.annotate(str(i), (xi, yi), textcoords="offset points",
                                xytext=(5, 4), fontsize=7)
                ax.set_xlabel("azimuth [lambda]"); ax.set_ylabel("elevation [lambda]")
                ax.grid(True, alpha=0.3); ax.set_aspect("equal", adjustable="datalim")
                info.insert("end", message or antenna_config.describe(cfg, masks=masks))
            except Exception as e:
                info.insert("end", f"Cannot preview: {e}")
        info.config(state="disabled")
        canvas.draw_idle()

    def fill_form(cfg):
        fields["name"].set(cfg.get("name", ""))
        fields["rx"].set(_format_pairs(cfg.get("rx_positions", [])))
        fields["tx"].set(_format_pairs(cfg.get("tx_offsets", [])))
        fields["masks"].set(" ".join(str(int(m)) for m in cfg.get("tx_masks", [])))
        fields["phase"].set(" ".join(f"{float(p):g}" for p in (cfg.get("element_phase_deg") or [])))
        state["notes"] = cfg.get("notes", "")

    def refresh_list(select_name=None):
        state["configs"] = antenna_config.all_configs()
        listbox.delete(0, "end")
        for c in state["configs"]:
            listbox.insert("end", ("[built-in] " if c.get("builtin") else "") + c["name"])
        names = [c["name"] for c in state["configs"]]
        if select_name and select_name in names:
            idx = names.index(select_name)
            listbox.selection_clear(0, "end")
            listbox.selection_set(idx)
            listbox.see(idx)
            on_select()

    def on_select(_evt=None):
        sel = listbox.curselection()
        if not sel:
            return
        cfg = state["configs"][sel[0]]
        fill_form(cfg)
        show_preview(cfg)

    listbox.bind("<<ListboxSelect>>", on_select)

    def do_preview():
        try:
            cfg = current_cfg_from_form()
        except Exception as e:
            show_preview(message=f"Cannot parse the form: {e}")
            return None
        ok, msg = antenna_config.validate(cfg)
        show_preview(cfg if ok else None, None if ok else f"Invalid configuration: {msg}")
        return cfg if ok else None

    def do_save():
        cfg = do_preview()
        if cfg is None:
            return
        user = [c for c in antenna_config.all_configs() if not c.get("builtin")]
        builtin_names = {c["name"] for c in antenna_config.BUILTIN_CONFIGS}
        if cfg["name"] in builtin_names:
            messagebox.showerror("Antenna configuration",
                                 "A built-in configuration cannot be overwritten.\n"
                                 "Give it a different name to save your own copy.",
                                 parent=win)
            return
        user = [c for c in user if c["name"] != cfg["name"]] + [cfg]
        if antenna_config.save_user_configs(user):
            refresh_list(cfg["name"])
            _refresh_antenna_cfg_combo(cfg["name"])
            messagebox.showinfo("Antenna configuration", f"Saved '{cfg['name']}'.", parent=win)

    def do_delete():
        sel = listbox.curselection()
        if not sel:
            return
        cfg = state["configs"][sel[0]]
        if cfg.get("builtin"):
            messagebox.showerror("Antenna configuration",
                                 "Built-in configurations cannot be deleted.", parent=win)
            return
        if not messagebox.askyesno("Antenna configuration",
                                   f"Delete '{cfg['name']}'?", parent=win):
            return
        user = [c for c in antenna_config.all_configs()
                if not c.get("builtin") and c["name"] != cfg["name"]]
        if antenna_config.save_user_configs(user):
            refresh_list()
            _refresh_antenna_cfg_combo()

    def do_use():
        cfg = do_preview()
        if cfg is None:
            return
        names = antenna_config.config_names()
        if cfg["name"] not in names:
            messagebox.showinfo("Antenna configuration",
                                "Save the configuration first, then it can be selected.",
                                parent=win)
            return
        _refresh_antenna_cfg_combo(cfg["name"])
        for label in ("Angle of Arrival", "Movement Map"):
            mark_settings_changed(label)
        win.destroy()

    btns = ttk.Frame(left); btns.pack(fill="x", pady=(8, 0))
    ttk.Button(btns, text="Preview", command=do_preview).pack(side="left")
    ttk.Button(btns, text="Save", command=do_save).pack(side="left", padx=4)
    ttk.Button(btns, text="Delete", command=do_delete).pack(side="left")
    ttk.Button(btns, text="Use this one", command=do_use).pack(side="right")

    # --- Decide between configurations by measurement ---
    check = ttk.LabelFrame(left, text="Check against the loaded capture")
    check.pack(fill="x", pady=(12, 0))
    ttk.Label(check, text="Known target angle [deg] (optional):").pack(anchor="w", padx=6)
    known_var = tk.StringVar(value="")
    ttk.Entry(check, textvariable=known_var, width=10).pack(anchor="w", padx=6, pady=(0, 4))

    def do_compare():
        try:
            known = float(known_var.get()) if known_var.get().strip() else None
        except ValueError:
            show_preview(message="Known target angle must be a number (or left empty).")
            return
        try:
            result = compare_antenna_configs(known_angle_deg=known)
        except ComputationCancelled:
            show_preview(message="Comparison cancelled.")
            return
        except Exception as exc:
            show_preview(message=f"Cannot compare: {exc}")
            return

        where = (f"range bin {result['range_bin']}"
                 + (f" (~{result['range_m']:.2f} m)" if result['range_m'] is not None else ""))
        lines = [f"Strongest target at {where}, averaged over {result['frames']} frames.",
                 ""]
        if known is None:
            lines.append("No known angle given - configurations are ordered by how sharp "
                         "the peak is, which is only a hint. Put a reflector at a known "
                         "angle and enter it above to decide properly.")
        else:
            lines.append(f"Ordered by distance from the known target angle ({known:g} deg).")
        lines.append("")
        lines.append(f"{'configuration':<40}{'peak[deg]':>10}{'err':>7}{'sharp':>9}")
        lines.append("-" * 65)
        for row in result['rows']:
            if row.get('failed'):
                lines.append(f"{row['name'][:39]:<40}{'failed':>10}")
                continue
            err = "   -" if row['error'] is None else f"{row['error']:4.1f}"
            lines.append(f"{row['name'][:39]:<40}{row['peak_deg']:>10.1f}{err:>7}"
                         f"{row['sharpness_db']:>7.1f}dB")
        sel = listbox.curselection()
        sel_cfg = state["configs"][sel[0]] if sel else None
        show_preview(sel_cfg, "\n".join(lines))

    ttk.Button(check, text="Compare all configurations",
               command=do_compare).pack(anchor="w", padx=6, pady=(0, 6))

    refresh_list(aoa_antenna_cfg_var.get() if aoa_antenna_cfg_var is not None else None)
    if not listbox.curselection() and listbox.size():
        # Nothing selected yet (e.g. the view is still on auto-detect): show the first
        # configuration so the form and the preview are never blank on open.
        listbox.selection_set(0)
        on_select()
    win.grab_set()
    return win

def _ensure_texas_axes_projection(need_3d: bool):
    """Recreate texas_aoa_ax with the requested projection (2D <-> 3D) if needed."""
    global texas_aoa_ax, texas_aoa_ax2
    is_3d = hasattr(texas_aoa_ax, 'get_zlim')
    if is_3d != need_3d or texas_aoa_ax2 is None:
        try:
            texas_aoa_fig.clear()
        except Exception:
            pass
        if need_3d:
            texas_aoa_ax = texas_aoa_fig.add_subplot(121, projection='3d')
        else:
            texas_aoa_ax = texas_aoa_fig.add_subplot(121)
        texas_aoa_ax2 = texas_aoa_fig.add_subplot(122)

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

    # RAE projection mode (Top / Slice / 3D) - read early to setup axes projection
    rae_proj_mode = "Top (R-Az)"
    try:
        if texas_rae_proj_var is not None:
            rae_proj_mode = texas_rae_proj_var.get()
    except Exception: pass
    _ensure_texas_axes_projection(view_type == "Range-Az-El" and rae_proj_mode.startswith("3D"))

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
        texas_aoa_ax.set_title(f"AoA Spectrum (Frame {f})")
        texas_aoa_ax.grid(True, alpha=0.3)
        if ang is not None and len(ang) > 0:
            texas_aoa_ax.set_xlim(ang[0], ang[-1])
        show_axis("aoa", texas_aoa_ax, -40, 0, texas_aoa_canvas)
    elif view_type == "Range-Azimuth":
        data = last_texas_aoa_spec[f]
        ang = last_texas_aoa_meta.get('angles_deg')
        rng = last_texas_aoa_meta.get('range_axis')
        
        # Apply range crop
        crop_mask = None
        if min_r > 0 and rng is not None:
            crop_mask = rng >= min_r
            if np.any(crop_mask):
                data = data[crop_mask, :]
                rng = rng[crop_mask]
            else:
                crop_mask = None

        # Fixed color scaling across ALL frames (crop-aware) so playback keeps a stable scale
        if crop_mask is not None:
            vmax_dyn = _global_vmax(last_texas_aoa_spec, crop_mask)
        else:
            vmax_dyn = _global_vmax(last_texas_aoa_spec)
        vmin_dyn = vmax_dyn - 35 # Use a fixed 35dB dynamic range from the global peak

        texas_aoa_im = texas_aoa_ax.imshow(
            data, aspect='auto', origin='lower',
            extent=[ang[0], ang[-1], rng[0], rng[-1]],
            vmin=vmin_dyn, vmax=vmax_dyn, cmap=display_cmap()
        )
        show_image("aoa", texas_aoa_im, vmin_dyn, vmax_dyn, texas_aoa_canvas)
        texas_aoa_cbar = texas_aoa_fig.colorbar(texas_aoa_im, ax=texas_aoa_ax, label='dB')
        texas_aoa_ax.set_xlabel("Angle [deg]")
        texas_aoa_ax.set_ylabel("Range [m]")
        texas_aoa_ax.set_title(f"Range-Azimuth (Frame {f})")
    elif view_type == "Azimuth-Elevation":
        data = last_texas_aoa_spec[f]
        az = last_texas_aoa_meta.get('angles_az_deg')
        el = last_texas_aoa_meta.get('angles_el_deg')
        texas_aoa_im = texas_aoa_ax.imshow(
            data, aspect='equal', origin='lower',
            extent=[az[0], az[-1], el[0], el[-1]],
            vmin=-20, vmax=0, cmap=display_cmap()
        )
        show_image("aoa", texas_aoa_im, -20, 0, texas_aoa_canvas)
        texas_aoa_cbar = texas_aoa_fig.colorbar(texas_aoa_im, ax=texas_aoa_ax, label='dB')
        texas_aoa_ax.set_xlabel("Azimuth [deg]")
        texas_aoa_ax.set_ylabel("Elevation [deg]")
        texas_aoa_ax.set_title(f"Azimuth-Elevation (Frame {f})")
    elif view_type == "Range-Az-El":
        data3d = last_texas_aoa_spec[f]  # (R, EL, AZ)
        az = last_texas_aoa_meta.get('angles_az_deg')
        el = last_texas_aoa_meta.get('angles_el_deg')
        rng = last_texas_aoa_meta.get('range_axis')

        proj_mode = rae_proj_mode

        dets_all = last_texas_aoa_meta.get('detections')
        dets = dets_all[f] if (dets_all is not None and f < len(dets_all)) else None

        if proj_mode.startswith("Slice") :
            # Az-El slice for the range bin selected by the slider
            sel_bin = data3d.shape[0] // 4
            try:
                if aoa_bin_slider is not None and aoa_bin_slider.cget("state") != "disabled":
                    sel_bin = int(aoa_bin_slider.get())
            except Exception: pass
            sel_bin = min(max(0, sel_bin), data3d.shape[0] - 1)
            sl = data3d[sel_bin]  # (EL, AZ)
            # Fixed color scale across all frames (global maximum of the whole map)
            vmax_dyn = _global_vmax(last_texas_aoa_spec)
            texas_aoa_im = texas_aoa_ax.imshow(
                sl, aspect='auto', origin='lower',
                extent=[az[0], az[-1], el[0], el[-1]],
                vmin=vmax_dyn - 35, vmax=vmax_dyn, cmap=display_cmap()
            )
            show_image("aoa", texas_aoa_im, vmax_dyn - 35, vmax_dyn, texas_aoa_canvas)
            texas_aoa_cbar = texas_aoa_fig.colorbar(texas_aoa_im, ax=texas_aoa_ax, label='dB')
            texas_aoa_ax.set_xlabel("Azimuth [deg]")
            texas_aoa_ax.set_ylabel("Elevation [deg]")
            r_txt = f", r={rng[sel_bin]:.2f} m" if (rng is not None and sel_bin < len(rng)) else ""
            texas_aoa_ax.set_title(f"Az-El Slice (Frame {f}, Bin {sel_bin}{r_txt})")
            if dets:
                for d in dets:
                    if abs(d['range_idx'] - sel_bin) <= 1:
                        texas_aoa_ax.scatter(d['azimuth_deg'], d['elevation_deg'],
                                             s=80, facecolors='none', edgecolors='white', linewidths=1.5)
        elif proj_mode.startswith("3D"):
            # 3D point cloud: strong voxels + detections in cartesian space (x, y, z)
            rng_c = rng if rng is not None else np.arange(data3d.shape[0], dtype=float)
            data = data3d
            crop_mask = None
            if min_r > 0:
                crop_mask = rng_c >= min_r
                if np.any(crop_mask):
                    data = data[crop_mask]
                    rng_c = rng_c[crop_mask]
                else:
                    crop_mask = None

            # Fixed color scale across all frames (global maximum of the whole map)
            if crop_mask is not None:
                vmax_g = _global_vmax(last_texas_aoa_spec, crop_mask)
            else:
                vmax_g = _global_vmax(last_texas_aoa_spec)
            vmin_g = vmax_g - 35.0

            # Strong voxels, by the same rule the axis scaling uses.
            max_pts = 3000
            strong = rae_strong_voxels(data, vmax_g)
            vox = np.argwhere(strong)
            vals = data[strong]
            if vals.size > max_pts:
                order = np.argsort(vals)[-max_pts:]
                vox = vox[order]
                vals = vals[order]
            # Colour across what is actually drawn, otherwise every point ends up the
            # same shade at the top of the colour map.
            if vals.size:
                vmax_g = float(vals.max())
                vmin_g = float(min(vals.min(), vmax_g - 3.0))

            if vox.size > 0:
                r_v = rng_c[vox[:, 0]]
                el_v = np.deg2rad(el[vox[:, 1]])
                az_v = np.deg2rad(az[vox[:, 2]])
                x_v = r_v * np.cos(el_v) * np.sin(az_v)
                y_v = r_v * np.cos(el_v) * np.cos(az_v)
                z_v = r_v * np.sin(el_v)
                # Few, meaningful points now reach the plot, so they are drawn large and
                # opaque instead of as a faint haze.
                texas_aoa_im = texas_aoa_ax.scatter(
                    x_v, y_v, z_v, c=vals, cmap=display_cmap(),
                    vmin=vmin_g, vmax=vmax_g, s=26, alpha=0.85, depthshade=False
                )
                show_image("aoa", texas_aoa_im, vmin_g, vmax_g, texas_aoa_canvas)
                # pad: in a 3D axes the colour bar otherwise sits on top of the Z label.
                texas_aoa_cbar = texas_aoa_fig.colorbar(texas_aoa_im, ax=texas_aoa_ax,
                                                        label='dB', shrink=0.7, pad=0.12)

            # Overlay CFAR detections as bold markers
            if dets:
                for d in dets:
                    if d['range_m'] < min_r:
                        continue
                    texas_aoa_ax.scatter([d['x_m']], [d['y_m']], [d['z_m']],
                                         s=90, c='black', marker='x', linewidths=2.5,
                                         depthshade=False)
                    texas_aoa_ax.text(d['x_m'], d['y_m'], d['z_m'],
                                      f" r={d['range_m']:.1f}m", fontsize=8, color='black')

            # Trail of the tracked target over the preceding frames. A point cloud alone
            # makes slow sideways movement very hard to read - the trail turns it into a
            # visible path, and the current position is marked on it.
            track = last_texas_aoa_meta.get('tracked_targets')
            if track and f < len(track):
                lo = max(0, f - TRACK_TRAIL_FRAMES)
                seg = [t for t in track[lo:f + 1] if t['range'] >= min_r]
                if len(seg) > 1:
                    texas_aoa_ax.plot([t['x'] for t in seg], [t['y'] for t in seg],
                                      [t['z'] for t in seg],
                                      color='magenta', linewidth=1.8, alpha=0.9)
                if seg:
                    now = seg[-1]
                    texas_aoa_ax.scatter([now['x']], [now['y']], [now['z']],
                                         s=120, facecolors='none', edgecolors='magenta',
                                         linewidths=2.0, depthshade=False)

            # Axis limits: fixed across frames (so movement reads as motion, not as a
            # rescaling plot) but scaled to where the returns actually are. Spanning the
            # whole +-16 m grid made a person walking 0.6 m sideways a sub-pixel wobble.
            lim = last_texas_aoa_meta.get('view_extent')
            if lim is None:
                r_max = float(rng_c[-1]) if len(rng_c) else 1.0
                az_max = np.deg2rad(max(abs(az[0]), abs(az[-1]))) if az is not None else np.pi / 2
                el_max = np.deg2rad(max(abs(el[0]), abs(el[-1]))) if el is not None else np.pi / 4
                lim = (-r_max * np.sin(az_max), r_max * np.sin(az_max),
                       0.0, r_max,
                       -max(r_max * np.sin(el_max), 0.1), max(r_max * np.sin(el_max), 0.1))
            texas_aoa_ax.set_xlim(lim[0], lim[1])
            texas_aoa_ax.set_ylim(lim[2], lim[3])
            texas_aoa_ax.set_zlim(lim[4], lim[5])
            texas_aoa_ax.set_xlabel("X [m]")
            texas_aoa_ax.set_ylabel("Y (Distance) [m]")
            texas_aoa_ax.set_zlabel("Z (Height) [m]")
            method_used = last_texas_aoa_meta.get('method_used', '')
            texas_aoa_ax.set_title(f"Range-Az-El 3D ({method_used}, Frame {f})")
        else:
            # Top projection: max over elevation -> Range-Azimuth heatmap
            data = data3d.max(axis=1)  # (R, AZ)
            rng_c = rng
            crop_mask = None
            if min_r > 0 and rng is not None:
                crop_mask = rng >= min_r
                if np.any(crop_mask):
                    data = data[crop_mask, :]
                    rng_c = rng[crop_mask]
                else:
                    crop_mask = None

            # Fixed color scale across all frames (crop-aware global maximum)
            if crop_mask is not None:
                vmax_dyn = _global_vmax(last_texas_aoa_spec, crop_mask)
            else:
                vmax_dyn = _global_vmax(last_texas_aoa_spec)
            vmin_dyn = vmax_dyn - 35
            texas_aoa_im = texas_aoa_ax.imshow(
                data, aspect='auto', origin='lower',
                extent=[az[0], az[-1], rng_c[0], rng_c[-1]],
                vmin=vmin_dyn, vmax=vmax_dyn, cmap=display_cmap()
            )
            show_image("aoa", texas_aoa_im, vmin_dyn, vmax_dyn, texas_aoa_canvas)
            texas_aoa_cbar = texas_aoa_fig.colorbar(texas_aoa_im, ax=texas_aoa_ax, label='dB')
            texas_aoa_ax.set_xlabel("Azimuth [deg]")
            texas_aoa_ax.set_ylabel("Range [m]")
            method_used = last_texas_aoa_meta.get('method_used', '')
            texas_aoa_ax.set_title(f"Range-Az-El, max over El ({method_used}, Frame {f})")

            if dets:
                # Detections are logged once per run (see run_texas_aoa); printing
                # them here as well meant console I/O on every frame redraw, which
                # made scrubbing and playback visibly slower.
                for d in dets:
                    if d['range_m'] < min_r:
                        continue
                    texas_aoa_ax.scatter(d['azimuth_deg'], d['range_m'],
                                         s=80, facecolors='none', edgecolors='white', linewidths=1.5)
                    texas_aoa_ax.annotate(f"{d['elevation_deg']:.0f}\u00b0",
                                          (d['azimuth_deg'], d['range_m']),
                                          textcoords="offset points", xytext=(6, 6),
                                          color='white', fontsize=8)

    # Overlay tracked target if available
    if 'tracked_targets' in last_texas_aoa_meta:
        target = last_texas_aoa_meta['tracked_targets'][f]
        r_val = target['range']
        az_val = target['azimuth']
        el_val = target['elevation']
        x_m = target['x']
        y_m = target['y']
        z_m = target['z']
        
        if view_type == "Range-Azimuth" or (view_type == "Range-Az-El" and rae_proj_mode == "Top (R-Az)"):
            texas_aoa_ax.scatter([az_val], [r_val], color='red', marker='*', s=150, edgecolors='white', linewidths=1.5, zorder=10, label='Tracked Target')
            txt = f"Target:\nr={r_val:.1f} m\naz={az_val:.1f}°\nx={x_m:.2f} m\ny={y_m:.2f} m"
            texas_aoa_ax.text(az_val + 2, r_val, txt, color='white', fontsize=9, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
        elif view_type == "Range-Az-El" and rae_proj_mode == "Slice (Az-El)":
            texas_aoa_ax.scatter([az_val], [el_val], color='red', marker='*', s=150, edgecolors='white', linewidths=1.5, zorder=10)
            txt = f"Target:\naz={az_val:.1f}°\nel={el_val:.1f}°\nx={x_m:.2f} m\nz={z_m:.2f} m"
            texas_aoa_ax.text(az_val + 2, el_val, txt, color='white', fontsize=9, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
        elif view_type == "Range-Az-El" and rae_proj_mode.startswith("3D"):
            texas_aoa_ax.scatter([x_m], [y_m], [z_m], color='red', marker='*', s=200, edgecolors='white', linewidths=1.5, zorder=10, depthshade=False)
            # In a 3D axes a label anchored to the point lands wherever the rotation puts
            # it, usually across the data. Pin it to the corner of the plot instead.
            txt = (f"Target: r={r_val:.1f} m, az={az_val:.1f}°, el={el_val:.1f}°\n"
                   f"x={x_m:.2f} m, y={y_m:.2f} m, z={z_m:.2f} m")
            texas_aoa_ax.text2D(0.02, 0.98, txt, transform=texas_aoa_ax.transAxes,
                                color='red', fontsize=9, fontweight='bold',
                                va='top', ha='left')

    # Draw time tracking plot on texas_aoa_ax2
    if texas_aoa_ax2 is not None:
        texas_aoa_ax2.clear()
        # what init shows until Track Moving fills it - clear() left a bare 0..1 frame
        texas_aoa_ax2.set_title("Target tracking over time")
        texas_aoa_ax2.set_xlabel("Frame")
        texas_aoa_ax2.set_ylabel("Azimuth [deg]")
        # Clean twin axes if any exist - also when tracking is off now, or the X axis of
        # an earlier run stayed on the panel
        for other_ax in list(texas_aoa_fig.axes):
            if other_ax is not texas_aoa_ax2 and other_ax is not texas_aoa_ax:
                if other_ax.get_subplotspec() == texas_aoa_ax2.get_subplotspec():
                    try:
                        texas_aoa_fig.delaxes(other_ax)
                    except Exception: pass
        if 'tracked_targets' in last_texas_aoa_meta:
            tracked_targets = last_texas_aoa_meta['tracked_targets']
            frames_axis = np.arange(len(tracked_targets))
            az_series = [t['azimuth'] for t in tracked_targets]
            x_series = [t['x'] for t in tracked_targets]

            ax2_left = texas_aoa_ax2

            # Recreate twinx cleanly
            ax2_right = ax2_left.twinx()
            
            ax2_left.plot(frames_axis, az_series, 'b-', linewidth=1.5, label='Azimut')
            ax2_right.plot(frames_axis, x_series, 'r-', linewidth=1.5, label='X coordinate')
            
            # Highlight current frame
            ax2_left.axvline(x=f, color='green', linestyle='--', linewidth=1.5, label=f'Current ({f})')
            
            ax2_left.set_xlabel("Frame")
            ax2_left.set_ylabel("Azimuth [deg]", color='b')
            ax2_left.tick_params(axis='y', labelcolor='b')
            ax2_right.set_ylabel("X [m]", color='r')
            ax2_right.tick_params(axis='y', labelcolor='r')
            
            ax2_left.grid(True, alpha=0.3)
            ax2_left.set_title("Target tracking over time")

    try:
        texas_aoa_canvas.draw_idle()
    except Exception: pass

# --- NEW: Range-Angle Functions ---
def init_range_angle_plot(parent):
    global ra_fig, ra_ax, ra_canvas, ra_nang_var, ra_clutter_var, ra_peaks_var, ra_remove_dc_var, ra_r_start_var, ra_r_end_var
    
    controls = FlowFrame(parent)
    controls.grid(row=0, column=0, padx=8, pady=4, sticky="ew")

    register_run_button("Range-Angle",
        theme.Button(controls.group(), text="Run Range-Angle", command=run_range_angle)).pack(side="left")

    _g = controls.group()
    ttk.Label(_g, text="Angles:").pack(side="left", padx=(0, 2))
    ra_nang_var = tk.StringVar(value="181")
    ttk.Entry(_g, textvariable=ra_nang_var, width=6).pack(side="left")

    _g = controls.group()
    ttk.Label(_g, text="Range Start:").pack(side="left", padx=(0, 2))
    ra_r_start_var = tk.StringVar(value="0")
    ttk.Entry(_g, textvariable=ra_r_start_var, width=6).pack(side="left")
    ttk.Label(_g, text="End:").pack(side="left", padx=(8, 2))
    ra_r_end_var = tk.StringVar(value="0")
    ttk.Entry(_g, textvariable=ra_r_end_var, width=6).pack(side="left")

    _g = controls.group()
    ra_clutter_var = tk.BooleanVar(value=False)
    tk.Checkbutton(_g, text="Clutter Removal", variable=ra_clutter_var,
                   command=lambda: mark_settings_changed("Range-Angle")).pack(side="left", padx=(0, 4))
    ra_remove_dc_var = tk.BooleanVar(value=True)
    tk.Checkbutton(_g, text="Remove DC", variable=ra_remove_dc_var,
                   command=lambda: mark_settings_changed("Range-Angle")).pack(side="left", padx=4)
    ra_peaks_var = tk.BooleanVar(value=False)
    tk.Checkbutton(_g, text="Show Peaks", variable=ra_peaks_var, command=lambda: update_range_angle_plot(int(global_frame_slider.get()))).pack(side="left", padx=4)

    ra_fig = Figure(figsize=(5, 3), dpi=100, constrained_layout=True)
    ra_ax = ra_fig.add_subplot(111)
    ra_ax.set_title("Range-Angle Map")
    ra_ax.set_xlabel("Angle [deg]")
    ra_ax.set_ylabel("Range [m]")
    ra_canvas = FigureCanvasTkAgg(ra_fig, master=parent)
    ra_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
    parent.rowconfigure(1, weight=1)
    parent.columnconfigure(0, weight=1)

def _aoa_angle_limit() -> float:
    try:
        return float(aoa_angle_limit_var.get()) if aoa_angle_limit_var is not None else 90.0
    except Exception:
        return 90.0


def _range_azimuth_frames(cube, label, remove_static, remove_dc=True, angles=None,
                          r_start=0, r_end=None):
    """
    Range-azimuth maps of every frame, (F, R, A) in dB, with the Angle of Arrival tab's
    beamformer: its antenna configuration, MIMO mode, method and TDM Doppler
    compensation. The Movement Map and the Range-Angle view are both drawn from it.
    Returns (maps, meta, (r_start, r_end)) - the range bins kept.
    """
    range_cube, rf_meta = compute_range_fft_cube(
        cube, last_params, window='hann', pad_pow2=False, remove_dc=remove_dc,
        n_fft_range=range_fft_size(), return_complex=True
    )
    R = range_cube.shape[-1]
    r_end = R - 1 if r_end is None else max(0, min(R - 1, int(r_end)))
    r_start = max(0, min(r_end, int(r_start)))
    if (r_start, r_end) != (0, R - 1):
        range_cube = range_cube[..., r_start:r_end + 1]
    r_axis = rf_meta.get('range_m')
    params_aoa = last_params.copy()
    params_aoa['is_frequency_domain'] = True
    params_aoa['range_m'] = None if r_axis is None else np.asarray(r_axis)[r_start:r_end + 1]
    params_aoa['antenna_config'] = analysis_antenna_config()
    params_aoa.update(clutter_settings())
    params_aoa.update(calibration_params())
    try:
        params_aoa['tdm_doppler_compensation'] = bool(texas_aoa_doppler_comp_var.get())
    except Exception:
        pass
    m_mode = str(texas_aoa_mimo_mode_var.get()).strip() if (texas_aoa_mimo_mode_var is not None) else "MIMO"
    rx_override = 'auto' if m_mode.upper() == 'FIRST TX' else None
    method = str(texas_aoa_method_var.get()).lower() if (texas_aoa_method_var is not None) else "bartlett"
    if angles is None:
        lim = _aoa_angle_limit()
        angles = np.arange(-lim, lim + 1.0, 1.0)
    if remove_static:
        method_cl = params_aoa.get('clutter_method') or 'mean'
        loops = int(derive_capture_dimensions(last_params).get('loops') or 0)
        if clutter_filters.few_loops_method(method_cl, loops)[0] != method_cl:
            print(f"[{label}] {loops} chirps per frame: the static scene is taken "
                  f"over frames (ema) instead of the frame mean")
    print(f"[{label}] Computing Range-Azimuth maps for all frames...")
    with progress(f"{label}: Range-Azimuth maps ...", total=range_cube.shape[0]) as p:
        maps, meta = run_in_background(
            compute_range_azimuth_map,
            range_cube, params_aoa, remove_static=remove_static, mimo=True,
            mimo_mode=m_mode, rx_order_override=rx_override, method=method,
            angles_deg=angles, progress_cb=p.callback(), handle=p
        )
    return maps, meta, (r_start, r_end)


RA_PEAK_THRESHOLD_DB = -15.0      # Show Peaks: local maxima this close to the frame's top


def _ra_frame_peaks(f: int) -> list:
    """Show Peaks of one frame (computed when first shown): local maxima of the map within
    RA_PEAK_THRESHOLD_DB of the frame's strongest point."""
    cache = last_ra_meta.setdefault('peaks', {})
    if f in cache:
        return cache[f]
    from scipy.ndimage import maximum_filter
    x = last_ra_map[f]                                   # (angle, range)
    mask = (x == maximum_filter(x, size=5)) & (x > x.max() + RA_PEAK_THRESHOLD_DB)
    theta, bins = last_ra_meta['theta'], last_ra_meta['r_bins']
    cache[f] = [{'range_bin': int(bins[r]), 'angle_deg': float(theta[a]),
                 'intensity_db': float(x[a, r] - x.max())} for a, r in np.argwhere(mask)]
    return cache[f]


def run_range_angle():
    """
    Range-angle map of every frame from the beamformer of Angle of Arrival -> Range-Azimuth
    (antenna configuration, MIMO mode, method), for playback with the frame slider.

    It used to be an FFT across the physical receivers as if they were a uniform line. The
    IWR6843AOP's receivers sit in a 2 x 2 square with two of them rotated, so on a
    synthetic AOP capture a target at 0 deg came out at -90 deg and one at 20 deg at 41.
    """
    clear_settings_changed("Range-Angle")
    cube = ensure_cube_loaded()
    if cube is None:
        return
    try:
        global last_ra_map, last_ra_meta, ra_im, ra_cbar, ra_peak_scatter
        try:
            n_ang = max(3, int(ra_nang_var.get()))
        except Exception:
            n_ang = 181
        try:
            r_start = int(ra_r_start_var.get())
            r_end = int(ra_r_end_var.get())
            if r_end == 0: r_end = None
        except Exception:
            r_start, r_end = 0, None
        lim = _aoa_angle_limit()
        angles = np.linspace(-lim, lim, n_ang)
        maps, meta, (r_start, r_end) = _range_azimuth_frames(
            cube, "Range-Angle", remove_static=bool(ra_clutter_var.get()),
            remove_dc=bool(ra_remove_dc_var.get()), angles=angles, r_start=r_start, r_end=r_end)

        last_ra_map = np.transpose(maps, (0, 2, 1))          # (F, angle, range) as drawn
        r_bins = np.arange(r_start, r_end + 1)
        last_ra_meta = {
            'theta': np.asarray(meta.get('angles_deg', angles), dtype=float),
            'r_bins': r_bins,
            'range_m': np.asarray(meta.get('range_axis', r_bins), dtype=float),
            'peaks': {},
            'peak_threshold_db': RA_PEAK_THRESHOLD_DB,
        }

        # Global color scale
        vmin, vmax = image_percentiles(last_ra_map, [5, 99.9])
        last_ra_meta['vmin'] = vmin
        last_ra_meta['vmax'] = vmax

        ra_im = None # Force redraw
        ra_peak_scatter = None
        update_range_angle_plot(int(global_frame_slider.get()))

    except Exception as e:
        report_error("Error", f"Range-Angle error: {e}", e)
        import traceback
        traceback.print_exc()

def update_range_angle_plot(frame_idx: int):
    global ra_im, ra_cbar, ra_ax, ra_fig
    if last_ra_map is None or last_ra_meta is None:
        return
        
    f = min(max(frame_idx, 0), last_ra_map.shape[0] - 1)
    data = last_ra_map[f]
    theta = last_ra_meta['theta']
    range_m = last_ra_meta['range_m']

    extent = bin_extent(theta, range_m)

    if ra_im is None:
        ra_ax.clear()
        ra_im = ra_ax.imshow(data.T, aspect='auto', origin='lower', cmap=display_cmap(),
                             extent=extent, vmin=last_ra_meta['vmin'], vmax=last_ra_meta['vmax'])
        show_image("ra", ra_im, last_ra_meta['vmin'], last_ra_meta['vmax'], ra_canvas)
        if ra_cbar is None:
            ra_cbar = ra_fig.colorbar(ra_im, ax=ra_ax)
        else:
            ra_cbar.update_normal(ra_im)
        ra_cbar.set_label("Intensity [dB]")
        ra_ax.set_xlabel("Angle [deg]")          # clear() takes the labels of init too
        ra_ax.set_ylabel("Range [m]")
    else:
        ra_im.set_data(data.T)
        ra_im.set_extent(extent)
        
    ra_ax.set_title(f"Range-Angle Map (Frame {f})")

    # Optional peak overlay (cached per frame by run_range_angle)
    global ra_peak_scatter
    if ra_peak_scatter is not None:
        try:
            ra_peak_scatter.remove()
        except Exception:
            pass
        ra_peak_scatter = None
    if ra_peaks_var.get():
        pk = _ra_frame_peaks(f)
        if pk:
            r_axis = last_ra_meta['range_m']
            bins = list(last_ra_meta['r_bins'])
            xs, ys = [], []
            for d in pk:
                try:
                    idx = bins.index(int(d['range_bin']))
                except ValueError:
                    continue
                xs.append(float(d['angle_deg']))
                ys.append(float(r_axis[idx]))
            if xs:
                ra_peak_scatter = ra_ax.scatter(xs, ys, s=60, facecolors='none',
                                                edgecolors='white', linewidths=1.4)

    ra_canvas.draw_idle()

# --- Micro-Doppler: spectrogram, envelopes, swings (limb_motion.py, roadmap phase 10) ---
md_fig = None
md_ax = None
md_canvas = None

def init_md_plot(parent):
    global md_fig, md_ax, md_canvas
    md_fig = Figure(figsize=(5, 3), dpi=100, constrained_layout=True)
    md_ax = md_fig.add_subplot(111)
    md_ax.set_title("Micro-Doppler (Time-Doppler)")
    md_ax.set_xlabel("Time [s]")
    md_ax.set_ylabel("Radial velocity [m/s]")
    md_canvas = FigureCanvasTkAgg(md_fig, master=parent)
    md_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
    parent.rowconfigure(1, weight=1)
    parent.columnconfigure(0, weight=1)


def _md_settings() -> dict:
    """The Micro-Doppler controls; an unreadable field takes its default."""
    def num(var, default, cast=float):
        try:
            return cast(str(var.get()).strip())
        except Exception:
            return default
    height = str(md_height_var.get()).strip() if md_height_var is not None else ""
    try:
        height = float(height) if height else None
    except ValueError:
        height = None
    return {
        "bins": (num(md_r0_var, 5, int), num(md_r1_var, 100, int)),
        "gate": "auto" if str(md_gate_var.get()).lower().startswith("auto") else "fixed",
        "gate_width_m": max(0.2, num(md_width_var, limb_motion.DEFAULT_GATE_WIDTH_M)),
        "window_s": max(0.002, num(md_window_var, 1e3 * limb_motion.DEFAULT_WINDOW_S) / 1e3),
        "threshold_db": num(md_thr_var, limb_motion.DEFAULT_THRESHOLD_DB),
        "swing_m_s": max(0.1, num(md_swing_var, limb_motion.DEFAULT_SWING_M_S)),
        "radar_height_m": height,
    }


def run_micro_doppler():
    clear_settings_changed("Micro-Doppler")
    cube = ensure_cube_loaded()
    if cube is None:
        return
    try:
        global last_md_map, last_md_meta, md_im
        s = _md_settings()
        rx_mode = (global_rx_mode_var.get() or "MRC").strip().upper()
        with progress("Micro-Doppler: spectrogram, envelopes and swings ..."):
            res = run_in_background(
                limb_motion.analyse, cube, {**last_params, **clutter_settings()},
                bins=s["bins"], gate=s["gate"], gate_width_m=s["gate_width_m"],
                window_s=s["window_s"], rx_mode=rx_mode, n_fft_range=range_fft_size(),
                t0_s=segment_t0_s(), threshold_db=s["threshold_db"], swing_m_s=s["swing_m_s"],
                radar_height_m=s["radar_height_m"])
        img, extent = limb_motion.display_image(res.spec)
        last_md_map = img.T                               # the picture shown: time columns x Doppler
        vmin, vmax = image_percentiles(img, [5, 99.5])
        last_md_meta = {"result": res, "image": img, "extent": extent,
                        "display_vmin": vmin, "display_vmax": vmax, "settings": s}
        print("[Micro-Doppler]\n" + limb_motion.describe(res))
        sm = res.summary

        def _f(x):
            return "-" if not np.isfinite(x) else f"{x:+.2f}"
        md_readout_var.set(f"Max speed: away {_f(sm['max_swing_away_m_s'])}, towards "
                           f"{_f(sm['max_swing_towards_m_s'])} m/s | {sm['swings']} swings | "
                           f"v_max {res.spec.v_max:.2f} m/s")
        md_controls.after_idle(md_controls._reflow)       # the label got wider
        md_im = None                                      # redraw everything
        set_global_frames(cube.shape[0])
        update_md_plot(_current_frame_index())
    except Exception as e:
        report_error("Error", f"Micro-Doppler error: {e}", e)


def _md_frame_time(frame_idx: int) -> float:
    """Recording time of a frame of the current segment (where the indicator goes)."""
    res = (last_md_meta or {}).get("result")
    if res is None:
        return 0.0
    period = capture_params.frame_period_s(last_params) or 0.0
    f = min(max(int(frame_idx), 0), max(0, int(res.spec.meta.get("frames", 1)) - 1))
    return float(res.spec.meta.get("t0_s", 0.0)) + f * period


def update_md_plot(frame_idx: int):
    global md_im, md_cbar, md_indicator_line
    meta = last_md_meta or {}
    res = meta.get("result")
    if res is None:
        return
    if md_im is None:
        img, extent = meta["image"], meta["extent"]
        vmin, vmax = meta.get("display_vmin"), meta.get("display_vmax")
        env, spec = res.env, res.spec
        md_ax.clear()
        for name in ("md-", "md+"):                       # copies of an earlier run
            _display_items.pop(name, None)
        md_im = md_ax.imshow(img, aspect="auto", origin="lower", cmap=display_cmap(),
                             extent=extent, vmin=vmin, vmax=vmax, interpolation="nearest")
        show_image("md", md_im, vmin, vmax, md_canvas)
        # Envelopes read beyond +-v_max (unfolded) sit on the spectrum's periodic
        # continuation, drawn above and below it.
        lo = np.nanmin(env.lower_m_s) if np.isfinite(env.lower_m_s).any() else -spec.v_max
        hi = np.nanmax(env.upper_m_s) if np.isfinite(env.upper_m_s).any() else spec.v_max
        y0, y1 = min(extent[2], lo - 0.3), max(extent[3], hi + 0.3)
        span = extent[3] - extent[2]
        for k, name in ((-1, "md-"), (1, "md+")):
            if (k < 0 and y0 < extent[2]) or (k > 0 and y1 > extent[3]):
                im2 = md_ax.imshow(img, aspect="auto", origin="lower", cmap=display_cmap(),
                                   extent=[extent[0], extent[1], extent[2] + k * span,
                                           extent[3] + k * span],
                                   vmin=vmin, vmax=vmax, interpolation="nearest")
                show_image(name, im2, vmin, vmax, md_canvas)
        md_ax.set_ylim(y0, y1)
        md_ax.set_xlim(extent[0], extent[1])
        if md_cbar is None:
            md_cbar = md_fig.colorbar(md_im, ax=md_ax)
        else:
            md_cbar.update_normal(md_im)
        md_cbar.set_label("Power [dB]")
        md_ax.set_xlabel("Time [s]")
        md_ax.set_ylabel("Radial velocity [m/s] (+ away)")
        if md_env_var is None or md_env_var.get():
            md_ax.plot(env.times_s, env.torso_m_s, color="white", lw=1.0, ls="--",
                       alpha=0.8, label="torso")
            md_ax.plot(env.times_s, env.upper_m_s, color="#ff4fa3", lw=1.0, label="envelope")
            md_ax.plot(env.times_s, env.lower_m_s, color="#ff4fa3", lw=1.0)
            for sw in res.swings:
                mark = "^" if sw.direction == "away" else "v"
                md_ax.plot(sw.t_peak_s, sw.v_peak_m_s, mark, color="#ffe14f", ms=7,
                           mec="black", mew=0.6)
                if len(res.swings) <= 30:
                    label = f"{sw.v_peak_m_s:+.1f}" + (f" {sw.limb}" if sw.limb else "")
                    md_ax.annotate(label, (sw.t_peak_s, sw.v_peak_m_s), fontsize=7,
                                   color="white", xytext=(3, 5 if sw.direction == "away" else -11),
                                   textcoords="offset points")
            md_ax.legend(loc="upper right", fontsize=7, framealpha=0.5)
        m = spec.meta
        gate = (f"auto gate {m['gate_width_m']:.1f} m" if m["gate"] == "auto"
                else f"bins {m['search_bins'][0]}-{m['search_bins'][1]}")
        md_ax.set_title(f"Micro-Doppler ({gate}, {spec.layout}, window "
                        f"{1e3 * m['window_s']:.3g} ms)")
        md_indicator_line = None

    if md_indicator_line is not None:
        try:
            md_indicator_line.remove()
        except Exception:
            pass
    md_indicator_line = md_ax.axvline(_md_frame_time(frame_idx), color="red", alpha=0.7,
                                      linestyle="--", linewidth=1)
    md_indicator_line.set_gid(frame_marker.GID)
    md_canvas.draw_idle()


def _md_redraw():
    """Envelopes on/off: redraw from what was computed, without recomputing."""
    global md_im
    if (last_md_meta or {}).get("result") is not None:
        md_im = None
        update_md_plot(_current_frame_index())


def _md_data_for_export():
    """Spectrogram in dB: one row per time column, the first column the time."""
    meta = last_md_meta or {}
    if meta.get("result") is None:
        return None
    img, extent = meta["image"], meta["extent"]
    n = img.shape[1]
    step = (extent[1] - extent[0]) / n
    t = extent[0] + step * (np.arange(n) + 0.5)
    v = meta["result"].spec.velocity_m_s
    return (np.column_stack([t, img.T]),
            "time_s," + ",".join(f"v{x:+.3f}" for x in v))


def _export_md_swings():
    res = (last_md_meta or {}).get("result")
    if res is None:
        messagebox.showwarning("Export", "Run Micro-Doppler first.")
        return
    path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")],
                                        title="Save the swings as CSV")
    if not path:
        return
    try:
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(limb_motion.SWING_FIELDS))
            w.writeheader()
            for row in limb_motion.swing_rows(res.swings):
                w.writerow(row)
        messagebox.showinfo("Export", f"{len(res.swings)} swings saved to {path}")
    except Exception as e:
        messagebox.showerror("Export", f"Error saving CSV: {e}")


def _export_md_envelopes():
    res = (last_md_meta or {}).get("result")
    if res is None:
        messagebox.showwarning("Export", "Run Micro-Doppler first.")
        return
    export_to_csv(limb_motion.envelope_table(res.env), "Micro-Doppler envelopes")

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
    clear_settings_changed("Movement Map")
    global last_mov_map, last_mov_meta, mov_im, mov_cbar, last_mov_ra_cube, last_mov_ra_meta
    cube = ensure_cube_loaded()
    if cube is None: return
    
    try:
        # We need Range-Azimuth maps for all frames
        # This can be slow, so reuse the cached cube when one is available.
        ra_cube = last_mov_ra_cube
        if ra_cube is None or ra_cube.shape[0] < 4:
            # Recompute Range-Azimuth map: the Angle of Arrival tab's beamformer, as the
            # Range-Angle view (MIMO mode, method, TDM Doppler compensation)
            ra_cube, meta_2d, _ = _range_azimuth_frames(cube, "Movement Map", remove_static=True)
            # Cache it so we don't recompute if we just change movement map settings
            last_mov_ra_cube, last_mov_ra_meta = ra_cube, meta_2d
        else:
            meta_2d = last_mov_ra_meta
            print("[Movement Map] Using cached Range-Azimuth maps.")

        F, R, A = ra_cube.shape
        
        # Get Temporal FFT params
        band = mov_band_var.get() # "Movement", "Breathing", "HR"
        accum = mov_accum_var.get() # "Full", "Sliding Window"
        win_size = int(mov_win_var.get())
        
        # Frequency analysis
        fps = derive_fps(last_params, "Movement Map")
        
        # Define bands (same as in 1D View)
        # Movement: 0.67-0.80 and >3.50 Hz
        # Breathing: 0.08-0.67 Hz (5-40 BPM)
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
        # For now, let's invert dB - block by block where it is used: the whole cube of a
        # 3-min take at 100 fps is 3.3 GB already in dB
        def ra_linear(sl):
            return 10 ** (ra_cube[sl] / 10.0)
        
        # Temporal FFT along Frame axis
        # sig shape: (F, R, A)
        # We want to know how much energy is in a certain band at each (R, A)
        
        br_lo, br_hi = vital_bands.current().resp_hz      # the shared bands (9.1)
        hr_lo, hr_hi = vital_bands.current().heart_hz

        def _band_mask(freqs):
            f_nyq = freqs[-1]
            if band == "Breathing":
                return (freqs >= br_lo) & (freqs <= br_hi)
            if band == "HR":
                return (freqs >= hr_lo) & (freqs <= min(hr_hi, f_nyq))
            # "Movement"
            return ((freqs > br_hi) & (freqs < hr_lo)) | (freqs > hr_hi)

        def _band_energy(segment):
            """Band energy per (range, angle) pixel for one block of frames."""
            n = segment.shape[0]
            seg = segment - np.mean(segment, axis=0, keepdims=True)
            seg = seg * np.hanning(n).astype(seg.dtype).reshape(-1, 1, 1)
            n_fft = int(2**np.ceil(np.log2(max(32, n))))
            spec = np.abs(radar_processing.rfft(seg, n=n_fft, axis=0))
            freqs = np.fft.rfftfreq(n_fft, d=1.0/fps)
            return np.sum(spec[_band_mask(freqs), :, :], axis=0)

        use_sliding = (str(accum).strip().lower().startswith("sliding")
                       and win_size >= 8 and F > win_size)
        if use_sliding:
            # Sliding window used to be a dead UI option: the value was read but never
            # used, so "Sliding Window" behaved exactly like "Full". Now each window is
            # analysed separately and the per-pixel maximum is kept, which keeps short
            # events visible instead of diluting them over the whole recording.
            step = max(1, win_size // 2)
            starts = list(range(0, F - win_size + 1, step))
            print(f"[Movement Map] Sliding window: {len(starts)} windows of {win_size} frames (step {step}).")
            def _sliding(handle):
                acc = None
                for s0 in starts:
                    e = _band_energy(ra_linear(slice(s0, s0 + win_size)))
                    acc = e if acc is None else np.maximum(acc, e)
                    handle.advance()
                    if _cancel_requested:
                        raise ComputationCancelled()
                return acc

            with progress("Movement Map: sliding windows ...", total=len(starts)) as p:
                band_energy = run_in_background(_sliding, p, handle=p)
        else:
            if str(accum).strip().lower().startswith("sliding"):
                print(f"[Movement Map] Sliding window skipped (needs win_size>=8 and "
                      f"more than {win_size} frames, have F={F}); using full recording.")
            # A few range bins at a time, off the window's thread: the FFT of all frames of
            # every pixel at once needed over 20 GB and froze the window for minutes
            step = max(1, int(4e8 // max(1, F * A * 12)))

            def _full(handle):
                out = np.empty((R, A))
                for r0 in range(0, R, step):
                    out[r0:r0 + step] = _band_energy(ra_linear((slice(None), slice(r0, r0 + step))))
                    handle.advance(min(step, R - r0))
                    if _cancel_requested:
                        raise ComputationCancelled()
                return out

            with progress("Movement Map: band energy ...", total=R) as p:
                band_energy = run_in_background(_full, p, handle=p)


        # Normalize for display
        band_energy = 10 * np.log10(band_energy + 1e-12)
        band_energy -= np.max(band_energy)
        
        last_mov_map = band_energy
        last_mov_meta = {
            'angles_deg': meta_2d.get('angles_deg'),
            'range_axis': meta_2d.get('range_axis'),
            'band': band,
            'fps': fps,
            'accumulation': "Sliding Window" if use_sliding else "Full",
            'window_frames': int(win_size) if use_sliding else int(F),
        }
        
        update_movement_map_plot()
    except Exception as e:
        import traceback
        traceback.print_exc()
        report_error("Error", f"Movement Map error: {e}", e)

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
        vmin=vmin, vmax=vmax, cmap=display_cmap("hot")
    )
    show_image("movement", mov_im, vmin, vmax, mov_canvas, default_cmap="hot")
    mov_cbar = mov_fig.colorbar(mov_im, ax=mov_ax, label='Relative Energy [dB]')
    mov_ax.set_xlabel("Angle [deg]")
    mov_ax.set_ylabel("Range [m]")
    band = last_mov_meta.get('band', "")
    mov_ax.set_title(f"Accumulated Movement Map ({band})")
    
    mov_canvas.draw_idle()

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
    # The slider starts at bin 0, the antenna coupling at 0 m: begin at the range with the
    # most moving energy instead (the map has the static scene taken off already).
    r_axis = last_rd_meta.get('range_axis')
    try:
        rbin = int(view1d_rbin_var.get())
    except Exception:
        rbin = 0
    if r_axis is not None and len(r_axis) == Kr and r_axis[min(rbin, Kr - 1)] < MIN_TARGET_RANGE_M:
        k = strongest_range_bin(last_rd_map.mean(axis=(0, 1)), r_axis)
        view1d_rbin_var.set(k)
        view1d_rbin_label_var.set(f"{k}")
        print(f"[1D View] bin {rbin} is the antenna coupling at {r_axis[min(rbin, Kr - 1)]:.2f} m: "
              f"showing the range that moves most, bin {k} ({r_axis[k]:.2f} m)")

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
            fps = derive_fps(last_params, "1D View")
            
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
        fps = derive_fps(last_params, "1D View")
        
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
        #  - Noise/DC:   0.00–0.08 Hz
        #  - Breathing:  0.08–0.67 Hz
        #  - HR:         0.80–3.50 Hz
        #  - Movement:   (0.67–0.80) ULF + >3.50 Hz (rychlé pohyby)
        f_nyq = float(freqs[-1]) if freqs.size > 0 else 0.0
        br_lo, br_hi = vital_bands.current().resp_hz      # the shared bands (9.1)
        hr_lo, hr_hi = vital_bands.current().heart_hz
        masks = {
            "Noise/DC": (freqs >= 0.00) & (freqs < br_lo),
            "Breathing": (freqs >= br_lo) & (freqs <= br_hi),
            "HR": (freqs >= hr_lo) & (freqs <= min(hr_hi, f_nyq)),
            "Movement": ((freqs > br_hi) & (freqs < hr_lo)) | (freqs > hr_hi)
        }

        band_energy = {}
        total = float(np.sum(psd) + 1e-12)
        for k, m in masks.items():
            band_energy[k] = float(np.sum(psd[m])) / total if total > 0 else 0.0

        # Najdi peaky pro BR/HR
        br_f, br_bpm, br_peak = find_bpm(br_lo, br_hi)
        hr_f, hr_bpm, hr_peak = find_bpm(hr_lo, min(hr_hi, f_nyq if f_nyq > 0 else hr_hi))

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

# --- NEW: Vital Signs Functions ---
def init_vital_signs_plot(parent):
    global vital_fig, vital_canvas, vital_ax_br, vital_ax_hr, vital_ax_phase, vital_ax_spec_br, vital_ax_spec_hr, vital_ax_rp
    global vital_br_val_var, vital_hr_val_var, vital_rbin_var, vital_win_br_var, vital_win_hr_var
    global vital_auto_bin_var, vital_rx_var, vital_auto_rx_var
    global vital_clutter_var, vital_dwt_var, vital_kalman_var, vital_bin_idx_var, vital_dist_text_var, vital_bin_slider
    global vital_dc_method_var, vital_phase_method_var
    global vital_roi_width_var, vital_roi_mode_var, vital_win_step_var, vital_conf_thresh_var, vital_hr_stats_var, vital_rr_stats_var, vital_roi_info_var, vital_ax_hr_time, vital_ax_rr_time
    global vital_toolbar, vital_plot_manager

    # All the controls in one strip that wraps with the window (flow_layout.FlowFrame).
    # They were four fixed rows, and at 1024 px the first lost its band limits and RX
    # box, the second its ROI read-out.
    controls = FlowFrame(parent)
    controls.pack(side="top", fill="x", padx=8, pady=4)
    vital_run_group = controls.group()
    register_run_button("Vital Signs",
        theme.Button(vital_run_group, text="Run Vital Signs Analysis", command=run_vital_signs)).pack(side="left")

    _g = controls.group()
    ttk.Label(_g, text="BR Win [s]:").pack(side="left", padx=(0, 2))
    vital_win_br_var = tk.StringVar(value=f"{vitalsigns.DEFAULT_CONFIG['win_br']:g}")
    tk.Spinbox(_g, from_=5, to=120, textvariable=vital_win_br_var, width=5).pack(side="left")
    ttk.Label(_g, text="HR Win [s]:").pack(side="left", padx=(10, 2))
    vital_win_hr_var = tk.StringVar(value=f"{vitalsigns.DEFAULT_CONFIG['win_hr']:g}")
    tk.Spinbox(_g, from_=5, to=120, textvariable=vital_win_hr_var, width=5).pack(side="left")
    ttk.Label(_g, text="Step [s]:").pack(side="left", padx=(10, 2))
    vital_win_step_var = tk.StringVar(value=f"{vitalsigns.DEFAULT_CONFIG['win_step']:g}")
    tk.Spinbox(_g, from_=0.1, to=10, increment=0.1, textvariable=vital_win_step_var, width=4).pack(side="left")

    # Breathing and heart search bands, shared with the AI Vital and HR tabs
    # (vital_bands.py); they replace the old "Resting" checkbox.
    vital_bands.tk_controls(controls, on_change=lambda _b: mark_settings_changed("Vital Signs"))

    # RX selection
    _g = controls.group()
    ttk.Label(_g, text="RX:").pack(side="left", padx=(0, 2))
    vital_rx_var = tk.StringVar(value="0")
    tk.Spinbox(_g, from_=0, to=15, textvariable=vital_rx_var, width=3).pack(side="left")
    vital_auto_rx_var = tk.BooleanVar(value=vitalsigns.DEFAULT_CONFIG['auto_rx'])
    tk.Checkbutton(_g, text="Auto", variable=vital_auto_rx_var).pack(side="left")

    # Stats Area
    stats_frame = controls.group()
    if vital_rr_stats_var is None: vital_rr_stats_var = tk.StringVar(value="RR: -- (mean: --, min: --, max: --)")
    if vital_hr_stats_var is None: vital_hr_stats_var = tk.StringVar(value="HR: -- (mean: --, min: --, max: --)")
    ttk.Label(stats_frame, textvariable=vital_rr_stats_var, font=theme.font(8), foreground=theme.BLUE).pack(side="top", anchor="w")
    ttk.Label(stats_frame, textvariable=vital_hr_stats_var, font=theme.font(8), foreground=theme.RED).pack(side="top", anchor="w")

    # ROI & Bin Selection
    _g = controls.group()
    ttk.Label(_g, text="Range Bin:").pack(side="left", padx=(0, 2))
    if vital_bin_idx_var is None:
        vital_bin_idx_var = tk.IntVar(value=0)
    vital_bin_slider = theme.Slider(
        _g, from_=0, to=100, orient="horizontal",
        command=on_vital_bin_change,
        state="disabled", length=250,
        variable=vital_bin_idx_var, showvalue=False
    )
    vital_bin_slider.pack(side="left", padx=2)

    if vital_dist_text_var is None:
        vital_dist_text_var = tk.StringVar(value="bin 0 / 0.00 m")
    ttk.Label(_g, textvariable=vital_dist_text_var, width=15, font=theme.font(9, "bold")).pack(side="left", padx=5)

    _g = controls.group()
    ttk.Label(_g, text="ROI half width:").pack(side="left", padx=(0, 2))
    if vital_roi_width_var is None: vital_roi_width_var = tk.IntVar(value=vitalsigns.DEFAULT_CONFIG['roi_hw'])
    tk.Spinbox(_g, from_=0, to=10, textvariable=vital_roi_width_var, width=3,
               command=lambda: mark_settings_changed("Vital Signs")).pack(side="left")
    ttk.Label(_g, text="ROI Mode:").pack(side="left", padx=(10, 2))
    if vital_roi_mode_var is None: vital_roi_mode_var = tk.StringVar(value=vitalsigns.DEFAULT_CONFIG['roi_mode'])
    cb_roi = ttk.Combobox(_g, textvariable=vital_roi_mode_var, values=["Single bin", "Best bin in ROI", "Weighted ROI average"], width=18, state="readonly")
    cb_roi.pack(side="left")
    cb_roi.bind("<<ComboboxSelected>>", lambda e: run_vital_signs() if last_params else None)

    if vital_roi_info_var is None: vital_roi_info_var = tk.StringVar(value="ROI: --")
    ttk.Label(controls.group(), textvariable=vital_roi_info_var, font=theme.font(9)).pack(side="left")

    # Advanced Options
    _g = controls.group()
    if vital_auto_bin_var is None:
        # on: the slider starts at bin 0, the antenna coupling at 0 m - never a person
        vital_auto_bin_var = tk.BooleanVar(value=vitalsigns.DEFAULT_CONFIG['auto_bin'])
    tk.Checkbutton(_g, text="Auto Bin", variable=vital_auto_bin_var, command=lambda: mark_settings_changed("Vital Signs")).pack(side="left", padx=(0, 10))

    if vital_clutter_var is None:
        vital_clutter_var = tk.BooleanVar(value=vitalsigns.DEFAULT_CONFIG['clutter_removal'])
    tk.Checkbutton(_g, text="Clutter Removal", variable=vital_clutter_var, command=lambda: mark_settings_changed("Vital Signs")).pack(side="left")

    if vital_kalman_var is None:
        vital_kalman_var = tk.BooleanVar(value=vitalsigns.DEFAULT_CONFIG['kalman_filter'])
    tk.Checkbutton(_g, text="Kalman Filter", variable=vital_kalman_var, command=lambda: mark_settings_changed("Vital Signs")).pack(side="left")

    if vital_dwt_var is None:
        vital_dwt_var = tk.BooleanVar(value=vitalsigns.DEFAULT_CONFIG['wiener_denoising'])
    tk.Checkbutton(_g, text="Wiener Denoising", variable=vital_dwt_var, command=lambda: mark_settings_changed("Vital Signs")).pack(side="left")

    # Confidence is the peak-to-average power ratio inside the band, so the threshold
    # is a plain "how many times above the band average the peak has to be". It used to
    # be a share of the band energy, which depended on the FFT length rather than on the
    # signal, and at the old default of 0.1 not even a noiseless sinusoid qualified.
    _g = controls.group()
    ttk.Label(_g, text="Min peak/avg:").pack(side="left", padx=(0, 2))
    if vital_conf_thresh_var is None:
        vital_conf_thresh_var = tk.StringVar(value=str(vitalsigns.DEFAULT_CONFIG['conf_thresh']))
    tk.Spinbox(_g, from_=1, to=30, increment=0.5, textvariable=vital_conf_thresh_var,
               width=5).pack(side="left")

    # Signal Processing Methods
    _g = controls.group()
    ttk.Label(_g, text="IQ DC Correction:").pack(side="left", padx=(0, 2))
    if vital_dc_method_var is None: vital_dc_method_var = tk.StringVar(value=vitalsigns.DEFAULT_CONFIG['dc_method'])
    cb_dc = ttk.Combobox(_g, textvariable=vital_dc_method_var, values=["none", "mean", "moving_average"], width=15, state="readonly")
    cb_dc.pack(side="left")
    cb_dc.bind("<<ComboboxSelected>>", lambda e: run_vital_signs() if last_params else None)

    _g = controls.group()
    ttk.Label(_g, text="Phase Extraction:").pack(side="left", padx=(0, 2))
    if vital_phase_method_var is None: vital_phase_method_var = tk.StringVar(value=vitalsigns.DEFAULT_CONFIG['phase_method'])
    cb_ph = ttk.Combobox(_g, textvariable=vital_phase_method_var, values=["unwrap", "dacm"], width=10, state="readonly")
    cb_ph.pack(side="left")
    cb_ph.bind("<<ComboboxSelected>>", lambda e: run_vital_signs() if last_params else None)

    # Figure
    vital_fig = Figure(figsize=(10, 12), dpi=100)
    gs = vital_fig.add_gridspec(6, 2, height_ratios=[1, 1, 1.2, 1, 1.2, 1])
    vital_ax_rp = vital_fig.add_subplot(gs[0, :])
    vital_ax_phase = vital_fig.add_subplot(gs[1, :])
    
    vital_ax_rr_time = vital_fig.add_subplot(gs[2, :])
    vital_ax_br = vital_fig.add_subplot(gs[3, 0])
    vital_ax_spec_br = vital_fig.add_subplot(gs[3, 1])
    
    vital_ax_hr_time = vital_fig.add_subplot(gs[4, :])
    vital_ax_hr = vital_fig.add_subplot(gs[5, 0])
    vital_ax_spec_hr = vital_fig.add_subplot(gs[5, 1])

    # laid out again at every draw: done once at 1200 px, a lower window put the
    # labels of one row on top of the next
    vital_fig.set_layout_engine("tight")

    # Interactive canvas with navigation toolbar (Home/Back/Forward/Pan/Zoom/Save)
    plot_frame = ttk.Frame(parent)
    plot_frame.pack(side="top", fill="both", expand=True)
    vital_canvas, vital_toolbar = create_interactive_canvas(plot_frame, vital_fig)
    # Info/Export - here, once the figure exists (it was passed as None and the PNG
    # export failed, the CSV one had no data to give)
    add_card_header_extras(vital_run_group, "Vital Signs", INFO_VITAL_SIGNS, vital_fig,
                           _vital_signals_for_export,
                           extra_items=[("Rates over time (CSV)", _export_vital_rates)],
                           manual_section="Vitals")

    # X-axis span-zoom / reset controls (synchronized time axes)
    vital_plot_manager = PlotInteractivityManager(vital_canvas)
    add_plot_control_bar(controls, vital_plot_manager)      # more groups of the strip


def _vital_signals_for_export():
    """(time, displacement, breathing, heart) per frame, with a header - or None."""
    d = last_vital_data
    if d is None:
        return None
    cols = [np.asarray(d['time']), np.asarray(d['phase']), np.asarray(d['sig_br']),
            np.asarray(d['sig_hr'])]
    n = min(len(c) for c in cols)
    return (np.column_stack([c[:n] for c in cols]),
            "time_s,displacement_m,breathing_signal,heart_signal")


def _export_vital_rates():
    """Breathing and heart rate per analysis window, one row per window."""
    d = last_vital_data
    if d is None:
        messagebox.showwarning("Export", "Run the Vital Signs analysis first.")
        return
    path = filedialog.asksaveasfilename(defaultextension=".csv",
                                        filetypes=[("CSV files", "*.csv")],
                                        title="Save breathing and heart rates as CSV")
    if not path:
        return
    th = float(d.get('conf_th', vitalsigns.DEFAULT_CONF_THRESH))
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("kind,time_s,rate_per_min,peak_to_average,valid\n")
            for kind, t, v, c in (("breathing", d['rr_times'], d['rr_values'], d['rr_confs']),
                                  ("heart", d['hr_times'], d['hr_values'], d['hr_confs'])):
                for ti, vi, ci in zip(t, v, c):
                    f.write(f"{kind},{ti:.3f},{vi:.2f},{ci:.2f},{int(ci >= th)}\n")
        messagebox.showinfo("Export", f"Rates saved to {path}")
    except Exception as e:
        messagebox.showerror("Export", f"Error saving CSV: {e}")


def on_vital_bin_change(val):
    global last_params, vital_dist_text_var, vital_bin_idx_var, last_vital_rf_meta, vital_auto_bin_var, in_run_vital_signs
    global vital_roi_info_var, vital_roi_width_var
    if in_run_vital_signs: return
    try:
        rbin = int(float(val))
        vital_bin_idx_var.set(rbin)
        
        # If user moves slider, disable auto-bin
        if vital_auto_bin_var and vital_auto_bin_var.get():
            vital_auto_bin_var.set(False)
            
        # Immediate label update if we have meta
        if last_params and last_vital_rf_meta is not None:
            r_axis = last_vital_rf_meta.get('range_m')
            if r_axis is None:
                r_axis = last_vital_rf_meta.get('range_axis')
            
            if r_axis is not None and rbin < len(r_axis):
                dist_m = r_axis[rbin]
                vital_dist_text_var.set(f"bin {rbin} / {dist_m:.2f} m")
                
                # Update ROI info label
                roi_hw = vital_roi_width_var.get()
                R = len(r_axis)
                roi_bins = np.arange(max(0, rbin - roi_hw), min(R, rbin + roi_hw + 1))
                r_start = r_axis[roi_bins[0]]
                r_end = r_axis[roi_bins[-1]]
                vital_roi_info_var.set(f"ROI: bins {roi_bins[0]}-{roi_bins[-1]} / {r_start:.2f}-{r_end:.2f} m")
        
        # Recomputing here fired on every pixel of the slider drag; the analysis now
        # waits for the Run button.
        mark_settings_changed("Vital Signs")
    except Exception:
        pass

def run_vital_signs():
    clear_settings_changed("Vital Signs")
    global last_cube, last_params, last_vital_data
    global last_vital_rfft_cube, last_vital_rf_meta, last_vital_cube_ptr
    global last_vital_clutter_cube, last_vital_clutter_val, in_run_vital_signs
    global vital_auto_bin_var, vital_rx_var, vital_auto_rx_var, vital_bin_idx_var
    global vital_clutter_var, vital_dwt_var, vital_kalman_var, vital_bin_slider, vital_dist_text_var
    global vital_dc_method_var, vital_phase_method_var
    global vital_roi_width_var, vital_roi_mode_var, vital_win_step_var, vital_conf_thresh_var
    global vital_hr_stats_var, vital_rr_stats_var, vital_roi_info_var
    global vital_win_br_var, vital_win_hr_var

    if ensure_cube_loaded() is None: return

    in_run_vital_signs = True
    try:
        params = last_params
        cube = last_cube 
        
        if last_vital_rfft_cube is not None and last_vital_cube_ptr is last_cube:
            rfft_cube_base = last_vital_rfft_cube
            rf_meta = last_vital_rf_meta
        else:
            rfft_cube_base, rf_meta = compute_range_fft_cube(cube, params, return_complex=True,
                                                             n_fft_range=range_fft_size())
            last_vital_rfft_cube = rfft_cube_base
            last_vital_rf_meta = rf_meta
            last_vital_cube_ptr = last_cube
            last_vital_clutter_cube = None
            last_vital_clutter_val = None

        F, C, RX, R = rfft_cube_base.shape
        
        if vital_bin_slider is not None:
            try:
                if int(vital_bin_slider.cget('to')) != R - 1:
                    vital_bin_slider.configure(to=R - 1)
                if vital_bin_slider.cget('state') != "normal":
                    vital_bin_slider.configure(state="normal")
            except: pass
            
        fps = derive_fps(params, "Vital Signs")
        
        r_axis = rf_meta.get('range_m')
        if r_axis is None: r_axis = rf_meta.get('range_axis')
        if r_axis is None: return

        clutter_on = vital_clutter_var.get()
        if last_vital_clutter_cube is not None and last_vital_clutter_val == clutter_on:
            rfft_cube = last_vital_clutter_cube
        else:
            if clutter_on:
                rfft_cube = rfft_cube_base - np.mean(rfft_cube_base, axis=0, keepdims=True)
            else:
                rfft_cube = rfft_cube_base
            last_vital_clutter_cube = rfft_cube
            last_vital_clutter_val = clutter_on

        config = {
            'auto_bin': vital_auto_bin_var.get(),
            'bin_idx': vital_bin_idx_var.get(),
            'roi_hw': vital_roi_width_var.get(),
            'roi_mode': vital_roi_mode_var.get(),
            'auto_rx': vital_auto_rx_var.get(),
            'rx_idx': int(vital_rx_var.get()) if vital_rx_var.get() else 0,
            'dc_method': vital_dc_method_var.get() if vital_dc_method_var else 'mean',
            'phase_method': vital_phase_method_var.get() if vital_phase_method_var else 'unwrap',
            'clutter_removal': False, # Already handled by caching above
            'kalman_filter': vital_kalman_var.get(),
            'wiener_denoising': vital_dwt_var.get(),
            'win_br': float(vital_win_br_var.get()) if vital_win_br_var.get() else 30.0,
            'win_hr': float(vital_win_hr_var.get()) if vital_win_hr_var.get() else vitalsigns.DEFAULT_HR_WIN_S,
            'win_step': float(vital_win_step_var.get()) if vital_win_step_var.get() else 1.0,
            'conf_thresh': float(vital_conf_thresh_var.get()) if vital_conf_thresh_var.get() else vitalsigns.DEFAULT_CONF_THRESH,
            'resp_bpm': vital_bands.current().resp_bpm,
            'heart_bpm': vital_bands.current().heart_bpm,
            # Measured on the raw cube at load time, so the plots can mark the
            # stretches that were recorded over lost packets.
            'damaged_frames': last_frame_damaged
        }

        with progress("Analysing vital signs ..."):
            results = run_in_background(
                vitalsigns.process_vital_signs_logic, rfft_cube, params, fps, r_axis, config)
        t0 = segment_t0_s()           # capture time, also inside a segment (roadmap 9.6)
        if t0:
            for key in ('time', 'rr_times', 'hr_times'):
                results[key] = np.asarray(results[key]) + t0
        
        # Sync GUI variables
        vital_bin_idx_var.set(results['best_bin'])
        vital_dist_text_var.set(f"bin {results['best_bin']} / {results['dist_m']:.2f} m")
        vital_rx_var.set(str(results['actual_best_rx']))
        
        r_start, r_end = results['roi_dist_range']
        roi_bins = results['roi_bins']
        vital_roi_info_var.set(f"ROI: bins {roi_bins[0]}-{roi_bins[-1]} / {r_start:.2f}-{r_end:.2f} m")
        
        rr_curr, rr_mean, rr_min, rr_max = results['rr_stats']
        hr_curr, hr_mean, hr_min, hr_max = results['hr_stats']
        
        # When no window clears the confidence bar the single-record estimate is still
        # shown, but say so and say why - "(identifying...)" left it looking as if the
        # analysis had not finished, with an empty rate-over-time graph beside it and no
        # hint that a longer recording or a lower bar is what it needs.
        def _rate_text(label, curr, mean, lo, hi, fallback, confs, th):
            if curr is not None:
                return (f"Current {label}: {curr:.1f} | Mean: {mean:.1f} | "
                        f"Min: {lo:.1f} | Max: {hi:.1f}")
            n = len(confs)
            best = float(np.max(confs)) if n else 0.0
            return (f"Current {label}: {fallback:.1f} (low confidence - 0 of {n} "
                    f"windows reached {th:g}, best {best:.1f}; try a longer recording "
                    f"or a lower Min peak/avg)")

        vital_rr_stats_var.set(_rate_text("RR", rr_curr, rr_mean, rr_min, rr_max,
                                          results['br_bpm'], results['rr_confs'],
                                          results['conf_th']))
        vital_hr_stats_var.set(_rate_text("HR", hr_curr, hr_mean, hr_min, hr_max,
                                          results['hr_bpm'], results['hr_confs'],
                                          results['conf_th']))
        
        last_vital_data = results
        update_vital_signs_plot()
    finally:
        in_run_vital_signs = False

# --- NEW: HR Detection Integration ---
def init_hr_detection(parent):
    def get_hr_data():
        global last_cube, last_params
        if ensure_cube_loaded() is None: return None
        rfft_cube, rf_meta = compute_range_fft_cube(last_cube, last_params, return_complex=True,
                                                    n_fft_range=range_fft_size())
        fps = derive_fps(last_params, "HR Analysis")
        r_axis = rf_meta.get('range_m')
        if r_axis is None: r_axis = rf_meta.get('range_axis')
        
        # Range Profile for visualization
        rp = np.mean(np.abs(rfft_cube), axis=(0, 1, 2))
        return rfft_cube, last_params, fps, r_axis, rp

    def get_raw():
        """The decoded segment for the Moving person part (moving_hr.py)."""
        if ensure_cube_loaded() is None:
            return None
        return (last_cube, last_params, derive_fps(last_params, "Moving HR"), segment_t0_s(),
                os.path.basename(bin_var.get()))

    def run_moving(fn, **kw):
        with progress("Moving HR ...", total=100) as p:
            return run_in_background(fn, progress_cb=p.callback(), handle=p, **kw)

    return hr_gui.HRDetectionTab(parent, get_hr_data,
                                 moving=dict(get_raw=get_raw, run=run_moving,
                                             report_error=report_error))

def plot_with_damage(ax, t, y, damaged, color=None, label=None, alpha=1.0,
                     shade=True, **kw):
    """
    Draw a time series, dashing the stretches recorded over lost packets.

    Where the DCA1000 filled in zeros the waveform is not a measurement, but on a plot
    it looks exactly like one. Those stretches are drawn dashed and lightly shaded, so a
    reading taken across them can be recognised for what it is. A capture with no loss
    is drawn as one ordinary solid line and gets no shading, so nothing changes for a
    clean recording.

    Returns True when anything was marked.
    """
    t = np.asarray(t)
    y = np.asarray(y)
    dmg = np.asarray(damaged, dtype=bool) if damaged is not None else None
    if dmg is None or dmg.size != y.size or not dmg.any():
        ax.plot(t, y, color=color, label=label, alpha=alpha, **kw)
        return False

    # Resolve the colour once. Left to matplotlib, every segment would take the next
    # entry from the property cycle and the trace came out as a rainbow.
    if color is None:
        color = next(ax._get_lines.prop_cycler)['color'] \
            if hasattr(ax._get_lines, 'prop_cycler') else ax._get_lines.get_next_color()

    spans = vitalsigns.damaged_spans(dmg)
    if shade:
        for a, b in spans:
            # One sample wide at minimum, so a single lost frame is still visible.
            x0 = t[a]
            x1 = t[min(b, len(t) - 1)]
            if x1 <= x0 and len(t) > 1:
                x1 = x0 + (t[1] - t[0])
            ax.axvspan(x0, x1, color='red', alpha=0.10, lw=0, zorder=0)

    # Solid over good runs, dashed over damaged ones. Each run is extended by one
    # sample so the line stays connected across the boundaries.
    first_solid, first_dash = True, True
    for value, style in ((False, '-'), (True, '--')):
        for a, b in vitalsigns.damaged_spans(dmg == value):
            seg = slice(max(0, a - 1), min(len(t), b + 1))
            lbl = None
            if value and first_dash:
                lbl, first_dash = "lost packets", False
            elif (not value) and first_solid:
                lbl, first_solid = label, False
            ax.plot(t[seg], y[seg], color=color, linestyle=style, label=lbl,
                    alpha=alpha, **kw)
    return True


def update_vital_signs_plot():
    global vital_ax_br, vital_ax_hr, vital_ax_phase, vital_ax_spec_br, vital_ax_spec_hr, vital_ax_rp, vital_canvas, last_vital_data
    global vital_ax_hr_time, vital_ax_rr_time, vital_plot_manager
    if last_vital_data is None: return
    d = last_vital_data
    
    # 0. Range Profile
    vital_ax_rp.clear()
    vital_ax_rp.plot(d['r_axis'], d['rp_db'], color='black', alpha=0.7)
    vital_ax_rp.axvline(d['dist_m'], color='red', linestyle='-', linewidth=2, label='Center Bin')
    
    roi_str = ""
    if 'roi_bins' in d and 'roi_dist_range' in d:
        vital_ax_rp.axvspan(d['roi_dist_range'][0], d['roi_dist_range'][1], color='red', alpha=0.15, label='ROI')
        roi_b_start = d['roi_bins'][0]
        roi_b_end = d['roi_bins'][-1]
        roi_str = f", ROI {roi_b_start}-{roi_b_end} / {d['roi_dist_range'][0]:.2f}-{d['roi_dist_range'][1]:.2f} m"
    
    vital_ax_rp.set_title(f"Range Profile (bin {d['best_bin']} / {d['dist_m']:.2f} m{roi_str})")
    vital_ax_rp.set_xlabel("Distance [m]")
    vital_ax_rp.set_ylabel("Magnitude [dB]")
    vital_ax_rp.grid(True, alpha=0.3)

    # 1. Phase/Displacement
    dmg = d.get('damaged_frames')
    dmg_pct = 100.0 * float(d.get('damaged_fraction', 0.0))
    vital_ax_phase.clear()
    marked = plot_with_damage(vital_ax_phase, d['time'], d['phase'], dmg,
                              label='Original',
                              alpha=0.4 if d['phase_filtered'] is not None else 1.0)
    if d['phase_filtered'] is not None:
        plot_with_damage(vital_ax_phase, d['time'], d['phase_filtered'], dmg,
                         label='Filtered', color='green', shade=False)
    vital_ax_phase.set_title(
        "Relative Displacement" if not marked else
        f"Relative Displacement - {dmg_pct:.1f} % of frames lost in transfer (dashed)")
    vital_ax_phase.set_xlabel("Time [s]")
    vital_ax_phase.set_ylabel("Displacement [m]")
    vital_ax_phase.legend(loc='upper right', fontsize='x-small')
    vital_ax_phase.grid(True, alpha=0.3)
    
    # 2. Breathing Wave & Spectrum
    vital_ax_br.clear()
    plot_with_damage(vital_ax_br, d['time'], d['sig_br'], dmg, color='blue')
    vital_ax_br.set_title("Respiratory Waveform"
                          + (" (dashed = lost packets)" if marked else ""))
    vital_ax_br.set_xlabel("Time [s]")
    vital_ax_br.set_ylabel("Amplitude")
    vital_ax_br.grid(True, alpha=0.3)
    
    vital_ax_spec_br.clear()
    vital_ax_spec_br.plot(d['f_axis_br'], d['psd_br'], color='blue')
    vital_ax_spec_br.axvline(d['br_bpm']/60.0, color='red', linestyle='--')
    vital_ax_spec_br.set_xlim(d['f_resp_range'])
    vital_ax_spec_br.set_title(f"BR Spectrum ({d['br_bpm']:.1f} BPM)")
    vital_ax_spec_br.set_xlabel("Frequency [Hz]")
    vital_ax_spec_br.set_ylabel("Magnitude")
    vital_ax_spec_br.grid(True, alpha=0.3)
    
    # 3. RR Over Time
    vital_ax_rr_time.clear()
    if len(d['rr_times']) > 0:
        mask = d['rr_confs'] >= d['conf_th']
        vital_ax_rr_time.plot(d['rr_times'], d['rr_values'], color='blue', alpha=0.2)
        # windows below Min peak/avg as rings: a single window (a 30 s take, 30 s
        # windows) left the plot empty, as if nothing had been computed
        vital_ax_rr_time.scatter(d['rr_times'][~mask], d['rr_values'][~mask], s=15,
                                 facecolors='none', edgecolors='blue', alpha=0.4)
        if np.any(mask):
            vital_ax_rr_time.scatter(d['rr_times'][mask], d['rr_values'][mask], color='blue', s=15, label='Valid')
            vital_ax_rr_time.axhline(np.mean(d['rr_values'][mask]), color='blue', linestyle='--', alpha=0.5, label='Mean')
    vital_ax_rr_time.set_title("Respiration Rate Over Time")
    vital_ax_rr_time.set_xlabel("Time [s]")
    vital_ax_rr_time.set_ylabel("RR [breaths/min]")
    vital_ax_rr_time.grid(True, alpha=0.3)

    # 4. Heart Wave & Spectrum
    vital_ax_hr.clear()
    plot_with_damage(vital_ax_hr, d['time'], d['sig_hr'], dmg, color='red')
    vital_ax_hr.set_title("Cardiac Waveform"
                          + (" (dashed = lost packets)" if marked else ""))
    vital_ax_hr.set_xlabel("Time [s]")
    vital_ax_hr.set_ylabel("Amplitude")
    vital_ax_hr.grid(True, alpha=0.3)
    
    vital_ax_spec_hr.clear()
    vital_ax_spec_hr.plot(d['f_axis_hr'], d['psd_hr'], color='red')
    vital_ax_spec_hr.axvline(d['hr_bpm']/60.0, color='black', linestyle='--')
    vital_ax_spec_hr.set_xlim(d['f_heart_range'])
    vital_ax_spec_hr.set_title(f"HR Spectrum ({d['hr_bpm']:.1f} BPM)")
    vital_ax_spec_hr.set_xlabel("Frequency [Hz]")
    vital_ax_spec_hr.set_ylabel("Magnitude")
    vital_ax_spec_hr.grid(True, alpha=0.3)
    
    # 5. HR Over Time
    vital_ax_hr_time.clear()
    if len(d['hr_times']) > 0:
        mask = d['hr_confs'] >= d['conf_th']
        vital_ax_hr_time.plot(d['hr_times'], d['hr_values'], color='red', alpha=0.2)
        vital_ax_hr_time.scatter(d['hr_times'][~mask], d['hr_values'][~mask], s=15,
                                 facecolors='none', edgecolors='red', alpha=0.4)
        if np.any(mask):
            vital_ax_hr_time.scatter(d['hr_times'][mask], d['hr_values'][mask], color='red', s=15, label='Valid')
            vital_ax_hr_time.axhline(np.mean(d['hr_values'][mask]), color='red', linestyle='--', alpha=0.5, label='Mean')
    vital_ax_hr_time.set_title("Heart Rate Over Time")
    vital_ax_hr_time.set_xlabel("Time [s]")
    vital_ax_hr_time.set_ylabel("HR [BPM]")
    vital_ax_hr_time.grid(True, alpha=0.3)
    
    vital_fig.tight_layout()

    # Re-attach interactivity after redraw: SpanSelectors on time axes
    # (synchronized X zoom) + store original view limits for Reset View.
    if vital_plot_manager is not None:
        vital_plot_manager.register_time_axes([
            vital_ax_phase, vital_ax_br, vital_ax_rr_time, vital_ax_hr, vital_ax_hr_time
        ])
    vital_canvas.draw_idle()

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

_cfar_settings = None     # the settings of the last Run, reused for every frame shown
_cfar_frame = None        # the frame last_detections / last_detection_peaks belong to


def _cfar_frame_detections(frame_idx: int, s: dict):
    """CFAR detections and their physical targets for one frame of the RD map."""
    rd_frame = last_rd_map[frame_idx]  # (Doppler, Range)
    # along range the cells are resolution cells: a zero-padded range FFT (80 samples on
    # 256 points) has 3.2 bins in each, and 2 guard bins sat inside a target's own peak
    osr = s.get('range_osr', 1.0)
    detections = cfar_2d_ca(rd_frame, guard_cells=s['guard'], training_cells=s['training'],
                            threshold_factor=s['threshold'], method=s['method'],
                            guard_cells_range=int(np.ceil(s['guard'] * osr)),
                            training_cells_range=int(np.ceil(s['training'] * osr)))
    peaks = extract_peaks(detections, rd_frame, min_distance=3)
    peaks_physical = detections_to_range_velocity(
        peaks, last_rd_meta.get('range_axis'), last_rd_meta.get('velocity_axis'))
    # Velocity beyond +-v_max from the TX phases of a TDM-MIMO capture (velocity.py)
    if s.get('params_v') is not None and last_cube is not None and frame_idx < last_cube.shape[0]:
        velocity.extend_detections(peaks_physical, last_cube[frame_idx], s['params_v'],
                                   last_rd_meta, v_limit=s.get('v_limit'))
    return detections, peaks_physical


def range_oversampling() -> float:
    """Range FFT bins per resolution cell of the last Range-Doppler map (1 without padding)."""
    try:
        n_fft = int(last_rd_meta.get('n_fft_range') or 0)
        n = int(last_params.get('numAdcSamples') or 0)
        return max(1.0, n_fft / n) if n_fft and n else 1.0
    except Exception:
        return 1.0


def run_cfar_detection():
    clear_settings_changed("CFAR")
    global last_detections, last_detection_peaks, detection_im, detection_cbar, detection_scatter
    global _cfar_settings, _cfar_frame

    if last_rd_map is None or last_rd_meta is None:
        messagebox.showerror("Error", "Please run Range-Doppler first.")
        return

    try:
        # CFAR parameters, kept so every frame shown later is detected the same way -
        # the detections of the frame Run was pressed on used to stay on the plot
        # whatever frame the slider moved to (and in the exported video)
        s = {'guard': int(cfar_guard_var.get()), 'training': int(cfar_training_var.get()),
             'threshold': float(cfar_threshold_var.get()), 'method': cfar_method_var.get(),
             'params_v': None, 'v_limit': None, 'range_osr': range_oversampling()}
        unfold = cfar_unfold_var is not None and cfar_unfold_var.get()
        if unfold and last_cube is not None:
            params_v = {**last_params, 'antenna_config': analysis_antenna_config(),
                        **calibration_params()}
            if velocity.tdm_layout(params_v, last_cube.shape[1]) is None:
                print("[CFAR] Velocity unfolding needs a TDM-MIMO capture (one TX per chirp, "
                      "all chirps loaded); skipped.")
            else:
                s['params_v'] = params_v
                try:
                    s['v_limit'] = float(cfar_vlimit_var.get())
                except Exception:
                    s['v_limit'] = None     # empty: any speed the hypotheses allow

        # Get current frame from slider
        frame_idx = _current_frame_index()
        frame_idx = min(max(frame_idx, 0), last_rd_map.shape[0] - 1)

        with progress("Running CFAR detection ..."):
            detections, peaks_physical = _cfar_frame_detections(frame_idx, s)

        _cfar_settings, _cfar_frame = s, frame_idx
        last_detections = detections
        last_detection_peaks = peaks_physical
        # Update plot
        update_detection_plot(frame_idx)

        # Print detections to console
        if s['range_osr'] > 1.01:
            print(f"[CFAR] the range FFT has {s['range_osr']:.1f} bins per resolution cell: "
                  f"along range {int(np.ceil(s['guard'] * s['range_osr']))} guard and "
                  f"{int(np.ceil(s['training'] * s['range_osr']))} training bins")
        n_dopp = last_rd_map.shape[1]
        if cfar_range_only(n_dopp, s['guard'], s['training']):
            print(f"[CFAR] {n_dopp} velocity bins are fewer than the "
                  f"{2 * (s['guard'] + s['training']) + 1}-cell window: detecting along range only")
        print(f"\n[CFAR Detection] Frame {frame_idx}: {len(peaks_physical)} targets detected")
        for i, peak in enumerate(peaks_physical):
            extra = ""
            if 'velocity_unfolded_ms' in peak:
                if peak['fold_decided']:
                    extra = (f" -> unfolded {peak['velocity_unfolded_ms']:.2f} m/s "
                             f"(fold {peak['fold']:+d}, margin {peak['fold_margin_db']:.1f} dB)")
                elif peak['fold_best'] == 0:
                    extra = f" (not folded, margin only {peak['fold_margin_db']:.1f} dB)"
                else:
                    extra = (f" (fold {peak['fold_best']:+d} = {peak['velocity_best_ms']:.2f} m/s "
                             f"not accepted: margin {peak['fold_margin_db']:.1f} dB or over max |v|)")
            print(f"  Target {i+1}: Range={peak.get('range_m', 'N/A'):.2f} m, "
                  f"Velocity={peak.get('velocity_ms', 'N/A'):.2f} m/s{extra}, "
                  f"Magnitude={peak['magnitude']:.2f}")

    except Exception as e:
        report_error("Error", f"CFAR Detection error: {e}", e)

def update_detection_plot(frame_idx: int):
    global detection_im, detection_cbar, detection_scatter

    # RELAXED: allow RD background update even if no detections computed yet
    if last_rd_map is None:
        return

    f = min(max(frame_idx, 0), last_rd_map.shape[0] - 1)
    rd_frame = last_rd_map[f]
    img_db = 20 * np.log10(rd_frame + 1e-9)

    # The detections belong to the frame shown: another frame is detected with the
    # settings of the last Run (a new RD map clears last_detection_peaks first).
    global last_detections, last_detection_peaks, _cfar_frame
    if _cfar_settings is not None and last_detection_peaks is not None and _cfar_frame != f:
        try:
            last_detections, last_detection_peaks = _cfar_frame_detections(f, _cfar_settings)
            _cfar_frame = f
        except Exception as e:
            print(f"[CFAR] frame {f}: {e}")

    range_axis = last_rd_meta.get('range_axis')
    velocity_axis = last_rd_meta.get('velocity_axis')
    extent = bin_extent(velocity_axis, range_axis)
    vmin = last_rd_meta.get('display_vmin')
    vmax = last_rd_meta.get('display_vmax')

    if detection_im is None:
        detection_ax.clear()
        detection_im = detection_ax.imshow(img_db.T, aspect='auto', origin='lower',
                                           cmap=display_cmap(), extent=extent, vmin=vmin, vmax=vmax,
                                           interpolation=few_bins_interpolation(len(velocity_axis)))
        show_image("cfar", detection_im, vmin, vmax, detection_canvas)
        if detection_cbar is None:
            detection_cbar = detection_fig.colorbar(detection_im, ax=detection_ax)
        detection_cbar.set_label("Magnitude [dB]")
    else:
        detection_im.set_data(img_db.T)
        detection_im.set_extent(extent)
        if vmin is not None and vmax is not None:
            show_image("cfar", detection_im, vmin, vmax, detection_canvas)

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
DPI_SCALE = theme.enable_dpi_awareness()      # sharp on a scaled Windows display (125 %, 200 %)
root = tk.Tk()
theme.apply(root)      # fonts, colours, plot style (theme.py)
from version import __version__ as DARTSCOPE_VERSION
root.title(f"DARTscope {DARTSCOPE_VERSION}")

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

theme.initial_geometry(root, 1400, 900, DPI_SCALE)

bin_var = tk.StringVar()

# NEW: Global frame controls (callbacks stay as-is)
def on_global_frame_change(val):
    try:
        idx = int(float(val))
    except ValueError:
        idx = 0
    apply_global_frame(idx)
_global_play_after_id = None   # pending root.after() handle of the playback loop
_global_play_clock = None      # (t0, frame at t0, frame last shown) of the running playback
_play_meter = Meter()          # how fast it really goes (the corner of the status bar)
_play_mode = "Playback"        # ... and how the frames come: drawn now or from their pictures
_play_shown_at = 0.0           # when the corner was last written


def _update_global_frame():
    global _global_play_after_id, _global_play_clock
    _global_play_after_id = None
    if not global_playing:
        return
    try:
        current_val = int(global_frame_slider.get())
        max_val = int(global_frame_slider.cget("to"))
    except Exception:
        current_val, max_val = 0, 0
    if max_val < 0:
        return
    t = time.perf_counter()
    view = None
    try:
        if main_notebook.tab(main_notebook.select(), "text") == "Analysis":
            view = _view_canvas(analysis_nb.tab(analysis_nb.select(), "text"))
    except Exception:
        view = None
    next_val, _global_play_clock = next_play_frame(
        _global_play_clock, current_val, max_val, t, global_play_interval_ms,
        step=frame_step(global_play_interval_ms) if view is not None else 1,
        cached=(lambda f: frame_cache.has(view, f)) if view is not None else None)
    apply_global_frame(next_val)
    _show_play_rate(next_val)
    spent_ms = (time.perf_counter() - t) * 1000.0
    _global_play_after_id = root.after(max(1, int(global_play_interval_ms - spent_ms)), _update_global_frame)


def _show_play_rate(frame: int):
    """The corner of the status bar while playing: how the frames come, how many are shown
    a second and how fast against the recording - a user sees when it is not real time."""
    global _play_shown_at
    now = time.perf_counter()
    _play_meter.add(now, frame)
    if now - _play_shown_at < 0.4:
        return
    _play_shown_at = now
    r = _play_meter.rates(global_play_interval_ms)
    _playback_status(_play_mode if r is None else
                     f"{_play_mode}  -  {r[0]:.0f} fps shown, {r[1]:.2f}x real time")

def play_global():
    global global_playing, _global_play_clock
    if int(global_frame_slider.cget("to")) <= 0:
        return
    if global_playing:
        # Already running - a second Play would start a parallel timer chain and
        # advance the frame twice per tick.
        return
    global_playing = True
    _global_play_clock = None
    _play_meter.reset()
    global_play_btn.config(state="disabled")
    global_stop_btn.config(state="normal")
    _update_global_frame()

def stop_global():
    global global_playing, _global_play_after_id
    global_playing = False
    # Cancel the pending callback as well: clearing the flag alone left a scheduled
    # tick alive, which resumed the loop if the user pressed Play again quickly.
    if _global_play_after_id is not None:
        try:
            root.after_cancel(_global_play_after_id)
        except Exception:
            pass
        _global_play_after_id = None
    if 'global_play_btn' in globals() and global_play_btn.winfo_exists():
        global_play_btn.config(state="normal")
    if 'global_stop_btn' in globals() and global_stop_btn.winfo_exists():
        global_stop_btn.config(state="disabled")
    _playback_status()

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

SETTLE_MS = 300               # a view shown from its pictures is drawn itself this long after the last frame


def _canvas_of(name: str):
    """A view's canvas kept in a module global, or None before the view was built."""
    return globals().get(name)


def _view_canvas(view: str):
    """The canvas the shown page of an Analysis view draws on (the frame cache's key)."""
    try:
        if view == "Multi Vitals":
            t = multi_vitals_tab
            page = t.viz_nb.tab(t.viz_nb.select(), "text")
            return t.map_canvas if page == "Map" else t.rates_canvas
        if view == "Imaging":
            return imaging_tab.canvases.get(imaging_tab.page())
        return _canvas_of({"Range FFT": "range_fft_canvas", "Profile": "rp_canvas",
                           "Range-Doppler": "rd_canvas", "CFAR": "detection_canvas",
                           "Angle of Arrival": "texas_aoa_canvas", "Micro-Doppler": "md_canvas",
                           "Range-Angle": "ra_canvas"}.get(view, ""))
    except Exception:
        return None


_settle_id = None
_settle_args = None


def _settle():
    """The frames stopped coming: the view shown from a picture draws the frame itself, so
    that zoom, the cursor and its other widgets are that frame's again."""
    global _settle_id
    _settle_id = None
    if _settle_args is None:
        return
    canvas, i, update = _settle_args
    try:
        frame_cache.render(canvas, i, update)
    except Exception:
        pass


def _playback_status(text: str = ""):
    """The corner of the status bar: how Play shows the frames while it runs, how many it
    keeps when it stops."""
    try:
        if not global_playing:
            n = frame_cache.count()
            text = f"Playback cache: {n} frames, {frame_cache.size_bytes() / 1e6:.0f} MB" if n else ""
        if playback_status_var.get() != text:
            playback_status_var.set(text)
    except Exception:
        pass


def _frame_view(canvas, i: int, update):
    """Show frame i of a view: its picture when it has one (frame_cache), else draw it
    and keep the picture."""
    global _settle_id, _settle_args, _play_mode
    if canvas is None:
        _play_mode = "Playback"
        update()
        return
    if frame_cache.show(canvas, i):
        _play_mode = "Cached playback"
        _settle_args = (canvas, i, update)
        if _settle_id is not None:
            try:
                root.after_cancel(_settle_id)
            except Exception:
                pass
        _settle_id = root.after(SETTLE_MS, _settle)
        return
    _play_mode = "Rendering playback (first run is slow)"
    frame_cache.render(canvas, i, update)


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

    # 2) Trigger plot updates selectively to optimize performance
    # Determine active tabs
    active_main = ""
    try:
        active_main = main_notebook.tab(main_notebook.select(), "text")
    except:
        pass

    # If we are not in Analysis and no background playback is active, skip plotting
    if active_main != "Analysis" and not rd_playing and not range_fft_playing:
        return

    active_an = ""
    try:
        active_an = analysis_nb.tab(analysis_nb.select(), "text")
    except:
        pass

    # Update Range FFT if playing or visible
    if range_fft_playing or active_an == "Range FFT":
        try:
            if last_range_fft is not None:
                rf_max = last_range_fft.shape[0] - 1
                _frame_view(_canvas_of('range_fft_canvas'), i, lambda: update_range_fft_plot(min(i, rf_max)))
        except Exception:
            pass

    # Update the range profile if it has been computed and is visible
    if active_an == "Profile" and last_range_profile is not None:
        _frame_view(_canvas_of('rp_canvas'), i, lambda: update_range_profile_plot(i))

    # Update Range-Doppler if playing or visible
    if rd_playing or active_an == "Range-Doppler":
        try:
            if last_rd_map is not None:
                rd_max = last_rd_map.shape[0] - 1
                _frame_view(_canvas_of('rd_canvas'), i, lambda: update_rd_plot(min(i, rd_max)))
        except Exception:
            pass

    # Update CFAR if visible
    if active_an == "CFAR":
        try:
            if last_rd_map is not None and detection_ax is not None:
                det_max = last_rd_map.shape[0] - 1
                _frame_view(_canvas_of('detection_canvas'), i, lambda: update_detection_plot(min(i, det_max)))
        except Exception:
            pass

    # Update Angle of Arrival if visible
    if active_an == "Angle of Arrival":
        try:
            if last_texas_aoa_spec is not None:
                tx_aoa_max = last_texas_aoa_spec.shape[0] - 1
                _frame_view(_canvas_of('texas_aoa_canvas'), i, lambda: update_texas_aoa_plot(min(i, tx_aoa_max)))
        except Exception:
            pass

    # Update Micro-Doppler if visible
    if active_an == "Micro-Doppler":
        try:
            if last_md_meta is not None:
                _frame_view(_canvas_of('md_canvas'), i, lambda: update_md_plot(i))
        except Exception:
            pass

    # Update Range-Angle if visible
    if active_an == "Range-Angle":
        try:
            if last_ra_map is not None:
                ra_max = last_ra_map.shape[0] - 1
                _frame_view(_canvas_of('ra_canvas'), i, lambda: update_range_angle_plot(min(i, ra_max)))
        except Exception:
            pass

    # The Multi Vitals map follows the frame
    if active_an == "Multi Vitals":
        try:
            _frame_view(_view_canvas('Multi Vitals'), i, lambda: multi_vitals_tab.show_frame(i))
        except Exception:
            pass

    # and the Imaging pages
    if active_an == "Imaging":
        try:
            _frame_view(_view_canvas('Imaging'), i, lambda: imaging_tab.show_frame(i))
        except Exception:
            pass

    # Update 1D View if visible
    if active_an == "1D Dopp":
        try:
            if last_rd_map is not None and view1d_fig is not None:
                update_view1d_plot(i)
        except Exception:
            pass

    # The frame line in the plots drawn over the whole segment (frame_marker.py)
    frame_marker.refresh()


def _frame_marker_state():
    """Where the frame slider stands, for frame_marker: None while nothing is loaded."""
    if not last_params or global_frame_slider is None or global_frames_total <= 0:
        return None
    try:
        frame = int(global_frame_slider.get())
    except Exception:
        return None
    period = capture_params.frame_period_s(last_params) or 0.0
    if period <= 0:
        return None
    return frame_marker.FrameState(frame=frame, frames=int(global_frames_total),
                                   period_s=period, t0_s=segment_t0_s())


frame_marker.set_state(_frame_marker_state)

# NEW: Main notebook with top-level tabs
main_notebook = ttk.Notebook(root)
frame_decode_main = ttk.Frame(main_notebook)
frame_analysis_main = ttk.Frame(main_notebook)
frame_helper_main = ttk.Frame(main_notebook)
frame_log_main = ttk.Frame(main_notebook)
frame_manual_main = ttk.Frame(main_notebook)

frame_fastcheck_main = ttk.Frame(main_notebook)
main_notebook.add(frame_decode_main, text="Decode")
main_notebook.add(frame_fastcheck_main, text="Fastcheck")
main_notebook.add(frame_analysis_main, text="Analysis")
main_notebook.add(frame_log_main, text="Log")
main_notebook.add(frame_helper_main, text="Helper")
main_notebook.add(frame_manual_main, text="Manual")

# --- Status bar with progress indicator ---
# Packed before the notebook: the notebook expands into the leftover space, so the
# status bar must claim its strip first or it gets squeezed out of the window.
status_bar = ttk.Frame(root)
status_bar.pack(side="bottom", fill="x", padx=6, pady=(0, 4))
progress_text_var = tk.StringVar(value="Ready. New to DARTscope? The Manual tab starts with a quick start.")
# Packed before the message: in a narrow window the message is cut, never the Cancel
# button or the progress bar.
progress_bar = ttk.Progressbar(status_bar, orient="horizontal", mode="determinate",
                               maximum=100.0, length=220)
progress_bar.pack(side="right", padx=4)
_cancel_button = ttk.Button(status_bar, text="Cancel", command=request_cancel,
                            state="disabled", width=8)
_cancel_button.pack(side="right", padx=(0, 6))
# how the global Play shows the frames: drawn now (the first run, slow) or from their pictures
playback_status_var = tk.StringVar(value="")
ttk.Label(status_bar, textvariable=playback_status_var, anchor="e",
          foreground=theme.ACCENT).pack(side="right", padx=(0, 10))
ttk.Label(status_bar, textvariable=progress_text_var, anchor="w").pack(side="left", padx=(4, 8),
                                                                       fill="x", expand=True)

main_notebook.pack(fill="both", expand=True, padx=6, pady=6)

# Decode tab (moved into frame_decode_main)
tk.Label(frame_decode_main, text="Select BIN file:").grid(row=0, column=0, sticky="e")
tk.Entry(frame_decode_main, textvariable=bin_var, width=60).grid(row=0, column=1, padx=4, pady=4)
tk.Button(frame_decode_main, text="Browse...", command=select_bin_file).grid(row=0, column=2, padx=4)
tk.Button(frame_decode_main, text="Decode Log", command=decode_log).grid(row=1, column=1, pady=8)
ttk.Button(frame_decode_main, text="Run Fastcheck",
           command=lambda: _run_fastcheck_from_decode()).grid(row=1, column=0, padx=4, sticky="e")
ttk.Button(frame_decode_main, text="Quick start (Manual)",
           command=lambda: open_manual("Quick start")).grid(row=1, column=2, padx=4)

# Recent files box
recent_frame = ttk.LabelFrame(frame_decode_main, text=f"Recent history (max {RECENTS.limit})")
recent_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=2, pady=2)

# Listbox with double-click to open; 12 rows visible, the rest by scrolling
recent_listbox = tk.Listbox(recent_frame, height=min(RECENTS.limit, 12))
recent_scroll = ttk.Scrollbar(recent_frame, orient="vertical", command=recent_listbox.yview)
recent_listbox.configure(yscrollcommand=recent_scroll.set)
recent_listbox.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=6)
recent_scroll.pack(side="left", fill="y", pady=6)
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

# The Analysis tab scrolls as a last resort. Its strips wrap, but in a low window the
# wrapped rows alone can fill the height - at 800 x 600 the Vitals settings took all
# of it, the plot collapsed and the last row was cut off. Now the content keeps the
# height it needs - the global bar, the shown tab's settings and a plot of
# ANALYSIS_MIN_PLOT px - and a scrollbar appears; in a higher window it just fills it
# (at 1024 x 700 no tab needs the scrollbar).
ANALYSIS_MIN_PLOT = 160
# A figure of several rows of subplots needs this much per row - its title, tick labels
# and a plot one can read (96-dpi pixels; times the display's scale)
ANALYSIS_ROW_MIN = 100

def _plot_min_height(widget):
    """The minimum of a plot of several rows of subplots (None for one row)."""
    for fc in frame_marker.canvases():
        if fc.get_tk_widget() is widget:
            rows = min(plot_interactivity.figure_rows(fc.figure), 8)
            return int(rows * ANALYSIS_ROW_MIN * DPI_SCALE) if rows > 1 else None
    return None

def _analysis_needed_height():
    """Height the Analysis content needs for the tab shown now, or None to ask again."""
    try:
        sel = analysis_nb.select()
        if not sel:
            return 0
        tab = analysis_nb.nametowidget(sel)
        if cap_plot_heights(tab, int(ANALYSIS_MIN_PLOT * DPI_SCALE), min_for=_plot_min_height):
            return None             # the new requests reach the tab at idle time
        # The tab gets what is left below the global bar and the tab strip; it needs
        # what it asks for: its settings whole and its plot at ANALYSIS_MIN_PLOT.
        return analysis_area.winfo_height() - tab.winfo_height() + tab.winfo_reqheight()
    except (NameError, tk.TclError):
        return 0

analysis_scroll = ScrollFrame(frame_analysis_main, fit=_analysis_needed_height)
analysis_scroll.pack(fill="both", expand=True)
analysis_area = analysis_scroll.inner

# NEW: Global controls bar INSIDE Analysis tab
# Wrapping strip: the frame slider gets a row of its own below it, so none of these
# settings is pushed off the edge on a narrow window.
global_bar = FlowFrame(analysis_area)
_g = global_bar.group()
ttk.Label(_g, text="Global Frame:", font=theme.font(10, "bold")).pack(side="left", padx=(4, 8))
global_play_btn = tk.Button(_g, text="Play All", command=play_global, state="disabled")
global_play_btn.pack(side="left")
global_stop_btn = tk.Button(_g, text="Stop", command=stop_global, state="disabled")
global_stop_btn.pack(side="left", padx=(6, 0))

# Antenna (RX combine) selector – uses on_rx_mode_change as before
def on_rx_mode_change(_evt=None):
    """
    The RX combine mode feeds the Range-Doppler map; flag it for recompute instead of
    recomputing straight away, so it can be changed together with other settings.
    """
    for view in ("Range-Doppler", "Micro-Doppler"):
        mark_settings_changed(view)
    if last_range_profile is not None:
        update_range_profile_plot(_current_frame_index())

_g = global_bar.group()
ttk.Label(_g, text="Antenna:").pack(side="left", padx=(0, 2))
global_rx_mode_var = tk.StringVar(value="MRC")
global_rx_mode_combo = ttk.Combobox(
    _g, textvariable=global_rx_mode_var,
    values=["MRC", "SUM", "RX0", "RX1"], state="readonly", width=6
)
global_rx_mode_combo.pack(side="left")
global_rx_mode_combo.bind("<<ComboboxSelected>>", on_rx_mode_change)

# Sub-bin peak interpolation (spectral.py) - used by every heart/breathing rate and by
# the CFAR and 3D target lists. "none" reproduces the old bin-centre values.
def on_peak_interp_change(_evt=None):
    spectral.set_default_method(peak_interp_var.get())
    for view in ("Vital Signs", "Angle of Arrival"):
        mark_settings_changed(view)

_g = global_bar.group()
ttk.Label(_g, text="Range FFT:").pack(side="left", padx=(8, 2))
range_fft_var = tk.StringVar(value="auto")
range_fft_combo = ttk.Combobox(_g, textvariable=range_fft_var, width=6, state="readonly",
                               values=["auto", "256", "512", "1024", "2048", "4096"])
range_fft_combo.pack(side="left")
range_fft_combo.bind("<<ComboboxSelected>>", on_range_fft_change)

_g = global_bar.group()
ttk.Label(_g, text="Peak interp:").pack(side="left", padx=(8, 2))
peak_interp_var = tk.StringVar(value=spectral.default_method())
peak_interp_combo = ttk.Combobox(_g, textvariable=peak_interp_var, values=list(spectral.METHODS),
                                 state="readonly", width=9)
peak_interp_combo.pack(side="left")
peak_interp_combo.bind("<<ComboboxSelected>>", on_peak_interp_change)

# Static clutter suppression (clutter.py): Range-Doppler always, Angle of Arrival with
# "Remove static", Range-Angle with its clutter box. "mean" is what they always did.
def on_clutter_change(_evt=None):
    for view in ("Range-Doppler", "Angle of Arrival", "Range-Angle", "Micro-Doppler"):
        mark_settings_changed(view)

_g = global_bar.group()
ttk.Label(_g, text="Clutter:").pack(side="left", padx=(8, 2))
clutter_method_var = tk.StringVar(value="mean")
clutter_method_combo = ttk.Combobox(_g, textvariable=clutter_method_var,
                                    values=list(clutter_filters.METHODS), state="readonly", width=6)
clutter_method_combo.pack(side="left")
clutter_method_combo.bind("<<ComboboxSelected>>", on_clutter_change)
ttk.Label(_g, text="EMA τ [s]").pack(side="left", padx=(6, 2))
clutter_tau_var = tk.StringVar(value="5")
ttk.Entry(_g, textvariable=clutter_tau_var, width=4).pack(side="left")
ttk.Label(_g, text="SVD rank").pack(side="left", padx=(6, 2))
clutter_rank_var = tk.StringVar(value=str(clutter_filters.DEFAULT_RANK))
ttk.Entry(_g, textvariable=clutter_rank_var, width=3).pack(side="left")

_g = global_bar.group()
ttk.Label(_g, text="Colours:").pack(side="left", padx=(8, 2))
display_cmap_var = tk.StringVar(value="auto")
_cm = ttk.Combobox(_g, textvariable=display_cmap_var, values=DISPLAY_CMAPS, state="readonly", width=8)
_cm.pack(side="left")
_cm.bind("<<ComboboxSelected>>", on_display_change)
ttk.Label(_g, text="Range [dB]:").pack(side="left", padx=(6, 2))
display_range_var = tk.StringVar(value="auto")
_dr = ttk.Combobox(_g, textvariable=display_range_var, values=DISPLAY_RANGES, width=5)
_dr.pack(side="left")
for _ev in ("<<ComboboxSelected>>", "<Return>", "<FocusOut>"):
    _dr.bind(_ev, on_display_change)

def _invalidate_cube_derived_data(reason: str, keep_cube: bool = False,
                                  outdates_fastcheck: bool = True):
    """
    Drop the cube and everything derived from it (keep_cube: only what is derived -
    for settings like the range FFT length that do not change the raw data).

    Anything computed from the raw cube becomes stale when the chirp selection or
    the lane order changes. Previously only a few of these were cleared, so the
    RTI, Range Profile, Angle of Arrival, Movement Map, Range-Angle and CFAR tabs
    kept showing results from the old settings with no visible hint.
    """
    global last_cube, last_frame_damaged
    global last_rd_map, last_rd_meta, last_rd_buffer, last_md_map, last_md_meta
    global last_range_fft, last_range_fft_meta, last_range_fft_bg
    global last_ai_rfft_complex, last_ai_rfft_meta
    global last_rti, last_rti_meta, last_range_profile, last_range_profile_meta
    global last_mov_ra_cube, last_mov_ra_meta
    global last_texas_aoa_spec, last_texas_aoa_meta
    global last_mov_map, last_mov_meta, last_ra_map, last_ra_meta
    global last_detections, last_detection_peaks
    global last_vital_data, last_vital_rfft_cube, last_vital_rf_meta, last_vital_cube_ptr

    if outdates_fastcheck and globals().get("fastcheck_tab") is not None:
        fastcheck_tab.mark_stale(reason)
    if globals().get("multi_vitals_tab") is not None:
        multi_vitals_tab.clear(f"{reason} - press Run Multi Vitals.")
    if globals().get("imaging_tab") is not None:
        imaging_tab.clear(f"{reason} - press Run Imaging.")
    if not keep_cube:
        global last_full_cube, last_frame_damaged_full
        last_cube = None
        last_frame_damaged = None
        last_full_cube = last_frame_damaged_full = None
    last_rd_map = last_rd_meta = last_rd_buffer = None
    last_md_map = last_md_meta = None
    last_range_fft = last_range_fft_meta = last_range_fft_bg = None
    last_ai_rfft_complex = last_ai_rfft_meta = None
    last_rti = last_rti_meta = None
    last_range_profile = last_range_profile_meta = None
    last_mov_ra_cube = last_mov_ra_meta = None
    last_texas_aoa_spec = last_texas_aoa_meta = None
    last_mov_map = last_mov_meta = None
    last_ra_map = last_ra_meta = None
    last_detections = last_detection_peaks = None
    last_vital_data = last_vital_rfft_cube = last_vital_rf_meta = last_vital_cube_ptr = None
    print(f"[GUI] {reason} - cached results invalidated, re-run the views you need.")
    try:
        if progress_text_var is not None:
            progress_text_var.set(f"{reason} - press Run to recompute")
    except Exception:
        pass

# NEW: Chirp selection change handler
def on_chirp_sel_change(_evt=None):
    _invalidate_cube_derived_data("Chirp selection changed")

# NEW: Chirp selection UI
ttk.Label(_g, text="Chirp:").pack(side="left", padx=(10, 2))
global_chirp_sel_var = tk.StringVar(value="All")
global_chirp_sel_combo = ttk.Combobox(
    _g, textvariable=global_chirp_sel_var, values=["All"], state="disabled", width=10
)
global_chirp_sel_combo.pack(side="left")
global_chirp_sel_combo.bind("<<ComboboxSelected>>", on_chirp_sel_change)

# NEW: Lane Swap control (Global)
def on_lane_swap_change():
    _invalidate_cube_derived_data("Lane swap changed")

_g = global_bar.group()
lane_swap_var = tk.BooleanVar(value=False)
tk.Checkbutton(_g, text="Lane Swap (1<->2)", variable=lane_swap_var,
               command=on_lane_swap_change).pack(side="left")
iq_correct_var = tk.BooleanVar(value=False)
tk.Checkbutton(_g, text="IQ correction", variable=iq_correct_var,
               command=lambda: _invalidate_cube_derived_data("IQ correction changed")
               ).pack(side="left", padx=(8, 0))

# Time segment (roadmap 9.6): process from..to seconds of the capture; empty = all
_g = global_bar.group()
ttk.Label(_g, text="Segment [s]:").pack(side="left", padx=(8, 2))
segment_from_var = tk.StringVar(value="")
segment_to_var = tk.StringVar(value="")
_sf = ttk.Entry(_g, textvariable=segment_from_var, width=6)
_sf.pack(side="left")
ttk.Label(_g, text="-").pack(side="left")
_st = ttk.Entry(_g, textvariable=segment_to_var, width=6)
_st.pack(side="left")
for _e in (_sf, _st):
    _e.bind("<Return>", on_segment_apply)
ttk.Button(_g, text="Apply", width=6, command=on_segment_apply).pack(side="left", padx=(4, 0))
segment_info_var = tk.StringVar(value="")
ttk.Label(_g, textvariable=segment_info_var, foreground=theme.SECONDARY).pack(side="left", padx=(6, 0))

# Real-time FPS label
global_fps_var = tk.StringVar(value="-- fps")
ttk.Label(_g, textvariable=global_fps_var).pack(side="left", padx=(10, 0))

# NEW: Global Range Crop
_g = global_bar.group()
ttk.Label(_g, text="Range Cut [m]:").pack(side="left", padx=(0, 2))
aoa_min_range_var = tk.StringVar(value="0.3")
tk.Entry(_g, textvariable=aoa_min_range_var, width=5).pack(side="left")

# NEW: Global Angle Limit
ttk.Label(_g, text="Angle Lim [±deg]:").pack(side="left", padx=(10, 2))
aoa_angle_limit_var = tk.StringVar(value="90")
tk.Entry(_g, textvariable=aoa_angle_limit_var, width=5).pack(side="left")

global_bar.pack(fill="x", padx=6, pady=(6, 0))

# The frame slider gets its own full-width row - inside the wrapping strip it would be
# squeezed to its minimum, and it is the control used most.
global_frame_slider = theme.Slider(analysis_area, from_=0, to=0, orient="horizontal",
                               label="", command=on_global_frame_change, state="disabled")
global_frame_slider.pack(fill="x", padx=6, pady=(0, 4))

# NEW: Analysis sub-notebook and sub-tabs
# Fifteen tabs share one strip and ttk has no second row for them - past its width it
# just clips the labels. The names are kept short: thirteen fitted a 1000 px window on
# Windows (about 1200 px before), on macOS they needed 1080 px, with Multi Vitals 1170
# and with Imaging about 1240.
analysis_nb = ttk.Notebook(analysis_area)
# a tab (also an inner one, e.g. in HR) shown: its plots get the frame line (frame_marker.py)
root.bind_all("<<NotebookTabChanged>>", lambda e: root.after_idle(frame_marker.refresh), add="+")
frame_range = ttk.Frame(analysis_nb)
frame_rti = ttk.Frame(analysis_nb)
frame_rp = ttk.Frame(analysis_nb)
frame_rd = ttk.Frame(analysis_nb)
frame_mov = ttk.Frame(analysis_nb) # NEW: Movement Map
frame_aoa = ttk.Frame(analysis_nb)   # Angle of Arrival (1D / Range-Az / Az-El / 3D)
frame_ra = ttk.Frame(analysis_nb)      # NEW: Range-Angle
frame_imaging = ttk.Frame(analysis_nb)  # 4D radar imaging: point cloud, radar vision
frame_md = ttk.Frame(analysis_nb)  # NEW: Micro-Doppler
frame_view1d = ttk.Frame(analysis_nb)  # NEW: 1D View
frame_detection = ttk.Frame(analysis_nb)
frame_vital = ttk.Frame(analysis_nb)
frame_multi_vitals = ttk.Frame(analysis_nb)
frame_hr = ttk.Frame(analysis_nb)
frame_ai_vital = ttk.Frame(analysis_nb) # NEW

analysis_nb.add(frame_range, text="Range FFT")
analysis_nb.add(frame_rti, text="RTI")
analysis_nb.add(frame_rp, text="Profile")
analysis_nb.add(frame_rd, text="Range-Doppler")
analysis_nb.add(frame_mov, text="Movement")
analysis_nb.add(frame_aoa, text="Angle of Arrival")
analysis_nb.add(frame_ra, text="Range-Angle")      # NEW
analysis_nb.add(frame_imaging, text="Imaging")
analysis_nb.add(frame_md, text="Micro-Doppler")  # NEW
analysis_nb.add(frame_view1d, text="1D Dopp")
analysis_nb.add(frame_detection, text="CFAR")
analysis_nb.add(frame_vital, text="Vitals")
analysis_nb.add(frame_multi_vitals, text="Multi Vitals")
analysis_nb.add(frame_hr, text="HR")
analysis_nb.add(frame_ai_vital, text="AI")

analysis_nb.pack(fill="both", expand=True, padx=6, pady=6)

# Force plot update when switching tabs to ensure the newly visible plot is current
def on_tab_changed(_event=None):
    try:
        idx = int(global_frame_slider.get())
        apply_global_frame(idx)
    except Exception:
        pass

main_notebook.bind("<<NotebookTabChanged>>", on_tab_changed)
analysis_nb.bind("<<NotebookTabChanged>>", on_tab_changed)


# Range FFT tab (under Analysis)
controls_frame = FlowFrame(frame_range)
controls_frame.grid(row=0, column=0, columnspan=5, padx=8, pady=4, sticky="ew")
range_fft_run_group = controls_frame.group()
register_run_button("Range FFT",
    theme.Button(range_fft_run_group, text="Run Range FFT", command=run_range_fft)).pack(side="left")
# NEW: Remove stationary checkbox (motion-only view)
range_fft_rm_stationary_var = tk.BooleanVar(value=False)
tk.Checkbutton(controls_frame.group(), text="Remove stationary (motion only)", variable=range_fft_rm_stationary_var,
               command=lambda: update_range_fft_plot(0)).pack(side="left")

# NEW: intensity slider (0..100%) mapped to 0.0..1.0
range_fft_rm_strength_var = tk.DoubleVar(value=1.0)

def _on_rm_strength_change(_val=None):
    try:
        update_range_fft_plot(int(global_frame_slider.get()))
    except Exception:
        update_range_fft_plot(0)

strength_frame = controls_frame.group()

ttk.Label(strength_frame, text="Intensity:").pack(side="left")
rm_scale = theme.Slider(strength_frame, from_=0, to=100, orient="horizontal", showvalue=True, length=150,
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
add_card_header_extras(range_fft_run_group, "Range FFT", INFO_RANGE_FFT, range_fft_fig, 
                       lambda: last_range_fft[int(global_frame_slider.get())].mean(axis=(0,1)) if last_range_fft is not None else None,
                       update_range_fft_plot)

# Angle of Arrival tab
#
# This used to be three tabs - "AoA" (a 1D spectrum), "2D AoA" (a Range-Azimuth map)
# and "Texas AoA". The first two were exactly the "1D Spectrum" and "Range-Azimuth"
# views of the third, computed by separate code with its own settings, so the same
# recording could show two different answers depending on which tab you opened. There
# is one tab now, and the view selector chooses what it shows.
aoa_controls = FlowFrame(frame_aoa)
aoa_controls.grid(row=0, column=0, padx=8, pady=4, sticky="ew")
frame_aoa.columnconfigure(0, weight=1)

def _rerun_texas_aoa(e=None):
    mark_settings_changed("Angle of Arrival")

def create_mimo_mode_cb(parent, var_name, on_change):
    ttk.Label(parent, text="Mode:").pack(side="left", padx=(0, 2))
    v = tk.StringVar(value="MIMO")
    globals()[var_name] = v
    cb = ttk.Combobox(parent, state="readonly", width=10,
                      values=["MIMO", "First TX", "All Chirps"], textvariable=v)
    cb.pack(side="left")
    cb.bind("<<ComboboxSelected>>", on_change)
    return cb

def create_method_cb(parent, var_name, on_change):
    ttk.Label(parent, text="Method:").pack(side="left", padx=(8, 2))
    v = tk.StringVar(value="Bartlett")
    globals()[var_name] = v
    cb = ttk.Combobox(parent, state="readonly", width=10,
                      values=["Bartlett", "Capon"], textvariable=v)
    cb.pack(side="left")
    cb.bind("<<ComboboxSelected>>", on_change)
    return cb

# --- run + export ---------------------------------------------------------
_g = aoa_controls.group()
register_run_button("Angle of Arrival",
    theme.Button(_g, text="Run Angle of Arrival", command=run_texas_aoa)).pack(side="left")

init_texas_aoa_plot(frame_aoa)
add_card_header_extras(_g, "Angle of Arrival", INFO_AOA, texas_aoa_fig,
                       lambda: last_texas_aoa_spec[int(global_frame_slider.get())] if last_texas_aoa_spec is not None else None,
                       update_texas_aoa_plot)

# --- what to show ---------------------------------------------------------
_g = aoa_controls.group()
ttk.Label(_g, text="View:").pack(side="left", padx=(0, 2))
texas_aoa_view_var = tk.StringVar(value="1D Spectrum")
texas_aoa_view_cb = ttk.Combobox(
    _g, textvariable=texas_aoa_view_var, state="readonly", width=17,
    values=["1D Spectrum", "Range-Azimuth", "Azimuth-Elevation", "Range-Az-El"]
)
texas_aoa_view_cb.pack(side="left", padx=2)
ttk.Label(_g, text="Proj:").pack(side="left", padx=(8, 2))
texas_rae_proj_var = tk.StringVar(value="Top (R-Az)")
texas_rae_proj_cb = ttk.Combobox(
    _g, textvariable=texas_rae_proj_var, state="readonly", width=12,
    values=["Top (R-Az)", "Slice (Az-El)", "3D (x,y,z)"]
)
texas_rae_proj_cb.pack(side="left", padx=2)

# --- antenna geometry -----------------------------------------------------
# Describes the radar hardware rather than one analysis, so every AoA view uses it.
_g = aoa_controls.group()
ttk.Label(_g, text="Array:").pack(side="left", padx=(0, 2))
aoa_antenna_cfg_var = tk.StringVar(value=antenna_config.AUTO_DETECT)
aoa_antenna_cfg_combo = ttk.Combobox(
    _g, textvariable=aoa_antenna_cfg_var, state="readonly", width=30,
    values=antenna_config.config_names()
)
aoa_antenna_cfg_combo.pack(side="left")
aoa_antenna_cfg_combo.bind("<<ComboboxSelected>>",
                           lambda e: [mark_settings_changed(v)
                                      for v in ("Angle of Arrival", "Movement Map")])
ttk.Button(_g, text="Config...",
           command=lambda: open_antenna_config_dialog(root)).pack(side="left", padx=(4, 0))

# --- beamformer -----------------------------------------------------------
_g = aoa_controls.group()
create_mimo_mode_cb(_g, "texas_aoa_mimo_mode_var", _rerun_texas_aoa)
create_method_cb(_g, "texas_aoa_method_var", _rerun_texas_aoa)

# --- range window ---------------------------------------------------------
# A single number picks one range; "0.3-2" limits the whole analysis to that window.
# Without it a wall several metres away is usually the strongest thing in the scene,
# it sets the colour scale and the 3D axes, and a person closer in is lost in it.
_g = aoa_controls.group()
tk.Label(_g, text="Range [m] (e.g. 1.5 or 0.3-2):").pack(side="left", padx=(0, 2))
texas_aoa_range_var = tk.StringVar(value="")
tk.Entry(_g, textvariable=texas_aoa_range_var, width=10).pack(side="left")

# --- preprocessing --------------------------------------------------------
_g = aoa_controls.group()
if texas_aoa_remove_static_var is None:
    texas_aoa_remove_static_var = tk.BooleanVar(value=False)
tk.Checkbutton(_g, text="Remove Static", variable=texas_aoa_remove_static_var).pack(side="left")

if texas_aoa_mimo_var is None:
    texas_aoa_mimo_var = tk.BooleanVar(value=True)
tk.Checkbutton(_g, text="MIMO", variable=texas_aoa_mimo_var).pack(side="left", padx=(8, 0))

if aoa_invert_phase_var is None:
    aoa_invert_phase_var = tk.BooleanVar(value=False)
tk.Checkbutton(_g, text="Invert TX Phase", variable=aoa_invert_phase_var).pack(side="left", padx=(8, 0))

if texas_aoa_shift_var is None:
    texas_aoa_shift_var = tk.BooleanVar(value=False)   # Default OFF

# --- detection and clutter ------------------------------------------------
_g = aoa_controls.group()
if texas_aoa_detect_var is None:
    texas_aoa_detect_var = tk.BooleanVar(value=False)
tk.Checkbutton(_g, text="Detect (CFAR)", variable=texas_aoa_detect_var).pack(side="left")
tk.Label(_g, text="Thr:").pack(side="left", padx=(4, 2))
texas_aoa_cfar_factor_var = tk.StringVar(value="6.0")
tk.Entry(_g, textvariable=texas_aoa_cfar_factor_var, width=5).pack(side="left")

if texas_aoa_sub_clutter_var is None:
    texas_aoa_sub_clutter_var = tk.BooleanVar(value=False)
tk.Checkbutton(_g, text="Sub Clutter", variable=texas_aoa_sub_clutter_var).pack(side="left", padx=(8, 0))
tk.Button(_g, text="Rec Clutter", command=record_clutter).pack(side="left", padx=(4, 0))

_g = aoa_controls.group()
if texas_aoa_doppler_comp_var is None:
    texas_aoa_doppler_comp_var = tk.BooleanVar(value=False)
tk.Checkbutton(_g, text="Doppler Comp", variable=texas_aoa_doppler_comp_var).pack(side="left")

if texas_aoa_track_moving_var is None:
    texas_aoa_track_moving_var = tk.BooleanVar(value=False)
tk.Checkbutton(_g, text="Track Moving", variable=texas_aoa_track_moving_var).pack(side="left", padx=(8, 0))

# --- channel calibration (corner reflector, calibration.py) -----------------
_g = aoa_controls.group()
tk.Label(_g, text="Calibration:").pack(side="left")
calib_status_var = tk.StringVar(value="none")
tk.Label(_g, textvariable=calib_status_var, fg=theme.SECONDARY).pack(side="left", padx=(2, 4))
tk.Button(_g, text="Estimate...", command=on_estimate_calibration).pack(side="left")
tk.Button(_g, text="Load...", command=on_load_calibration).pack(side="left", padx=(4, 0))
tk.Button(_g, text="Clear", command=lambda: _set_active_calibration(None)).pack(side="left", padx=(4, 0))

def on_rae_proj_change(e=None):
    try:
        if last_texas_aoa_spec is not None and last_texas_aoa_meta is not None \
                and last_texas_aoa_meta.get('view_type') == 'Range-Az-El':
            fi = int(global_frame_slider.get()) if global_frame_slider is not None else 0
            update_texas_aoa_plot(fi)
    except Exception:
        pass

texas_rae_proj_cb.bind("<<ComboboxSelected>>", on_rae_proj_change)

# Movement Map tab
mov_controls = FlowFrame(frame_mov)
mov_controls.grid(row=0, column=0, padx=8, pady=4, sticky="ew")
_g = mov_controls.group()
register_run_button("Movement Map",
    theme.Button(_g, text="Run Movement Map", command=run_movement_map)).pack(side="left")

init_mov_plot(frame_mov)
add_card_header_extras(_g, "Movement Map", INFO_MOV_MAP, mov_fig, 
                       lambda: last_mov_map if last_mov_map is not None else None,
                       update_movement_map_plot, manual_section="Movement")

_g = mov_controls.group()
ttk.Label(_g, text="Band:").pack(side="left", padx=(0, 2))
if mov_band_var is None:
    mov_band_var = tk.StringVar(value="Movement")
mov_band_cb = ttk.Combobox(
    _g, state="readonly", width=12,
    values=["Movement", "Breathing", "HR"],
    textvariable=mov_band_var
)
mov_band_cb.pack(side="left")

_g = mov_controls.group()
ttk.Label(_g, text="Accumulation:").pack(side="left", padx=(0, 2))
if mov_accum_var is None:
    mov_accum_var = tk.StringVar(value="Full")
mov_accum_cb = ttk.Combobox(
    _g, state="readonly", width=12,
    values=["Full", "Sliding Window"],
    textvariable=mov_accum_var
)
mov_accum_cb.pack(side="left")

_g = mov_controls.group()
ttk.Label(_g, text="Win [frames]:").pack(side="left", padx=(0, 2))
if mov_win_var is None:
    mov_win_var = tk.StringVar(value="32")
tk.Entry(_g, textvariable=mov_win_var, width=5).pack(side="left")

# Micro-Doppler tab: spectrogram, envelopes, swings and their speed (limb_motion.py)
md_controls = FlowFrame(frame_md)
md_controls.grid(row=0, column=0, padx=8, pady=4, sticky="ew")
_g = md_controls.group()
register_run_button("Micro-Doppler",
    theme.Button(_g, text="Run Micro-Doppler", command=run_micro_doppler)).pack(side="left")

init_md_plot(frame_md)
init_range_angle_plot(frame_ra)
add_card_header_extras(_g, "Micro-Doppler", INFO_MD, md_fig, _md_data_for_export, update_md_plot,
                       extra_items=[("Swings (CSV)", _export_md_swings),
                                    ("Envelopes (CSV)", _export_md_envelopes)])

def _md_changed(_evt=None):
    mark_settings_changed("Micro-Doppler")

def _md_entry(parent, value, width=5):
    var = tk.StringVar(value=value)
    e = tk.Entry(parent, textvariable=var, width=width)
    e.pack(side="left")
    e.bind("<KeyRelease>", _md_changed)
    return var

_g = md_controls.group()
tk.Label(_g, text="Range bins from:").pack(side="left", padx=(0, 2))
md_r0_var = _md_entry(_g, "5", width=6)
tk.Label(_g, text="to:").pack(side="left", padx=(6, 2))
md_r1_var = _md_entry(_g, "100", width=6)

_g = md_controls.group()
tk.Label(_g, text="Gate:").pack(side="left", padx=(0, 2))
md_gate_var = tk.StringVar(value="Auto")
_cb = ttk.Combobox(_g, textvariable=md_gate_var, values=["Auto", "Fixed"], state="readonly", width=6)
_cb.pack(side="left")
_cb.bind("<<ComboboxSelected>>", _md_changed)
tk.Label(_g, text="Width [m]:").pack(side="left", padx=(6, 2))
md_width_var = _md_entry(_g, f"{limb_motion.DEFAULT_GATE_WIDTH_M:g}")

_g = md_controls.group()
tk.Label(_g, text="Window [ms]:").pack(side="left", padx=(0, 2))
md_window_var = _md_entry(_g, f"{1e3 * limb_motion.DEFAULT_WINDOW_S:g}")

_g = md_controls.group()
tk.Label(_g, text="Threshold [dB]:").pack(side="left", padx=(0, 2))
md_thr_var = _md_entry(_g, f"{limb_motion.DEFAULT_THRESHOLD_DB:g}")
tk.Label(_g, text="Min swing [m/s]:").pack(side="left", padx=(6, 2))
md_swing_var = _md_entry(_g, f"{limb_motion.DEFAULT_SWING_M_S:g}")

_g = md_controls.group()
tk.Label(_g, text="Radar height [m]:").pack(side="left", padx=(0, 2))
md_height_var = _md_entry(_g, "")
md_env_var = tk.BooleanVar(value=True)
tk.Checkbutton(_g, text="Envelopes", variable=md_env_var, command=_md_redraw).pack(side="left", padx=(8, 0))

_g = md_controls.group()
md_readout_var = tk.StringVar(value="")
ttk.Label(_g, textvariable=md_readout_var).pack(side="left")

# 1D View tab
view1d_controls = FlowFrame(frame_view1d)
view1d_controls.grid(row=0, column=0, padx=8, pady=4, sticky="ew")
frame_view1d.columnconfigure(0, weight=1)
_g = view1d_controls.group()
tk.Button(_g, text="Run 1D View", command=run_view1d).pack(side="left")

init_view1d_plot(frame_view1d)
add_card_header_extras(_g, "1D View", INFO_VIEW1D, view1d_fig, 
                       lambda: last_rd_map[int(global_frame_slider.get()), :, int(view1d_rbin_var.get())] if last_rd_map is not None else None,
                       update_view1d_plot, manual_section="1D Dopp")

_g = view1d_controls.group()
ttk.Label(_g, text="Range bin:").pack(side="left", padx=(0, 2))
if view1d_rbin_var is None:
    view1d_rbin_var = tk.IntVar(value=0)
view1d_slider = theme.Slider(
    _g, from_=0, to=0, orient="horizontal",
    command=on_view1d_rbin_change, state="disabled", length=300,
    variable=view1d_rbin_var, showvalue=False
)
view1d_slider.pack(side="left", padx=2)
if view1d_rbin_label_var is None:
    view1d_rbin_label_var = tk.StringVar(value="0")
ttk.Label(_g, textvariable=view1d_rbin_label_var, width=5).pack(side="left")

_g = view1d_controls.group()
ttk.Label(_g, text="Source:").pack(side="left", padx=(0, 2))
if view1d_band_src_var is None:
    view1d_band_src_var = tk.StringVar(value="Range Bin")
view1d_band_src_cb = ttk.Combobox(
    _g, state="readonly", width=15,
    values=["Range Bin", "Neighborhood ±N", "Whole Range"],
    textvariable=view1d_band_src_var
)
view1d_band_src_cb.pack(side="left")
view1d_band_src_cb.bind("<<ComboboxSelected>>", lambda e: update_view1d_plot(int(global_frame_slider.get())))

ttk.Label(_g, text="±N:").pack(side="left", padx=(8, 2))
if view1d_band_w_var is None:
    view1d_band_w_var = tk.StringVar(value="6")
ttk.Entry(_g, textvariable=view1d_band_w_var, width=4).pack(side="left")

if view1d_just_frame_var is None:
    view1d_just_frame_var = tk.BooleanVar(value=False)
tk.Checkbutton(view1d_controls.group(), text="Current Frame Only", variable=view1d_just_frame_var,
               command=lambda: update_view1d_plot(int(global_frame_slider.get()))).pack(side="left")

# --- NEW: AoA range-bin slider ---
# Container row under the plot controls (kept separate from global frame slider)
# Range-bin selector for the views that analyse one range bin (1D Spectrum, and the
# Az-El slice of the Range-Az-El map).
aoa_bin_controls = ttk.Frame(frame_aoa)
aoa_bin_controls.grid(row=2, column=0, padx=8, pady=(2, 6), sticky="ew")

# Label showing current selected bin and (if available) meters
if aoa_bin_label_var is None:
    aoa_bin_label_var = tk.StringVar(value="bin -")
ttk.Label(aoa_bin_controls, textvariable=aoa_bin_label_var).pack(side="right", padx=(8, 4))

def on_aoa_bin_change(val=None):
    """
    React to the range-bin slider.

    Dragging this used to fire a full AoA recomputation per slider step. For the Az-El
    slice nothing needs recomputing at all - the slice is just a cut through the map
    that is already in memory - so it is redrawn directly. Anything else is flagged for
    the next Run, in line with the rest of the application: settings are collected and
    applied when the button is pressed.
    """
    try:
        _aoa_bin_label(int(float(val)))
    except (TypeError, ValueError):
        pass
    try:
        if (last_texas_aoa_meta is not None
                and last_texas_aoa_meta.get('view_type') == "Range-Az-El"
                and texas_rae_proj_var is not None
                and texas_rae_proj_var.get().startswith("Slice")):
            fi = int(global_frame_slider.get()) if global_frame_slider is not None else 0
            update_texas_aoa_plot(fi)
            return
    except Exception:
        pass
    mark_settings_changed("Angle of Arrival")

# Slider itself (disabled until first run when we know number of bins)
if aoa_bin_var is None:
    aoa_bin_var = tk.IntVar(value=0)

if aoa_bin_slider is None:
    aoa_bin_slider = theme.Slider(
        aoa_bin_controls,
        from_=0,
        to=0,
        orient="horizontal",
        label="Range Bin",
        command=lambda _v: on_aoa_bin_change(_v),
        state="disabled",
        length=400,
        variable=aoa_bin_var,
    )
    aoa_bin_slider.pack(side="left", fill="x", expand=True)

# RTI tab
rti_controls_frame = FlowFrame(frame_rti)
rti_controls_frame.grid(row=0, column=0, padx=8, pady=4, sticky="ew")
frame_rti.columnconfigure(0, weight=1)
_g = rti_controls_frame.group()
register_run_button("RTI",
    theme.Button(_g, text="Run RTI", command=run_rti)).pack(side="left")

init_rti_plot(frame_rti)
add_card_header_extras(_g, "RTI", INFO_RTI, rti_fig, 
                       lambda: last_rti if last_rti is not None else None)

# Range Profile tab
rp_controls_frame = FlowFrame(frame_rp)
rp_controls_frame.grid(row=0, column=0, padx=8, pady=4, sticky="ew")
rp_run_group = rp_controls_frame.group()
theme.Button(rp_run_group, text="Run Profile", command=run_range_profile).pack(side="left")
_g = rp_controls_frame.group()
rp_zoom_var = tk.BooleanVar(value=False)
tk.Checkbutton(_g, text="Zoom FFT", variable=rp_zoom_var).pack(side="left", padx=(0, 2))
ttk.Label(_g, text="from [m]").pack(side="left")
rp_zoom_lo_var = tk.StringVar(value="0.5")
ttk.Entry(_g, textvariable=rp_zoom_lo_var, width=6).pack(side="left", padx=2)
ttk.Label(_g, text="to [m]").pack(side="left")
rp_zoom_hi_var = tk.StringVar(value="2.0")
ttk.Entry(_g, textvariable=rp_zoom_hi_var, width=6).pack(side="left", padx=2)

init_range_profile_plot(frame_rp)
add_card_header_extras(rp_run_group, "Range Profile", INFO_RP, rp_fig, 
                       lambda: last_range_profile if last_range_profile is not None else None,
                       manual_section="Profile")

# Range-Doppler tab
rd_controls_frame = FlowFrame(frame_rd)
rd_controls_frame.grid(row=0, column=0, padx=8, pady=4, sticky="ew")
_g = rd_controls_frame.group()
register_run_button("Range-Doppler",
    theme.Button(_g, text="Run Range-Doppler", command=run_range_doppler)).pack(side="left")

init_rd_plot(frame_rd)
add_card_header_extras(_g, "Range-Doppler", INFO_RD, rd_fig, 
                       lambda: last_rd_map[int(global_frame_slider.get())] if last_rd_map is not None else None,
                       update_rd_plot)

# Play/Stop of its own are gone: Play All of the frame bar plays this view as well. The
# buttons stay as objects (not shown), the RD code still sets their state.
rd_play_btn = tk.Button(rd_controls_frame, text="Play", command=play_rd, state="disabled")
rd_stop_btn = tk.Button(rd_controls_frame, text="Stop", command=stop_rd, state="disabled")
# NEW: Doppler window toggle (Hann)
rd_hann_var = tk.BooleanVar(value=True)
tk.Checkbutton(rd_controls_frame.group(), text="Hann window (Doppler)", variable=rd_hann_var).pack(side="left")
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
tk.Checkbutton(rd_controls_frame.group(), text="Auto velocity range", variable=rd_auto_vrange_var,
               command=_on_rd_vrange_toggle).pack(side="left")
# Zoom controls (manual only)
_g = rd_controls_frame.group()
tk.Label(_g, text="V_min [m/s]:").pack(side="left")
rd_vmin_var = tk.StringVar(value="-5")
rd_vmin_entry = tk.Entry(_g, textvariable=rd_vmin_var, width=5)  # NEW: keep ref for enable/disable
rd_vmin_entry.pack(side="left")
tk.Label(_g, text="V_max [m/s]:").pack(side="left", padx=(10, 0))
rd_vmax_var = tk.StringVar(value="5")
rd_vmax_entry = tk.Entry(_g, textvariable=rd_vmax_var, width=5)  # NEW: keep ref for enable/disable
rd_vmax_entry.pack(side="left")
# Initialize fields state for Auto
_on_rd_vrange_toggle()

# ODSTRANĚNO: lokální RD “Frame” slider (redundantní, nahrazen globálním)
# rd_frame_slider = tk.Scale(frame_rd, from_=0, to=0, orient="horizontal",
#                            label="Frame", command=on_rd_frame_change, state="disabled")
# rd_frame_slider.grid(row=2, column=0, sticky="ew", padx=4, pady=4)

# CFAR tab
detection_controls_frame = FlowFrame(frame_detection)
detection_controls_frame.grid(row=0, column=0, columnspan=5, padx=8, pady=4, sticky="ew")
cfar_run_group = detection_controls_frame.group()

init_detection_plot(frame_detection)
init_vital_signs_plot(frame_vital)
# Keep the reference: the tab owns the analysis state, and without it there is
# no way to reach the HR tab from outside its own callbacks (tests included).
hr_detection_tab = init_hr_detection(frame_hr)


def init_multi_vitals(parent):
    """The Multi Vitals view (multi_vitals_gui.py) on the decoded segment."""
    def get_raw():
        if ensure_cube_loaded() is None:
            return None
        params = last_params.copy()
        params['antenna_config'] = analysis_antenna_config()
        params.update(calibration_params())
        # the frame slider and Play All need the frame count - the other views set it as
        # they run, and with only this one run Play did nothing
        set_global_frames(int(last_cube.shape[0]))
        return (last_cube, params, derive_fps(last_params, "Multi Vitals"), segment_t0_s(),
                os.path.basename(bin_var.get()))

    def run(fn, **kw):
        with progress("Multi Vitals ...", total=100) as p:
            return run_in_background(fn, progress_cb=p.callback(), handle=p, **kw)

    def extras(group, fig, data_func, extra_items, update_func=None):
        add_card_header_extras(group, "Multi Vitals", INFO_MULTI_VITALS, fig, data_func,
                               update_func, extra_items=extra_items)

    return multi_vitals_gui.MultiVitalsTab(
        parent, get_raw, run=run, report_error=report_error,
        register_run=lambda b: register_run_button("Multi Vitals", b),
        header_extras=extras, fft_size=range_fft_size)


multi_vitals_tab = init_multi_vitals(frame_multi_vitals)


def init_imaging(parent):
    """The Imaging view (radar_imaging_gui.py) on the decoded segment."""
    def get_raw():
        if ensure_cube_loaded() is None:
            return None
        params = last_params.copy()
        params['antenna_config'] = analysis_antenna_config()
        params.update(calibration_params())
        set_global_frames(int(last_cube.shape[0]))       # for the slider and Play All
        return (last_cube, params, derive_fps(last_params, "Imaging"), segment_t0_s(),
                os.path.basename(bin_var.get()))

    def run(fn, **kw):
        with progress("Imaging ...", total=100) as p:
            return run_in_background(fn, progress_cb=p.callback(), handle=p, **kw)

    def extras(group, fig, data_func, extra_items, update_func=None):
        add_card_header_extras(group, "Imaging", INFO_IMAGING, fig, data_func,
                               update_func, extra_items=extra_items)

    return radar_imaging_gui.ImagingTab(
        parent, get_raw, run=run, report_error=report_error,
        register_run=lambda b: register_run_button("Imaging", b),
        header_extras=extras, fft_size=range_fft_size, export_video=export_to_video)


imaging_tab = init_imaging(frame_imaging)

# AI Vital Signs tab initialization
def get_ai_vital_data():
    global last_ai_rfft_complex, last_ai_rfft_meta
    
    # 1. Ensure Radar Cube is loaded
    if last_cube is None:
        try:
            ensure_cube_loaded()
        except Exception as e:
            report_error("Error", f"Failed to load radar cube: {e}", e)
            return None

    # 2. Compute complex Range FFT if not available
    if last_ai_rfft_complex is None and last_cube is not None:
        try:
            # We need complex data for AI vital signs (phase extraction)
            from range_fft import compute_range_fft_cube
            rfft_cube, rfft_meta = compute_range_fft_cube(last_cube, last_params, return_complex=True,
                                                          n_fft_range=range_fft_size())
            last_ai_rfft_complex = rfft_cube
            last_ai_rfft_meta = rfft_meta
        except Exception as e:
            report_error("Error", f"Failed to compute complex Range FFT: {e}", e)
            return None

    # 3. Get Range Axis
    r_axis = None
    if last_ai_rfft_meta is not None:
        r_axis = last_ai_rfft_meta.get('range_m')
    
    if r_axis is None and last_ai_rfft_complex is not None:
        r_axis = np.arange(last_ai_rfft_complex.shape[-1])
    elif r_axis is None:
        r_axis = np.arange(64)

    # 4. Calculate proper FPS (same helper as every other view, so the tabs cannot
    # disagree about the frame rate of the same recording)
    fps = derive_fps(last_params, "AI Vital")

    return {
        'rfft_cube': last_ai_rfft_complex,
        'fps': fps,
        'range_axis': r_axis,
        # the segment's start, so a reference synchronised to it lines up (roadmap 9.6)
        'radar_start_timestamp': (last_params.get('radar_start_timestamp') + segment_t0_s()
                                  if last_params and last_params.get('radar_start_timestamp') is not None
                                  else None),
        'radar_start_time': ((last_params.get('radar_start_time') or "") +
                             (f" + {segment_t0_s():.1f} s segment" if segment_t0_s() else "")
                             if last_params else None),
        'radar_duration_s': (last_ai_rfft_complex.shape[0] - 1) / fps if last_ai_rfft_complex is not None and last_ai_rfft_complex.shape[0] else 0.0,
        'params': last_params
    }

ai_vital_tab = ai_vital_gui.AIVitalSignsTab(frame_ai_vital, get_ai_vital_data)
# All plots exist now: lower what they ask for at once, so no tab shows the Analysis
# scrollbar for a moment on its first visit (see ANALYSIS_MIN_PLOT).
cap_plot_heights(analysis_area, int(ANALYSIS_MIN_PLOT * DPI_SCALE), min_for=_plot_min_height)

add_card_header_extras(cfar_run_group, "CFAR", INFO_CFAR, detection_fig, 
                       lambda: last_detections if last_detections is not None else None,
                       update_detection_plot)

register_run_button("CFAR",
    theme.Button(cfar_run_group, text="Run CFAR Detection", command=run_cfar_detection)).pack(side="left")
tk.Label(detection_controls_frame.group(), text="CFAR Parameters:",
         font=theme.font(10, "bold")).pack(side="left")
_g = detection_controls_frame.group()
tk.Label(_g, text="Guard Cells:").pack(side="left", padx=(0, 2))
cfar_guard_var = tk.StringVar(value="2")
tk.Entry(_g, textvariable=cfar_guard_var, width=5).pack(side="left")
_g = detection_controls_frame.group()
tk.Label(_g, text="Training Cells:").pack(side="left", padx=(0, 2))
cfar_training_var = tk.StringVar(value="8")
tk.Entry(_g, textvariable=cfar_training_var, width=5).pack(side="left")
_g = detection_controls_frame.group()
tk.Label(_g, text="Threshold Factor:").pack(side="left", padx=(0, 2))
cfar_threshold_var = tk.StringVar(value="3.0")
tk.Entry(_g, textvariable=cfar_threshold_var, width=5).pack(side="left")
_g = detection_controls_frame.group()
tk.Label(_g, text="Method:").pack(side="left", padx=(0, 2))
cfar_method_var = tk.StringVar(value="average")
cfar_method_combo = ttk.Combobox(_g, textvariable=cfar_method_var,
                                 values=["average", "ordered"], state="readonly", width=10)
cfar_method_combo.pack(side="left")
_g = detection_controls_frame.group()
cfar_unfold_var = tk.BooleanVar(value=False)
tk.Checkbutton(_g, text="Unfold velocity (TDM)",
               variable=cfar_unfold_var).pack(side="left", padx=(0, 2))
tk.Label(_g, text="max |v| [m/s]:").pack(side="left")
cfar_vlimit_var = tk.StringVar(value="")
tk.Entry(_g, textvariable=cfar_vlimit_var, width=5).pack(side="left")
# No frame slider of its own: the one above all Analysis views drives this view too.
detection_frame_slider = None

# Helper tab (reparented)
init_helper_tab(frame_helper_main, get_params=lambda: last_params or None)


# ---------------- Fastcheck tab: a quick check of the whole recording ----------------
def _fastcheck_data():
    """The whole capture for the Fastcheck tab, decoded on demand as for the Analysis views."""
    if ensure_cube_loaded() is None or last_full_cube is None:
        return None
    bin_path = bin_var.get()
    return {"stem": os.path.basename(bin_path), "params": last_params, "cube": last_full_cube,
            "bin_paths": get_all_bin_parts(bin_path), "fps": derive_fps(last_params, "Fastcheck"),
            "n_fft_range": range_fft_size()}


def _fastcheck_run(data):
    """fastcheck.run off the Tk thread, with the progress bar and Cancel."""
    with progress("Fastcheck ...", total=int(data["cube"].shape[0])) as p:
        return run_in_background(fastcheck.run, data["stem"], data["params"], data["cube"],
                                 fps=data["fps"], bin_paths=data["bin_paths"],
                                 n_fft_range=data["n_fft_range"], bands=vital_bands.current(),
                                 progress_cb=p.callback(), handle=p)


def _run_fastcheck_from_decode():
    if _busy:                    # a computation runs: its event loop must not start another
        try:
            progress_text_var.set("Busy - Fastcheck can run once this finishes")
        except Exception:
            pass
        return
    main_notebook.select(frame_fastcheck_main)
    fastcheck_tab.run_check()


fastcheck_tab = fastcheck_gui.FastcheckTab(
    frame_fastcheck_main, _fastcheck_data, _fastcheck_run, report_error=report_error,
    open_manual=lambda section=None: open_manual(section),
    register_run_button=register_run_button, cmap=display_cmap)

# ---------------- Measure tab: record with the radar (acquisition/, measure_gui.py) ----------------
frame_measure_main = ttk.Frame(main_notebook)
main_notebook.insert(0, frame_measure_main, text="Measure")
main_notebook.select(frame_decode_main)


def _open_recording_from_measure(bin_path: str):
    """A recording just made in the Measure tab: open it as if chosen with Browse."""
    if not os.path.isfile(bin_path):
        messagebox.showerror("Measure", f"{bin_path} does not exist")
        return
    bin_var.set(bin_path)
    try:
        RECENTS.add(bin_path)
        populate_recent_list()
    except Exception:
        pass
    main_notebook.select(frame_decode_main)
    decode_log()


# two radars, one sub-tab each: the IWR6843AOPEVM (measure_gui.py) and the
# AWR2243-2X-CAS-EVM cascade (cascade_gui.py); they share the DCA1000
measure_nb = ttk.Notebook(frame_measure_main)
measure_nb.pack(fill="both", expand=True)
frame_measure_single = ttk.Frame(measure_nb)
frame_measure_cascade = ttk.Frame(measure_nb)
measure_nb.add(frame_measure_single, text="Single-chip radar")
measure_nb.add(frame_measure_cascade, text="Cascade radar")

measure_tab = None
try:
    import measure_gui
    measure_tab = measure_gui.MeasureTab(frame_measure_single, open_recording=_open_recording_from_measure,
                                         report_error=report_error)
except Exception as e:                      # the rest of DARTscope works without it
    print(f"[Measure] could not be built: {e}")
    ttk.Label(frame_measure_single, text=f"The Measure tab could not be built: {e}",
              wraplength=900).pack(padx=20, pady=20)


def _release_single_chip():
    """Before a cascade take: the single-chip tab lets go of the DCA1000."""
    return measure_tab.release_hardware() if measure_tab is not None else None


cascade_tab = None
try:
    import cascade_gui
    cascade_tab = cascade_gui.CascadeTab(frame_measure_cascade, open_recording=_open_recording_from_measure,
                                         release_hardware=_release_single_chip, report_error=report_error)
except Exception as e:
    print(f"[Cascade] could not be built: {e}")
    ttk.Label(frame_measure_cascade, text=f"The Cascade radar tab could not be built: {e}",
              wraplength=900).pack(padx=20, pady=20)

# ---------------- Manual tab: docs/MANUAL.md with a collapsible contents ----------------
manual_view = None
try:
    manual_view = ManualView(frame_manual_main, resource_path(os.path.join("docs", "MANUAL.md")))
except Exception as e:
    print(f"[Manual] could not be built: {e}")


def open_manual(section: str | None = None):
    """Show the Manual tab - at `section` (a heading or its anchor) when given."""
    try:
        main_notebook.select(frame_manual_main)
        if manual_view is not None and section:
            manual_view.show(section)
    except Exception:
        pass


# ---------------- Log tab: everything the application prints ----------------
log_view_widget = log_view.LogView(frame_log_main, main_notebook, frame_log_main)

# A check of a build (acquisition/diagnostics.py): with DARTSCOPE_SELFTEST=<report.json> the
# program writes what it finds and closes; DARTSCOPE_SELFTEST_TAKE=<folder> adds one short
# take with the radar through the Measure tab.
if os.environ.get("DARTSCOPE_SELFTEST"):
    from acquisition.diagnostics import run_selftest
    run_selftest(root, measure_tab, os.environ["DARTSCOPE_SELFTEST"],
                 os.environ.get("DARTSCOPE_SELFTEST_TAKE"))

root.mainloop()