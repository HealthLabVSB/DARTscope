"""
radar_imaging_gui.py

The Imaging view of the Analysis tab (radar_imaging.py): 4D radar imaging - a point
cloud of every frame (where every reflection is in three dimensions and how fast it
moves) and images of the scene. Its settings and Run in one strip, the display options
in another, a summary line and five pages:

  3D             the point cloud in space, the radar at the origin, objects as boxes
                 with their track numbers; drag to turn it.
  Bird's-eye     the room from above: the image of the frame (range x azimuth) under
                 its points, objects with their tracks of the last seconds.
  Radar vision   what the radar sees, azimuth x elevation, the points on it coloured
                 by their distance (a depth image).
  Range-Doppler  the frame's range-Doppler map with every point (velocities unfolded
                 beyond one TX slot's limit lie outside the map's own axis).
  Tracks         every track of the recording from above, and the points and objects
                 of every frame.

The pages follow the frame slider and its playback; points of the last frames can stay
(Trail). Colour by velocity (blue towards the radar, red away), height, SNR, range or
object.

The host passes get_raw() -> (cube, params, fps, t0_s, name) or None, run(fn, **kw) that
calls fn(progress_cb=..., **kw) off the Tk thread with the progress bar, and optionally
report_error, register_run(button), header_extras(group, fig, data_func, extra_items,
update_func) for the ⓘ icon and the Export menu, export_video(fig, update_func, name),
and fft_size() -> the Analysis bar's Range FFT size.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import tkinter as tk
from collections import OrderedDict
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Circle

import radar_imaging
import theme
from flow_layout import FlowFrame
from plot_interactivity import create_interactive_canvas

MODE_LABELS = {"People (moving + micro-motion)": "people", "Moving (Doppler in a frame)": "moving",
               "Micro-motion (breathing)": "micro", "Still scene (room)": "still",
               "Everything": "all"}
COLOR_BY = ("Velocity", "Height", "SNR", "Range", "Object", "Kind")
KIND_COLORS = {radar_imaging.KIND_MOVING: "#d62728", radar_imaging.KIND_MICRO: "#1f77b4",
               radar_imaging.KIND_STILL: "#7f7f7f", radar_imaging.KIND_ECHO: "#c7c7c7"}
PAGES = ("3D", "Bird's-eye", "Radar vision", "Range-Doppler", "Tracks")
TRAIL_S = 3.0                  # seconds of an object's track drawn behind it
IMAGE_RANGE_DB = 20.0          # the images: this far below their strongest point
IMAGE_CACHE = 48               # frames whose images are kept
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2",
          "#17becf", "#bcbd22", "#7f7f7f"]


def track_color(tid: int) -> str:
    return COLORS[(int(tid) - 1) % len(COLORS)] if tid >= 0 else "0.6"


def obj_color(o: dict) -> str:
    """An object's colour: its track's, grey for an echo of another or no person."""
    return "0.6" if o.get("echo_of", -1) >= 0 or o.get("person") is False else track_color(o.get("track", -1))


def colour_values(points: np.ndarray, by: str, res) -> tuple:
    """(values or colours, cmap, vmin, vmax, label) of points for the colouring chosen."""
    if by == "Velocity":
        vmax = max(0.5, float(np.percentile(np.abs(res.points["velocity_m_s"]), 99))) if res.points.size else 1.0
        return points["velocity_m_s"], "coolwarm", -vmax, vmax, "Velocity [m/s] (+ away)"
    if by == "Height":
        z = res.points["z_m"] if res.points.size else np.zeros(1)
        lo, hi = (np.percentile(z, [2, 98]) if z.size else (-1, 1))
        return points["z_m"], "viridis", float(lo), float(max(hi, lo + 0.5)), "Height z [m]"
    if by == "SNR":
        return points["snr_db"], "magma", res.settings.threshold_db, 50.0, "SNR [dB]"
    if by == "Range":
        return points["range_m"], "turbo", 0.0, float(res.range_m[-1]) if res.range_m.size else 10.0, "Range [m]"
    if by == "Kind":
        return [KIND_COLORS.get(int(k), "0.5") for k in points["kind"]], None, None, None, "Kind"
    cols = [track_color(int(c)) for c in points["cluster"]]
    return cols, None, None, None, "Object"


def write_ply(path: str, points: np.ndarray) -> None:
    """The points as an ASCII PLY cloud (x, y, z, velocity, snr), for CloudCompare/MeshLab."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.size}\n")
        for k in ("x", "y", "z", "velocity", "snr", "frame"):
            f.write(f"property float {k}\n")
        f.write("end_header\n")
        for q in points:
            f.write(f"{q['x_m']:.4f} {q['y_m']:.4f} {q['z_m']:.4f} {q['velocity_m_s']:.4f} "
                    f"{q['snr_db']:.2f} {int(q['frame'])}\n")


class ImagingTab:
    def __init__(self, parent, get_raw, run=None, report_error=None, register_run=None,
                 header_extras=None, fft_size=None, export_video=None):
        self.get_raw = get_raw
        self.run_fn = run or (lambda fn, **kw: fn(**kw))
        self.report_error = report_error or (lambda title, msg, exc=None:
                                             messagebox.showerror(title, msg))
        self.fft_size = fft_size or (lambda: None)
        self.export_video = export_video
        self.result: radar_imaging.ImagingResult | None = None
        self.cube = self.params = None
        self.name = ""
        self._frame = 0
        self._images = OrderedDict()
        self._drawn = {}                 # page -> frame drawn (lazy pages)
        self._art = {}                   # page -> persistent artists

        # --- what to compute (Run) ---
        controls = FlowFrame(parent)
        controls.pack(side="top", fill="x", padx=8, pady=(4, 0))
        run_group = controls.group()
        self.run_button = theme.Button(run_group, text="Run Imaging", command=self.run)
        self.run_button.pack(side="left")
        if register_run is not None:
            register_run(self.run_button)
        s0 = radar_imaging.Settings()
        self.mode = tk.StringVar(value="People (moving + micro-motion)")
        self.threshold = tk.StringVar(value=f"{s0.threshold_db:g}")
        self.r_max = tk.StringVar(value="")
        self.height = tk.StringVar(value=f"{s0.height_m:g}")
        self.tilt = tk.StringVar(value=f"{s0.tilt_deg:g}")
        g = controls.group()
        ttk.Label(g, text="Show:").pack(side="left", padx=(0, 2))
        ttk.Combobox(g, textvariable=self.mode, state="readonly", width=27,
                     values=list(MODE_LABELS)).pack(side="left")
        g = controls.group()
        ttk.Label(g, text="Threshold [dB]:").pack(side="left", padx=(0, 2))
        tk.Entry(g, textvariable=self.threshold, width=4).pack(side="left")
        ttk.Label(g, text="Range to [m]:").pack(side="left", padx=(8, 2))
        tk.Entry(g, textvariable=self.r_max, width=5).pack(side="left")
        g = controls.group()
        ttk.Label(g, text="Radar height [m]:").pack(side="left", padx=(0, 2))
        tk.Entry(g, textvariable=self.height, width=4).pack(side="left")
        ttk.Label(g, text="Tilt down [°]:").pack(side="left", padx=(8, 2))
        tk.Entry(g, textvariable=self.tilt, width=4).pack(side="left")

        # --- how to show it (immediate) ---
        display = FlowFrame(parent)
        display.pack(side="top", fill="x", padx=8, pady=(0, 2))
        self.color_by = tk.StringVar(value="Velocity")
        self.trail = tk.StringVar(value="0")
        self.min_snr = tk.StringVar(value="")
        self.show_objects = tk.BooleanVar(value=True)
        self.show_image = tk.BooleanVar(value=True)
        self.show_echoes = tk.BooleanVar(value=False)
        self.show_others = tk.BooleanVar(value=False)
        g = display.group()
        ttk.Label(g, text="Colour by:").pack(side="left", padx=(0, 2))
        cb = ttk.Combobox(g, textvariable=self.color_by, state="readonly", width=9, values=COLOR_BY)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda _e: self.redraw())
        g = display.group()
        ttk.Label(g, text="Trail [frames]:").pack(side="left", padx=(0, 2))
        tk.Spinbox(g, from_=0, to=100, textvariable=self.trail, width=4,
                   command=self.redraw).pack(side="left")
        ttk.Label(g, text="Min SNR [dB]:").pack(side="left", padx=(8, 2))
        e = tk.Entry(g, textvariable=self.min_snr, width=4)
        e.pack(side="left")
        e.bind("<Return>", lambda _e: self.redraw())
        e.bind("<FocusOut>", lambda _e: self.redraw())
        g = display.group()
        tk.Checkbutton(g, text="Objects", variable=self.show_objects,
                       command=self.redraw).pack(side="left")
        tk.Checkbutton(g, text="Image under the points", variable=self.show_image,
                       command=self.redraw).pack(side="left")
        tk.Checkbutton(g, text="Echoes", variable=self.show_echoes,
                       command=self.redraw).pack(side="left")
        tk.Checkbutton(g, text="Other points (noise, short-lived objects)", variable=self.show_others,
                       command=self.redraw).pack(side="left")

        self.summary = tk.Label(parent, text="(press Run Imaging: the 4D point cloud of every "
                                             "frame - x, y, z and velocity - and images of the scene)",
                                anchor="w", justify="left", font=theme.font(9), wraplength=900)
        self.summary.pack(side="top", fill="x", padx=10)

        self.viz_nb = ttk.Notebook(parent)
        self.viz_nb.pack(side="top", fill="both", expand=True, padx=6, pady=4)
        self.figs, self.canvases = {}, {}
        for name in PAGES:
            page = ttk.Frame(self.viz_nb)
            self.viz_nb.add(page, text=name)
            fig = Figure(figsize=(8, 6))
            canvas, _ = create_interactive_canvas(page, fig)
            fig.text(0.5, 0.5, "Press Run Imaging.", ha="center", va="center", color="gray")
            self.figs[name], self.canvases[name] = fig, canvas
        self.viz_nb.bind("<<NotebookTabChanged>>", lambda _e: self.show_frame(self._frame))
        if header_extras is not None:
            header_extras(run_group, self.figs["Bird's-eye"], self.points_for_export,
                          [("Shown page, all frames (Video)", self.export_page_video),
                           ("Shown page (PNG)", self.export_page_png),
                           ("Point cloud of the frame (PLY)", self.export_ply_frame),
                           ("Point cloud of every frame (PLY)", self.export_ply_all),
                           ("Objects and tracks (CSV)", self.export_objects)],
                          lambda i: self._draw_bev(int(i)))

    # --- running -------------------------------------------------------------------------
    def settings(self) -> radar_imaging.Settings | None:
        try:
            thr = float(self.threshold.get())
            r_max = float(self.r_max.get()) if self.r_max.get().strip() else None
            h = float(self.height.get() or 0)
            tilt = float(self.tilt.get() or 0)
        except ValueError as e:
            messagebox.showerror("Imaging", f"Invalid setting: {e}")
            return None
        if thr <= 0 or (r_max is not None and r_max <= 0.5) or not -90 < tilt < 90:
            messagebox.showerror("Imaging", "Threshold above 0 dB, range to above 0.5 m and a tilt "
                                            "between -90 and 90 degrees.")
            return None
        return radar_imaging.Settings(mode=MODE_LABELS[self.mode.get()], threshold_db=thr,
                                      r_max_m=r_max, height_m=h, tilt_deg=tilt,
                                      n_fft_range=self.fft_size())

    def run(self):
        s = self.settings()
        if s is None:
            return
        data = self.get_raw()
        if data is None:
            return
        cube, params, fps, t0_s, self.name = data
        try:
            res = self.run_fn(radar_imaging.analyse, cube=cube, params=params, fps=fps,
                              settings=s, stem=self.name, t0_s=t0_s)
        except Exception as e:
            self.report_error("Imaging", f"Imaging failed: {e}", e)
            return
        if res is None:
            return
        self.result, self.cube, self.params = res, cube, params
        self._images.clear()
        self._drawn.clear()
        self._art.clear()
        text = res.summary()
        print(f"[Imaging] {self.name}: {text}")
        for n in res.notes:
            print(f"[Imaging]   {n}")
        self.summary.configure(text=text + ("" if not res.notes else "  (" + "; ".join(res.notes) + ")"))
        self.redraw()

    def clear(self, text: str):
        """Forget the result (a new recording, or one whose data changed)."""
        if self.result is None:
            return
        self.result = self.cube = self.params = None
        self._images.clear()
        self._drawn.clear()
        self._art.clear()
        self.summary.configure(text=text)
        for name in PAGES:
            fig = self.figs[name]
            fig.clear()
            fig.text(0.5, 0.5, text, ha="center", va="center", color="gray")
            self.canvases[name].draw_idle()

    # --- what is shown -----------------------------------------------------------------------
    def page(self) -> str:
        try:
            return self.viz_nb.tab(self.viz_nb.select(), "text")
        except tk.TclError:
            return PAGES[0]

    def layered(self) -> bool:
        """Whether the people are a layer of their own (moving and micro-motion modes with angles):
        the rest - noise, short-lived reflections - only with Other points."""
        res = self.result
        return bool(res is not None and res.has_angles and res.settings.mode in ("people", "moving", "micro"))

    def shown_points(self, f: int, others: bool = False) -> np.ndarray:
        """Points of frame f and its trail, above the display's SNR: the people's, or with
        others=True the rest of them (none unless Other points is on)."""
        res = self.result
        try:
            trail = max(0, int(float(self.trail.get() or 0)))
        except ValueError:
            trail = 0
        q = res.frame(f, trail)
        try:
            lo = float(self.min_snr.get()) if self.min_snr.get().strip() else None
        except ValueError:
            lo = None
        if lo is not None and q.size:
            q = q[q["snr_db"] >= lo]
        if self.layered() and q.size:
            # a person's points are theirs even when taken for an echo of someone else
            m = res.person_mask(q)
            if not others:
                return q[m]
            q = q[~m] if self.show_others.get() else q[:0]
        elif others:
            return q[:0]
        if not self.show_echoes.get() and q.size:
            q = q[q["kind"] != radar_imaging.KIND_ECHO]
        return q

    def images(self, f: int) -> dict:
        f = int(f)
        if f in self._images:
            self._images.move_to_end(f)
            return self._images[f]
        q = self.result.frame(f)
        q = q[q["kind"] != radar_imaging.KIND_ECHO] if q.size else q
        im = radar_imaging.frame_images(self.cube, self.params, f, self.result.settings, fps=self.result.fps,
                                        at_ranges_m=q["range_m"] if q.size else None)
        self._images[f] = im
        while len(self._images) > IMAGE_CACHE:
            self._images.popitem(last=False)
        return im

    def redraw(self):
        if self.result is None:
            return
        self._drawn.clear()
        self._art.clear()
        self.show_frame(self._frame)

    def show_frame(self, frame: int):
        """Draw the shown page at the frame on the frame slider (the others when shown)."""
        self._frame = int(frame)
        res = self.result
        if res is None:
            return
        f = int(np.clip(self._frame, 0, res.frames - 1))
        page = self.page()
        if self._drawn.get(page) == f and page != "Tracks":
            return
        try:
            {"3D": self._draw_3d, "Bird's-eye": self._draw_bev, "Radar vision": self._draw_front,
             "Range-Doppler": self._draw_rd, "Tracks": self._draw_tracks}[page](f)
        except Exception as e:                      # a view must not stop the playback
            print(f"[Imaging] {page} at frame {f}: {e}")
            return
        self._drawn[page] = f

    def _title(self, f: int, extra: str = "") -> str:
        res = self.result
        q = res.frame(f)
        if not res.has_angles:
            extra += ("\nNO ANGLES - range only, every point on the axis (the array is unknown, or the "
                      "cascade sends on all TX at once / has no calibration)")
        if self.layered():
            n_p = int(res.person_mask(q).sum())
            what = f"{n_p} of them people's, {len(res.objects(f, people_only=True))} people"
        else:
            what = f"{len(res.objects(f))} objects"
        return (f"{res.stem}  t = {res.t0_s + f / res.fps:.2f} s (frame {f})  -  {q.size} points, "
                f"{what}{extra}")

    def _objects(self, f: int) -> list:
        if not self.show_objects.get():
            return []
        return self.result.objects(f, echoes=self.show_echoes.get(),
                                   people_only=self.layered() and not self.show_others.get())

    # --- pages ---------------------------------------------------------------------------------
    def _limits(self):
        res = self.result
        r = float(res.range_m[-1]) if res.range_m.size else 8.0
        if res.points.size:
            r = min(r, max(2.0, float(np.percentile(res.points["range_m"], 99.5)) + 0.5))
        lim = np.deg2rad(min(res.settings.az_limit_deg, 89.0))
        half = r * np.sin(lim)
        return r, half

    def _draw_3d(self, f: int):
        res = self.result
        fig = self.figs["3D"]
        q = self.shown_points(f)
        by = self.color_by.get()
        art = self._art.get("3D")
        if art is None:
            view = getattr(self, "_view3d", (22, -60))
            fig.clear()
            ax = fig.add_subplot(111, projection="3d")
            r, half = self._limits()
            h = res.settings.height_m
            ax.set_xlim(-half, half)
            ax.set_ylim(0, r)
            zs = res.points["z_m"] if res.points.size else np.zeros(1)
            z0 = min(-h if h > 0 else float(np.percentile(zs, 1)) if zs.size else -1.0, -0.5)
            z1 = max(float(np.percentile(zs, 99)) if zs.size else 1.0, 0.5)
            ax.set_zlim(z0, z1)
            ax.set_xlabel("x across [m]")
            ax.set_ylabel("y away [m]")
            ax.set_zlabel("z up [m]")
            ax.plot([0], [0], [h], marker="^", color="black", ms=8)
            lim = np.deg2rad(res.settings.az_limit_deg)
            for a in (-lim, lim):
                ax.plot([0, r * np.sin(a)], [0, r * np.cos(a)], [h, h], color="0.75", lw=0.8)
            if h > 0:
                gx, gy = np.meshgrid(np.linspace(-half, half, 9), np.linspace(0, r, 9))
                ax.plot_wireframe(gx, gy, np.zeros_like(gx), color="0.88", lw=0.5)
            vals, cmap, vmin, vmax, label = colour_values(res.points[:1], by, res)
            sc_o = ax.scatter([], [], [], s=5, color="0.7", depthshade=False)
            sc = ax.scatter([], [], [], s=12, depthshade=False)
            if cmap is not None:
                sc.set_cmap(cmap)
                sc.set_clim(vmin, vmax)
                fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.08, label=label)
            ax.view_init(*view)
            title = ax.set_title("", fontsize=9)
            art = self._art["3D"] = dict(ax=ax, sc=sc, sc_o=sc_o, title=title, extra=[])
        ax, sc = art["ax"], art["sc"]
        o_pts = self.shown_points(f, others=True)
        art["sc_o"]._offsets3d = (o_pts["x_m"], o_pts["y_m"], o_pts["z_m"])
        self._view3d = (ax.elev, ax.azim)
        for a in art["extra"]:
            a.remove()
        art["extra"] = []
        vals, cmap, vmin, vmax, _ = colour_values(q, by, res)
        sc._offsets3d = (q["x_m"], q["y_m"], q["z_m"])
        sizes = 6 + 3 * np.clip((q["snr_db"] - res.settings.threshold_db) / 3, 0, 10) if q.size else []
        sc.set_sizes(sizes if q.size else [6])
        if cmap is not None:
            sc.set_array(np.asarray(vals, float))
        else:
            sc.set_array(None)
            sc.set_facecolors(vals if q.size else [])
            sc.set_edgecolors(vals if q.size else [])
        if q.size:
            age = f - q["frame"]
            alpha = np.clip(1.0 - age / (age.max() + 1.0), 0.25, 1.0)
            sc.set_alpha(None)
            fc = sc.to_rgba(np.asarray(vals, float)) if cmap is not None else \
                np.array([to_rgba(c) for c in vals])
            fc[:, 3] = alpha
            sc.set_facecolors(fc)
            sc.set_edgecolors(fc)
        for o in self._objects(f):
            sx, sy, sz = (max(0.3, v) for v in o["size_m"])
            x0, x1 = o["x_m"] - sx / 2, o["x_m"] + sx / 2
            y0, y1 = o["y_m"] - sy / 2, o["y_m"] + sy / 2
            z0, z1 = o["z_m"] - sz / 2, o["z_m"] + sz / 2
            c = obj_color(o)
            for xs, ys, zs in (([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [z0] * 5),
                               ([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [z1] * 5)):
                art["extra"] += ax.plot(xs, ys, zs, color=c, lw=1.2)
            for xx, yy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
                art["extra"] += ax.plot([xx, xx], [yy, yy], [z0, z1], color=c, lw=1.2)
            art["extra"].append(ax.text(o["x_m"], o["y_m"], z1 + 0.1, f"#{o['track']} {o['velocity_m_s']:+.1f} m/s",
                                        color=c, fontsize=8, fontweight="bold"))
            tr = res.trail(o["track"], f, TRAIL_S)
            if tr.shape[0] > 1:
                art["extra"] += ax.plot(tr[:, 0], tr[:, 1], np.full(tr.shape[0], z0), color=c, lw=1, alpha=0.7)
        art["title"].set_text(self._title(f))
        self.canvases["3D"].draw_idle()

    def _draw_bev(self, f: int):
        res = self.result
        fig = self.figs["Bird's-eye"]
        fig.clear()
        ax = fig.add_subplot(111)
        r, half = self._limits()
        if self.show_image.get() and res.has_angles:
            im = self.images(f)
            bev = im.get("bev")
            if bev is not None:
                rr, az = im["range_m"], im["az_deg"]
                dr = float(np.median(np.diff(rr))) if rr.size > 1 else 0.1
                da = float(np.median(np.diff(az))) if az.size > 1 else 1.0
                re = np.concatenate([rr - dr / 2, [rr[-1] + dr / 2]])
                ae = np.deg2rad(np.concatenate([az - da / 2, [az[-1] + da / 2]]))
                keep = re <= r + dr
                re = re[keep]
                B = 10 * np.log10(np.maximum(bev[:re.size - 1], 1e-30))
                hi = float(B.max())
                X = re[:, None] * np.sin(ae)[None, :]
                Y = re[:, None] * np.cos(ae)[None, :]
                ax.pcolormesh(X, Y, B, cmap="Greys", vmin=hi - IMAGE_RANGE_DB, vmax=hi, shading="flat",
                              alpha=0.55, zorder=0, rasterized=True)
        lim = np.deg2rad(res.settings.az_limit_deg)
        for a in (-lim, lim):
            ax.plot([0, r * np.sin(a)], [0, r * np.cos(a)], color="0.7", lw=0.8, zorder=1)
        th = np.linspace(-lim, lim, 60)
        for rr in np.arange(1.0, r + 0.01, 1.0):
            ax.plot(rr * np.sin(th), rr * np.cos(th), color="0.85", lw=0.6, zorder=1)
        ax.plot([0], [0], marker="^", color="black", ms=10, zorder=3)
        o_pts = self.shown_points(f, others=True)
        if o_pts.size:
            ax.scatter(o_pts["x_m"], o_pts["y_m"], s=7, color="0.65", edgecolors="none", zorder=3)
        q = self.shown_points(f)
        by = self.color_by.get()
        vals, cmap, vmin, vmax, label = colour_values(q, by, res)
        if q.size:
            age = f - q["frame"]
            sizes = 10 + 4 * np.clip((q["snr_db"] - res.settings.threshold_db) / 3, 0, 10)
            sc = ax.scatter(q["x_m"], q["y_m"], c=vals, cmap=cmap, vmin=vmin, vmax=vmax, s=sizes,
                            alpha=None, edgecolors="none", zorder=4)
            if age.max() > 0:
                sc.set_alpha(np.clip(1.0 - age / (age.max() + 1.0), 0.25, 1.0))
            if cmap is not None:
                fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label=label)
        for o in self._objects(f):
            c = obj_color(o)
            tr = res.trail(o["track"], f, TRAIL_S)
            if tr.shape[0] > 1:
                ax.plot(tr[:, 0], tr[:, 1], "-", color=c, lw=1.5, alpha=0.8, zorder=5)
            rad = max(0.25, 0.5 * max(o["size_m"][0], o["size_m"][1]))
            ax.add_patch(Circle((o["x_m"], o["y_m"]), rad, fill=False, color=c, lw=2, zorder=6))
            ax.text(o["x_m"] + rad, o["y_m"] + rad, f"#{o['track']}  {o['velocity_m_s']:+.1f} m/s\n"
                    f"z {o['z_m']:+.1f} m, {o['n']} pts", color=c, fontsize=8, fontweight="bold", zorder=7,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.2), clip_on=True)
        ax.set_aspect("equal")
        ax.set_xlim(-half - 0.3, half + 0.3)
        ax.set_ylim(-0.2, r + 0.2)
        ax.set_xlabel("x across [m] (angle sign as in the Angle of Arrival views)")
        ax.set_ylabel("y away from the radar [m]")
        ax.set_title(self._title(f), fontsize=9)
        self.canvases["Bird's-eye"].draw_idle()

    def _draw_front(self, f: int):
        res = self.result
        fig = self.figs["Radar vision"]
        fig.clear()
        ax = fig.add_subplot(111)
        if not res.has_angles:
            ax.text(0.5, 0.5, "No antenna geometry: no angles.", ha="center", transform=ax.transAxes)
            self.canvases["Radar vision"].draw_idle()
            return
        im = self.images(f) if self.show_image.get() else {}
        if "front" in im:
            Fm = 10 * np.log10(im["front"])
            hi = float(Fm.max())
            az, el = im["front_az_deg"], im["front_el_deg"]
            # white where the radar sees most, like a camera's picture of it
            ax.imshow(Fm, origin="lower", aspect="auto", cmap="gray", vmin=hi - IMAGE_RANGE_DB, vmax=hi,
                      extent=[az[0], az[-1], el[0], el[-1]], zorder=0, interpolation="bilinear")
        elif "bev" in im:
            prof = 10 * np.log10(np.maximum(im["bev"].sum(axis=0), 1e-30))
            ax2 = ax.twinx()
            ax2.plot(im["az_deg"], prof, color="0.6", lw=1)
            ax2.set_ylabel("power over all ranges [dB]", color="0.5")
        o_pts = self.shown_points(f, others=True)
        if o_pts.size:
            ax.scatter(o_pts["azimuth_deg"], o_pts["elevation_deg"], s=8, color="0.65", edgecolors="none", zorder=2)
        q = self.shown_points(f)
        if q.size:
            sizes = 14 + 5 * np.clip((q["snr_db"] - res.settings.threshold_db) / 3, 0, 10)
            sc = ax.scatter(q["azimuth_deg"], q["elevation_deg"], c=q["range_m"], cmap="turbo_r",
                            vmin=0, vmax=self._limits()[0], s=sizes, edgecolors="k", linewidths=0.3, zorder=3)
            fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="Range [m] (depth)")
        for o in self._objects(f):
            rng = np.sqrt(o["x_m"] ** 2 + o["y_m"] ** 2 + (o["z_m"] - res.settings.height_m) ** 2)
            a = np.rad2deg(np.arctan2(o["x_m"], o["y_m"]))
            e = np.rad2deg(np.arcsin(np.clip((o["z_m"] - res.settings.height_m) / max(rng, 1e-6), -1, 1)))
            c = obj_color(o)
            ax.text(a, e, f"#{o['track']} {rng:.1f} m", color=c, fontsize=9, fontweight="bold", zorder=5,
                    ha="center", va="bottom", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
        s = res.settings
        ax.set_xlim(-s.az_limit_deg, s.az_limit_deg)
        ax.set_ylim(-s.el_limit_deg if res.has_elevation else -10, s.el_limit_deg if res.has_elevation else 10)
        ax.set_xlabel("Azimuth [°]")
        ax.set_ylabel("Elevation [°]" + ("" if res.has_elevation else " (the array has no vertical extent)"))
        ax.grid(alpha=0.25)
        ax.set_title("Radar vision - what the radar sees, the points by distance\n" + self._title(f), fontsize=9)
        self.canvases["Radar vision"].draw_idle()

    def _draw_rd(self, f: int):
        res = self.result
        fig = self.figs["Range-Doppler"]
        fig.clear()
        ax = fig.add_subplot(111)
        im = self.images(f)
        rd = 10 * np.log10(np.maximum(im["rd"], 1e-30))
        r, v = im["range_m"], im["velocity_m_s"]
        hi = float(rd.max())
        mesh = ax.imshow(rd, origin="lower", aspect="auto", cmap="viridis", vmin=hi - 50, vmax=hi,
                         extent=[r[0], r[-1], v[0], v[-1]], zorder=0)
        fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.02, label="Power, all channels [dB]")
        q = res.frame(f)
        if q.size:
            moving = q["kind"] == radar_imaging.KIND_MOVING
            ax.scatter(q["range_m"][moving], q["velocity_m_s"][moving], s=40, facecolors="none",
                       edgecolors="red", linewidths=1.2, zorder=3, label="points (unfolded velocity)")
            for kind, mk, col, lab in ((radar_imaging.KIND_MICRO, "D", "cyan", "micro-motion (across frames)"),
                                       (radar_imaging.KIND_STILL, "s", "orange", "still scene"),
                                       (radar_imaging.KIND_ECHO, "x", "0.6", "echoes")):
                m = q["kind"] == kind
                if m.any() and mk == "x":
                    ax.scatter(q["range_m"][m], q["velocity_m_s"][m], s=25, marker=mk, color=col,
                               zorder=3, label=lab)
                elif m.any():
                    ax.scatter(q["range_m"][m], q["velocity_m_s"][m], s=25, marker=mk,
                               facecolors="none", edgecolors=col, zorder=3, label=lab)
            vm = float(np.abs(q["velocity_m_s"]).max())
            if vm > abs(v[-1]):
                ax.axhspan(v[-1], vm + 0.5, color="0.9", zorder=-1)
                ax.axhspan(-vm - 0.5, v[0], color="0.9", zorder=-1)
                ax.set_ylim(-vm - 0.5, vm + 0.5)
            ax.legend(fontsize=8, loc="upper right")
        if v.size <= radar_imaging.MIN_LOOPS_DOPPLER:
            ax.text(0.5, 0.92, f"{v.size} chirps of a TX a frame: no Doppler in a frame - people show as "
                    "micro-motion (across frames)", transform=ax.transAxes, ha="center", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
        ax.set_xlim(0, self._limits()[0])
        ax.set_xlabel("Range [m]")
        ax.set_ylabel("Velocity [m/s] (+ away)")
        ax.set_title("Range-Doppler of the frame (moving part) and its points\n" + self._title(f), fontsize=9)
        self.canvases["Range-Doppler"].draw_idle()

    def _draw_tracks(self, f: int):
        res = self.result
        fig = self.figs["Tracks"]
        if self._art.get("Tracks") is None:
            fig.clear()
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
            ax = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            r, half = self._limits()
            layered = self.layered()
            persons = getattr(res, "person_tracks", set())
            for tid, tr in sorted(res.trails.items()):
                if tr.shape[0] < 2:
                    continue
                if layered and tid not in persons:
                    if self.show_others.get():
                        ax.plot(tr[:, 1], tr[:, 2], "-", color="0.75", lw=0.7, zorder=1)
                    continue
                c = track_color(tid)
                ax.plot(tr[:, 1], tr[:, 2], "-", color=c, lw=1.2, alpha=0.85, zorder=2)
                ax.plot(tr[0, 1], tr[0, 2], "o", color=c, ms=5)
                ax.text(tr[-1, 1], tr[-1, 2], f" #{tid}", color=c, fontsize=8, fontweight="bold")
            ax.plot([0], [0], marker="^", color="black", ms=9)
            ax.set_aspect("equal")
            ax.set_xlim(-half - 0.3, half + 0.3)
            ax.set_ylim(-0.2, r + 0.2)
            ax.grid(alpha=0.3)
            ax.set_xlabel("x across [m]")
            ax.set_ylabel("y away [m]")
            ax.set_title(f"{res.stem}: " + (f"{len(persons)} people's tracks of {len(res.trails)}" if layered
                                             else f"{len(res.trails)} tracks") + " (dot: where each began)", fontsize=9)
            dots, = ax.plot([], [], "o", ms=11, mfc="none", mew=2, color="k")
            n_pts = np.diff(res.starts)
            t = res.t0_s + np.arange(res.frames) / res.fps
            ax2.plot(t, n_pts, color="0.4", lw=0.8, label="points")
            ax2b = ax2.twinx()
            ax2b.plot(t, [len(res.objects(i, people_only=layered)) for i in range(res.frames)],
                      color=theme.ACCENT, lw=1.2, label="people" if layered else "objects")
            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("Points", color="0.4")
            ax2b.set_ylabel("People" if layered else "Objects", color=theme.ACCENT)
            line = ax2.axvline(t[0], color="red", lw=1)
            fig.tight_layout()
            self._art["Tracks"] = dict(dots=dots, line=line, t=t)
        art = self._art["Tracks"]
        objs = res.objects(f, people_only=self.layered())
        art["dots"].set_data([o["x_m"] for o in objs], [o["y_m"] for o in objs])
        art["line"].set_xdata([art["t"][f], art["t"][f]])
        self.canvases["Tracks"].draw_idle()

    # --- export ------------------------------------------------------------------------------
    def points_for_export(self):
        """Every point of every frame: (array, header) or None."""
        res = self.result
        if res is None or not res.points.size:
            return None
        cols = [k for k in radar_imaging.POINT_FIELDS]
        data = np.column_stack([res.points[k].astype(float) for k in cols])
        t = res.t0_s + res.points["frame"].astype(float) / res.fps
        return np.column_stack([t, data]), "time_s," + ",".join(c if c != "cluster" else "track" for c in cols)

    def _need(self) -> bool:
        if self.result is None:
            messagebox.showwarning("Export", "Run Imaging first.")
            return False
        return True

    def export_ply_frame(self):
        if not self._need():
            return
        path = filedialog.asksaveasfilename(defaultextension=".ply", filetypes=[("PLY point cloud", "*.ply")],
                                            title="Save the frame's point cloud")
        if path:
            write_ply(path, self.shown_points(int(np.clip(self._frame, 0, self.result.frames - 1))))
            messagebox.showinfo("Export", f"Point cloud saved to {path}")

    def export_ply_all(self):
        if not self._need():
            return
        path = filedialog.asksaveasfilename(defaultextension=".ply", filetypes=[("PLY point cloud", "*.ply")],
                                            title="Save the point cloud of every frame")
        if path:
            write_ply(path, self.result.points)
            messagebox.showinfo("Export", f"Point cloud saved to {path}")

    def export_objects(self):
        if not self._need():
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")],
                                            title="Save the objects of every frame as CSV")
        if not path:
            return
        res = self.result
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("time_s,frame,track,x_m,y_m,z_m,velocity_m_s,size_x_m,size_y_m,size_z_m,points,snr_db\n")
            for f in range(res.frames):
                for o in res.objects(f, confirmed=False):
                    fh.write(f"{res.t0_s + f / res.fps:.3f},{f},{o['track']},{o['x_m']:.3f},{o['y_m']:.3f},"
                             f"{o['z_m']:.3f},{o['velocity_m_s']:.3f},{o['size_m'][0]:.3f},{o['size_m'][1]:.3f},"
                             f"{o['size_m'][2]:.3f},{o['n']},{o['snr_db']:.1f}\n")
        messagebox.showinfo("Export", f"Objects saved to {path}")

    def export_page_png(self):
        if not self._need():
            return
        page = self.page()
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")],
                                            title=f"Save the {page} page")
        if path:
            self.figs[page].savefig(path, dpi=150)

    def export_page_video(self):
        if not self._need():
            return
        page = self.page()
        draw = {"3D": self._draw_3d, "Bird's-eye": self._draw_bev, "Radar vision": self._draw_front,
                "Range-Doppler": self._draw_rd, "Tracks": self._draw_tracks}[page]
        if self.export_video is None:
            messagebox.showwarning("Export", "Video export is not available here.")
            return
        self.export_video(self.figs[page], lambda i: draw(int(np.clip(i, 0, self.result.frames - 1))),
                          f"Imaging {page}")
