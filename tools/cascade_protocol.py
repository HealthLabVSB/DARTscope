"""
Verification protocol for the AWR2243 cascade presets (cascade/params.py PRESETS): every
preset recorded with the hardware and measured from its own data, so the protocol states
what a configuration does, not what it was designed to do.

    python -m tools.cascade_protocol --out ~/radar_measurements/cascade/protocol_2026-10-09
    python -m tools.cascade_protocol --out <folder> --analyse-only      # re-measure takes

Per preset (one take of --duration seconds):

    integrity     frames recorded / expected, packets the recorder had to zero-fill, the
                  firmware's own frame count
    timing        the frame period from takes of different length (timing_*s takes of the
                  first preset): the span from the first to the last packet is a + T x frames,
                  a being the DCA1000 holding its last short packet until its timeout (~2 s);
                  every take's frame rate is then frames / (span - a)
    noise         per RX, from the difference of consecutive chirps of the same TX - the
                  still scene cancels, what is left is the receiver noise (dBFS per complex
                  sample, full scale = int16 limit)
    range         the chirp-averaged range profile (Hann, 8x zero padding): its strongest
                  reflectors, their -3 dB width against the theoretical 1.44 bins of a Hann
                  window (c / 2B), their SNR; the same reflector at the same range in every
                  preset checks slope and sampling rate
    displacement  on each strong reflector, the phase across frames as displacement
                  (lambda / 4 pi); the steadiest reflector gives the instrument's floor in
                  the breathing (0.1-0.5 Hz) and heart (0.8-2.5 Hz) bands, set against the
                  floor its SNR predicts
    Doppler       the strongest reflector's velocity from the chirps of one TX: a still
                  scene has to come out at 0 m/s
    TDM           with several TX in turn: each TX slot's power on the strongest reflector
                  (does every TX transmit?) and how steady the phase of every virtual
                  channel stays against the first (needed for angles)
    DARTscope     the capture health report (capture_health.assess) and the Fastcheck's
                  vital-signs finding, as DARTscope itself shows them

Output in --out: the takes (<preset>_Raw_0.bin, _Raw_LogFile.csv, _cascade.json),
protocol.json with every number, protocol.md (a summary table) and one PNG per take.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

C = 299_792_458.0
HANN_3DB_BINS = 1.44            # -3 dB width of a Hann window's main lobe, in bins
PAD = 8                         # range FFT zero padding for the width measurement
MIN_RANGE_M = 0.4               # nearer is the TX-RX coupling
BANDS = {"breathing": (0.1, 0.5), "heart": (0.8, 2.5)}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _load(bin_path: str):
    from cascade.decode import cascade_params
    from radar_processing import derive_capture_dimensions, load_raw_cube
    with contextlib.redirect_stdout(io.StringIO()):
        params, _ = cascade_params(bin_path)
        dims = derive_capture_dimensions(params)
        cube = load_raw_cube(bin_path, params, dims)
    return params, cube


def _band_rms(x: np.ndarray, fs: float, band) -> float:
    """RMS of x within a frequency band (periodogram, Hann)."""
    n = len(x)
    w = np.hanning(n)
    spec = np.fft.rfft((x - x.mean()) * w)
    f = np.fft.rfftfreq(n, 1 / fs)
    psd = np.abs(spec) ** 2 / (fs * np.sum(w ** 2))          # one-sided after doubling
    psd[1:] *= 2
    m = (f >= band[0]) & (f <= band[1])
    return float(np.sqrt(np.sum(psd[m]) * (f[1] - f[0]))) if m.any() else float("nan")


def _width_3db(profile_db: np.ndarray, k: int) -> float:
    """-3 dB width of the peak at index k, in samples (linear interpolation)."""
    top = profile_db[k] - 3.0
    i = k
    while i > 0 and profile_db[i] > top:
        i -= 1
    j = k
    while j < len(profile_db) - 1 and profile_db[j] > top:
        j += 1
    left = i + (top - profile_db[i]) / (profile_db[i + 1] - profile_db[i])
    right = j - 1 + (profile_db[j - 1] - top) / (profile_db[j - 1] - profile_db[j])
    return float(right - left)


def analyse_take(bin_path: str) -> dict:
    """Every measurement of the protocol for one take."""
    from scipy.signal import find_peaks
    params, cube = _load(bin_path)
    meta = json.loads(Path(params["cascade_sidecar"]).read_text(encoding="utf-8"))
    rec = meta["record"]
    lay = meta["layout"]
    n_ch, loops = int(lay["chirps_in_loop"]), int(lay["num_loops"])
    F, Ctot, RX, N = cube.shape
    fps_set = float(meta["radar"]["frame_rate_hz"])
    lam = float(meta["radar"]["wavelength_m"])
    fs = float(params["digOutSampleRate_MHz"]) * 1e6
    slope = float(params["freqSlopeMHz_us"]) * 1e12
    b_eff = slope * N / fs
    dr = C / (2 * b_eff)
    out: dict = {"take": Path(bin_path).name, "preset": meta.get("notes", ""), "frames": F,
                 "chirps_per_frame": Ctot, "chirps_in_loop": n_ch, "loops": loops, "rx": RX, "samples": N,
                 "range_bin_m": dr, "bandwidth_sampled_ghz": b_eff / 1e9, "wavelength_mm": lam * 1e3}

    # --- integrity and timing
    first, last = rec.get("first_packet_ns"), rec.get("last_packet_ns")
    out["integrity"] = {"frames_recorded": rec["frames_in_file"], "frames_expected": meta["radar"]["num_frames"],
                        "zero_filled_packets": rec["zero_filled_packets"],
                        "received_packets": rec.get("received_packets"),
                        "firmware": rec.get("firmware_status"), "complete": rec["complete"]}
    if first and last and F > 1:
        span = (last - first) / 1e9
        bytes_frame = Ctot * RX * N * 4
        drain = bytes_frame / (1456 / meta["dca1000"]["packet_delay_us"] * 1e6)
        out["timing"] = {"fps_set": fps_set, "frames": F, "packet_span_s": span, "frame_drain_s": drain}

    # the chirps of the first TX slot: (frames, loops, rx, samples)
    x0 = cube[:, 0::n_ch]
    # --- noise: consecutive chirps of the same TX, the still scene cancels
    d = x0[:, 1:] - x0[:, :-1]
    noise_var = np.mean(np.abs(d) ** 2, axis=(0, 1, 3)) / 2                       # per RX
    out["noise_dbfs_per_rx"] = [float(v) for v in 10 * np.log10(noise_var)]

    # --- range: chirp-averaged profile
    win = np.hanning(N).astype(np.float32)
    nfft = PAD * int(2 ** np.ceil(np.log2(N)))
    avg = x0.mean(axis=1)                                                        # (F, RX, N)
    spec = np.fft.fft((avg - avg.mean(-1, keepdims=True)) * win, n=nfft, axis=-1)
    prof = np.mean(np.abs(spec) ** 2, axis=(0, 1))                              # (nfft,)
    prof_db = 10 * np.log10(prof + 1e-30)
    r_axis = np.arange(nfft) * dr * N / nfft
    usable = (r_axis >= MIN_RANGE_M) & (r_axis <= 0.85 * r_axis[-1] * 1.0)
    # noise level of the averaged profile: what the noise variance gives after loops averaging
    noise_bin = float(np.mean(noise_var)) / loops * float(np.sum(win ** 2))
    peaks, _ = find_peaks(np.where(usable, prof_db, -300), prominence=10, distance=PAD * 2)
    peaks = sorted(peaks, key=lambda k: prof_db[k], reverse=True)[:6]
    refl = []
    pad = nfft / N                           # FFT points per range bin (8 unless N is not a power of 2)
    for k in peaks:
        w_bins = _width_3db(prof_db, k) / pad
        refl.append({"range_m": float(r_axis[k]), "power_db": float(prof_db[k]),
                     "snr_db": float(prof_db[k] - 10 * np.log10(noise_bin)),
                     "width_3db_cm": w_bins * dr * 100, "width_theory_cm": HANN_3DB_BINS * dr * 100,
                     "width_ratio": w_bins / HANN_3DB_BINS})
    out["reflectors"] = refl
    out["range_profile"] = {"range_m": r_axis[::PAD // 2].tolist(), "db": prof_db[::PAD // 2].tolist()}

    # --- displacement on each strong reflector (frame-rate slow time)
    spec0 = np.fft.fft((avg - avg.mean(-1, keepdims=True)) * win, axis=-1)          # (F, RX, N)
    disp_rows = []
    t = np.arange(F) / fps_set
    for r in refl:
        b = int(round(r["range_m"] / dr))
        rx_best = int(np.argmax(np.mean(np.abs(spec0[:, :, b]) ** 2, axis=0)))
        z = spec0[:, rx_best, b]
        ph = np.unwrap(np.angle(z))
        disp = ph * lam / (4 * np.pi)
        disp = disp - np.polyval(np.polyfit(t, disp, 1), t)
        snr_lin = np.mean(np.abs(z) ** 2) / (noise_var[rx_best] / loops * np.sum(win ** 2))
        predicted = lam / (4 * np.pi) / np.sqrt(2 * snr_lin)
        row = {"range_m": r["range_m"], "rx": rx_best, "rms_um": float(np.std(disp) * 1e6),
               "predicted_rms_um": float(predicted * 1e6), "snr_db": float(10 * np.log10(snr_lin))}
        for name, band in BANDS.items():
            row[f"{name}_band_um"] = float(_band_rms(disp, fps_set, band) * 1e6)
        # thermal floor in a band: white phase noise spread over fps/2
        for name, band in BANDS.items():
            row[f"{name}_band_predicted_um"] = float(predicted * np.sqrt((band[1] - band[0]) / (fps_set / 2)) * 1e6)
        disp_rows.append(row)
    out["displacement"] = disp_rows
    if disp_rows:
        steady = min(disp_rows, key=lambda r: r["heart_band_um"])
        out["displacement_floor"] = steady
        # the most moving reflector, as an indication of a person
        mover = max(disp_rows, key=lambda r: r["breathing_band_um"])
        b = int(round(mover["range_m"] / dr))
        z = spec0[:, mover["rx"], b]
        disp = np.unwrap(np.angle(z)) * lam / (4 * np.pi)
        disp = disp - np.polyval(np.polyfit(t, disp, 1), t)
        out["displacement_trace"] = {"steady_range_m": steady["range_m"], "mover_range_m": mover["range_m"],
                                     "t": t[:: max(1, F // 1500)].tolist(),
                                     "mover_um": (disp[:: max(1, F // 1500)] * 1e6).tolist()}
        bs = int(round(steady["range_m"] / dr))
        zs = spec0[:, steady["rx"], bs]
        ds = np.unwrap(np.angle(zs)) * lam / (4 * np.pi)
        ds = ds - np.polyval(np.polyfit(t, ds, 1), t)
        out["displacement_trace"]["steady_um"] = (ds[:: max(1, F // 1500)] * 1e6).tolist()

    # --- Doppler of the strongest reflector, chirps of one TX
    if refl and loops >= 4:
        b = int(round(refl[0]["range_m"] / dr))
        rp = np.fft.fft(x0 * win, axis=-1)[:, :, :, b]                              # (F, loops, RX)
        nd = 8 * loops
        dsp = np.fft.fftshift(np.fft.fft(rp * np.hanning(loops)[None, :, None], n=nd, axis=1), axes=1)
        pw = np.mean(np.abs(dsp) ** 2, axis=(0, 2))
        v_axis = np.fft.fftshift(np.fft.fftfreq(nd, d=meta["radar"]["chirp_time_us"] * 1e-6 * n_ch)) * lam / 2
        dv = lam / (2 * loops * meta["radar"]["chirp_time_us"] * 1e-6 * n_ch)
        out["doppler"] = {"range_m": refl[0]["range_m"], "velocity_m_s": float(v_axis[int(np.argmax(pw))]),
                          "bin_width_m_s": float(dv), "v_max_m_s": float(abs(v_axis[0]))}

    # --- TDM: does every TX slot transmit, and are the virtual channels coherent?
    if n_ch > 1 and refl:
        b = int(round(refl[0]["range_m"] / dr))
        slots = cube.reshape(F, loops, n_ch, RX, N).mean(axis=1)                       # (F, slot, RX, N)
        sp = np.fft.fft((slots - slots.mean(-1, keepdims=True)) * win, axis=-1)[..., b]   # (F, slot, RX)
        pwr = np.mean(np.abs(sp) ** 2, axis=(0, 2))
        masks = [int(a) | (int(c) << 3) for a, c in lay["tx_chirps"]]
        rel = np.angle(sp * np.conj(sp[:, :1, :1]))                                     # vs channel (0, 0)
        rel = np.angle(np.exp(1j * (rel - np.angle(np.mean(np.exp(1j * rel), axis=0, keepdims=True)))))
        out["tdm"] = {"range_m": refl[0]["range_m"],
                      "slot_masks": masks,
                      "slot_power_db": [float(v) for v in 10 * np.log10(pwr / pwr.max())],
                      "channel_phase_std_deg": float(np.degrees(np.std(rel, axis=0)).mean()),
                      "channel_phase_std_deg_max": float(np.degrees(np.std(rel, axis=0)).max()),
                      "virtual_channels": int(n_ch * RX)}

    # --- DARTscope's own verdicts
    try:
        import capture_health
        from radar_processing import get_all_bin_parts
        with contextlib.redirect_stdout(io.StringIO()):
            rep = capture_health.assess(Path(bin_path).name, params, bin_paths=get_all_bin_parts(bin_path),
                                        cube=cube)
        out["dartscope_health"] = [{"check": f.check, "severity": f.severity, "message": f.message}
                                   for f in rep.findings]
    except Exception as e:
        out["dartscope_health"] = [{"check": "error", "severity": "error", "message": str(e)}]
    try:
        import fastcheck
        from capture_params import derive_fps
        with contextlib.redirect_stdout(io.StringIO()):
            chk = fastcheck.run(Path(bin_path).name, params, cube, fps=derive_fps(params))
        vit = [f for f in chk.findings if f.check == "vital signs"]
        out["dartscope_vital_signs"] = vit[0].message if vit else None
    except Exception as e:
        out["dartscope_vital_signs"] = f"error: {e}"
    return out


def plot_take(res: dict, png: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    rp = res["range_profile"]
    a = ax[0, 0]
    a.plot(rp["range_m"], rp["db"], lw=0.8)
    for r in res["reflectors"]:
        a.axvline(r["range_m"], color="r", ls=":", lw=0.8)
    a.set(title=f"Range profile (bin {res['range_bin_m'] * 100:.2f} cm)", xlabel="range [m]", ylabel="dB")
    a.grid(alpha=0.3)
    a = ax[0, 1]
    a.bar([f"RX{i + 1}" for i in range(len(res["noise_dbfs_per_rx"]))], res["noise_dbfs_per_rx"])
    a.set(title="Receiver noise per complex sample", ylabel="dBFS", ylim=(min(res["noise_dbfs_per_rx"]) - 5,
                                                                       max(res["noise_dbfs_per_rx"]) + 3))
    a = ax[1, 0]
    tr = res.get("displacement_trace")
    if tr:
        a.plot(tr["t"], tr["mover_um"], lw=0.7, label=f"most moving reflector {tr['mover_range_m']:.2f} m")
        a.plot(tr["t"], tr["steady_um"], lw=0.7, label=f"steadiest reflector {tr['steady_range_m']:.2f} m")
        a.legend(fontsize=8)
    a.set(title="Displacement (phase x lambda / 4 pi, detrended)", xlabel="time [s]", ylabel="µm")
    a.grid(alpha=0.3)
    a = ax[1, 1]
    if res.get("tdm"):
        td = res["tdm"]
        names = [f"0x{m:02x}" for m in td["slot_masks"]]
        a.bar(names, td["slot_power_db"])
        a.set(title=f"Power per TX slot at {td['range_m']:.2f} m; channel phase std "
                    f"{td['channel_phase_std_deg']:.1f} deg", ylabel="dB rel. strongest", xlabel="TX mask")
    elif res.get("doppler"):
        dop = res["doppler"]
        a.text(0.05, 0.6, f"Doppler of the strongest reflector ({dop['range_m']:.2f} m):\n"
                          f"{dop['velocity_m_s']:+.3f} m/s (bin {dop['bin_width_m_s']:.3f} m/s, "
                          f"max ±{dop['v_max_m_s']:.2f} m/s)", transform=a.transAxes)
        a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png, dpi=90)
    plt.close(fig)


def summary_md(results: dict, envelope: dict | None, timing_fit: dict | None = None) -> str:
    lines = ["# AWR2243 cascade - preset verification", "",
             "Generated by `python -m tools.cascade_protocol`; every number below is measured from the takes "
             "in this folder (protocol.json holds them all). Method: see the module docstring of "
             "tools/cascade_protocol.py.", "",
             "| preset | frames | lost pkts | fps set / measured | noise dBFS (RX range) | narrowest -3 dB width (theory) | displacement floor heart band | Doppler of still reflector | TDM slots / phase std |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        if "error" in r:
            lines.append(f"| {name} | error: {r['error']} |")
            continue
        integ = r["integrity"]
        tim = r.get("timing", {})
        nz = r["noise_dbfs_per_rx"]
        best = min(r["reflectors"], key=lambda q: q["width_3db_cm"]) if r["reflectors"] else None
        fl = r.get("displacement_floor")
        dop = r.get("doppler")
        td = r.get("tdm")
        lines.append(
            f"| {name} | {integ['frames_recorded']}/{integ['frames_expected']} | {integ['zero_filled_packets']} | "
            f"{tim.get('fps_set', 0):.2f} / {tim.get('fps_measured', float('nan')):.2f} | "
            f"{min(nz):.1f} .. {max(nz):.1f} | "
            + (f"{best['width_3db_cm']:.2f} cm at {best['range_m']:.2f} m ({best['width_theory_cm']:.2f}) | " if best else "- | ")
            + (f"{fl['heart_band_um']:.2f} µm (thermal {fl['heart_band_predicted_um']:.2f}) | " if fl else "- | ")
            + (f"{dop['velocity_m_s']:+.3f} m/s (bin {dop['bin_width_m_s']:.2f}) | " if dop else "- | ")
            + (f"{len(td['slot_masks'])} TX, min {min(td['slot_power_db']):.1f} dB / {td['channel_phase_std_deg']:.2f}° |" if td else "- |"))
    if timing_fit:
        lines += ["", f"Frame period from {len(timing_fit['takes'])} takes of {timing_fit['preset']}: "
                      f"{timing_fit['frame_period_ms']:.4f} ms ({timing_fit['fps']:.3f} fps); DCA1000 tail "
                      f"{timing_fit['dca_tail_s']:.3f} s; largest residual {timing_fit['residual_ms']:.1f} ms."]
    if envelope:
        lines += ["", "## Capability envelope (2 s takes at the edges)", "",
                  "| test | settings | result |", "|---|---|---|"]
        for name, row in envelope.items():
            if row.get("host_errors"):
                res = "refused by the checks"
            elif row.get("ok") and row.get("frames") == row.get("expected") and not row.get("lost"):
                res = f"ran: {row['frames']}/{row['expected']} frames, 0 lost"
            elif row.get("ok") and row.get("frames") == 0:
                res = "stalled after the first chirp"
            elif row.get("ok"):
                res = f"ran: {row.get('frames')}/{row.get('expected')} frames, {row.get('lost')} packets lost (first frame)"
            else:
                res = f"failed: {row.get('error')}"
            lines.append(f"| {name} | {json.dumps(row.get('settings'))} | {res} |")
    lines += ["", "## Not verified here", "",
              "- absolute range (needs a reflector at a taped distance); the presets agree with each other",
              "- velocity scale (needs a moving target); a still reflector reads 0 m/s",
              "- angles: the antenna configuration follows TI's SWRU639 figures; needs a corner-reflector calibration",
              "- breathing and heart rate against a reference (chest belt, pulse oximeter)"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    from cascade.params import PRESETS, CascadeParams
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="folder for the takes and the protocol")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds per take")
    ap.add_argument("--presets", nargs="*", help="preset names (default: all but the SDK profile)")
    ap.add_argument("--analyse-only", action="store_true", help="measure takes already in --out")
    ap.add_argument("--report-only", action="store_true", help="rewrite protocol.md from protocol.json")
    args = ap.parse_args(argv)
    out = Path(os.path.expanduser(args.out))
    if args.report_only:
        d = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
        (out / "protocol.md").write_text(summary_md(d["results"], d.get("envelope"), d.get("timing_fit")),
                                          encoding="utf-8")
        return
    out.mkdir(parents=True, exist_ok=True)
    names = args.presets or [n for n in PRESETS if not n.startswith("SDK")]
    results = {}
    for name in names:
        prefix = slug(name)
        first = out / f"{prefix}_Raw_0.bin"
        if not args.analyse_only and not first.exists():
            from cascade.recording import CascadeRecording, CascadeRecordingError
            p = CascadeParams.preset(name)
            p.duration_s = args.duration
            print(f"[record] {name}: {args.duration:g} s", flush=True)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    CascadeRecording(p, out, prefix, notes=name).run()
            except CascadeRecordingError as e:
                results[name] = {"error": str(e)}
                print(f"[record] {name}: FAILED {e}", flush=True)
                continue
        if not first.exists():
            continue
        print(f"[analyse] {name}", flush=True)
        t0 = time.time()
        try:
            res = analyse_take(str(first))
            plot_take(res, out / f"{prefix}.png", name)
        except Exception as e:
            import traceback
            traceback.print_exc()
            res = {"error": f"{type(e).__name__}: {e}"}
        res["analysis_s"] = round(time.time() - t0, 1)
        results[name] = res
    # the frame period, from takes of different length: the packet span is a + T * frames,
    # where a is the DCA1000 holding its last short packet until its timeout (~2 s)
    pts = []
    for side in out.glob("*_cascade.json"):
        m = json.loads(side.read_text(encoding="utf-8"))
        r = m.get("record") or {}
        if (m.get("params") == CascadeParams.preset(names[0]).to_dict() | {"duration_s": m["params"]["duration_s"]}
                and r.get("first_packet_ns") and r.get("last_packet_ns")):
            pts.append((r["frames_in_file"], (r["last_packet_ns"] - r["first_packet_ns"]) / 1e9, side.name))
    timing_fit = None
    if len(pts) >= 2:
        f_arr, s_arr = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
        T, a = np.polyfit(f_arr, s_arr, 1)
        timing_fit = {"preset": names[0], "takes": [p[2] for p in pts], "frames": f_arr.tolist(),
                      "span_s": s_arr.tolist(), "frame_period_ms": T * 1e3, "fps": 1 / T, "dca_tail_s": a,
                      "residual_ms": float(np.max(np.abs(np.polyval([T, a], f_arr) - s_arr)) * 1e3)}
        print(f"[timing] frame period {T * 1e3:.4f} ms ({1 / T:.3f} fps), DCA1000 tail {a:.3f} s, from {len(pts)} takes")
        for res in results.values():
            tim = res.get("timing") if isinstance(res, dict) else None
            if tim:
                tim["fps_measured"] = tim["frames"] / (tim["packet_span_s"] - a)
                tim["dca_tail_s"] = a
    env = out / "envelope.json"
    envelope = json.loads(env.read_text(encoding="utf-8")) if env.exists() else None
    (out / "protocol.json").write_text(json.dumps({"results": results, "envelope": envelope,
                                                   "timing_fit": timing_fit}, indent=2,
                                                  ensure_ascii=False, default=float), encoding="utf-8")
    (out / "protocol.md").write_text(summary_md(results, envelope, timing_fit), encoding="utf-8")
    print(summary_md(results, envelope, timing_fit))


if __name__ == "__main__":
    main()
