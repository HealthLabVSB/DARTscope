"""
python -m acquisition <command>     (run from the Radar folder)

    show CONFIG                  what a configuration records (Lua script, _LogFile.txt or .json)
    lua CONFIG -o OUT.lua        the radar-only Lua script for mmWave Studio's Lua shell
    check [CONFIG]               network, ports, DCA1000, mmWave Studio, disk
    capture CONFIG --out DIR --name NAME [--radar native|studio|external]
    verify RECORDING [--config CONFIG]
    simulate CONFIG --out DIR --name NAME     the whole path against a DCA1000 on localhost
    gui                          the Measure window (settings, takes you start, series)
    series CONFIG --out DIR --name NAME --vary PARAM=VALUES ...
                                 a series over a range of settings, each take started by
                                 you (Enter), with a report ranking them for a movement

CONFIG is one of the lab's Lua scripts (C:/Data/Radar/config/*.lua), the _LogFile.txt of
an earlier recording (to record again with the same settings), or a .json sidecar.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

from . import lua, studio_log
from .capture import CaptureEngine, CaptureRequest
from .config import RadarConfig
from .measurement import Measurement
from .serialport import com_number


from .configfile import load_config  # noqa: E402  (shared with the Measure tab)


def _print_config(cfg: RadarConfig, warnings: list[str]):
    e, ph = cfg.expectation(), cfg.physical()
    print(f"source: {cfg.source}")
    print(f"device: {cfg.device.board} ({cfg.device.chip}, {cfg.device.band}, COM{cfg.device.com_port} "
          f"in the script)")
    print(f"profile: {ph['start_ghz']:.4f} GHz, slope {ph['slope_mhz_us']:.3f} MHz/us, idle "
          f"{ph['idle_us']:g} us, ADC start {ph['adc_start_us']:g} us, ramp {ph['ramp_end_us']:g} us, "
          f"{cfg.profile.num_adc_samples} samples at {cfg.profile.dig_out_sample_rate} ksps, RX gain "
          f"{cfg.profile.rx_gain} dB")
    print(f"chirps: TX pattern {e['tx_pattern']} x {e['loops']} loops = {e['chirps_per_frame']} "
          f"chirps/frame; {e['frames'] or 'until stopped'} frames every {e['frame_period_ms']:g} ms")
    print(f"data: {e['rx']} RX, {e['adc_format']} {e['adc_bits']}-bit {e['iq_order']}, "
          f"{e['radar_lanes']} LVDS lane(s) into a {e['container_lanes']}-lane DCA1000")
    print(f"size: {e['bytes_per_frame']} B/frame, {e['total_bytes']} B total, "
          f"{e['data_rate_mb_s']:.2f} MB/s, {e['packets']} packets")
    if ph["range_res_m"]:
        print(f"gives: range resolution {ph['range_res_m'] * 100:.2f} cm, v_max "
              f"{ph['max_velocity_m_s']:.2f} m/s")
    for w in warnings:
        print(f"note: {w}")
    for p in cfg.problems():
        print(p)


def _events(phase: str, info: dict):
    if phase == "recording":
        exp = info.get("expected")
        pct = f" ({100 * info['bytes'] / exp:5.1f} %)" if exp else ""
        print(f"\r  {info['bytes'] / 1e6:9.1f} MB{pct}  {info['packets']} packets, "
              f"{info['zero_filled']} zero-filled, {info['elapsed_s']:.1f} s", end="", flush=True)
    elif phase == "done":
        print(f"\n{phase}: {info.get('status')} ({info.get('stop_reason')})")
    else:
        msg = info.get("message") or ", ".join(f"{k}={v}" for k, v in info.items()
                                                if k not in ("expected",))
        print(f"{phase}: {msg}")


def _report(res) -> int:
    print(f"recording: {os.path.join(res.measurement.folder, res.measurement.stem)}")
    st = res.stats
    if st:
        print(f"  {st.get('bytes_written', 0)} bytes in {len(st.get('files', []))} file(s); "
              f"{st.get('received')} packets, {st.get('zero_filled_packets')} zero-filled, "
              f"{st.get('late_recovered')} late recovered, {st.get('duplicates')} duplicates")
    if res.verification:
        print("verification with the DARTscope loader:")
        for c in res.verification["checks"]:
            print(f"  {'OK  ' if c['ok'] else 'FAIL'} {c['name']}: {c['detail']}")
        for k, v in res.verification.get("info", {}).items():
            print(f"       {k}: {v}")
    if res.error:
        print(f"error: {res.error}")
    return 0 if res.ok else 1


def cmd_show(a):
    cfg, warnings, _ = load_config(a.config)
    _print_config(cfg, warnings)
    return 0


def cmd_lua(a):
    cfg, warnings, _ = load_config(a.config)
    text = lua.radar_script(cfg)
    if a.output:
        with open(a.output, "x", encoding="utf-8") as f:
            f.write(text)
        print(f"written {a.output}")
    else:
        print(text)
    return 0


def cmd_check(a):
    from .status import preflight
    cfg = load_config(a.config)[0] if a.config else RadarConfig()
    rows = preflight(cfg, folder=a.folder, bind_ip=a.bind_ip,
                     studio_port=a.studio_port if a.studio else None)
    for r in rows:
        print(r)
    return 0 if all(r.ok for r in rows) else 1


def _apply_overrides(cfg: RadarConfig, a):
    if getattr(a, "frames", None) is not None:
        cfg.frame.num_frames = int(a.frames)
    if getattr(a, "com_port", None) is not None and com_number(a.com_port) is not None:
        cfg.device.com_port = com_number(a.com_port)


def cmd_capture(a):
    from .radar import ExternalRadar, StudioRemoteRadar
    cfg, warnings, text = load_config(a.config)
    _apply_overrides(cfg, a)
    _print_config(cfg, warnings)
    if a.radar == "studio":
        radar = StudioRemoteRadar(host=a.studio_host, port=a.studio_port, boot=not a.no_boot,
                                  studio_dir=cfg.device.studio_dir,
                                  on_command=lambda c: print(f"  mmWave Studio <- {c}", flush=True))
    elif a.radar == "native":
        from .native import NativeRadar
        port = a.com_port if a.com_port is not None and com_number(a.com_port) is None else None
        radar = NativeRadar(com_port=port or cfg.device.com_port,
                            on_command=lambda c: print(f"  radar <- {c}", flush=True))
    else:
        radar = ExternalRadar(prompt=lambda msg: input(msg + " Press Enter when started... "))
    notes = {}
    for kv in a.note:                       # KEY=VALUE, or free text kept under "note"
        key, _, value = kv.partition("=") if "=" in kv else ("note", "", kv)
        notes[key.strip()] = value.strip()
    stop = threading.Event()
    engine = CaptureEngine(radar, on_event=_events, stop_event=stop)
    req = CaptureRequest(config=cfg, folder=a.out, name=a.name, auto_increment=a.auto_name,
                         notes=notes, duration_s=a.duration, verify=not a.no_verify,
                         bind_ip=a.bind_ip, source_text=text)
    try:
        res = engine.run(req)
    except KeyboardInterrupt:
        print("\nstopped by the user; the files written so far are kept")
        return 130
    return _report(res)


def cmd_verify(a):
    from .verify import verify
    meas = Measurement.from_path(a.recording)
    cfg = None
    if a.config:
        cfg = load_config(a.config)[0]
    else:
        side = meas.sidecar()
        if side and side.get("config"):
            cfg = RadarConfig.from_dict(side["config"])
        elif os.path.isfile(meas.log_path):
            cfg = studio_log.read_config(meas.log_path)
            cfg.dca.lvds_mode = 2 if cfg.lvds.n_lanes <= 2 else 1
    stats = (meas.sidecar() or {}).get("stream") or None
    rep = verify(meas, cfg, recorder_stats=stats)
    for line in rep.lines():
        print(line)
    return 0 if rep.ok else 1


def cmd_simulate(a):
    from .dca1000 import Dca1000Controller
    from .layout import encode_for, random_cube
    from .radar import NullRadar
    from .simulator import Dca1000Simulator, Faults, free_udp_port
    cfg, warnings, text = load_config(a.config)
    cfg.frame.num_frames = int(a.frames)
    cfg.dca.host_ip = "127.0.0.1"
    cfg.dca.config_port, cfg.dca.data_port = free_udp_port(), free_udp_port()
    cube = random_cube(cfg, a.frames, seed=a.seed)
    faults = Faults(drop=set(range(40, 40 + a.drop)), swap={20} if a.swap else set())
    sim = Dca1000Simulator(encode_for(cfg, cube), host_data_addr=("127.0.0.1", cfg.dca.data_port),
                           faults=faults, rate_mb_s=a.rate)
    dca = Dca1000Controller(cfg.dca, bind_ip="127.0.0.1", dca_addr=sim.addr)
    engine = CaptureEngine(NullRadar(on_start=sim.start_stream), dca=dca, on_event=_events)
    try:
        res = engine.run(CaptureRequest(config=cfg, folder=a.out, name=a.name,
                                        auto_increment=a.auto_name, bind_ip="127.0.0.1",
                                        notes={"simulated": "DCA1000 simulator, random samples"},
                                        source_text=text))
    finally:
        dca.close()
        sim.close()
    code = _report(res)
    if res.ok and not a.drop:
        _, _, got = res.measurement.load_cube()
        same = got.shape == cube.shape and bool(abs(got * 32767 - cube).max() < 0.02)
        print(f"loader returned the simulated samples exactly: {same}")
        code = code or (0 if same else 1)
    return code


def cmd_gui(a):
    import measure_gui                       # at the top of the Radar folder
    measure_gui.main()
    return 0


def _radar_port(a):
    """The radar's UART: a COM number on Windows, a device path on macOS/Linux."""
    if a.com_port:
        n = com_number(a.com_port)
        return n if n is not None else a.com_port
    from .ports import detect
    det = detect()
    for note in det.notes:
        print(f"note: {note}")
    if det.radar is None:
        return None
    print(f"radar UART: {det.radar.name} ({det.radar.board})")
    return det.radar.number or det.radar.name


def cmd_series(a):
    from .backend import HardwareBackend, SimulatorBackend
    from .params import MeasureParams
    from .quality import OBJECTIVES, analyze_recording, save as save_quality
    from .series import Series, parse_values, plan
    cfg, warnings, _text = load_config(a.config)
    base = MeasureParams.from_config(cfg)
    if a.duration is not None:
        base.duration_s = float(a.duration)
    variables = []
    for spec in a.vary:
        name, _, values = spec.partition("=")
        variables.append((name.strip(), parse_values(name.strip(), values)))
    steps = plan(base, cfg, variables, repeats=a.takes, shuffle=a.shuffle)
    ok = [s for s in steps if s.ok]
    print(f"{len(steps)} take(s), {len(ok)} can be recorded:")
    for s in steps:
        d = s.derived or {}
        extra = (f"v max {d.get('max_vel_ms', 0):.2f} m/s, v res {d.get('vel_res_ms', 0):.3f} m/s, "
                 f"range res {d.get('range_res_cm', 0):.2f} cm, {d.get('rate_mb_s', 0):.1f} MB/s"
                 if s.ok else "; ".join(f"{p[1]}: {p[2]}" for p in s.problems if p[0] == "error"))
        print(f"  {s.index:3d} {'ok ' if s.ok else 'NO '} {s.label}: {extra}")
    if not ok or a.plan_only:
        return 0 if ok else 1
    notes = dict(kv.partition("=")[::2] if "=" in kv else ("note", kv) for kv in a.note)
    series = Series(a.out, a.name, steps, objective=a.objective, notes=notes, base_params=base,
                    variables=variables)
    if a.radar == "simulator":
        backend = SimulatorBackend()
    else:
        port = _radar_port(a)
        if port is None:
            print("error: no radar COM port - plug the radar in or give --com-port", file=sys.stderr)
            return 2
        backend = HardwareBackend(com_port=port)
    print(f"series '{series.name}' in {series.folder}, ranked for {OBJECTIVES[series.objective][0]}")
    quit_all = False
    try:
        for s in ok:
            if quit_all:
                break
            gate, stop, ready = threading.Event(), threading.Event(), threading.Event()
            out: dict = {}

            def on_event(phase, info, _ready=ready):
                if phase == "ready":
                    _ready.set()
                elif phase == "recording":
                    _events(phase, info)

            def work(_s=s, _gate=gate, _stop=stop, _out=out, _on=on_event):
                try:
                    _out["res"] = backend.run(series.request(_s, start_gate=_gate), on_event=_on,
                                              stop_event=_stop)
                except BaseException as e:
                    _out["err"] = e
            t = threading.Thread(target=work, daemon=True)
            print(f"\ntake {s.index}/{len(steps)}: {s.label} - preparing the radar ...")
            t.start()
            while not ready.is_set() and t.is_alive():
                ready.wait(0.1)
            if ready.is_set():
                if a.auto_start:
                    answer = ""
                else:
                    answer = input("  ready - Enter starts, s + Enter skips, q + Enter ends the series: ")
                answer = answer.strip().lower()
                if answer in ("s", "q"):
                    quit_all = answer == "q"
                    stop.set()
                else:
                    for n in range(int(a.countdown), 0, -1):
                        print(f"  {n} ...", flush=True)
                        time.sleep(1.0)
                    gate.set()
            t.join()
            if "err" in out:
                raise out["err"]
            res = out["res"]
            qual = None
            if res.status in ("complete", "incomplete") and res.stats.get("bytes_written"):
                stem = os.path.join(res.measurement.folder, res.measurement.stem)
                qual = analyze_recording(stem, stats=res.stats, verification=res.verification)
                save_quality(qual, stem)
            r = series.add(s, None if res.status == "cancelled" else res, qual,
                           status="skipped" if res.status == "cancelled" else None)
            print(f"\n  {r.status}: {qual.headline() if qual else res.error or ''}"
                  + (f" - score {r.score:.1f}" if r.score is not None else ""))
            if qual:
                for level, what, msg in qual.findings:
                    print(f"  {level.upper()} {what}: {msg}")
            series.write()
    finally:
        backend.close()
    paths = series.write()
    print("\nwhat to take from it:")
    for line in series.recommendations():
        print(f"  - {line}")
    print(f"report: {paths['html']}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m acquisition", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("gui", help="the Measure window (also a tab of DARTscope)")
    s.set_defaults(fn=cmd_gui)

    s = sub.add_parser("series", help="a series over a range of settings, each take started by you")
    s.add_argument("config", help="the base settings: a Lua script, a log or a sidecar")
    s.add_argument("--out", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--vary", action="append", default=[], metavar="PARAM=VALUES",
                   help="e.g. loops=32,64,128 or tx='1|1,2,3' or rx='1234|12' or "
                        "slope_mhz_us=50..200/50 (repeatable)")
    s.add_argument("--takes", type=int, default=1, help="takes per configuration")
    s.add_argument("--shuffle", action="store_true", help="random order of the takes")
    s.add_argument("--objective", default="general",
                   choices=("general", "walking", "limbs", "vitals", "angles"))
    s.add_argument("--duration", type=float, help="seconds a take (default: the config's frames)")
    s.add_argument("--radar", choices=("native", "simulator"), default="native")
    s.add_argument("--com-port", help="the radar's UART: 15 / COM15 on Windows, "
                                      "/dev/cu.usbserial-...0 on macOS (default: detected)")
    s.add_argument("--countdown", type=int, default=3)
    s.add_argument("--auto-start", action="store_true", help="start every take at once")
    s.add_argument("--plan-only", action="store_true")
    s.add_argument("--note", action="append", default=[], metavar="KEY=VALUE")
    s.set_defaults(fn=cmd_series)

    s = sub.add_parser("show", help="what a configuration records")
    s.add_argument("config")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("lua", help="radar-only Lua script for mmWave Studio")
    s.add_argument("config")
    s.add_argument("-o", "--output")
    s.set_defaults(fn=cmd_lua)

    s = sub.add_parser("check", help="network, ports, DCA1000, mmWave Studio, disk")
    s.add_argument("config", nargs="?")
    s.add_argument("--folder")
    s.add_argument("--bind-ip", default="")
    s.add_argument("--studio", action="store_true", help="also check mmWave Studio's RSTD port")
    s.add_argument("--studio-port", type=int, default=2777)
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("capture", help="record")
    s.add_argument("config")
    s.add_argument("--out", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--radar", choices=("external", "studio", "native"), default="external",
                   help="external: you start the frames (default); studio: mmWave Studio over "
                        "its RSTD port does it (needs pythonnet); native: no mmWave Studio - "
                        "TI's RadarLinkDLL from the 32-bit helper (Radar/.py32) on Windows, the "
                        "same calls in Python over the DCA1000's USB on macOS/Linux; mmWave "
                        "Studio closed")
    s.add_argument("--frames", type=int, help="override numFrames (0 = until stopped)")
    s.add_argument("--com-port",
                   help="the radar's own UART (the bootloader; ar1.Connect): 15 / COM15 on "
                        "Windows, /dev/cu.usbserial-...0 on macOS (default there: detected)")
    s.add_argument("--duration", type=float, help="seconds, for numFrames 0")
    s.add_argument("--note", action="append", default=[], metavar="KEY=VALUE")
    s.add_argument("--auto-name", action="store_true", help="NAME_2, NAME_3 ... if NAME exists")
    s.add_argument("--no-verify", action="store_true")
    s.add_argument("--bind-ip", default="")
    s.add_argument("--studio-host", default="127.0.0.1")
    s.add_argument("--studio-port", type=int, default=2777)
    s.add_argument("--no-boot", action="store_true",
                   help="radar already powered and firmware loaded: skip reset and download")
    s.set_defaults(fn=cmd_capture)

    s = sub.add_parser("verify", help="read a recording back with the DARTscope loader")
    s.add_argument("recording", help="the .bin, the _LogFile.txt or the sidecar")
    s.add_argument("--config")
    s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("simulate", help="the whole path against a DCA1000 simulator")
    s.add_argument("config")
    s.add_argument("--out", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--frames", type=int, default=5)
    s.add_argument("--drop", type=int, default=0, help="packets to drop")
    s.add_argument("--swap", action="store_true", help="send one pair of packets swapped")
    s.add_argument("--rate", type=float, help="MB/s")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--auto-name", action="store_true")
    s.set_defaults(fn=cmd_simulate)

    a = p.parse_args(argv)
    from .capture import CaptureError
    from .config import ConfigError
    from .dca1000 import Dca1000Error
    from .measurement import NamingError
    from .radar import RadarError
    try:
        return a.fn(a)
    except (ConfigError, CaptureError, Dca1000Error, RadarError, NamingError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
