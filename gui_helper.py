"""
gui_helper.py

Helper tab (GUI) for radar experiment design and command generation.
Translates measurement goals (max range, range/velocity resolution) into hardware parameters (bandwidth,
slope, ADC rate/samples, chirp/frame timing), validates against typical device limits, computes duty cycles,
and generates mmWave Studio commands. Provides presets for macro/micro motion use cases.

Includes:
- Parameter calculator with actionable limit guidance (what to increase/decrease).
- Duty cycle metrics (frame duty, active‑ramp duty).
- Bandwidth/ADC‑rate checks and profile count hints.
- Command block generation for mmWave Studio.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
import tkinter as tk
from tkinter import ttk, messagebox

# --- Module-level variables for the Helper tab ---
helper_vars = {}
helper_outputs = {}
helper_commands_text = None

def init_helper_tab(parent):
    """Creates the UI for the Helper tab."""
    global helper_vars, helper_outputs, helper_commands_text

    # --- Main layout frames ---
    top_frame = ttk.Frame(parent)
    top_frame.pack(side="top", fill="x", padx=10, pady=5)
    left_frame = ttk.Frame(parent)
    left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=5)
    right_frame = ttk.Frame(parent)
    right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=5)

    # --- Top: Presets and Actions ---
    ttk.Label(top_frame, text="Preset:", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 5))
    preset_var = tk.StringVar()
    preset_combo = ttk.Combobox(top_frame, textvariable=preset_var, values=["Custom", "Long Range (Macro)", "Short Range (Respiration)"], state="readonly")
    preset_combo.pack(side="left", padx=5)
    preset_combo.set("Custom")
    preset_combo.bind("<<ComboboxSelected>>", lambda e: apply_helper_preset(preset_var.get()))

    ttk.Button(top_frame, text="Calculate Parameters", command=calculate_radar_params).pack(side="left", padx=20)
    ttk.Button(top_frame, text="Generate Commands", command=generate_helper_commands).pack(side="left", padx=5)

    # --- Left Side: Inputs and Outputs ---
    # Input Group
    input_group = ttk.LabelFrame(left_frame, text="1. Desired Performance")
    input_group.pack(fill="x", pady=5)
    inputs = {
        "max_range_m": ("Max Range [m]", "20"),
        "range_res_cm": ("Range Resolution [cm]", "5"),
        "max_vel_ms": ("Max Velocity [m/s]", "1"),
        "vel_res_ms": ("Velocity Resolution [m/s]", "0.05"),
        "start_freq_ghz": ("Start Frequency [GHz]", "77"),
    }
    for key, (label, default) in inputs.items():
        row = ttk.Frame(input_group)
        row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text=label, width=25).pack(side="left")
        var = tk.StringVar(value=default)
        ttk.Entry(row, textvariable=var, width=15).pack(side="left")
        helper_vars[key] = var

    # Output Group
    output_group = ttk.LabelFrame(left_frame, text="2. Calculated Radar Parameters")
    output_group.pack(fill="x", pady=5)
    outputs = {
        "bandwidth_mhz": "Bandwidth [MHz]", "ramp_slope_mhz_us": "Ramp Slope [MHz/us]",
        "adc_samples": "ADC Samples", "adc_rate_msps": "ADC Rate [Msps]",
        "ramp_time_us": "Ramp Time [us]", "idle_time_us": "Idle Time [us]",
        "chirp_time_us": "Total Chirp Time [us]", "num_chirps": "Chirps per Frame",
        "frame_period_ms": "Frame Periodicity [ms]",
        "frame_duty_cycle": "Frame Duty Cycle [%]",
        "active_ramp_duty_cycle": "Active Ramp Duty Cycle [%]",
        "max_bw_supported": "Max Bandwidth Supported [GHz]",
        "max_profiles": "Max Chirp Profiles (HW Limit)",
    }
    for key, label in outputs.items():
        row = ttk.Frame(output_group)
        row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text=label, width=25).pack(side="left")
        var = tk.StringVar(value="--")
        ttk.Label(row, textvariable=var, font=("Segoe UI", 9, "bold")).pack(side="left")
        helper_outputs[key] = var

    # --- Right Side: Generated Commands ---
    cmd_group = ttk.LabelFrame(right_frame, text="3. Generated mmWave Studio Commands")
    cmd_group.pack(fill="both", expand=True, pady=5)
    helper_commands_text = tk.Text(cmd_group, wrap="word", height=25, width=70, font=("Consolas", 9))
    ys = ttk.Scrollbar(cmd_group, orient="vertical", command=helper_commands_text.yview)
    helper_commands_text.config(yscrollcommand=ys.set)
    ys.pack(side="right", fill="y")
    helper_commands_text.pack(side="left", fill="both", expand=True, padx=5, pady=5)

def apply_helper_preset(preset_name):
    if preset_name == "Long Range (Macro)":
        helper_vars["max_range_m"].set("50")
        helper_vars["range_res_cm"].set("20")
        helper_vars["max_vel_ms"].set("5")
        helper_vars["vel_res_ms"].set("0.5")
    elif preset_name == "Short Range (Respiration)":
        helper_vars["max_range_m"].set("2")
        helper_vars["range_res_cm"].set("1")
        helper_vars["max_vel_ms"].set("0.2")
        helper_vars["vel_res_ms"].set("0.02")
    calculate_radar_params()

def calculate_radar_params():
    C0 = 299792458.0
    try:
        # Clear previous results
        for var in helper_outputs.values():
            var.set("--")

        # Read inputs
        r_max = float(helper_vars["max_range_m"].get())
        r_res = float(helper_vars["range_res_cm"].get()) / 100.0  # cm to m
        v_max = float(helper_vars["max_vel_ms"].get())
        v_res = float(helper_vars["vel_res_ms"].get())
        f_start = float(helper_vars["start_freq_ghz"].get()) * 1e9 # GHz to Hz

        # --- Determine radar band and limits ---
        is_77ghz_band = (f_start > 70e9)
        max_supported_bw_ghz = 4.0 # Typical for both 60 and 77 GHz TI devices
        # ADC Rate limit is device-specific. Using a general high value for xWR1642/6843.
        max_adc_rate_msps = 40.0
        max_profiles = 4 # Typical for AWR1x/6x

        # --- Calculations ---
        # Bandwidth from Range Resolution
        bandwidth = C0 / (2 * r_res)

        # --- VALIDATION CHECKS ---
        # 1. Bandwidth Check
        if (bandwidth / 1e9) > max_supported_bw_ghz:
            msg = (f"Calculated bandwidth ({bandwidth/1e9:.2f} GHz) exceeds the typical supported limit of {max_supported_bw_ghz} GHz.\n\n"
                   f"This is caused by a very high 'Range Resolution' requirement.\n\n"
                   f"To fix this, please INCREASE the 'Range Resolution [cm]' value.")
            messagebox.showwarning("Bandwidth Limit Exceeded", msg)
            return


        # Chirp time from Max Velocity
        # Vmax = lambda / (4 * T_inter_chirp)
        wavelength = C0 / f_start
        t_inter_chirp = wavelength / (4 * v_max)

        # Number of chirps from Velocity Resolution
        # Vres = lambda / (2 * Tframe) = lambda / (2 * Nd * T_inter_chirp)
        nd = round(wavelength / (2 * v_res * t_inter_chirp))
        nd = max(16, 1 << (nd - 1).bit_length()) # Make it power of 2, min 16

        # Ramp time and ADC sampling rate
        # Assume ADC sampling time is 80% of ramp time.
        ramp_time = 0.8 * t_inter_chirp
        idle_time = 0.2 * t_inter_chirp

        # Ramp slope from bandwidth and ramp time
        ramp_slope = bandwidth / ramp_time

        # ADC rate from max range and slope
        f_beat_max = (ramp_slope * 2 * r_max) / C0
        adc_rate = f_beat_max * 1.25 # 25% margin over max beat freq

        # 2. ADC Sampling Rate Check
        if (adc_rate / 1e6) > max_adc_rate_msps:
            msg = (f"Calculated ADC Sampling Rate ({adc_rate/1e6:.2f} Msps) exceeds the typical limit of {max_adc_rate_msps} Msps for this type of device.\n\n"
                   f"This is caused by a combination of high 'Max Range', high 'Max Velocity', and/or high 'Range Resolution'. The required beat frequency is too high.\n\n"
                   f"To fix this, try one of the following:\n"
                   f"  - DECREASE 'Max Range [m]'\n"
                   f"  - DECREASE 'Max Velocity [m/s]'\n"
                   f"  - INCREASE 'Range Resolution [cm]'")
            messagebox.showwarning("ADC Rate Limit Exceeded", msg)
            return

        # Number of ADC samples
        adc_samples = int(adc_rate * ramp_time)
        adc_samples = 1 << (adc_samples - 1).bit_length() # round to next power of 2

        # Final check: recalculate ADC rate based on rounded samples
        adc_rate = adc_samples / ramp_time

        # Frame periodicity
        frame_period = nd * t_inter_chirp

        # --- Duty Cycle Calculations ---
        # Total active time for all chirps in a frame
        total_chirp_time_in_frame = nd * t_inter_chirp
        # Frame duty cycle: (total active time) / (total frame time)
        frame_duty_cycle = (total_chirp_time_in_frame / frame_period) * 100 if frame_period > 0 else 0
        # Active ramp duty cycle: (ramp time) / (total chirp time)
        active_ramp_duty_cycle = (ramp_time / t_inter_chirp) * 100 if t_inter_chirp > 0 else 0


        # --- Update GUI ---
        helper_outputs["bandwidth_mhz"].set(f"{bandwidth / 1e6:.2f}")
        helper_outputs["ramp_slope_mhz_us"].set(f"{ramp_slope / 1e12:.2f}")
        helper_outputs["adc_samples"].set(f"{adc_samples}")
        helper_outputs["adc_rate_msps"].set(f"{adc_rate / 1e6:.2f}")
        helper_outputs["ramp_time_us"].set(f"{ramp_time * 1e6:.1f}")
        helper_outputs["idle_time_us"].set(f"{idle_time * 1e6:.1f}")
        helper_outputs["chirp_time_us"].set(f"{t_inter_chirp * 1e6:.1f}")
        helper_outputs["num_chirps"].set(f"{nd}")
        helper_outputs["frame_period_ms"].set(f"{frame_period * 1e3:.2f}")
        # NEW: Update new fields
        helper_outputs["frame_duty_cycle"].set(f"{frame_duty_cycle:.1f}")
        helper_outputs["active_ramp_duty_cycle"].set(f"{active_ramp_duty_cycle:.1f}")
        helper_outputs["max_bw_supported"].set(f"{max_supported_bw_ghz:.1f}")
        helper_outputs["max_profiles"].set(f"{max_profiles}")


    except (ValueError, ZeroDivisionError) as e:
        for var in helper_outputs.values():
            var.set("-- Error --")
        messagebox.showerror("Calculation Error", f"Invalid input or calculation failed: {e}")

def generate_helper_commands():
    if "--" in helper_outputs["bandwidth_mhz"].get():
        messagebox.showwarning("Warning", "Please calculate parameters first.")
        return

    # Read calculated values
    start_freq = float(helper_vars["start_freq_ghz"].get())
    idle_time_us = float(helper_outputs["idle_time_us"].get())
    adc_start_time_us = 6 # A common default
    ramp_time_us = float(helper_outputs["ramp_time_us"].get())
    adc_samples = int(helper_outputs["adc_samples"].get())
    adc_rate_ksps = float(helper_outputs["adc_rate_msps"].get()) * 1000
    ramp_slope_mhz_us = float(helper_outputs["ramp_slope_mhz_us"].get())
    num_chirps = int(helper_outputs["num_chirps"].get())
    frame_period_ms = float(helper_outputs["frame_period_ms"].get())

    # Assuming 1 TX, 4 RX
    tx_mask = 1
    rx_mask = 15

    # Format commands
    commands = f"""
% Stop sensor if running
sensorStop

% Flush any existing configuration
flushCfg

% Chirp and Profile Configuration
% Profile: 0, Start Freq: {start_freq:.3f} GHz, Idle Time: {idle_time_us:.1f} us, ADC Start: {adc_start_time_us:.1f} us, Ramp End: {ramp_time_us:.1f} us, TX Power: 0, TX Phase: 0, Slope: {ramp_slope_mhz_us:.3f} MHz/us, ADC Samples: {adc_samples}, ADC Rate: {adc_rate_ksps:.0f} ksps, HPF1: 0, HPF2: 0, RX Gain: 30 dB
profileCfg 0 {start_freq:.3f} {idle_time_us:.1f} {adc_start_time_us:.1f} {ramp_time_us:.1f} 0 0 {ramp_slope_mhz_us:.3f} 0 {adc_samples} {adc_rate_ksps:.0f} 0 0 30

% Chirp: Start: 0, End: 0, Profile: 0, Start Freq Var: 0, Slope Var: 0, Idle Var: 0, ADC Start Var: 0, TX Mask: {tx_mask}
chirpCfg 0 0 0 0 0 0 0 {tx_mask}

% Frame Configuration
% Start Chirp: 0, End Chirp: 0, Loops: {num_chirps}, Frames: 0 (infinite), Periodicity: {frame_period_ms:.2f} ms, Trigger: SW
frameCfg 0 0 {num_chirps} 0 {frame_period_ms:.2f} 1 0

% Data format and path
% Using default real data, 16 bits, non-interleaved
adcbufCfg -1 0 1 1 1
adcCfg 2 1
% LVDS path for DCA1000
dataPathConfig {rx_mask} 1 0
lvdsStreamCfg -1 0 1 0
% Data format: 16-bit, Real, non-interleaved, IQ order doesn't matter for real
dataFmtConfig -1 2 0 0 0 0

% Start sensor
sensorStart
"""
    helper_commands_text.delete("1.0", tk.END)
    helper_commands_text.insert("1.0", commands.strip())
