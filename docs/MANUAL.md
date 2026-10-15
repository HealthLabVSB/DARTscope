# DARTscope User Manual

DARTscope turns the raw data of a Texas Instruments mmWave radar, recorded with a DCA1000 capture board, into range, velocity, angle and vital-sign measurements. This manual describes every tab, button, checkbox and setting of the application, and the order in which to use them. It assumes no previous knowledge of the application; the few radar terms it needs are explained in the [Glossary](#13-glossary).

> **How to read this manual.** In the application, pick a chapter or a section in the contents on the left - a chapter shows everything under it, a section only itself; **Expand all** / **Collapse all** open or close the whole contents. Type into **Search** to list every section that mentions a word (the words are highlighted; **Clear** ends the search). The ⓘ icon next to each view's Run button opens that view's section here. **Save as HTML...** writes the whole manual into one file you can send to someone who does not have the application. **Reload** reads the manual file again. The same text is `docs/MANUAL.md` in the source repository.

## 1. Getting started

### What DARTscope does

- Reads a raw recording from the DCA1000 together with the radar configuration that mmWave Studio logged, and decodes it into a data cube: frames x chirps x receivers x samples.
- Checks the recording: lost packets, clipping, dead receivers, IQ balance, noise, whether the files belong together.
- Shows the data as range profiles, range over time (RTI), range-velocity maps (Range-Doppler), movement maps, angle of arrival (1D, range-azimuth, azimuth-elevation, 3D), micro-Doppler, and detects targets (CFAR).
- Measures breathing and heart rate from the chest motion of a person, in three independent tabs (Vitals, HR, AI), and compares them with a reference measurement (PPG / pulse oximeter of a somnograph).
- Exports images (PNG), data (CSV, JSON) and videos (MP4).

### What you need

#### Hardware and software used for recording

- A TI mmWave radar evaluation board. The antenna layouts of the IWR6843AOP (antenna-on-package) and the AWR2544LOPEVM are built in; other boards can be described in the antenna configuration editor.
- A DCA1000EVM capture board and mmWave Studio to configure the radar and record - or, without mmWave Studio, the [Measure tab](#measure-tab) (IWR6843AOPEVM) and its [Cascade radar](#cascade-radar) sub-tab (AWR2243-2X-CAS-EVM).
- DARTscope itself runs on a normal PC; it does not talk to the radar.

#### The files of one recording

Keep all files of one recording in the same folder, with the same name prefix. Do not rename one of them without the others.

| File | Written by | Needed |
|---|---|---|
| `<name>_Raw_0.bin` | DCA1000 - the raw ADC samples | yes |
| `<name>_Raw_1.bin`, `_Raw_2.bin`, ... | DCA1000 - further parts of a long recording | read automatically when present |
| `<name>_LogFile.txt` | mmWave Studio - the chirp, frame and data-format configuration | yes |
| `<name>_Raw_LogFile.csv` | DCA1000 recorder - lane mode, lost packets, exact start time | strongly recommended |

Without the `_Raw_LogFile.csv` the application still works, but it cannot know the exact start time of the recording (needed to line it up with a reference), how many packets were lost, or which LVDS lane width the recorder used.

### Installing and starting

- **Windows build:** run `DARTscope.exe` (GitHub Releases of the repository). Nothing has to be installed. It has no console window; everything the application reports appears in the [Log tab](#5-log-tab). To record with the [Measure tab](#measure-tab), mmWave Studio 2.1.1 has to be installed (the radar's library and firmware come from it - it does not run), and the first recording makes Windows ask whether DARTscope may use the network: allow it, for public networks too (the DCA1000's network usually counts as one), or the data from the DCA1000 never arrive.
- **From source:** Python 3.10 or newer. Install the dependencies once with `pip install -r requirements.txt`, then start the application with `python DCA_processing.py` from the repository folder.
- **Video export** additionally needs `ffmpeg` installed and on the system PATH. Everything else works without it.

### The main window

- **Tabs at the top:** **Measure** (record with the radar and the DCA1000 without mmWave Studio, single takes and series, see [Measure tab](#measure-tab); the cascade radar has its own sub-tab, see [Cascade radar](#cascade-radar)), **Decode** (choose and decode a recording), **Fastcheck** (a quick check of the whole recording: is it complete, is the signal sane, is somebody breathing), **Analysis** (all views), **Log** (everything the application reports), **Helper** (a radar configuration calculator), **Manual** (this text).
- **Status bar at the bottom:** a message on the left, a progress bar and a **Cancel** button on the right. Long computations show their progress there - what is being computed, how long it has run and, where the share done is known, about how long it will still take (*Imaging ...  (37 %, 1 min 05 s, about 1 min 50 s left)*); **Cancel** stops them. Every step that took a second or more is written to the [Log tab](#5-log-tab) with its time (*[time] Computing RTI: 2.4 s*), so it shows afterwards where a long run spent its time.
- **Analysis tab:** a bar of global settings at the top, the frame slider under it, and one sub-tab per view: Range FFT, RTI, Profile, Range-Doppler, Movement, Angle of Arrival, Range-Angle, Micro-Doppler, 1D Dopp, CFAR, Vitals, HR, AI.
- **Small windows and displays:** no setting is ever cut off.
  - Every row of settings wraps onto further rows when the window is narrow.
  - The tall settings columns scroll: the capture parameters in the Decode tab, the left column of the HR and AI tabs, and the Helper tab.
  - When the window is too low for the settings of the shown view plus a plot 160 px high, the Analysis tab gets a scrollbar on the right. In a higher window there is none and the plot takes all the room that is left.
  - The mouse wheel scrolls the panel under the pointer.
  - The left columns of the HR and AI tabs keep their width and the plots take the rest; drag the divider between them to change it.

### Quick start

The shortest path from a recording to a result:

1. **Decode tab:** click **Browse...** and choose the `<name>_Raw_0.bin` of your recording. The log next to it is found and read at once; the parameter panel fills with the configuration.
2. Check a few parameters: the frame rate (`framePeriodicity_ms_selected`), the number of frames (`numFrames`), the range resolution (`rangeRes_cm`) and the maximum range (`maxRange_m`). See [Capture parameters panel](#capture-parameters-panel).
3. Press **Run Fastcheck** (in the Decode tab or in the **Fastcheck** tab). The raw data are read - for a long recording this takes from a few seconds up to a minute (watch the status bar) - and checked in one go. Read the coloured verdict and the findings, worst first: every line is **ok**, **info**, **warning** or **error**. The findings are also written to the **Log** tab. See [Fastcheck tab](#4-fastcheck-tab).
4. **Analysis tab**, sub-tab **Profile**, click **Run Profile**. The data were already read by the Fastcheck, so it is quick. (Without the Fastcheck, the first Run of any view reads them.)
5. Drag the frame slider under the global bar to move through the recording, or press **Play All**.
6. Try other views: **Range-Doppler** -> **Run Range-Doppler**; **Vitals** -> **Run Vital Signs Analysis** for a seated person; **Angle of Arrival** -> **Run Angle of Arrival**.
7. Export what you need from each view's **Export** menu (next to its ⓘ icon).

For complete step-by-step procedures see [Workflows](#10-workflows).

### How settings take effect

- **Most settings wait for Run.** Settings are collected and applied when you press the view's Run button, so several can be changed together. When a change affects a view that has already been computed, its Run button turns **yellow** and the status bar says *settings changed - press Run to recompute*.
- **Some act immediately:** the frame slider, **Colours**, **Range [dB]**, **Antenna** (on the Profile view), **Remove stationary** (Range FFT), **Proj** (3D views), **Show Peaks** (Range-Angle).
- **Settings that change how the raw data are read** - **Chirp**, **Lane Swap**, **IQ correction** - discard everything computed; the next Run reads the data again.
- **Segment** and **Range FFT** discard the computed results but keep the data already read, so the next Run is quick.
- **Results are kept** until something invalidates them: switching between tabs does not recompute anything.
- **Cancel** in the status bar stops a long computation; the view keeps its previous result.

## 2. Recording data for DARTscope

### Before you record

- Put the radar on a stable stand; any motion of the radar itself looks like motion of the scene.
- Keep metal and moving objects (fans, curtains, monitors with moving parts) out of the range you are interested in, or record an empty scene first so it can be subtracted (see [Angle of Arrival](#angle-of-arrival), **Rec Clutter**).
- Synchronise the clock of the recording PC with the clock of any reference device (somnograph, camera). DARTscope lines recordings up by their absolute start time.
- Record at least 60 seconds for vital signs; the default analysis windows are 15 s (heart) and 30 s (breathing).

### Chirp configuration for vital signs

- **Frame rate:** at least 20 frames per second; 50-100 fps is comfortable. The chest motion is sampled once per frame, and the default heart band reaches 150 beats per minute (2.5 Hz).
- **Transmitters:** one TX is enough for one person. Several TX in time-division (TDM-MIMO) are needed only for measuring angles.
- **Chirps per frame:** a few; DARTscope averages the chirps of a frame, which improves the signal-to-noise ratio.
- **Range:** the person 0.5-2 m in front of the radar, seated and still, chest facing the radar. Nothing else moving at the same distance.
- The [Helper tab](#9-helper-tab), **Purpose** **Vital signs (seated person)**, designs such a configuration and writes the Lua script for mmWave Studio.

### Chirp configuration for people and angles

- **TDM-MIMO:** each chirp of the pattern must switch on exactly one transmitter (masks 1, 2, 4, ...). DARTscope then forms the virtual array; the pattern it found is listed under *Chirp TX Pattern* on the Decode tab.
- **Maximum velocity:** `maxVelocity_m_s` in the parameter panel must exceed the fastest motion you want to measure (walking is about 1-1.5 m/s). Faster targets fold over (alias); see [CFAR](#cfar), **Unfold velocity (TDM)**.
- **Range resolution** (`rangeRes_cm`) decides whether two people at almost the same distance can be separated by range; the angle separates them otherwise.
- The [Helper tab](#9-helper-tab), **Purpose** **People and angles (AoA, tracking)**, designs a TDM-MIMO configuration with all transmitters.

### Chirp configuration for fast limb movements

For the [Micro-Doppler](#micro-doppler) view - swings of legs and arms and their maximum speed:

- **Maximum velocity** (`maxVelocity_m_s`) should exceed the fastest limb: a foot swings forward at roughly 4-5 m/s in walking, a kick reaches 10-20 m/s, a hand 1-2 m/s in walking and 5-10 m/s in a punch. With one transmitter and a 50 us chirp (idle time about 10 us instead of the 100 us of many configurations) it is about 25 m/s at 60 GHz; three transmitters in TDM divide it by three. A swing faster than `maxVelocity_m_s` is still measured as long as the whole body spans less than twice that (see [Micro-Doppler](#micro-doppler)).
- **Frames back to back** (frame period = chirps x chirp time) give a continuous slow time: the spectrogram then slides across frames every few milliseconds. Frames with gaps give one spectrum per frame, and need enough chirps per frame for the velocity resolution (a frame of 5 chirps resolves only about 3.5 m/s).
- **Data rate:** 256 samples x 4 RX every 50 us is about 82 MB/s, far above the 30.6 MB/s the lab's recordings reached without losing packets. Use fewer samples or receivers, a slightly longer idle time, or gaps between frames (see the [capture health report](#capture-health-report)).
- The [Helper tab](#9-helper-tab), **Purpose** **Fast limb movements (micro-Doppler)**, designs such a configuration within 30 MB/s.
- For telling a leg from an arm by height, record with an array that has vertical extent (IWR6843AOP, TDM-MIMO) and note the height of the radar above the floor.
- To validate speeds, record a reference at the same time (motion capture such as Captury, or an IMU on the limb) and make a sharp event at the start - a clap or a stamp - to line the two up.

### Chirp configuration for the heart rate of a moving person

For the [Moving person (phase 12)](#moving-person-phase-12) analysis of the HR view (roadmap 12.1):

- **Frames back to back** (continuous chirping), or at least short gaps: across a gap the chest's phase has to be predicted, and a walker's chest turns it by tens of radians in 10 ms.
- **At least 100 frames per second**, better more: the envelope of the heart's 5-25 Hz content needs 50 Hz at the very least.
- **Several chirps of one transmitter per frame** - the tracker measures the velocity from them; 4 or more.
- **TDM-MIMO on a 2D array** (IWR6843AOP) for the beam on the trunk and to tell the legs apart by elevation, as long as `maxVelocity_m_s` stays above the walking speed (about 1.5 m/s).
- **A reference:** an ECG chest belt or a pulse oximeter recording at the same time, and a sharp event at the start (a clap) to line them up. Export the reference as a CSV with the columns `time_s` and `hr_bpm` for `tools.heart_rate`.
- Record the same person at rest, swaying, doing squats and walking, each for a minute or more, so each motion class can be judged on its own.
- The [Helper tab](#9-helper-tab), **Purpose** **Heart rate of a moving person**, designs such a configuration (all TX, frames back to back, 100 fps).

### Recording a reference measurement

- A somnograph (e.g. the PPG and pulse channels) exports each signal as a text file whose header contains the start time. Export at least **Pleth** (the PPG waveform) and **Pulse** (the heart rate).
- Keep the exports of one session in one folder, by default `somno_vysledky` next to the radar recordings. `python -m tools.pair_reference <radar folder>` then tells which session covers which recording and at which offset. See [Command-line tools](#11-command-line-tools).
- The comparison itself is done in the [AI tab](#ai), see [Comparing with a reference](#comparing-with-a-reference-ppg-and-pulse).

### Measure tab

The tab has two sub-tabs: **Single-chip radar** (the IWR6843AOPEVM, described here) and **Cascade radar** (the AWR2243-2X-CAS-EVM, see [Cascade radar](#cascade-radar)). Both use the same DCA1000; the cascade takes it over only while the single-chip sub-tab has no take prepared.

The **Single-chip radar** sub-tab records with the IWR6843AOPEVM and the DCA1000 **without mmWave Studio** - it boots and configures the radar itself (TI's RadarLinkDLL through the 32-bit Python in `Radar/.py32`). Close mmWave Studio first: it holds the radar's COM port and the DCA1000's USB. The same window opens on its own with `python -m acquisition gui`.

- **Devices:**
  - **Radar (no mmWave Studio)** records with the hardware; **Simulator (no hardware)** streams a synthetic scene (a person swaying about 1.5 m away) through a simulated DCA1000, to try everything at the desk.
  - **Radar COM port:** found by the USB identity of the board, not by its number - on the IWR6843AOPEVM the *Enhanced COM Port* of its Silicon Labs CP2105. **Detect** looks again (after replugging); the line under it says whether the radar and the DCA1000's USB were found and warns while mmWave Studio runs.
  - **Check DCA1000** checks the network route to 192.168.33.180, the UDP ports and the DCA1000's FPGA version. The label on the left (**radar: not powered** at first) says whether the firmware is loaded.
- **Single measurement - Settings:** the chirp settings in physical units. **Load script / log ...** takes one of the lab's Lua scripts, the `_LogFile.txt` of a recording (to record again the same way) or a sidecar; **Preset ...** designs a configuration for a purpose as the [Helper tab](#9-helper-tab) does, with the purpose's own recording length (vital signs 120 s) - shortened to whole seconds only when the recording would not fit in one 1 GiB file, which the grey line under the buttons then says. What the form does not show (HPF, LVDS, DCA1000) comes from the loaded script.
  - **TX pattern:** the transmitters of each chirp of a loop, in order - `1` one transmitter, `1, 2, 3` TDM-MIMO over three, `2, 1+3` two chirps of which the second uses TX1 and TX3 together (examples next to the field: `1 | 1, 2, 3 | 2, 1+3`). **RX:** the receivers.
  - **Ramp end time** 0 follows the sampling (ADC start + sampling time + 1 us). **Duration** 0 records until **Stop / cancel**.
- **What they give:** range resolution and maximum range, maximum velocity and velocity resolution, frame rate, chirps per frame and virtual antennas, data rate and file size - and every problem, errors first (device limits, the loader, the DCA1000). A configuration with an error cannot be prepared.
- **Recording:** folder (**...** chooses it), name (numbered `_2`, `_3` ... rather than overwritten), subject, **Movement** and **Note** go into the recording's sidecar. **prepare the next take automatically** makes the radar ready again after every take with the same settings.
- **Take:**
  1. **1. Prepare radar** configures the radar: the first time it loads the firmware and powers the radar up (about 20 s), afterwards about a second.
  2. When the line says READY, take your place and press **2. START** - or the space bar (not while typing in a field). **Countdown** gives you seconds to get into position, with a **beep** each second, a high one at the start and a low one at the end.
  3. **Stop / cancel** cancels a prepared take, or ends a recording early (an open-ended one normally).
  - Changing a setting while READY prepares the radar again with the new one.
- **Last take:** a short report within seconds - frames and lost packets, where the motion is (range and its spread), how far above the noise, how fast (|v|, 95 %) against the maximum velocity - and what to change: velocity folding (the motion is faster than the maximum velocity: fewer TX in TDM or shorter chirps), a target at the end of the covered range, ADC clipping (lower the RX gain), too few frames per second for vital signs. **Open in DARTscope** opens the recording in the Decode tab; **Open folder** shows its files. The report is also saved as `<name>_quality.json`.
- The settings of the tab (folder, port, parameters, series) are kept for the next start.

### Series of measurements

The **Series (parameter sweep)** sub-tab of the Measure tab finds which settings suit a movement: it records the same movement once (or more) with each combination of settings, every take started by you, and ranks them.

- **What to vary:** up to four parameters with their values - numbers `32, 64, 128` or `50..200/50` (from..to/step), TX patterns and RX sets separated by `|` (`1 | 1, 2, 3`, `1234 | 12`). Everything else is the Single measurement settings.
- **Takes per configuration** records each configuration more than once; **random order** shuffles the takes, so getting tired or better at the movement does not favour one setting.
- **Kind of movement (ranking):** what the score favours - *General movement*, *Walking / whole-body movement*, *Fast limbs (micro-Doppler)* (no velocity folding above all, many velocity bins), *Vital signs* (a steady target, 20 fps or more, small phase steps between frames), *Angles* (more virtual antennas). It can be changed afterwards; the ranking follows.
- **Plan** lists every combination with what it gives; combinations the radar or the loader cannot do are grey, with the reason. **Start series** records into `<folder>/<Series name>/`: the radar is prepared for each take, you press **2. START** (or space) when ready. **Skip take**, **Repeat last** (one more take of the configuration just recorded) and **End series** steer it. Double-click a row to open that take in DARTscope.
- **Plan and results:** the takes with their score and report; the best configuration green, takes with a warning (e.g. velocity folding) orange. On the left, **what to take from it:** the best configuration, how much each parameter mattered (the mean score of each of its values), folding, lost data, clipping, and whether takes of the same configuration disagree (the movement varied).
- **Open report** shows the HTML report (`<series>_report.html`, also `_series.csv` and `_series.json`); **Use the best configuration** copies the winner into the Single measurement settings.
- The score is motion SNR in dB minus penalties (folding, lost data, clipping, a target at the end of the range) plus what the kind of movement needs - a heuristic to compare takes of the same movement, not a physiological measure. Look at the best takes in the Analysis views before settling on a configuration.

### Cascade radar

The **Cascade radar** sub-tab of the Measure tab records with the **AWR2243-2X-CAS-EVM** (two cascaded AWR2243 front-ends, 6 TX and 8 RX, and an AM2732R processor) and the DCA1000 - without mmWave Studio, on macOS or Windows. The AM2732R runs TI's cascade test from the mmWave MCU+ SDK, loaded over the board's own USB (XDS110, JTAG) by TI's debugger (from UniFlash or Code Composer Studio 20+); it configures both front-ends with the settings of the tab and passes their data to the DCA1000 over LVDS. The settings are written into the firmware before every take, so any change takes effect at the next **START**.

- **What it needs:** TI UniFlash or Code Composer Studio 20+ installed (they provide the JTAG debugger; on Windows CCS is taken first, it needs no links; the firmware comes with DARTscope in `cascade/firmware`), the cascade's 12 V and its micro-USB straight into the PC (a charging-only cable shows nothing), the J6 jumper on RUN, the DCA1000 on the Samtec connector with 5 V and Ethernet (192.168.33.30). Tested on macOS (UniFlash) and Windows 11 (CCS 20.4; the very first JTAG run there took about a minute).
- **Cascade devices:** **DCA1000** answers with its FPGA version, **Cascade JTAG** finds the XDS110 on USB; next the firmware and the TI installation found. **Check devices** looks again. **Test JTAG** connects to the AM273x and lets go - TI's debugger, the drivers, the cascade's USB and 12 V checked without loading the firmware or starting the radar. On the right, in sight before every take: *✓ Calibrated* (green from a corner reflector, orange from the autofocus) with its date, residual and check, or *✗ Not calibrated* - see [Calibration](#calibration-of-the-cascade).
- **Preset:** each one recorded on the hardware and measured from its own data (protocol: `docs/CASCADE_PROTOCOL.md`, made with `python -m tools.cascade_protocol`):
  - *Vital signs - 1 person, 100 fps*: TX1, 8 RX, 5.9 cm, 32 chirps averaged per frame; displacement noise 0.12 µm in the heart band.
  - *Vital signs - multi-person (TDM 48 ch)*: all six TX in turn, 48 virtual channels to tell people apart by angle, 50 frames per second.
  - *People - multi-person tracking (TDM 48 ch)*: 6.0 cm, ±4.9 m/s, 0.31 m/s velocity bins, 15 frames per second.
  - *Imaging - best range resolution (3.9 cm)*: 76.0 GHz start, 4.95 GHz sweep, 48 virtual channels.
  - *People - high resolution (3.9 cm, 0.23 m/s, TDM 48 ch)*: the imaging profile's 3.9 cm (to 7.5 m) with 32 loops of all six TX: 0.23 m/s velocity bins, ±3.7 m/s, 20 frames per second, 23.6 MB/s - not yet run through the protocol. The DCA1000 -> PC link carries about 25 MB/s safely, and that fixes loops x frames per second: 64 loops give 0.12 m/s at 10 frames per second, 128 loops 0.06 m/s at 5.
  - *Motion / micro-Doppler (fast)*, *Long range (to ~40 m)*, and *SDK test profile (what the repo ran)* (2.1 m range resolution - what the cascade recordings made with the lab's repository have).
  **Load ...** takes saved settings or the `_cascade.json` of a take (to record again the same way), **Save ...** keeps the settings.
- **Chirp profile:** start frequency, slope, idle time, ADC start, ramp end, TX start, samples per chirp, sampling rate (complex), RX gain, HPF1/HPF2, TX backoff - one profile for both front-ends.
- **Transmitters:** a TX mode, or the **TX masks per chirp** of a loop as primary/secondary masks (1 = TX0, 2 = TX1, 4 = TX2): `7/7` all six at once, `1/0` one, `1/0; 2/0; 4/0; 0/1; 0/2; 0/4` TDM-MIMO over all six (48 virtual channels). Up to 12 chirps per loop.
- **Frame and duration:** loops per frame, frame period, and the **Duration** of the take - the number of frames follows.
- **What they give:** the quantized slope, sampling time, sampled bandwidth, range resolution and maximum range, chirp time, chirps per frame, the active part of the frame, maximum velocity and velocity resolution, frame rate, virtual channels, data rate, file size, how long a chirp's data take on the CSI-2 and LVDS links, and the DCA1000's packet gap.
- **Checks:** red - cannot be recorded (outside 76-81 GHz, sampling past the ramp end, chirps longer than the frame, a chirp shorter than 1.2x its LVDS transfer - 512 samples at 20 Msps stalled after one chirp otherwise -, data rate above what the DCA1000 -> PC link carried without loss, ~35 MB/s, ...); orange - allowed but probably not wanted; green - ready.
- **Recording:** **Folder** (**...** chooses it), **Name** (**New name** makes a time-stamped one) and **Note**.
- **Take:** **START** arms the DCA1000, loads the firmware with the settings, records the whole duration (progress, lost packets) and stops the firmware; **Stop** ends a take early. A take takes about 10-15 s longer than its duration (loading and stopping the firmware). If the radar refuses the settings, the log says at which step and why (e.g. *MMWave_config: EPROFILECFG (profile)*) and nothing is kept.
- **Files:** `<name>_Raw_0.bin` (`_Raw_1.bin` ... beyond 1 GiB) and `<name>_Raw_LogFile.csv` as for the single-chip radar, and `<name>_cascade.json` with the settings, what they give, the layout - int16 I/Q, `[frame][loop][chirp in loop][RX 0..7][sample]`, RX0-3 from the primary AWR2243, RX4-7 from the secondary - and the board's calibration at the time of the take, if there was one.
- **Open in DARTscope** (after a take, and in the Preview for the take shown) opens it in the [Decode tab](#3-decode-tab) like any recording - also **Browse** there, the `_cascade.json` stands in for the mmWave Studio log. The parameter panel shows the cascade's profile, 8 RX (mask 255) and the transmitters as TX1-TX3 (primary) and TX4-TX6 (secondary); Fastcheck and the Analysis views work as for the single-chip radar. A take is decoded with its calibration (the one kept in its `_cascade.json`, else the board's current one - the Decode log says which): its antenna configuration and channel corrections go with it, and the angle views and [Multi Vitals](#multi-vitals) use them without anything selected. Without a calibration the angle views stop with a message until the antenna configuration is selected on the Analysis bar, and Multi Vitals finds people by range only.
- **Preview** (opens after every take; **Open take ...** opens an older one, until then it says *No take open*): the range profile of the frame on the **Frame** slider for all 8 RX, the range-time map (a click picks the range), range-velocity of that frame, the displacement at the picked range for the chosen **RX**, and its spectrum with a rough breathing and heart rate.
- The DCA1000's packet gap is set from the data rate (36-100 µs): with the minimum 25 µs the PC's USB Ethernet adapter lost packets in bursts. The firmware is rebuilt only for changes beyond the settings (`cascade/firmware/README.md`).

#### Calibration of the cascade

An angle is read off the phase differences of one echo between the antennas - across the cascade's 18.5-wavelength aperture one degree is 116 degrees of phase at the far end (9 on the IWR6843AOP). The radar measures that phase plus the phase of its own electronics: feed lines, every TX and RX channel, and the second chip with its own front end. The calibration measures the electronics once and takes it off every take. Breathing and heart rate do not need it (a constant phase drops out of a change over time); where a person is, and telling two apart at one range, do. On the lab's six-TX takes the raw channels already focused a seated person into one beam (coherence 0.88 at +0.7 deg) - the calibration takes off the last tens of degrees per channel. With one TX (the vital-signs preset) the 8 RX form two groups 8 wavelengths apart and a person shows at several angles 6 deg apart however well calibrated: angles need six TX.

**The antenna layout** (the built-in configuration *AWR2243-2X-CAS-EVM (6TX x 8RX cascade)*): the positions are those of TI's user guide (SWRU639, Figure 2-3) - TX at 0, 1 (raised 0.8 wavelengths), 2, 3.5, 5.5 and 7.5 wavelengths, RX at half-wavelength pitch, the secondary's 8 wavelengths after the primary's - but within each AWR2243 the TX channels run the other way than the guide's labels: chirp masks 4, 2, 1 drive the antennas at 0, 1 and 2 wavelengths, masks 32, 16, 8 those at 3.5, 5.5 and 7.5. Measured on 2026-10-11 with a corner reflector on 8 marked floor positions (1 and 2 m away, 0.25 and 0.75 m to either side, one take per position): split into a TX and an RX part, the echo put the RX where the guide has them and the TX in this order. With the guide's order every six-TX take smeared one person over tens of degrees and fitted a point only 0.30-0.49 (a person standing on the 8 marks came out at mirrored angles near the centre); with the measured one people fit 0.65-0.91, the person on the 8 marks lands on them to about 10 cm and the reflector's 8 positions to a few degrees (it stood on the marks only to a few cm). Takes with all six TX on one chirp (mask 63, the older presets) use only the 8 RX and are not affected. The raised TX (mask 2) is raised upwards: with the reflector 25 cm above the radar 2 m away (+7.1 deg) its phase turned by +37 deg against the takes at the radar's height (+36 expected), the other TX by 9 deg at most. The board sees only +-10 deg in elevation (SWRU639): 50 cm below the radar at 1 and 2 m (-27 and -14 deg) the reflector did not show at all - keep the radar at the height of the chests it is to watch. Positive angles are on the radar's left, as on the IWR6843AOP.

The **Calibration** tab of the Cascade radar:
- **The board's calibration:** what is in use (*✓ Calibrated ...* or *✗ Not calibrated*); **Load ...** a calibration file (or a take's `_cascade.json`) as a new result, **Export ...** the one in use, **Forget** it (it stays in the archive and in the takes made with it).
- **Corner reflector (absolute angles):** place a trihedral corner reflector (5-10 cm edges; three squares of cardboard covered in aluminium foil will do) on a non-metallic stand at the radar's height, enter its **Distance [m]** and **Angle [deg]** (0 = straight ahead) and press **Record and calibrate**: 5 s with all six TX and the chirp profile of the *Imaging - best range resolution (3.9 cm)* preset whatever the take settings are (with a coarse profile the room falls into a few range cells and nothing is a point; the phases are moved to every other profile by the path offsets). Then move it to another angle (20-30 deg), enter it and press **Record and verify**: the reflector's angle without and with the calibration - the fit on its own take always looks right, the second position says whether it holds.
- **Person (no reflector):** someone sits still, alone, facing the radar 1.5-3 m away and breathes quietly; the breathing chest - the strongest moving echo - is taken as a point at the **Angle [deg]** entered (where they sit; empty keeps the raw channels' pointing and the angles are relative). **Record a person and calibrate** (20 s, six TX) or **Calibrate on a person in a take ...**. The other moving echoes of the take are kept out and reported; a calibration that spoils one of them is refused. On the takes of 2026-10-09 (someone seated about 2 m in front) a calibration from the person in one take took the person in the other two from 0.87-0.88 to 0.92-0.98, the three calibrations agreed to 2.5-5.9 deg, and in the Angle of Arrival view the strongest sidelobe next to the person went from -4.1...-5.2 dB to -5.2...-6.7 dB (a perfect point in this array: -6.7 dB), angle and width unchanged.
- **Autofocus (no reflector, angles relative):** **Record the room and autofocus** (10 s) or **Autofocus a take ...** (any six-TX take): the still point-like reflectors of the room are fitted; the people in it and every third still reflector are left out and judge the result. It is kept only when two or more of the fitted reflectors, each taken alone, agree, the ones left out end up focused and none that the raw channels focus well gets worse - otherwise it says why and leaves the board as it was. A room of walls and furniture has no points: in the empty room of 2026-10-11 ten takes gave calibrations that repeated to 1-2 deg, each from one echo forced into a point - and 85 deg off the person's; they took the person from 0.89 to 0.23-0.31. Repeating is not being right; it refuses such scenes now. A turn of the whole picture is invisible to it - it keeps the raw channels' pointing.
- **Result:** what was measured - the phase of every TX and RX, the path offset of each (mm), how plane the echo is before and after (1 = one clean beam), the residual of the model and any warning; **Save as the board's calibration** makes it the one in use.
- **How stable it is:** ten takes of the empty room, each with a fresh firmware start, gave every channel's phase to 1.5 deg (worst channel 2.7 deg), and the Angle of Arrival view put the room's objects at the same angles to 0.1-0.2 deg: a calibration holds over restarts. Whether it holds over switching the board off is still to be checked.

How it works: every TX and every RX has its own complex gain (14 unknowns against 48 channels; what the product does not explain is the *residual* - a large one means a reflection near the reflector or a misplaced one). So a calibration from six TX serves any TX pattern, the one-TX preset included. The echo is modelled along the exact paths TX -> reflector -> RX (the far field of this aperture begins at 2.7 m; a plane wave is 30 deg off at the array's ends at 2 m). The phases hold at the centre frequency they were measured at; from where each channel's peak lies the extra path of every TX and RX is measured, and the phases are moved to the frequency of each take (1 cm of extra path and 1 GHz: 24 deg). The board's calibration is kept in DARTscope's configuration folder with an archive of every saved one (`cascade_calibration.json`, `cascade_calibrations/`); calibration takes go to `calibration/` in the recording folder.

## 3. Decode tab

### Selecting a recording

- **Select BIN file** / **Browse...**: choose the `<name>_Raw_0.bin`. Choosing a file decodes its log at once.
- You can also type or paste a path into the field and press **Decode Log**.
- A recording split into `_Raw_0.bin`, `_Raw_1.bin`, ... is always opened by its `_Raw_0.bin`; the other parts are read automatically.
- The log is looked for next to the BIN file as `<name>_LogFile.txt` (the `_Raw_0` part is taken off the name). If it is not there, the message *Corresponding LOG file not found.* appears.

### Decode Log

Pressing **Decode Log** (or choosing a file):

1. clears everything computed for the previous recording,
2. reads the mmWave Studio log and the recorder log and derives the physical parameters (frequencies, resolutions, frame rate),
3. fills the [parameter panel](#capture-parameters-panel), the **Chirp** list on the Analysis bar, and the playback speed,
4. clears the **Segment** fields and applies the range bias of an active calibration.

The raw data are **not** read yet; **Run Fastcheck** or the first Run of any view in the Analysis tab reads them. That is also when the [capture health report](#capture-health-report) is produced.

If the log does not contain a profile configuration, the message *ProfileConfig not found in the log file.* appears - the log belongs to something else than a recording.

**Quick start (Manual)** opens the [Quick start](#quick-start) of this manual. **Run Fastcheck** switches to the [Fastcheck tab](#4-fastcheck-tab) and checks the recording - usually the first thing to do after choosing one.

### Recent history

The list of the last 20 recordings opened, newest first.

- **Double-click** a line, press **Enter**, or select it and click **Open**: opens and decodes that recording.
- **Dialog...**: the same list in a separate window.
- **Refresh**: reloads the list. **Clear**: empties it.
- A recording whose file no longer exists is removed from the list when you try to open it.

### Capture parameters panel

Everything the logs say about the recording, in groups:

- **General:** chip, start and stop frequency, chirp slope, idle / ADC-start / ramp-end times, number of ADC samples (`numAdcSamples`) and the sampling rate (`digOutSampleRate_MHz`).
- **Bandwidth & Resolution:** bandwidth, range resolution (`rangeRes_cm`, the smallest distance between two targets that can be separated), range bin spacing, maximum range (`maxRange_m`).
- **Velocity (Doppler):** wavelength, chirp repetition time, number of Doppler bins, maximum unambiguous velocity (`maxVelocity_m_s`) and velocity resolution.
- **Data Format:** ADC format (Real, Complex1X, Complex2X), bits, IQ order, enabled receivers, number of LVDS lanes and where that number came from (`numLanes_source`).
- **RX/TX & Gains:** receiver gain, transmitter power back-off, phase shifter.
- **Frame (derived/log):** the frame period that is used (`framePeriodicity_ms_selected`) and where it came from, and the number of distinct chirps per frame.
- **Recorder (DCA1000 log):** logging mode, lane mode, duration, packets received, **zero-filled packets** (lost data) and packet loss in per cent.
- **FrameConfig:** first and last chirp index, loops, chirps per frame, **numFrames**, frame periodicity, trigger.
- **Chirp TX Pattern:** which transmitter each chirp of the frame uses - the basis of TDM-MIMO.
- **Chirp N-M:** one group per chirp configuration.

Values the application derives itself (for example the resolutions) are computed from the logged configuration, not measured.

### Capture health report

When the raw data are read for the first time (**Run Fastcheck** or the first Run after decoding), DARTscope checks the recording and writes a report to the [Log tab](#5-log-tab), one line per check, marked **ok**, **info**, **warning** or **error**. An error is also shown in the status bar. The report never stops the loading; it tells you what the results can be trusted for.

| Check | What it looks at | What to do when it is not ok |
|---|---|---|
| recorder log | whether `<name>_Raw_LogFile.csv` is present | copy it next to the recording; without it start time and packet loss are unknown |
| capture start | where the start time comes from | a start time taken from the mmWave Studio log can be minutes off - do not use it to line up a reference |
| lanes | the LVDS lane count and its source | an error means no usable lane count: check the recorder log and the lane configuration |
| packet loss | zero-filled packets reported by the recorder | lost packets read as phase jumps; see *zero-filled frames* |
| ADC window | whether the ADC samples fit inside the chirp ramp | an error means samples were taken outside the ramp - fix the chirp timing |
| frame rate | whether a frame period could be derived | a warning means 15 fps was assumed; time axes and rates are then wrong |
| antenna | which antenna layout is assumed | choose the right one in [Angle of Arrival](#angle-of-arrival), **Array** |
| raw files | whether the BIN parts were written with this log | an error means the BIN file belongs to a different recording with the same name |
| raw data | whether the BIN file holds data at all | an empty file: record again |
| ADC saturation | samples at full scale | warning or error: lower the RX gain or move the radar away; clipped samples lose their phase |
| REAL container | how REAL-format data are packed | information only |
| frame count | decoded frames against the declared number and the recorder duration | a mismatch usually means the wrong lane count or format - check the Data Format parameters |
| finite | invalid numbers in the data | an error means corrupt data |
| RX channels | receivers that carry nothing | a dead receiver ruins angles - check the RX enable mask and the hardware |
| DC offset | the DC level of each receiver | a large DC offset hides nearby targets; usually harmless with DC removal |
| IQ balance | how far targets stand above their mirror images (image rejection) | an error saying *I and Q look swapped* means a wrong IQ order; a low value can be improved with **IQ correction** |
| noise floor | the noise level of each receiver | one receiver much noisier than the others: check that receiver |
| noise spectrum | lines and a rising floor in the noise after the static scene is removed | information: interference, or something that moved |
| zero-filled frames | frames the recorder filled with zeros | these frames are marked in the Vitals plots; choose a clean part with **Segment** |

## 4. Fastcheck tab

A quick check of the whole recording before you analyse it: is the data complete, does the signal look sane from the first frame to the last, and is somebody breathing in front of the radar. One button; the check itself takes a few seconds even for a recording of several gigabytes.

**Use:**
1. Choose the recording in the **Decode** tab (**Browse...** or the **Recent history**).
2. Press **Run Fastcheck** - here, or the **Run Fastcheck** button of the Decode tab, which switches to this tab. The first run reads the raw data, like the first Run of an Analysis view; the Analysis views then reuse them. The status bar shows the progress; **Cancel** stops the check.
3. Read the verdict at the top, then the findings, worst first.

**The verdict** is the coloured line at the top:
- green **OK - nothing wrong found**;
- orange **Usable with care** - there are warnings (for example a few lost packets): read them before relying on a result that they affect;
- red **Problems** - at least one error: find out what it is before using the recording.

Under it are the key numbers: length, frames and frame rate, chirps x RX x samples, **complete frames** (the share of frames without zero-filled packets), the number of **dropouts** with the longest one, and the **longest clean stretch** - how long a piece of the recording a vital-signs analysis can use without a gap.

**The findings** - columns **Status**, **Check** and **Result**; click a row to read all of it under the list. They are the [capture health report](#capture-health-report) (the logs, the files, the start of the raw data and the first frames) and what the pass over every frame found:

| Check | What it looks at | When it is not ok |
|---|---|---|
| zero-filled frames | frames holding a long run of exact zeros - packets the recorder lost and filled in (the same test as the red shading in the Vitals view) | a warning for any; an **error** above 25 % of the frames |
| recorder vs data | the recorder's own count of zero-filled packets against the frames found | a warning when the recorder reports none, yet frames hold runs of zeros |
| clipping | samples at full scale, in every frame | a warning for any frame; an **error** above 10 % of the frames |
| receivers over time | the power of every receiver in every frame | a receiver 20 dB below its own usual level: it fell silent |
| signal level | the total received power over the recording | information when it changes by more than 10 dB: something moved close to the radar, or the gain changed |
| frame glitches | how much the range profile changes from one frame to the next, as the median over all ranges - a person moving changes a few ranges, a glitch all of them | more than 3 dB and 6 times the recording's usual change: a burst of interference, or the data stream out of step |
| vital signs | the range with the most slow motion in the breathing band, and whether it has a clear rhythm | information only |

On the recordings it was tried on, the share of zero-filled samples it found agreed with the recorder's packet loss to the hundredth of a per cent (13.63 % and 5.36 %).

**The plots** (zoom and pan with the toolbar under them):
- **RTI** - range against time, the mean power of every frame. Dropouts are marked red along the top edge; the dashed line is the range with the most breathing. **Motion only (RTI)** removes the static scene (the average of the whole recording) and leaves what moves: a person shows as a bright band.
- **Integrity** - the power of every receiver in every frame, in dBFS. Zero-filled frames are shaded red and the lines break there; orange triangles mark clipped frames, purple crosses glitches. The receivers normally run parallel; a line that drops or steps shows a fault at that time.
- **Vital-band energy by range** - for every range, the power of the slow motion in the breathing band and in the heart band, relative to the noise above the heart band, in dB. The bands are the shared **Bands** of the [Vitals](#vitals) view. A seated person gives a peak of 30-45 dB at their distance; the dashed line marks the range chosen: the most breathing-band power, 0.2 m or further.
- **Breathing and heart band over time** - at that range, in windows of 30 s (a third of a shorter recording, at least 12 s) every 2 s; lost frames shaded red. The chest displacement is taken from the phase of the good frames only and stitched across the dropouts, as in the [Moving person (phase 12)](#moving-person-phase-12) analysis.
  - **Top:** the breathing rate of every window (left axis, lower part) - a black line where the rhythm is clear, grey rings where it is not. A window is clear when its breathing peak stands out of the band (about 6 times the band average for 30 s windows, as the Vitals view's **Min peak/avg**, a little less for shorter windows), or at 60 % of that when its rate is within 2 /min of the windows around it: noise gives a different rate in every window, breathing the same one. A window is also clear when its breaths, counted one by one as peaks of the chest displacement, stand above the noise: irregular breathing (3-9 s between breaths) has no clear spectral peak, and its strongest peak falls on a harmonic or on the longest pause. In such a window - and where a clear peak lies more than a quarter above the counted rate, its second harmonic - the rate is the counted one: the breath cycles of the whole recording that fall in the window, over the time they cover. The heart rate (red, right axis, upper part) - solid where its peak counts, dotted where not.
  - **Bottom:** how far the breathing band and the heart band stand above the noise floor (the band above the heart band), in dB. A seated person typically gives 30-40 dB for breathing and 10-15 dB for the heart band; a drop of both means the person moved away or the signal was lost.

**The finding** gives the median rate of the clear windows and, beside it, the breaths counted over the whole recording - *"about 9 /min (9 breaths counted in 60 s, irregular)"* - to compare with a count of your own; *irregular* when the intervals between breaths spread by more than a quarter. On a validation take where the subject counted 9 breaths in a minute, Fastcheck had said 18 /min with a clear rhythm in 2 of 21 windows; now 9 /min, 9 breaths, 20 of 21 windows.

**Breathing and heart rate:** both come from the phase of the chosen range (the chest displacement), stitched across lost frames. The heart rate is followed as the [Moving person (phase 12)](#moving-person-phase-12) analysis does it - the heart band (the breathing harmonics notched out) and the envelope of its 5-25 Hz content, a harmonic sum and a Viterbi path through the windows - not read off the largest peak, which missed the pulse of a reference oximeter by about 20 bpm - in the displacements of every receiver in the range that breathes most and in its neighbours, averaged. On six seated recordings with a somnograph pulse it was 3.3 bpm off on average (1.0-3.2 bpm on five of them, 12 on the sixth, which holds a foreign line), with 99 % of the windows counting; the reference was averaged over the same window. For detail - heart rate variability, other methods, a chosen bin - use the [Vitals](#vitals), [HR](#hr) or [AI](#ai) view. Recordings shorter than 8 s are not judged for vital signs.

**Export** menu: **Report (TXT)** - the verdict, the key numbers and every finding; **Copy report** - the same to the clipboard; **Frames (CSV)** - one row per frame: time, zero-filled, the fraction of zero samples, the fraction of clipped samples, the change from the frame before in dB, glitch, and the power of every receiver; **Figure (PNG)** - the four plots. The report is also written to the [Log tab](#5-log-tab).

**When it goes stale:** opening another recording clears the check. **Chirp**, **Lane Swap**, **IQ correction** and **Range FFT** leave it on screen with a note next to the Export menu; press **Run Fastcheck** again. **Segment** does not affect it - the check always covers the whole recording.

## 5. Log tab

Everything the application reports: the capture health report, lost-packet warnings, the CFAR target lists, calibration and IQ-correction summaries, messages about the time segment, warnings and errors (with the technical details of an error).

- **Clear** empties the log. **Save...** writes it to a text file - attach it when you report a problem. **Copy all** puts it on the clipboard.
- **Auto-scroll** keeps the newest line in view; switch it off to read something older while the application works.
- Errors are red, warnings orange, the lines of the health report and the Fastcheck blue.
- When a warning or an error arrives while another tab is shown, the tab title changes to **Log (!)** until you open it.

The Windows build has no console window, so this tab is the only place where these messages can be seen.

- Lines **[time]** give how long each step took that needed a second or more.

## 6. Analysis tab: global settings

The bar at the top of the Analysis tab holds the settings that apply to several views. It wraps onto several rows in a narrow window.

### Global frame: Play All and Stop

- The **frame slider** under the bar selects the frame shown by every view that shows one frame at a time. It becomes active once a view has been computed and covers the frames of that view.
- **Play All** plays the frames at the speed of the recording (the frame period from the log); **Stop** stops. At most 25 frames a second are shown (of a 50 fps cascade take every second one).
- **The first run is slow, the next ones in real time.** A view needs 70-150 ms to draw a frame (matplotlib draws the whole figure, ~65-70 ms; Imaging's Bird's-eye computes the image under the points and rebuilds its axes on top), a cascade take has one every 20 ms. So every frame a view draws is kept as a picture: the first Play draws every shown frame at the pace of drawing it, the next ones show the pictures by the clock - a frame in ~45 ms (Bird's-eye: 150 ms drawn; most of the 45 ms is macOS putting the picture on the screen). The corner of the status bar says which, and how fast it really goes - frames shown a second and the speed against the recording, e.g. *Cached playback - 22 fps shown, 1.00x real time* or *Rendering playback (first run is slow) - 7 fps shown, 0.27x real time* (below 1.00x it is slower than the recording); when stopped, how many frames are kept (*Playback cache: 640 frames, 60 MB*). A frame dragged to with the slider comes from its picture too.
- The pictures are what the view drew: anything that makes it draw otherwise - another setting, a new Run, zoom, a resized window, another recording - forgets that view's pictures, and the next run draws again. All of them together stay under 2 GB (about 5000 frames of a full-window view), the oldest go first. When the frames stop coming, the view draws the last one itself (zoom and the cursor work on it).
- The label next to the Segment shows that speed, for example *100.00 fps (10 ms)*.
- Views update while playing only when their tab is shown.
- The plots drawn over the whole recording or segment (RTI, Vitals, HR, AI, Moving person and the like) show where that frame is with a dashed vertical line; it moves with the slider and with Play All. Micro-Doppler keeps its red line.
- No view has a frame slider or Play button of its own: this slider and **Play All** drive them all.

### Antenna

How the receivers (RX channels) are combined in the Range-Doppler map, the Range Profile, the views built on the Range-Doppler map (CFAR, 1D Dopp) and the Micro-Doppler spectrogram:

- **MRC** (default) - power average of all receivers. Needs no phase alignment; the usual choice.
- **SUM** - coherent sum. Stronger only for targets straight in front of a calibrated array; weaker elsewhere.
- **RX0**, **RX1**, ... - one receiver only; the list has as many entries as the recording has receivers.

The angle views always use all receivers.

### Range FFT

The number of points of the range FFT, for every view.

- **auto** (default) - the next power of two of the samples per chirp, but at least 256. This is what all views always used, so older results look the same.
- **256 ... 4096** - zero padding: a finer grid of range bins. It places a target more finely but does not separate two targets closer than the range resolution.
- The FFT is never shorter than the chirp: a size that would cut samples off is raised automatically and a note appears in the Log.
- Changing it discards the computed results; the data already read are kept.

### Peak interpolation

**Peak interp** decides how the position of a spectral peak is read between two bins, everywhere a peak is measured: breathing and heart rates, CFAR targets, 3D targets, the Zoom FFT.

- **gaussian** (default) - the most accurate (about 0.02 bin error).
- **parabolic** - about 0.05 bin.
- **none** - the centre of the strongest bin, as in older versions.

### Clutter suppression

Static objects (walls, furniture) usually reflect far more than a person. **Clutter** chooses how they are suppressed along slow time (from chirp to chirp and frame to frame):

- **mean** (default) - subtracts the average of each frame. Removes everything that does not move within one frame, including a person sitting still.
- **ema** - subtracts a background averaged over the previous frames with the time constant **EMA τ [s]** (default 5 s). Removes what stays unchanged for longer than τ; a breathing person stays visible. The best choice for Range-Doppler views of seated people.
- **mti2** and **mti3** - two- and three-pulse moving target indicators. Strong suppression of anything static; slow targets are attenuated too. Good for walking people.
- **svd** - removes the **SVD rank** (default 1) strongest components; follows static clutter whose strength drifts a little.
- **none** - no suppression.

It applies to the Range-Doppler map (and so CFAR and 1D Dopp), to the Micro-Doppler spectrogram, to the Angle of Arrival views when **Remove Static** is ticked, and to the Range-Angle view when its **Clutter Removal** is ticked. Changing it marks those views for recomputing.

### Colours and Range [dB]

- **Colours** - the colour map of all images. **auto** (default) keeps each view's own: *viridis* for most, *hot* for the Movement map. *jet* is offered but draws false edges into smooth data.
- **Range [dB]** - **auto** (default) keeps each view's own colour scale; a number *N* shows the top *N* dB below the strongest point, which hides weak clutter or, with a large *N*, reveals it. You can type any number and press Enter.
- Both act at once on every view already computed; nothing is recomputed.

### Chirp

Which transmitter's chirps are used: **All** (default) or one transmitter, for example **TX:TX0**. The list is filled from the chirp pattern of the recording when the log is decoded. Choosing one TX reads the data again with only that transmitter's chirps (and so without MIMO).

### Lane Swap (1<->2)

Swaps LVDS lanes 1 and 2 while decoding. Only for recordings made with swapped lane wiring: the symptom is data that look like noise although the health report finds the right frame count. Ticking it discards everything computed; the next Run reads the data again.

### IQ correction

Estimates the gain and phase mismatch between the I and Q branches of every receiver from the targets and their mirror images, and corrects it - but only on the receivers where the image rejection of frames not used for the estimate does not get worse. The result per receiver is written to the Log. Off by default: most boards are already well balanced (image rejection 31-42 dB was measured on the recordings this was developed with). Ticking it discards everything computed; the next Run reads the data again.

### Segment [s]

Processes only a part of the recording, for example seconds 30 to 90.

- Type the start and the end in seconds of the recording time (0 = the first frame) and press **Apply** or Enter. An empty field means the start or the end of the recording.
- The label next to it shows the recording length and the chosen part, for example *30.0-90.0 s of 180.0 s (frame 0 = 30.0 s)*. Frame numbers on the slider then count from the start of the part.
- Changing the part takes effect immediately and needs no new decoding; the results computed on the old part are discarded.
- Time axes of the Vitals tab stay in recording time, and the AI tab's reference synchronisation takes the part's start into account.
- The capture health report and the lost-packet check always look at the whole recording.
- A new recording starts with the whole recording again.

### Range Cut and Angle Lim

- **Range Cut [m]** (default 0.3) - ignores everything closer than this in the Angle of Arrival and Movement views, and in target tracking. The first few centimetres are dominated by leakage from the transmitter.
- **Angle Lim [±deg]** (default 90) - the angle span computed in the Angle of Arrival, Movement and Range-Angle views. A smaller span computes faster and gives finer detail within it.

## 7. Analysis views

Every view has a Run button (or its Run starts it), an ⓘ icon with a short description of its processing - click the icon to open its section of this manual - and an **Export** menu. Views that show one frame at a time follow the global frame slider.

### Range FFT

**Shows:** the range spectrum of one frame - reflected strength against distance, averaged over the chirps and receivers of the frame.

**Use:** press **Run Range FFT**, then move the frame slider or press **Play All**.

**Controls:**
- **Remove stationary (motion only)** - subtracts the average over all frames, so only what changes stays visible. Acts at once.
- **Intensity** (0-100 %, default 100) - how much of the average is subtracted.

**Export:** PNG of the plot; CSV of the current frame's spectrum (one value per range bin); video of all frames.

### RTI

**Shows:** Range-Time Intensity - range on the vertical axis against chirp number (time) on the horizontal one, for the whole recording at once. Moving people draw curves; static objects draw horizontal lines.

**Use:** press **Run RTI**.

**Export:** PNG; CSV of the whole map (one row per chirp, one column per range bin).

### Profile

**Shows:** the range profile of one frame in dBFS (decibels relative to the full scale of the ADC), which tells how strong the reflections are compared with clipping.

**Use:** press **Run Profile**. The profile then follows the frame slider and the **Antenna** selector.

**Controls:**
- **Zoom FFT** with **from [m]** / **to [m]** (defaults 0.5 and 2.0) - computes the profile on a much finer grid inside that band and marks the exact peak; its range is printed in the title. Useful to measure the distance to a target precisely.

**Export:** PNG; CSV of the profile (one value per range bin, in dBFS).

### Range-Doppler

**Shows:** a map of range (vertical) against radial velocity (horizontal) for one frame. Positive velocity is motion away from the radar. Static objects appear at zero velocity unless clutter suppression removes them.

**Use:** press **Run Range-Doppler**; then the frame slider or **Play All**.

**Controls:**
- **Hann window (Doppler)** (default on) - a window over the chirps; lower sidelobes at the cost of slightly wider peaks.
- **Auto velocity range** (default on) - shows the full velocity span. Untick it to zoom to **V_min [m/s]** / **V_max [m/s]** (defaults -5 and 5).
- The **Antenna**, **Clutter** and **Range FFT** settings of the Analysis bar apply.
- **Title and colours:** the second line of the title gives the number of Doppler bins and their width (**Δv**) - what the map can tell apart; slower motion lands in the 0 m/s bin, which the clutter suppression empties. The colour scale starts just below the noise floor (the median of the map) and spans at least 20 dB, so a map without motion is dark rather than noise drawn as structure; with fewer than 64 Doppler bins each bin is drawn as a block. **Range [dB]** on the Analysis bar overrides the span.
- **A few chirps per frame** (fewer than 8 per transmitter - the vital-signs settings use 4): the frame has that many velocity bins, and the default clutter suppression (the mean over the frame's chirps) would remove everything, a seated person included. The view then takes the background over frames instead (**ema** with the Analysis bar's time constant, from the first frame on) and no window, so what moved in the last seconds shows at 0 m/s, as in the [RTI](#rti); the [Log tab](#5-log-tab) notes it. Breathing and heart rate themselves are measured in the [Vitals](#vitals) and HR views.
- **A short frame** (the chirps of one transmitter spanning less than 3 ms - the cascade's multi-person preset: 8 chirps in 1.9 ms): nothing a person does moves measurably within it, and the frame mean took someone seated 2 m away out of the map entirely (what was left was noise in streaks). The background is then taken over frames as well (**ema**), the window kept: the person shows at 0 m/s, 48 dB above the floor, a walker at their velocity; the Log notes it. Longer frames keep the frame mean: with the background over frames the remains of the still scene filled 0 m/s at every range of a walking take.

**Export:** PNG; CSV of the current frame (one row per Doppler bin, one column per range bin, linear magnitude); video.

### Movement

**Shows:** a map of range against angle where colour is how much of the chosen kind of motion happened there - where somebody breathed, where the heart beat was visible, where something moved.

**Use:** choose the **Band** and press **Run Movement Map**. The angle settings of the Angle of Arrival tab (Array, Mode, Method) and the Analysis bar's **Range Cut** and **Angle Lim** apply.

**Controls:**
- **Band** - **Breathing** or **HR** use the shared search bands of the vital-sign tabs (see [Vitals](#vitals), **Bands**); **Movement** is everything between them and above the heart band.
- **Accumulation** - **Full** analyses the whole recording at once; **Sliding Window** analyses windows of **Win [frames]** (default 32, half-overlapping) and keeps the strongest value of each point, so short events are not diluted.

**Export:** PNG; CSV of the map; video.

### Angle of Arrival

**Shows:** where the reflections come from - the angle of arrival computed from the phase differences between the receivers (and, with TDM-MIMO, the virtual array of all transmitter-receiver pairs).

**Use:** choose the **View**, check the **Array**, optionally enter a **Range**, press **Run Angle of Arrival**, then use the frame slider.

#### Views and projections

- **View: 1D Spectrum** - power against azimuth at one range: the range typed in **Range [m]**, or the bin on the **Range Bin** slider under the plot. With neither (or the slider nearer than 0.3 m, the antenna coupling) the range that moves most; after the Run the slider stands on the range analysed.
- **View: Range-Azimuth** - a top view: range against azimuth for all ranges.
- **View: Azimuth-Elevation** - azimuth against elevation at one range (needs an array with vertical extent, like the IWR6843AOP).
- **View: Range-Az-El** - the full 3D map of range x azimuth x elevation. **Proj** chooses how it is drawn: **Top (R-Az)** (the maximum over elevation), **Slice (Az-El)** (the angle map at the bin on the **Range Bin** slider - moving the slider redraws it without recomputing) or **3D (x,y,z)** (a rotatable point cloud; x sideways, y forward, z height).
- The right-hand plot follows the strongest target over time (Range-Azimuth and Range-Az-El).

#### Array

The antenna layout the angles are computed for. **Auto-detect (from log)** (default) chooses it from the chip named in the log. Built-in layouts include the IWR6843AOP (3 TX x 4 RX, 2D) and the AWR2544LOPEVM (4 TX x 4 RX). **Config...** opens the antenna configuration editor:

- The list of configurations; select one to see its virtual array in **Virtual array preview**.
- **Name**, **RX positions** and **TX offsets** in wavelengths (`x,y; x,y; ...`), **TX masks** (for example `1 2 4`), optional **Element phase** in degrees for elements mounted rotated.
- **Preview** draws the virtual array; **Save** stores a new or changed configuration (it is kept between sessions); **Delete** removes a saved one; **Use this one** selects it for all angle views.
- **Check against the loaded capture:** with a recording loaded, **Compare all configurations** computes the angle of the strongest target with every configuration. Enter the **Known target angle [deg] (optional)** of a reflector placed at a known angle to rank the configurations by their error; without it they are ranked by peak sharpness, which is only a hint.

#### Mode and Method

- **Mode: MIMO** (default) forms the virtual array from all transmitters; **First TX** uses the first transmitter only; **All Chirps** treats every chirp as coming from the same transmitter (for single-TX recordings).
- **Method: Bartlett** (default) - the classical beamformer, robust. **Capon** - adaptive, narrower peaks, separates two close targets better; needs more chirps than virtual antennas and falls back to Bartlett otherwise.

#### Range

**Range [m] (e.g. 1.5 or 0.3-2):** a single number selects the range of the 1D Spectrum and Azimuth-Elevation views; a span (`0.3-2`) limits the Range-Azimuth and Range-Az-El maps to it. Limiting the span keeps a strong wall far away from setting the colour scale.

#### Options

- **Remove Static** - suppresses static reflections with the method chosen under **Clutter** on the Analysis bar.
- **MIMO** (default on) - form the virtual array when the recording allows it.
- **Invert TX Phase** - inverts the phase of every second transmitter; only for boards that need it.
- **Detect (CFAR)** with **Thr** (default 6.0) - for the **Range-Az-El** view: detects targets in the 3D map of every frame; the number per frame goes to the Log and the targets are marked in the plots. A higher threshold gives fewer, surer detections.
- **Rec Clutter** - records the current recording as an *empty scene*; **Sub Clutter** then subtracts it from the recordings analysed afterwards. Record the empty room first, then load the recording with the person.
- **Doppler Comp** - compensates the phase a moving target adds between the transmitters of TDM-MIMO; correct angles for moving targets. Has no effect on static ones.
- **Track Moving** - the tracker follows the strongest *moving* target instead of the strongest one.

#### Calibration

The receivers and transmitters of a board differ slightly in gain and phase, which biases angles. A calibration measured with a corner reflector corrects it (see [Antenna calibration](#antenna-calibration-with-a-corner-reflector)).

- **Estimate...** - with a recording of a corner reflector loaded: asks for the reflector's range, azimuth and elevation, and optionally the measured distance (for the range bias); shows the result and offers to save it as a JSON file.
- **Load...** - loads a saved calibration. **Clear** - switches it off.
- The label shows the active calibration: channels, the range bias and the file.
- It applies to all angle views, the Movement map and velocity unfolding; its range bias shifts every range axis.

**Export:** PNG; CSV of the current frame's map; video.

### Range-Angle

**Shows:** the range-angle map of every frame, for playback with the frame slider, from the beamformer of Angle of Arrival -> **Range-Azimuth**: its antenna configuration (Array), **Mode**, **Method** and **Doppler Comp**, over the Analysis bar's **Angle Lim**. (It used to be an FFT across the receivers as if they stood in a line; on the IWR6843AOP, whose receivers form a 2 x 2 square, that put a target at 0 deg at -90 deg.)

**Use:** press **Run Range-Angle**, then the frame slider.

**Controls:**
- **Angles** (default 181) - the number of angle points across **Angle Lim** (181 over +-90 deg: 1 deg apart).
- **Range Start** / **End** - the range bins shown (0 and 0 = all).
- **Clutter Removal** - static suppression with the Analysis bar's **Clutter** method.
- **Remove DC** (default on) - removes the DC of every chirp before the range FFT.
- **Show Peaks** - marks the strongest points (acts at once).

### Imaging

**Shows:** 4D radar imaging ("radar vision"): every frame a point cloud - where every reflection is in three dimensions (x across, y away, z up) and how fast it moves towards or away from the radar - the objects it forms, followed through the recording, and images of the scene. Five pages:
- **3D** - the point cloud in space, the radar at the origin (black triangle), the floor when **Radar height [m]** is set; every object a box with its number and velocity, its track of the last 3 s on the floor. Drag to turn it.
- **Bird's-eye** - the room from above: the image of the frame (range x azimuth, grey: darker is stronger) under its points, range rings every metre and the edges of the angle grid; objects as circles with their number, velocity, height and points, their tracks of the last 3 s.
- **Radar vision** - what the radar sees, azimuth x elevation (white where it sees most, from the ranges of the frame's points), the points on it coloured by their distance: a depth image. Objects are labelled with their number and distance.
- **Range-Doppler** - the frame's range-Doppler map (all channels, the still scene taken out) with its points: red circles moving points (a velocity unfolded beyond one TX slot's limit lies beyond the map's own axis, on a grey band), cyan diamonds micro-motion, orange squares the still scene, grey crosses echoes.
- **Tracks** - every track of the recording from above (dot: where it began), the objects of the frame circled; below, the points and objects of every frame with the frame line.

All pages follow the frame slider and **Play All**.

**Use:**
1. Decode a recording. Angles need an array: a MIMO recording (3 TX of the IWR6843AOP, the cascade with its calibration) gives azimuth and elevation, a single TX of the IWR6843AOP four channels in a 2 x 2 square (coarse angles, still enough to place a person).
2. Set **Radar height [m]** (and **Tilt down [°]** if the radar points down) for heights above the floor, choose what to **Show:** and press **Run Imaging**. The summary line: points and objects a frame, the fastest movers and how far the velocity can be unfolded.
3. Move the frame slider or press **Play All**; change **Colour by:**, **Trail [frames]** and **Min SNR [dB]** without running again.

**Show:**
- *People (moving + micro-motion)* (default) - what moves within a frame (Doppler) and what moves from frame to frame (a breathing chest, someone shifting on a chair): walking and seated people, not the room.
- *Moving (Doppler in a frame)* - only what moves within a frame. Needs 8 chirps of a TX a frame or more; a recording for vital signs (4-5 chirps a frame) gives none - the summary says so.
- *Micro-motion (breathing)* - only what moves from frame to frame, over 1 s of frames around the frame: a seated person is no Doppler peak in a frame of a few chirps and is lost among the walls of the still scene (on seated recordings 2 % of the still points lay at the person), but their breathing stands out.
- *Still scene (room)* - the mean of the chirps of a frame: walls, furniture, the radar's surroundings, up to 4 points a range bin.
- *Everything* - all three; objects are made of the moving and micro-motion points only.

**How it works:** the range FFT of every chirp (the Range FFT size of the Analysis bar), the chirps of every TX slot as slow time; the still scene (their mean) apart from what moves; a Doppler FFT. *Moving:* the range-Doppler cells standing **Threshold [dB]** above their surroundings (the 25th percentile of the cells around them - a room's reflectors close together raised each other's average until none stood out), less the window sidelobes of stronger cells, less the small echo the removal of the still scene leaves of every mover beside zero velocity. Every cell's snapshot is corrected for the time between the TX slots; a velocity beyond one slot's limit is unfolded where the array explains the echo clearly better with it (two directions: the sharpest single beam picked a wrong velocity for two people at one range and speed) and never beyond 10 m/s. *Micro-motion:* the change of the still scene over the frames around the frame, against the thermal noise and 6 dB more than the threshold (everything in a room moves a little); its direction is the main direction of that change. *Still:* range bins above the thermal noise. Angles: Bartlett over azimuth and elevation with the antenna configuration's element positions and the channel calibration as in [Angle of Arrival](#angle-of-arrival) (a chirp sent on several TX is the sum of their responses); up to 2 sources a cell - a second one only where it explains half of what the first leaves (a sidelobe, or the steering model's error near the radar, explains little); near echoes with the exact paths TX -> point -> RX, not plane waves (the cascade's far field begins at 2.7 m); sources of one cell refined against each other (RELAX) and between grid points on a quadratic surface (the cascade's elevation, one row 0.8 wavelengths up at one side, is a diagonal ridge in azimuth and elevation). A source is kept in the cell where its own echo peaks in range and Doppler - a weaker person beside a stronger one is no peak of the summed power, but is one in their own direction - and its range and velocity are refined there. Points beyond the angle grid are not reported. Objects: DBSCAN over the points of the last 3 frames (a person gives the IWR6843AOP's 12 channels two or three points a frame), height counting 0.3 and velocity 0.15 m per m/s; a Kalman tracker follows them (a track shows after 3 frames, waits 1.5 s for its object, 6 s when it stood still).

**Echoes:** a wall or the floor reflects a person, and the reflection moves with them. A micro-motion point whose motion over 10 s correlates with that of a stronger point at least 0.4 m nearer (|r| >= 0.7) is marked as its echo; so is a track whose velocity swings in step with a nearer, stronger track's, and a track made mostly of echo points (all of it: point by point a fifth to a third of a reflection's points were missed). Echoes are hidden unless **Echoes** is ticked (grey). On a cascade recording of two people seated at 1.1 and 2 m this left the two of them and about one reflection a frame.

**People and the rest:** in the People, Moving and Micro-motion modes the people are a layer of their own; the rest of the points - noise, a reflection or an object seen for a moment, the room's reflections of a person - is a second layer, hidden unless **Other points** is ticked (grey). A track is a person when it is followed for 3 s or more, is no echo, and is no reflection of a person - 6 dB or more weaker, behind them (0.4 m further, within 20 deg of their direction) most of their common time; a second track on someone (an arm, a body split into two clusters) becomes part of theirs, so a person is one object with one number. With angles, a person who stays (slower than 0.3 m/s over a second) must also lie at someone [Multi Vitals](#multi-vitals) finds (within 0.8 m, half of the time it finds anyone there): its maps over half a second with Capon's beams reject the room's reflections far better than single frames, so Run Imaging runs its detection too (about 50 s for a minute of the cascade). A frame where such a track sits on a reflection more than 0.8 m from anyone Multi Vitals finds shows no person there - unless it is within 3 dB of the strongest person there, and was so most of the time (someone standing where Multi Vitals lost them: on the take of one person standing on 8 floor marks it found them at some marks only, and without this rule half of the frames showed nobody). Walkers Multi Vitals does not follow on the cascade, and among a walker's reflections the tracker swaps them: a walking track is a person in the frames where it is within 3 dB of the strongest person there (the reflections were 6 dB or more weaker than the strongest in 98 % of the frames). On the cascade take of someone walking around a 1.7 x 2.2 m rectangle the frames had 9 objects; with the layer one person (median), a second in 10 % of the frames. Standing and walking between marks is the hardest: a second person in 18 % (8 marks) and 35 % (4 marks) of the frames, where Multi Vitals took a reflection for the person. The people's points are those of their objects and every point within 0.6 m of them - also those taken for an echo of someone else (a second person 2 m from the cascade breathed enough like the first that half of their points were). On the cascade recording of one person 2 m away a second person showed in 20 % of the frames and 65 % of the points lay at the person; with the layer 1 % and 94 %; of two people seated at 1.1 and 2 m both showed (98 and 82 % of the seconds) and a third in 3 % of the frames; on seated IWR6843AOP recordings 88-100 % of the layer's points lay at the person, and someone swinging on a chair was one person instead of two or more in 80 % of the frames (without Multi Vitals' confirmation: a second person in 21, 64 and 80 % of the frames of these three).

**Controls:**
- **Run Imaging** - runs the analysis on the decoded segment.
- **Show:** - see above.
- **Threshold [dB]** (13) - how far a cell must stand above its surroundings (micro-motion: 6 dB more, against the thermal noise).
- **Range to [m]** - the farthest range (empty: all).
- **Radar height [m]** (0) and **Tilt down [°]** (0) - z is height above the floor with the radar's height set; the tilt turns the points back to level.
- **Colour by:** *Velocity* (blue towards the radar, red away), *Height*, *SNR*, *Range*, *Object* (its track's colour) or *Kind* (red moving, blue micro-motion, grey still, light grey echo).
- **Trail [frames]** - the points of this many frames before stay, fading.
- **Min SNR [dB]** - shows only points this strong (no new run).
- **Objects** - boxes, circles and tracks.
- **Image under the points** - the images of Bird's-eye and Radar vision.
- **Echoes** - draw the points and objects marked as echoes too.
- **Other points (noise, short-lived objects)** - draw the layer under the people too: its points grey and small, its objects and tracks grey (Tracks page: thin grey lines).

**Accuracy:** on synthetic scenes (every chirp and every element's path exact): reflectors all over the field of view within 0.4 cm and 0.05 deg in azimuth and elevation; movers at 5.5 and 6.5 m/s with one slot's limit at 4.15 m/s unfolded to 0.03 m/s (without unfolding half of them are missing); two at one range and speed 30 deg apart two points, 0.2 deg each; the cascade 0.04 deg in azimuth and 0.16 deg in elevation; three seated people breathing found in every frame within 0.5 cm, a wall's echo of one marked as such; two walkers crossing keep their tracks. On six IWR6843AOP recordings of a person seated at a measured place (1 TX, 100 fps), the person's object lay 0.08-0.20 m from it (median 0.13 m).

**Export:** **Current frame (PNG)** and **All frames (Video)** - Bird's-eye; **Data (CSV)** - every point of every frame: time, frame, range, velocity, azimuth, elevation, x, y, z, power, SNR, coherence (how much of the cell's echo its sources explain), kind (0 moving, 1 micro-motion, 2 still, 3 echo) and track; **Shown page, all frames (Video)** and **Shown page (PNG)**; **Point cloud of the frame (PLY)** and **Point cloud of every frame (PLY)** for CloudCompare or MeshLab; **Objects and tracks (CSV)** - every object of every frame.

**Wavefront check (cascade):** the strongest echoes of an array of more than 16 channels must fit its model - most of a person's motion from one direction. With the cascade's TX in the user guide's order (wrong, see [Calibration of the cascade](#calibration-of-the-cascade)) every six-TX take fitted 0.30-0.49 and one person came out at five angles; with the measured order people fit 0.65-0.91 (someone seated 0.9 m away, wider than the beam there, 0.46). Below 0.4 the summary says so: the angles and the number of people are unreliable - check the antenna configuration and the calibration, and that nothing close to the antennas (a surface, an object, a cover) distorts the wavefront. Multi Vitals checks the same.

**Limits:** a room reflects walking people too: on a recording of someone stepping in place 1.7 m away the strongest point of every frame was them, but their reflections up to 4.5 m stood 30 dB above the noise and made several objects; for walking people these are not yet recognised as echoes. The angles of a single TX of the IWR6843AOP (4 channels) are coarse; the cascade's elevation comes from one row 0.8 wavelengths above the others and is coarse too, and its angles need a calibration ([Calibration of the cascade](#calibration-of-the-cascade); without one: range and velocity only, on the axis). Two people closer together than the array's beam (about 30 deg for 12 channels) at one range and speed may be one object. A frame takes 5-120 ms on the IWR6843AOP and 15-80 ms on the cascade (*Everything*): a minute of 20 fps in half a minute to two.

### Micro-Doppler

**Shows:** the micro-Doppler spectrogram of the person - radial velocity (vertical, positive = away from the radar) against time in seconds - with the **torso** line (white, dashed), the upper and lower **envelope** (the fastest velocity still on the body, pink) and every detected **swing** of a leg or an arm (yellow triangle at its fastest point: up = away, down = towards, labelled with its speed and, when known, *leg* or *arm*). A vertical line marks the current frame. The label at the end of the controls reads the maximum speeds, the number of swings and `v_max`; the full list of swings goes to the [Log](#5-log-tab).

**Use:** press **Run Micro-Doppler**. The defaults follow the person automatically; set **Range bins from** / **to** only to keep other moving things out.

**Controls:**
- **Range bins from** / **to** (defaults 5 and 100) - the range bins searched for the person (**Gate: Auto**) or taken as they are (**Gate: Fixed**). Read the bin numbers in the Profile or Range-Doppler view.
- **Gate** - **Auto** (default) follows the moving person frame by frame: the bin with the most moving energy, **Width [m]** (default 1.6) around it. **Fixed** uses all bins from / to.
- **Window [ms]** (default 20) - the length of each spectrum. Shorter shows faster changes, longer resolves velocity more finely (resolution = wavelength / (2 x window)). Frames back to back: windows slide across frames every quarter window. Frames with gaps and too few chirps: one spectrum per frame (a note in the Log says so).
- **Threshold [dB]** (default 8) - how far above the noise floor a Doppler bin counts as part of the body. Raise it for a single receiver (**Antenna: RX0**...) or a noisy recording.
- **Min swing [m/s]** (default 1.5) - how far an envelope has to rise above its usual distance from the torso to count as a swing. A still body has a Doppler width of its own (breathing, sway), so the rise is measured from that, not from the torso line itself.
- **Radar height [m]** (default empty) - the height of the radar above the floor. With it, a recording from an array with vertical extent (IWR6843AOP in TDM-MIMO) tells a *leg* (fastest point below 0.6 m) from an *arm*; without it the height is relative to the radar and not classified.
- **Envelopes** (default on) - draws the torso line, the envelopes and the swings over the spectrogram (acts at once).
- The **Antenna**, **Clutter** and **Range FFT** settings of the Analysis bar apply. **mean** takes the static part out of every window. With frames back to back **mti2**, **mti3**, **ema** and **svd** run along the whole slow time (frame by frame they would leave a step at every frame boundary); with gaps between frames they work frame by frame as in Range-Doppler.

**How the speed is read:** the body is the connected band of bins above the threshold around the torso; its edge is where the outer part of the band falls 3 dB below its own level. On synthetic walkers this read the fastest point of a limb 0.03-0.15 m/s too fast with a 20 ms window; the edge at the noise threshold would be 0.3-0.45 m/s too fast.

**Faster than v_max:** the band is followed around the velocity axis, so a foot at 4 m/s in front of a torso at 1 m/s is measured correctly even with `v_max` = 2.94 m/s; the spectrum is then drawn repeated above or below itself, and the swing is marked *unfolded* in the Log. When the body spans more than twice `v_max` this is impossible - such moments are counted as *saturated* and swings next to them say so.

**Limits:** the radar measures only the radial component, v cos(angle): a limb moving across the line of sight looks slower than it is. For the true speed place the radar in the direction of the motion, or compare with motion capture (`python -m tools.limb_speed ... --reference`).

**Export:** PNG; **Data (CSV)** - the spectrogram in dB, one row per time column, first column the time; video; **Swings (CSV)** - every swing with its start, peak and end time, duration, direction, speed (absolute and relative to the torso), range, elevation, height and limb; **Envelopes (CSV)** - time, torso line, upper and lower envelope and the saturated flag per spectrum.

### 1D Dopp

**Shows:** for one range bin, the range profile, the Doppler spectrum of the current frame, and how the energy of that range over time splits into *Noise/DC*, *Breathing*, *HR* and *Movement*, with the breathing and heart-rate peaks.

**Use:** press **Run 1D View** (it computes the Range-Doppler map first if needed), choose the bin with the **Range bin** slider. While the slider is at the antenna coupling (nearer than 0.3 m, bin 0 after opening a recording), Run starts at the range that moves most; the Log says which.

**Controls:**
- **Source** - **Range Bin**, **Neighborhood ±N** (the bins within **±N** of it, default 6) or **Whole Range**.
- **Current Frame Only** - uses only the last 5 seconds up to the current frame instead of the whole recording.
- The bands are the shared vital-sign bands (see [Vitals](#vitals)).

**Export:** PNG; CSV of the current frame's Doppler spectrum at the bin; video.

### CFAR

**Shows:** targets detected in the Range-Doppler map, marked with red crosses. The list of targets (range, velocity, strength) is written to the Log.

**Use:** run the Range-Doppler view first, then press **Run CFAR Detection**. The targets of the frame on the slider are listed in the Log; moving the slider (or playing, or exporting a video) detects every frame shown with the settings of that Run. Change a setting and press Run again to apply it.

**Controls** (the **CFAR Parameters** row):
- **Guard Cells** (default 2) - cells around the tested one left out of the noise estimate, so a target's own energy does not raise its threshold.
- **Training Cells** (default 8) - cells used to estimate the noise. Along range both counts are resolution cells: with a zero-padded range FFT (more points than samples per chirp, e.g. 80 samples on 256 points) each cell is that many FFT bins (3.2 here; the Log says how many). With a few velocity bins (fewer than the window, e.g. the 4 of the vital-signs settings) the test runs along range only, which the Log notes too.
- **Threshold Factor** (default 3.0) - how many times above the noise a cell must be to count. Higher = fewer false detections.
- **Method** - **average** (cell-averaging CFAR) or **ordered** (order-statistic CFAR, robust next to other targets).
- **Unfold velocity (TDM)** with **max |v| [m/s]** - for TDM-MIMO recordings: finds velocities beyond the maximum unambiguous velocity from the phase steps between the transmitters. A fold is accepted only when the decision is clear (3 dB margin) and the speed is below **max |v|** (empty = no limit); the Log says for every target whether it was unfolded or why not. Trust an unfolded velocity only when it stays the same over several frames.

**Export:** PNG; CSV of the detection mask of the current frame; video.

### Vitals

**Shows:** breathing and heart rate of a person, with all intermediate signals: the range profile with the chosen range, the chest displacement, breathing and heart waveforms with their spectra, and both rates over time (dots: windows whose peak passed **Min peak/avg**, rings: the others, a dashed line: the mean of the dots). The heart spectrum is the harmonic sum of the last window the heart is followed in (as in [Moving person (phase 12)](#moving-person-phase-12)), its dashed line the rate the track reached - not always the tallest peak of that one window, which can be a breathing harmonic or noise. Stretches recorded over lost packets are shaded red and drawn dashed; they are marked only, the rates are computed the same way.

**Use:**
1. Press **Run Vital Signs Analysis** once with the defaults.
2. Look at the range profile at the top: the red marker must sit on the person. **Auto Bin** (on by default) takes the range that breathes most above its noise; if it still misses - move the **Range Bin** slider onto the person (the label shows the bin and its distance; this switches Auto Bin off) or set a **Segment** without the movement, and press Run again.
3. Read the rates in the two statistics lines of the first control row (**RR:** breathing, **HR:** heart): the latest value and the *mean*, *min* and *max* over the windows that passed the **Min peak/avg** threshold.

**Controls - first row:**
- **BR Win [s]** (default 30), **HR Win [s]** (default 30), **Step [s]** (default 1) - the sliding windows the rates are estimated in. Longer windows are steadier but react slower; the radar's heart line needs about 30 s (on six seated recordings against a somnograph pulse 15 s windows were 6.5 bpm off on average, 30 s windows 3.5).
- **Bands** - the breathing and heart-rate search bands, shared with the HR and AI tabs and with the Movement and 1D Dopp views. Presets: **Adult** (default: breathing 5-40 /min, heart 50-150 bpm), **Slow heart (athlete, sleep)** (heart 40-120 bpm - more risk of picking a breathing harmonic), **Children, exercise** (breathing 10-60 /min, heart 60-200 bpm). Or type your own limits and press Enter; an invalid pair (low >= high) is not applied until corrected.
- **RX** with **Auto** (default on) - which receiver. With Auto every receiver counts: the breathing rate comes from their spectra summed, the heart rate from the mean displacement of every receiver in the chosen bin and in its neighbours either side - the chest spans more than one range bin, and a neighbour counts where its signal stands above its noise; the waveform shown is the receiver with the strongest fluctuation. A receiver chosen by hand is used alone.

**Controls - second row:**
- **Range Bin** slider - the range of the person (disabled until the first Run).
- **ROI half width** (default 1) - how many neighbouring bins are considered.
- **ROI Mode** - **Single bin** (the heart from that bin alone), **Best bin in ROI** (default: of the bin and its neighbours, the one that breathes most above its noise; the heart from it and its neighbours) or **Weighted ROI average**. Changing it runs the analysis at once.

**Controls - third row:**
- **Auto Bin** (default on, and on again for every newly opened recording) - chooses the range whose signal holds the most breathing-band power above its own noise, all receivers together - as [Fastcheck](#4-fastcheck-tab) finds the person; never nearer than 0.3 m (the antenna coupling). When no range breathes clearly, the range where breathing-band energy persists (its median over 10 s windows). Moving the slider switches it off.
- **Clutter Removal** - subtracts the time average of each range bin.
- **Kalman Filter**, **Wiener Denoising** - optional smoothing of the displacement.
- **Min peak/avg** (default 6.0) - how many times the spectral peak must exceed the band average for a window to count. A clean signal gives about 12, pure noise about 4. The heart windows are judged on the same scale (the tracker's own bar of 3 is 6 here).

**Controls - fourth row:**
- **IQ DC Correction** - **none**, **mean** (default) or **moving_average**.
- **Phase Extraction** - **unwrap** (default) or **dacm**. Both run the analysis at once.
- **X Zoom Mode**, **Reset X Zoom**, **Reset View** and the plot toolbar - see [Zooming and panning](#zooming-and-panning).

**Export:** PNG of the whole figure; **Data (CSV)**: per frame time, displacement, breathing and heart signals; **Rates over time (CSV)**: breathing and heart rate of every window with its confidence on the **Min peak/avg** scale (for breathing the peak-to-average ratio of the spectrum) and whether it counted.

> **Tip:** if the heart rate jumps around, first check the range bin, then lengthen **HR Win**, then check that the heart band fits the person. `python -m tools.vital_range_scan` finds the range of a breathing person automatically.

### Multi Vitals

**Shows:** everyone in the room, followed through the recording, with the breathing and heart rate of each of them. Two pages:
- **Map** - the room from above, the radar at the bottom (black triangle), range rings every metre and the edges of the angle grid. Every person is a dot with its number, the rates of the window around the frame on the frame slider (*BR 12 · HR 72*; *BR -* where that window has no clear rhythm, *moving* while the person walks) and the last 5 s of their path. A paler dot: that block's position is predicted, not detected. Underneath, with **Detection map** on, the map the people were found in, for that moment. The dots move with the frame slider and with **Play All**.
- **Rates** - breathing (rings: windows without a clear rhythm), heart rate (where it counts) and chest displacement of every person over time, with the frame line.

**Use:**
1. Decode a recording; for angles it needs more than one azimuth position - a MIMO recording is best, a single TX of the IWR6843AOP gives two (enough to place people, too few to separate two at one range).
2. Press **Run Multi Vitals** with the defaults. The summary line lists everyone: *#1 at 1.41 m, -34 deg: breathing 12 /min, heart 62 bpm*.
3. Move the frame slider or press **Play All** with the **Map** page open; read the rates over time on **Rates**.

**How it works:** the chirps of every TX slot are averaged per frame into the virtual array, with the element phase correction and the channel calibration of [Angle of Arrival](#angle-of-arrival) (the **Antenna** selection of the Analysis bar applies); near the cascade - whose far field begins at 2.7 m - the steering of every range bin follows the exact paths TX -> person -> RX, not plane waves (with plane waves someone seated 1.1 m in front showed at four angles 35 deg apart and smeared a second person at 2 m away). A chirp that transmits on several TX at once (mask 5: TX1 and TX3) counts as one element whose response is the sum of theirs - the AoA views drop it. Every 0.5 s a map of what moves in range and azimuth is made and its peaks become detections - those standing 12 dB above the motion floor of their range, 6 dB above the median of their range over all angles (Capon spreads a walker's motion evenly around their range ring 12 dB below them, and every ripple of that ring was a peak: four "people" a block around the walker on the cascade, the dots nowhere near the map's spot) and within 25 dB of the strongest motion of the block in power (compared by their standing above their own floor, a second person 2 m away behind the reflections of someone 1.1 m in front was 37 dB down and lost; in power they were 12 dB apart); a peak no more than 6 dB above what the array's sidelobes leak there from a stronger person at the same range is dropped, and so is a peak 70 deg or further out (the antennas hardly see there: a still spot at -74 deg was the strongest peak of a walk's map half the time). A tracker (Kalman filter, Hungarian assignment) follows them: a track counts after 3 detections in its first 5 blocks, ends after 2 s without one, shorter than 3 s or detected in less than 60 % of its blocks it is not reported, nor is a target without vital signs followed for less than 8 s unless it walks - gets 1.5 m from where it began along a path at most 1.5 times as long (on the 36 single-person seated recordings 43 of the 69 extra "people" were such short targets: 3-7 s, 2-4 m behind the person, a reflection, a hand, a moment of noise jumping about; the summary counts them). A track that jumps more than 0.45 m and 2.5 times its recent steps is cut there (a weak reflection behind a seated person once jumped onto them and took over their detections); a track that starts within 1.5 s of another's end where that person could have got at their recent speed + 0.5 m/s continues it, and so does one that starts at the same spot (closer than **Min separation**) within 20 s - a still person lost for a moment keeps their number; a track that comes near someone followed from before takes over from them when that person's track ends only if it began after them (else a reflection's track took over the whole past of a person standing on four floor marks for two minutes). Two tracks that follow the same person at once are merged, and so are two that stay at the same spot at different times: one place is one person - someone who leaves and sits down again in the same place keeps their number (and so would someone else taking that chair). Vital signs are measured only where a person stays (slower than 0.2 m/s for 8 s or more): the range bins within 0.15 m of them, seen through three beams across them, go through the breathing and heart chain of the [Fastcheck tab](#4-fastcheck-tab) - the bin that breathes most, the breathing rate and the breaths counted in 30 s windows (a third of a shorter stay), the heart followed through the windows. Walking moves the chest by centimetres a step: no rates there.

**Accuracy:** on 36 recordings of one seated person with a somnograph (pulse from its PPG, breathing from three PPG surrogates that agree), the person's windows against the reference over the same span: heart rate within 5 bpm in 73 % of 749 windows (MAE 8.1 bpm), breathing within 2 /min in 85 % (MAE 1.2 /min) with the defaults; *Bartlett* 56 % of 611 windows and 78 % - with a single TX of the IWR6843AOP its beams are too wide to keep reflections out of the person's cell. The [Fastcheck tab](#4-fastcheck-tab) on the same recordings: 56 % and 81 %. The heart misses are mostly the same as Fastcheck's: windows read at 50-55 bpm where the pulse was 80-85. Synthetic rooms (three seated people; two at one distance 30 deg apart, one 12 dB weaker; two crossing at 1-1.4 m/s): everyone found, rates within 1 /min and 3 bpm.

**Echoes:** a wall or the floor behind a person reflects them, and the reflection breathes exactly like them. A target whose breathing correlates with a stronger one's (|r| >= 0.8; on the seated recordings the reflections correlated 0.85-0.99, unrelated targets -0.2 to 0.05) is marked as its echo and hidden; **Show echoes** draws them (grey, hollow). Between 0.5 and 0.8 the summary says *possibly an echo*. A target 10 dB or more weaker than a person and behind them - at least 0.4 m further, within 20 deg of their direction - for half of the time both are followed is their echo too, whatever its breathing: of the 26 extra "people" still left on the seated recordings 18 lay so (14-36 dB below the person, their breathing correlating with the person's anywhere from 0.04 to 0.68); 10 remained, in 9 of the 36 recordings, the person's heart and breathing rates unchanged (74 % of the heart windows within 5 bpm, 84 % of the breathing ones within 2 /min). Someone sitting right behind another person, 10 dB weaker, would be taken for their echo too.

**Controls:**
- **Run Multi Vitals** - runs the analysis on the decoded segment.
- **Find people by:**
  - *Motion (tracking)* (default) - every 0.5 s: the chirps of a frame against their mean (a walking person) and the frames of those 0.5 s against theirs (a breathing chest, a fidget); a peak of the frames over 1.5 s is added where nobody was found within 1 m (a seated person in the still second at the end of a breath - over 1.5 s a walker is smeared along their path). Follows people who walk in, sit down and leave, and two who cross each other's path.
  - *Breathing (still people)* - the motion of the frames over 10 s: a person sitting or lying still for minutes stays found, walls and furniture never are; a target whose cell does not breathe is dropped. Slower to react to someone walking in.
  - *Range only* - no angles: people at different distances, each confirmed by its breathing. For an array whose angles cannot be trusted - the cascade without a calibration (chosen automatically then) - or that has none. Two people at one distance are one.
- **Beam:** *Capon* (default) - unit gain towards the person while nulling what else moves: the maps find a person 12 dB weaker 30 deg beside another at the same distance (Bartlett's map has one peak there), the heart is measured through Capon beams and the breathing through Bartlett beams - unless someone else leaks into the person's cell at a tenth of their power or more, then Capon gives both (Capon beams distort a strong breathing signal slightly off their direction). *Bartlett* - the steering vector alone, and the array pattern to tell a sidelobe from a person.
- **Max people** - at most this many detections every 0.5 s (6).
- **Range to [m]** - the farthest range searched (6 m).
- **Min separation [m]** - two peaks closer than this are one person (0.5 m).
- **Show echoes** - draw the targets marked as echoes too.
- **Detection map** - the map underneath the dots on **Map**.
- The breathing and heart **Bands** of [Vitals](#vitals) apply.

**Angles:** positive and negative angles are those of the [Angle of Arrival](#angle-of-arrival) views (on both the IWR6843AOP and the cascade: someone on the radar's left at a positive angle). With the cascade the summary warns when the strongest motion fits the array poorly (see the wavefront check in [Imaging](#imaging)); the maps are then Bartlett's even with **Beam: Capon** (Capon nulls an echo that does not fit its steering vector in its own direction too - one person in front lit their whole range ring evenly, Bartlett kept the spot where they sat), and a peak 6 dB or more weaker than a stronger one within 0.25 m in range is taken for its sidelobe (such echoes leak to -5 dB, the array's model to -14 dB; one person 2 m away left them 12 cm behind at +-16 to 61 deg, and the tracker jumped onto them). The map under the dots shows the 15 dB below the strongest motion of the moment. On the IWR6843AOP recordings a person sitting 1 m in front of the radar and 1 m to its right came out at 1.37 m and -44 deg (true 1.41 m, 45 deg), at 2 m and 1 m to the right at -24 deg (26.6 deg).

**Export:** **Current frame (PNG)** - the map; **All frames (Video)** - the map through the whole recording, the dots moving; **Data (CSV)** - every person in every 0.5 s block: number, time, x, y, range, angle, detected, moving, the breathing and heart rate shown and which target it is an echo of; **Rates over time (CSV)** - every window of every person; **Rates figure (PNG)**.

**Limits:** a single TX of the IWR6843AOP has two azimuth positions half a wavelength apart: people are placed, but two at one distance closer than about 60 deg are one. People beyond the angle grid (+-75 deg) are not reported - the edge of the grid cannot place them. Walking people with a small array break into several tracks and leave reflections of themselves. The cascade measures angles with a calibration ([Calibration of the cascade](#calibration-of-the-cascade)); its takes bring it along.

### HR

A laboratory for heart-rate estimation on one range bin: several phase-extraction methods, receiver fusions and estimators side by side, and the heart followed through the recording as in [Vitals](#vitals).

**Use:** press **Run HR Detection**. **Bin** 0, the default, is the antenna coupling and never a heart: the tab then takes the range that breathes most (as **Auto Bin** in [Vitals](#vitals) does), sets **Bin** to it and says so in the Log. For another range enter its bin and press Run again.

**Controls:**
- **Range Bin & ROI:** **Bin** (bins nearer than 0.3 m are replaced by the range that breathes most), **ROI ±** (neighbouring bins, default 1), **ROI Mode** (**average displacement** - the default: each receiver's displacement averaged over the bin and its neighbours, a neighbour where its signal stands above its noise - **center only**, **average complex**, **best bin**, **weighted bins**).
- **Source & Fusion:** **RX** (**RX1**-**RX4** - numbered from 1 in this tab - **Fused** (default), **PCA**, **ICA**), **Fusion** (**best RX**, **average** (default: the receivers' mean displacement), **weighted**, **PCA**, **ICA** - used when RX is **Fused**), **Phase** (**unwrap**, **dacm**, **circlefit**).
- **Analysis Parameters:** **Win [s]** (default 30) and **Overlap %** (default 80) of the heart track - a 30 s window every 6 s.
- **Search bands (shared by all vital tabs)** - the same bands as in [Vitals](#vitals).
- **Filters:** **Set respiration branch**, **Set cardiac hr branch**, **Set cardiac waveform branch** open the filter parameters (type, band, order, zero-phase or causal). The respiration and heart branches *are* the shared search bands: changing them there changes the bands of all vital tabs.
- **Clutter Removal** (default on) - subtracts the time average of every range bin, which puts the chest's circle in the IQ plane around the origin: with the static reflection left in, the phase is a short arc around it and the heart all but vanishes (19.5 instead of 0.6 bpm off the pulse on one recording). **Notch Resp Harmonics** - notches the 2nd and 3rd harmonic of the breathing rate inside the heart band.
- **Run Synthetic Test** - a generated chest at 1.6 m breathing 15 /min, each receiver with its own heart rate (RX1 60, RX2 80, RX3 100 bpm, RX4 none): the **RX Comparison** table must show them.

**Results:** the line **HR Result** under the plots is the heart track - the heart followed through the windows as in [Vitals](#vitals) (harmonic sum, Viterbi path, the breathing harmonics notched out) - as the median of its windows whose peak passes 6 on the **Min peak/avg** scale. The FFT peak of the whole signal stands beside it; it can be a breathing harmonic or noise (58 bpm on one recording where the pulse went 78). Sub-tabs **Main Plots** (displacement, breathing and heart branches, the heart track over time with its median - green points count, red ones do not - and the spectrum with the FFT peak in red and the heart track in green), **RX Comparison** (a table and plots per receiver, from the FFT peak of the whole signal), **Method Comparison** (estimators side by side; the row **track** is the HR Result, with the median peak-to-average of its windows), **Diagnostic Map** (quality and heart rate for every receiver over 35 bins around the selected one).

**Export:** **Export CSV** (one row per receiver and method with the settings used, and the row **track**) and **Export JSON** (all results, the heart track window by window, settings and filters).

**How good it is:** on six seated recordings with a somnograph (`testovani/vitals`) the heart track with the defaults was 2.2 bpm off the pulse oximeter's beats, 84 % of the windows within 5 bpm - as the Vitals view (2.2 bpm, 85 %). With the former defaults (bin 0, RX1 alone, the static reflection left in, 15 s windows, the largest peak of each window) it was 11.1 bpm off.

#### Moving person (phase 12)

The heart rate of a person who may be moving - swaying, fidgeting, walking in place - window by window, with the part of the recording in which that is not possible said so. It works on the raw data by itself: the **Bin** above does not apply. Press **Run Moving HR** in the box **Moving person (phase 12)** at the end of the settings column; the results go to the page **Moving Person** among the plots and the summary to the box and the Log.

What it does, in order:
1. **Follows the trunk in range** - the range with the most moving energy, as a smooth path: it moves only when the energy elsewhere stays higher for long enough, so it does not jump between the chest and another reflection of the body. When the person sits still - the path within one bin of its median for 90 % of the time - it is moved onto the bin that breathes most among the median and two bins either side (as **Auto Bin** in [Vitals](#vitals) finds it): the strongest moving reflection of a seated person was a bin in front of the chest, 21 instead of 1 bpm off the pulse on one recording.
2. **Takes the chest signal** from the three range bins around the trunk and all receivers combined into a beam on the trunk. The phase of every bin is referred to the middle of the chirp, so a trunk moving into the next bin keeps its phase. At rest the heart is followed in another signal: every receiver of the three bins on its own, their displacements averaged, as in [Vitals](#vitals) - the beam is what a moving trunk needs, but at rest it was 2.2 instead of 1.3 bpm off the oximeter's beats on five seated recordings.
3. **Follows the phase of every chirp** with a Kalman filter (phase, rate, acceleration). A walker's chest turns the phase by tens of radians from frame to frame; the chirps inside a frame and the prediction across the gap to the next frame follow it. After a dropout the filter starts afresh, and the offset across the gap is taken out by fitting a smooth curve on both sides.
4. **Sorts every window** into rest (trunk slower than 0.02 m/s RMS), light motion (up to 0.3 m/s) and strong motion (faster, or too many lost frames).
5. **Finds the steps** while moving: the bounce of the trunk at the step frequency; it and its harmonics are notched out of the heart band, as are the 2nd-4th harmonics of breathing out of the displacement (the envelope has none; notched there too, a heart sitting on one of them was lost and followed at twice its rate).
6. **Looks for the heart** in the heart band of the chest displacement and in the envelope of its 5-25 Hz content (the heart's mechanical events are short; breathing and slow body motion carry no energy there), sums the spectrum at f, 2f and 3f, and follows the heart rate through the windows (Viterbi) with a limited change per second.

**Settings:**
- **Method** - **auto** (default: the displacement and the envelope at rest, the envelope alone in motion - the bounce of the steps sits some 50 dB above the heart in the displacement), **envelope**, **displacement** or **both**.
- **Window [s]** (default 15) - the analysis window; a new window every second. While the person stays still for 30 s, the heart's window widens to those 30 s about the same centre (inside the recording at its ends): a seated person's heart line needs that long; in motion it stays as set.
- **Max change [bpm/s]** (default 3) - how fast the heart rate may change from one window to the next.
- **Trunk from [m]** (default 0.3) and **Trunk to [m]** (empty = to the end) - where the trunk is looked for; narrow it when someone else moves in the room.
- **Estimate in strong motion** - give an estimate while walking too. Off by default: it is untested on real recordings (see below).
- **Export Moving HR (CSV)** - one row per window: time, motion class (0 rest, 1 light, 2 strong), trunk speed, breathing and step rate, heart rate, the largest peak of the window alone, confidence and whether the estimate counts.

**The page Moving Person:** the trunk range (windows with light motion shaded yellow, strong red), the chest displacement, the heart band of every window with the tracked heart rate (white: thick where it counts, dotted where not) and the step frequency (orange), and the trunk speed with the two class limits.

**How good it is:**
- On six seated recordings with a somnograph pulse (`testovani/vitals`): MAE 3.5 bpm over the valid windows, which are all of them - 0.9-2.5 bpm on five of them; the sixth (13.8 bpm) holds a strong 65.6 bpm line of something else in the room, which the heart track follows. Before October 2026 - without moving a still trunk onto the bin that breathes and widening the heart's window at rest - it was 9.0 bpm with 69 % valid. The [Vitals](#vitals) view, which since October 2026 follows the heart the same way in 30 s windows on the range Fastcheck would choose, is 2.9 bpm off the pulse on these recordings over all windows (2.2 off the PPG beats, 0.5-1.5 on five recordings), 99 % of them valid (before: 8.4 bpm, 46 % valid).
- On synthetic recordings of known heart rate (ten people each): about 1 bpm at rest and while swaying (worst 1.6). While walking in place or walking to and fro, the median was 3.5 bpm, but some lost the heart altogether (up to 55 bpm off). The chest displacement of a walker was tracked to 0.07 mm over 2 m.
- **Not solved: an arm or a leg in the same range cell as the chest while walking.** Its fast phase spoils the velocity a frame measures from its own chirps, and with gaps between frames the phase then locks onto a wrong velocity. Frames without gaps (continuous chirping) remove the ambiguity; a 2D array can also separate the legs by elevation. Until recordings with an ECG reference exist, treat any heart rate during walking as a hypothesis; see [Heart rate of a moving person](#heart-rate-of-a-moving-person).

### AI

Automatic selection of the best range bin and receiver by a quality score, sliding-window trends, and comparison with a reference PPG and pulse measurement.

**Use:** press **RUN AI ANALYSIS**; for the reference comparison see [Comparing with a reference](#comparing-with-a-reference-ppg-and-pulse).

**How it decides:** with **Auto Select Best Bin** the candidates are the bins within 0.15 m of the range that breathes most (the **Auto Bin** of [Vitals](#vitals)), and the score picks the receiver at that range; the neighbours stay in the feature table and the maps. Over all receivers of that bin, a window's breathing rate comes from their spectra summed and its heart rate from their mean displacement together with that of the neighbouring bins' receivers, followed through the windows as in Vitals (harmonic sum, Viterbi path). A window counts for the breathing when its peak passes 6 on the **Min peak/avg** scale of Vitals, for the heart when the heart track's peak does. The rates reported (**Overview**, **Spectra**) are the medians of the windows that count.

**Controls:**
- **Actions:** **RUN AI ANALYSIS**; **Run Synthetic Test** (a generated chest at 1.5 m with 72 bpm and 16 breaths/min, seen by four receivers, with the antenna coupling at 0 m; the Log says what was found).
- **AI Mode & Selection:** **Scoring Method** (**Rule-based AI scoring** by default; **PCA scoring**, **KMeans clustering**, **Gaussian Mixture**, **Isolation Forest** need the scikit-learn package), **Phase Extraction**, **Auto Select Best Bin** (default on, see above; off = use **Manual Bin Index**). The receiver is always the one with the best score.
- **ROI & RX Fusion:** **Enable ROI Fusion** (default on; off = the single best bin and receiver). **ROI Source** (**Best RX only**, **All RX at selected bin** (default), **ROI bins at best RX**, **ROI bins x all RX**, **Top-K bin/RX candidates**), **ROI Width (bins)**, **Fusion Method** (**best bin**, **weighted complex**, **weighted displacement** (default: every receiver's circle centred, its phase unwrapped, weighted by its score), **PCA fusion**, **ICA fusion**).
- **Sliding Window Settings:** **Window Length [s]** (default 30 - the heart track needs about 30 s), **Overlap [%]** (default 90 - a window every 3 s).
- **Temporal Window Selection:** click windows in the **Temporal Map** plot to select them, then **Use Selected Windows** computes the rates from those windows only (with no selection, from all windows scoring at least **Min AI Score**, default 0.5): every selected bin with its own receivers as above, the weighted median of the windows that count. The windows are those of the last analysis, even if the window fields have changed since. **Clear Selection** removes the selection.
- **AI Score Weights:** how much each quality feature contributes to the score (heart and breathing SNR, phase stability, receiver coherence, penalties for motion, entropy and harmonics).
- **Search bands (shared by all vital tabs)** - the same bands as in [Vitals](#vitals).
- **Reference PPG/Pleth:** see the workflow.
- **Export:** **Export Results (JSON)** - everything the analysis produced; **Export Features (CSV)** - one row per scored bin and receiver.

**Plots:** **Overview**, **Signals**, **Spectra**, **Spatial Map**, **Temporal Map** (click to select windows), **Features**, **Trends** (dots: the windows that count, rings: the others), **Comparison** (radar against reference), **Reference PPG**.

**How good it is:** on the six seated recordings: heart 2.2 bpm off the pulse oximeter's beats (84 % of the windows within 5 bpm), breathing 1.2 /min off the reference from the PPG (91 % within 2 /min; the windows that count 0.7 /min, 98 %). With the former defaults (the best score over every bright bin of the room, one receiver, 10 s windows, Welch segments of 512 frames, the largest peak of each window) heart 11.8 bpm, breathing 4.3 /min.

## 8. Plots, playback and export

### Playing through the recording

- The global frame slider and **Play All** drive all frame-by-frame views; **Stop** stops.
- Playback runs at the frame rate of the recording.
- Only the view whose tab is shown is redrawn while playing.

### Zooming and panning

The Vitals, HR and AI plots have the Matplotlib toolbar under the figure:

- **Home** resets the view, **Back** / **Forward** step through previous views, **Pan** drags, **Zoom** draws a rectangle, **Save** saves the figure.
- **X Zoom Mode** - drag in a time plot to select a time range; all time plots of the figure zoom to it together and their vertical axes adapt.
- **Reset X Zoom** returns the time axes; **Reset View** returns everything.

### The Export menu

Each view's **Export** button (next to ⓘ) offers:

- **Current frame (PNG)** - the figure as shown.
- **Data (CSV)** - the numbers behind the plot, as described in each view's section (usually without axis values - the range and velocity axes follow from the parameters).
- **All frames (Video)** - an MP4 of every frame at the recording's frame rate; needs ffmpeg. The application may not respond while the video is written.
- Vitals adds **Rates over time (CSV)**.

### Other exports

- **HR tab:** **Export CSV**, **Export JSON**.
- **AI tab:** **Export Results (JSON)**, **Export Features (CSV)**.
- **Calibration:** saved as a JSON file from **Estimate...**; `tools/calibrate.py --out`.
- **Log:** **Save...** in the Log tab.
- **This manual:** **Save as HTML...** in the Manual tab.

## 9. Helper tab

A chirp configuration for mmWave Studio from what you want to measure: the values to type into mmWave Studio (or a Lua script that does it all), what they give, and whether the radar, the DCA1000 and DARTscope's analyses can cope with them.

**Quick use:**
1. **Device** - **IWR6843AOP**, **IWR6843 (ISK, ODS)** or **AWR2544**.
2. **Purpose** - **Vital signs (seated person)**, **People and angles (AoA, tracking)**, **Fast limb movements (micro-Doppler)**, **Heart rate of a moving person** or **Tremor (hand)** fills in the goals and designs at once; **Custom** leaves them to you.
3. Adjust the goals in **1. What to measure** if you need to and press **Design**.
4. Read **Result** on the right: the values, what they give and the checks - **ok**, **WARNING** or **ERROR**, each saying what to change.
5. Page **Lua script**: set **3. Script** (COM port of the board, mmWave Studio folder, where the recording goes), then **Copy Lua** and paste it into mmWave Studio's Lua shell, or **Save Lua...** and run the file there. Or type the values of **Result** into mmWave Studio's SensorConfig tab by hand.
6. After the recording save the mmWave Studio log next to the .bin as `<name>_LogFile.txt` - DARTscope reads the configuration from it. Then **Run Fastcheck**.

**1. What to measure:**
- **Max range [m]**, **Range resolution [cm]**, **Max velocity [m/s]** - the farthest target, the smallest distance between two targets still told apart, and the fastest radial motion (walking about 1.5 m/s; faster targets fold over).
- **Velocity resolution [m/s]** or **Loops per frame** - either decides how many times the chirp pattern repeats in a frame; **Loops per frame** wins when both are filled. For vital signs a few loops are enough (DARTscope averages them).
- **Frame rate [fps]**, **Recording length [s]** (sets the number of frames), **RX gain [dB]**, **Start frequency [GHz]**, **Idle time [us]**, **ADC start time [us]**.
- **Transmitters** - **one TX** (**TX for one TX** picks which) or **all TX (TDM-MIMO)**: every transmitter in turn, needed for angles. The time between two chirps of the same transmitter then grows by the number of TX, and the maximum velocity drops with it.
- **Frames back to back (continuous chirping)** - no gap between frames: the chirps fill the frame (at most 255 loops), as the micro-Doppler and moving-person analyses want.

**How it designs:** the number of ADC samples from the maximum range and the resolution (the range bins needed); the time a chirp may take from the maximum velocity; within it a sampling window long enough for the resolution at a comfortable slope (100 MHz/us or less when there is time) and inside the IF band; then the slope, the ramp end time (the end of sampling plus 1 us), the loops and the frame period. The slope and the times are rounded as the chip rounds them. The resolution is the one of the *sampled* part of the ramp, which is what the data have - the ramp beyond sampling adds nothing (the [capture health report](#capture-health-report) points that out as *ADC window*).

**2. Chirp for mmWave Studio** holds the values in mmWave Studio's own terms: **Start Freq [GHz]**, **Idle Time [us]**, **ADC Start Time [us]**, **Ramp End Time [us]**, **Frequency Slope [MHz/us]**, **TX Start Time [us]**, **ADC Samples**, **Sample Rate [ksps]**, **RX Gain [dB]**, **TX of each chirp** (e.g. `TX1, TX2, TX3`), **No of Chirp Loops**, **No of Frames**, **Periodicity [ms]**. Edit any of them and press **Check these values**: the result and the script follow what you typed. **From the decoded recording** fills them from the recording opened in the Decode tab - to see what an old configuration gives, or to start from it.

**The checks:** the goals (range resolution, maximum range, maximum velocity, velocity resolution, frame rate); the device (band and sweep, slope, sample rate, IF band, ADC buffer, at most 255 loops); the frame (the chirps must fit in the period); the data rate - on the lab's recordings the DCA1000 recorded up to 30.6 MB/s without losing packets, so more is a warning and more than Gigabit Ethernet carries an error; the LVDS lanes while chirping; and whether the frame rate is enough for the heart band.

**Device limits used:** IWR6843 60-64 GHz, 4 GHz sweep, complex samples up to 12.5 Msps, IF band 10 MHz, slope up to 200 MHz/us (the highest used in the lab's recordings); AWR2544 76-81 GHz, 5 GHz sweep, real samples (as in the lab's script) up to 20 Msps, slope up to 100 MHz/us (as recorded - the data sheet may allow more). They are listed under **Result** and kept in `chirp_design.py` (`DEVICES`).

**The Lua script** follows the lab's working scripts (`C:/Data/Radar/config`): reset and connect, band and chip, firmware, `ChanNAdcConfig` (the transmitters used, 4 RX, 16 bit), LVDS, `ProfileConfig`, one `ChirpConfig` per transmitter, the DCA1000, `FrameConfig(first chirp, last chirp, frames, loops, periodicity ms, ...)`, the recording, a wait for its length and the stop. Numbers are written with a decimal point - Lua reads a decimal comma as two arguments (`ProfileConfig(0, 60,25, ...)` shifts every argument after it).

## 10. Workflows

### Vital signs of one seated person

1. Record at least 60 s with the person seated 0.5-2 m in front of the radar, at 20 fps or more (see [Chirp configuration for vital signs](#chirp-configuration-for-vital-signs)).
2. **Decode:** open the `_Raw_0.bin`.
3. **Analysis -> Profile -> Run Profile.** Check the Log: no errors; note lost packets.
4. Find the person's range: in **Profile** the person is usually one of the strongest peaks between 0.5 and 2 m; `python -m tools.vital_range_scan <recording>` finds it reliably by its breathing.
5. **Vitals:** press **Run Vital Signs Analysis** (Auto Bin finds the range); if the marker is not on the person, move the **Range Bin** slider to that range, check the **Bands** preset and run again.
6. Check that the breathing waveform looks regular and that the heart spectrum has one clear peak. If the rates jump around, see the tip in [Vitals](#vitals).
7. Optionally cross-check in **HR** (with **Bin** 0 it takes the same range as Auto Bin) and **AI**.
8. Export: **Export -> Rates over time (CSV)** and **Current frame (PNG)**.

### Comparing with a reference (PPG and pulse)

1. Find the matching reference session: `python -m tools.pair_reference <radar folder>` lists which somnograph session covers which recording and the offset into it.
2. Open the recording and run **AI -> RUN AI ANALYSIS**.
3. **Load PPG (TXT/CSV)**: choose the session's Pleth export. **Load Pulse (TXT/CSV)**: choose its Pulse export.
4. **Invert PPG Signal** only if the PPG waveform is upside down.
5. **Process PPG & Sync**: the reference is lined up with the radar by the absolute start times; the result appears under **Comparison** (heart rate against the reference, correlation) and **Reference PPG**.
6. **Manual Offset [s]** shifts the reference later (positive) or earlier (negative) if the clocks were not synchronised.
7. With a **Segment** set, the start of the part is taken into account automatically.

### Two people in the room

- [Multi Vitals](#multi-vitals) finds them, follows them and measures each one's breathing and heart rate: press **Run Multi Vitals**, then play through the recording on **Map**.
- Two people at clearly different distances can also be analysed one range bin at a time in **Vitals** (or **HR**); **Movement** with **Band: Breathing** shows where each of them breathes.
- Two people side by side at the same distance can only be told apart by angle: a MIMO recording, **Find people by: Motion** or **Breathing**, and **Beam: Capon** when they sit close together.

### Where is the person? Angle of arrival

1. Check the **Array** (Auto-detect or the right board).
2. **View: Range-Azimuth**, **Range [m]** e.g. `0.3-3`, press **Run Angle of Arrival**.
3. For a moving person tick **Doppler Comp**, for the moving one only **Track Moving**.
4. For height as well: **View: Range-Az-El** with **Proj: Top**, **Slice** or **3D**; tick **Detect (CFAR)** for a list of targets in the Log.

### Moving people: Doppler compensation, CFAR, velocity unfolding

1. **Clutter: mti2** or **ema** on the Analysis bar.
2. **Range-Doppler -> Run Range-Doppler**: people walking appear away from zero velocity.
3. **CFAR -> Run CFAR Detection**: the targets of each frame go to the Log.
4. If people move faster than `maxVelocity_m_s`, tick **Unfold velocity (TDM)** and set **max |v|** to a realistic limit.
5. **Angle of Arrival** with **Doppler Comp** gives their angles.

### Who is where: radar imaging

1. Decode a MIMO recording (or a single-TX one for seated people) and note the radar's height.
2. **Imaging**: **Radar height [m]**, **Show: People**, **Run Imaging**.
3. **Bird's-eye** and **Tracks** show who is where and how they went; **3D** and **Radar vision** what the radar sees; **Colour by: Kind** tells walkers (moving) from seated people (micro-motion).
4. **Export -> Point cloud of every frame (PLY)** or **Objects and tracks (CSV)**.

### Fast leg or arm swings and their maximum speed

1. Record with a configuration made for it (see [Chirp configuration for fast limb movements](#chirp-configuration-for-fast-limb-movements)); note the radar height.
2. **Clutter: mean** (default) or **mti2** on the Analysis bar.
3. **Micro-Doppler -> Run Micro-Doppler**. Check that the white torso line follows the body and the pink envelopes the edge of the band.
4. The label shows the maximum speeds; the Log lists every swing. With **Radar height [m]** set and a 2D array, each swing says *leg* or *arm*.
5. If swings are missed, lower **Min swing [m/s]**; if noise makes false ones, raise **Threshold [dB]**; if the person is far from others, **Gate: Fixed** with narrow **Range bins from / to** is also fine.
6. **Export -> Swings (CSV)** for the numbers; `python -m tools.limb_speed <recording> --csv swings.csv` does the same for many recordings, and `--reference` compares them with motion capture.

### Heart rate of a moving person

1. Record as in [Chirp configuration for the heart rate of a moving person](#chirp-configuration-for-the-heart-rate-of-a-moving-person), with an ECG belt or an oximeter.
2. **HR -> Moving person (phase 12) -> Run Moving HR**. On the page **Moving Person** check that the trunk range follows the person and the chest displacement has no steps.
3. Read the heart rate in the windows where it counts (the thick white line); red windows (strong motion) give none unless **Estimate in strong motion** is ticked.
4. **Export Moving HR (CSV)**, or for many recordings `python -m tools.heart_rate <recording> --reference ecg.csv` (or `--reference somno`): MAE, bias, limits of agreement and coverage for each motion class.

### Antenna calibration with a corner reflector

1. Put a trihedral corner reflector (5-10 cm edge at 60 GHz) on a non-metallic stand at radar height, 1-3 m in front of the radar, with nothing else within ±0.5 m of its distance. Straight ahead (azimuth 0, elevation 0) is easiest.
2. Measure its distance precisely.
3. Record 5-10 s with the same chirp and TX configuration as your measurements. Optionally record the same scene without the reflector.
4. Open the reflector recording, **Angle of Arrival -> Calibration: Estimate...**: enter the approximate range, the azimuth and elevation, and the measured distance. Save the result.
5. Check it on a second recording with the reflector at another angle: `python -m tools.calibrate <second recording> --verify <calibration.json> --range 2.0 --az 25` must show a smaller angle error with the calibration than without it. A calibration always fits the recording it was made from; only the second position proves it.
6. For your measurements: **Calibration: Load...** the file. It stays valid for that board and configuration; it is not needed for every recording.

### Working on part of a long recording

1. Open the recording and run any view once (the label next to **Segment** shows the length).
2. Type e.g. `30` and `90` into **Segment [s]** and press **Apply**.
3. Run the views you need; they now use only that part and are much faster. Clear both fields and press Apply to return to the whole recording.

### Checking many recordings at once

- `python -m tools.health_report <folder> --problems` - the health report of every recording in a folder, only warnings and errors.
- `python -m tools.decode_check <folder>` - decodes every recording and checks that frames, lanes and formats are consistent.

## 11. Command-line tools

All tools run from the repository folder with `python -m tools.<name> ...`; `--help` lists their options.

- **health_report** - the capture health report of one or many recordings. `python -m tools.health_report <recording or folder> [--quick] [--problems] [--json OUT]`; `--random N` checks N recordings picked at random. Exit status 2 when any recording has an error.
- **decode_check** - decodes every recording of a folder and compares the results. `python -m tools.decode_check <folder> [--pattern GLOB] [--no-compare] [--json OUT]`.
- **pair_reference** - which somnograph session covers which radar recording, and the offset into it. `python -m tools.pair_reference <radar folder> [--reference DIR] [--json OUT]`; the reference folder defaults to `somno_vysledky` inside the radar folder.
- **vitals_check** - how the [Vitals](#vitals) view does against a somnograph on every recording one of its sessions covers: the view's chain with its default settings, each window against the reference over the same span. Heart: the somnograph's Pulse and the beats of its Pleth (PPG) waveform. Breathing: three PPG surrogates (baseline, pulse amplitude, beat rate) and only windows where all three agree within 2 /min - the baseline and the beat rate also carry the ~0.1 Hz Mayer wave of the blood pressure. `python -m tools.vitals_check <radar folder> [--reference DIR] [--csv OUT] [--win-br S] [--win-hr S] [--only STEM ...]`; prints MAE, bias, the share within 5 bpm or 2 /min and the coverage per recording and pooled.
- **vital_range_scan** - finds the range of a breathing person by scoring every range bin, and reports breathing rate, displacement and a heart-rate readout per bin. A bin scores by how much of its slow motion is one breathing peak, weighted by how much its echo moves at breathing rates (column *move*, dB against the bin that moves most) and by its neighbours sharing the rate or the waveform; a bin moving less than 0.05 mm scores 0. Breaths are also counted one by one in the shared breathing band (*counted*, */min*); for irregular breathing that is the rate to believe, as the spectral peak (*peak/min*) falls on a harmonic. On 21 recordings of a seated person at a known distance it was within 0.3 m in 19 (9 before these guards). `python -m tools.vital_range_scan <recording> [--min-range M] [--max-range M] [--top N] [--csv OUT]`.
- **limb_speed** - swings of legs and arms and their maximum speed (the [Micro-Doppler](#micro-doppler) view without the GUI). `python -m tools.limb_speed <recording> [--from S --to S] [--bins 5-100] [--fixed] [--width M] [--window-ms 20] [--threshold-db 8] [--min-swing 1.5] [--clutter mean] [--antenna MRC] [--radar-height M] [--csv OUT] [--envelopes OUT]`. With `--reference mocap.csv --radar-pos X,Y,Z [--joint NAME] [--offset S] [--ref-units mm]` it compares every swing with the radial speed of a motion-capture joint (CSV with a time column and x, y, z per joint).
- **heart_rate** - the heart rate of a person who may be moving (the [Moving person (phase 12)](#moving-person-phase-12) analysis without the GUI). `python -m tools.heart_rate <recording> [--from S --to S] [--method auto] [--win 15] [--max-change 3] [--range 0.3-3.0] [--allow-strong] [--csv OUT]`. With `--reference somno` (the somnograph pulse, paired by time; `--reference-dir DIR`) or `--reference ecg.csv [--offset S]` (columns `time_s` and `hr_bpm`) it prints MAE, bias, the Bland-Altman limits of agreement, the share within 5 and 10 bpm and the coverage, for all windows and for rest, light and strong motion.
- **calibrate** - antenna calibration from a corner reflector and its verification. `python -m tools.calibrate <recording> --range 1.5 [--az 0 --el 0] [--true-range 1.50] [--empty <empty recording>] [--phase-only] [--antenna NAME] --out cal.json`; `--verify cal.json` checks a calibration on another recording.

## 12. Troubleshooting

- ***Please decode the log file first.*** - a Run was pressed before a recording was opened: use the Decode tab first.
- ***Corresponding LOG file not found.*** - the `<name>_LogFile.txt` is missing or named differently from the BIN file.
- ***ProfileConfig not found in the log file.*** - the log file is not a recording log.
- **The first Run takes long** - reading a long recording takes time; use **Segment** to work on a part.
- **Nothing in the plots / noise only** - check the Log: lanes, frame count and IQ balance. Wrong lane count or format shows as a frame-count error; swapped lanes as noise (try **Lane Swap**).
- ***I and Q look swapped*** in the Log - the IQ order of the recording is wrong; targets appear at mirrored ranges.
- **Frame rate warning (15 fps assumed)** - the log has no usable frame period; time axes and rates are then wrong.
- **Lost packets** - the recorder filled parts with zeros. They are shaded in the Vitals plots; choose a clean part with **Segment** for vital signs.
- **Angles look wrong or mirrored** - check the **Array**; compare configurations with a target at a known angle (**Config... -> Compare all configurations**); calibrate the board.
- **Heart rate unrealistic** - wrong range bin (check the range profile), band not suited to the person (**Bands**), or too much motion (look at the displacement plot). Compare with the HR and AI tabs.
- **CFAR says *Please run Range-Doppler first.*** - run the Range-Doppler view before CFAR.
- **Video export fails** - ffmpeg is not installed or not on the PATH.
- **A Run button stays yellow** - its settings changed after it was computed; press it.
- **Something failed silently** - open the Log tab: errors are listed there with details. Save the log when asking for help.
- **A computation takes long** - the status bar shows how long it has run and the estimate of what is left; the Log tab's **[time]** lines show afterwards which step took the time. A long recording at a high frame rate costs most in the views that work frame by frame (Imaging, Movement, Range-Angle); choose a shorter part with **Segment**.
- **Blurred text and plots on Windows** (a display scaled to 125 % or more) - DARTscope draws at the scaling of the main display; on a second display with another scaling Windows still stretches the window. `DARTSCOPE_DPI_AWARE=0` in the environment brings back the stretched window of older versions.

## 13. Glossary

- **ADC** - analog-to-digital converter; its samples are the raw data.
- **Chirp** - one frequency ramp transmitted by the radar; the samples of one chirp give one range spectrum.
- **Frame** - a group of chirps repeated at the frame rate; one frame gives one Range-Doppler map.
- **Loop** - one repetition of the chirp pattern within a frame.
- **TX / RX** - transmitting and receiving antennas.
- **TDM-MIMO** - transmitters take turns chirp by chirp, which gives a virtual array of TX x RX antennas and better angular resolution.
- **Virtual array** - the set of all transmitter-receiver pairs, treated as one larger antenna array.
- **Range bin** - one point of the range spectrum; the spacing between bins is shown in the parameters.
- **Range resolution** - the smallest distance between two targets that can still be separated.
- **Doppler / radial velocity** - velocity towards or away from the radar, measured from the phase change between chirps.
- **Maximum unambiguous velocity (v_max)** - faster targets fold over to a wrong velocity.
- **dBFS** - decibels relative to the full scale of the ADC; 0 dBFS is clipping.
- **Clutter** - reflections from static objects.
- **Micro-Doppler** - the Doppler of parts of a target that move relative to it (arms, legs), seen as a spectrogram of velocity against time.
- **Envelope** - the fastest velocity (away or towards the radar) still on the body at each moment; its peaks are the maximum speeds of the limbs.
- **Swing** - a limb moving clearly faster than the torso for a moment, from its start to its end.
- **CFAR** - constant false alarm rate detection: a threshold adapted to the local noise.
- **Dropout** - a stretch of frames the recorder filled with zeros because it lost packets; see the [Fastcheck tab](#4-fastcheck-tab).
- **RTI** - range-time intensity: the range profile of every frame (or chirp) side by side, range against time.
- **MRC** - maximum ratio combining; here, power averaging of the receivers.
- **Bartlett / Capon** - two beamformers for angle of arrival; Capon is adaptive and sharper.
- **IQ** - the in-phase and quadrature components of a complex signal.
- **Image rejection** - how much weaker the mirror image of a target is than the target; a measure of IQ balance.
- **LVDS lanes** - the data lines between the radar and the DCA1000.
- **Zero-filled packets** - data the recorder did not receive and replaced with zeros.
- **PPG / Pleth** - photoplethysmogram, the optical pulse waveform of a pulse oximeter; **Pulse** is its heart rate.
- **BR / RR** - breathing (respiration) rate, breaths per minute. **HR** - heart rate, beats per minute (bpm).
