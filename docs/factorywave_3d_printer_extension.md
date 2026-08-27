# FactoryWave 3D Printer Extension

## Overview

This document specifies an extension to the **FactoryWave** dataset that adds a **3D printer telemetry modality** alongside the existing robotic-arm streams. The printer is a **Bambu Lab P2S**, recorded under the same FactoryWave-style protocol — repeatable jobs, systematic fault injection, an optional counterfactual experiment, and paired metadata.

The extension covers the following configuration:

- **Bambu Lab P2S** — repeatable print jobs with normal, fault, and counterfactual recordings

**Goal**: record dense multimodal traces of repeatable print jobs under normal and fault conditions so downstream models can be evaluated on the same four-level (state / intervention / counterfactual / decision) questions as the robotic-arm tasks. Approximately TBD episodes.

**Printer:** Bambu Lab P2S (telemetry sampling rate **TBD**)
**Print jobs / "tasks":** TBD (e.g. one canonical calibration print + one functional print + one AMS multi-material print — final selection to confirm)
**Faults:** TBD (see [Fault Types](#fault-types) — catalogue to be filled in)
**Experiments:** 3 experiment types — normal, fault injection, counterfactual

---

## General Principles

- **One print job = one episode.** A complete print from heat-up to cool-down is the unit of recording.
- **Print jobs are repeatable.** The same g-code under nominal conditions should produce a near-identical telemetry trace; this is what makes fault injection and counterfactual reasoning tractable.
- **Phases must be recorded.** Every logged step carries a `print_phase` label (e.g. `heatup`, `bed_level`, `purge`, `first_layer`, `print`, `cooldown`, `idle`) so downstream models can condition on or segment over print structure.
- **Record everything.** All telemetry channels exposed by the printer's MQTT API (or equivalent local interface) should be logged. Feature selection happens at analysis time.
- **Vision and telemetry must be time-synced.** Frame timestamps and MQTT message timestamps must share a clock (or have a measurable offset) so visual fault evidence can be lined up with sensor signals.
- **Early data is useful even if trivial.** A short calibration print, a failed setup, or a normal print are all valuable for pipeline validation.

---

## Before You Start: Checklist

- [ ] Recording laptop is on the same network as the printer; MQTT broker reachable
- [ ] Telemetry-recording script is tested and receiving messages (run a 30-second test)
- [ ] Chamber camera RTSP stream (and LiDAR scan capture if applicable to P2S — confirm TBD) is reachable
- [ ] Time-sync check between MQTT messages and vision frames passes
- [ ] Filament reels for the planned blocks are loaded into the AMS (or external feeder) and logged: material, brand, colour, spool mass at start
- [ ] Bed surface is the planned type (textured PEI / smooth PEI / engineering plate) and logged
- [ ] Slicer settings file (.3mf or equivalent) for each print job is saved and version-controlled
- [ ] Spare nozzle / hotend / build plate within reach for fault recovery
- [ ] Fire-safe environment (no leaving the printer unattended during fault blocks)

---

## Safety Rules

1. **NEVER leave the printer unattended during fault blocks.** Thermal faults can cascade.
2. **Have an extinguisher rated for electrical / lithium fires within reach.**
3. **Kill the printer at the wall if a thermal anomaly persists** — the firmware's thermal runaway protection is the first line of defence, not the only one.
4. **The hotend reaches 250–300 °C.** Never reach into the chamber during a print.
5. **AMS / filament jams produce force buildup at the extruder.** Disconnect printer power before clearing a jam to avoid an uncommanded extrude when the obstruction releases.
6. **Verify the printer is at a clean nominal state** (chamber clear, bed clean, nozzle wiped) before starting each block.

---

## Modalities

| Modality | What | Sampling rate |
| --- | --- | --- |
| Telemetry (MQTT / local API) | Hotend / bed / chamber temperatures, fan speeds, motor states, print progress, AMS state, error/warning codes, commanded position, … | **TBD** (native MQTT publish rate + any local poll loop on top) |
| 2D vision | Chamber camera RGB stream | **TBD** (chamber cam native RTSP; resolution + fps TBD) |
| 3D vision (if applicable to P2S) | LiDAR first-layer scan / micro-LiDAR if present on this model | **TBD** (per-print scans, once per first layer) |
| Power | Wall-meter (optional) | **TBD** |

Each episode produces one synchronised bundle: telemetry stream + vision stream(s) + episode metadata block.

---

## Vision Setup

**TBD.** Confirm: which cameras are recorded (built-in chamber camera only, or external add-on cameras too), resolution, framerate, codec, mounting (built-in is fixed; any external camera's position needs documenting), and time-synchronisation method between the camera stream(s) and the MQTT telemetry stream.

---

## Print Jobs ("Tasks")

A "task" in this extension is **one canonical g-code**. Each task is run many times under different conditions (normal, fault-injected, counterfactual) to produce repeated episodes with a known nominal trace.

Final task selection is **TBD**. Candidate jobs (to confirm):

- **Calibration print** — short (~20–30 min), well-known model used industry-wide (e.g. calibration cube, Voron tune cube). Useful for baseline data and quick fault recordings.
- **Functional / standard print** — medium (~50 min–1 h), structurally interesting (e.g. 3DBenchy). Captures more diverse layer types (overhangs, bridges, small features).
- **AMS multi-material print** — medium-long, exercises the AMS filament swap mechanism (multi-colour or multi-material). Adds a fault surface that single-material prints do not.

Per task, the g-code file (.3mf) used must be committed alongside the dataset so the nominal trace is reproducible.

### Phases (per print)

| Phase index | Phase name      | Description                                                                  |
| ----------- | --------------- | ---------------------------------------------------------------------------- |
| 0           | `idle`          | Printer is powered, no print active                                          |
| 1           | `heatup`        | Bed and hotend heating to setpoint                                           |
| 2           | `bed_level`     | Mesh-level / auto-level routine (and LiDAR first-layer scan if applicable)   |
| 3           | `purge`         | Prime line / purge extrusion before the print starts                          |
| 4           | `first_layer`   | First print layer (highest adhesion-failure risk)                            |
| 5           | `print`         | Continuous layer-by-layer printing                                           |
| 6           | `ams_swap`      | Filament swap event (only if AMS is active)                                  |
| 7           | `cooldown`      | Print complete, hotend + bed cooling                                         |
| 8           | `idle`          | Print finished, printer waiting                                              |

---

## Fault Types

**TBD.** A printer-specific fault catalogue is to be assembled. Candidate categories to populate (each should follow the same structure as the FactoryWave robot-arm fault tables: ID, `root_cause`, explanation, how to inject, automation level, fault-specific metadata, event timing):

- Bed adhesion / first-layer faults (e.g. detached corner, wrong bed temperature, wrong adhesive)
- Motion faults (e.g. layer shift from loose belt or excessive acceleration, stepper skip under load)
- Extrusion faults (e.g. partial nozzle clog, filament tangle, AMS misfeed, wrong filament loaded)
- Thermal faults (e.g. thermistor offset, hotend setpoint misconfigured, part-cooling fan disconnect)
- Calibration faults (e.g. Z-offset drift, extrusion multiplier off, bed-level drift)
- Catastrophic faults (e.g. spaghetti failure / detached print, power-glitch pause-resume)
- AMS / multi-material faults (e.g. wrong filament for active step, jam at the feeder, runout mid-swap)

Each fault entry will specify whether it is software-injectable (e.g. setpoint changes, calibration offsets) or physically injectable (e.g. tape on the bed, partial nozzle obstruction). The `RECORD TIMESTEP` faults — where the injection moment can be precisely logged — are candidates for the counterfactual experiment.

### Default metadata per episode

Every episode (regardless of condition) logs the following default metadata fields alongside the telemetry stream:

| Field                       | Description                                                                     |
| --------------------------- | ------------------------------------------------------------------------------- |
| `episode_id`                | Unique identifier for the episode                                               |
| `printer_model`             | `bambu_lab_p2s`                                                                 |
| `print_job`                 | Print-job identifier (which canonical g-code was run)                            |
| `gcode_hash`                | SHA of the g-code file used, to bind the trace to an exact slicer output         |
| `condition`                 | Episode condition (normal, fault, counterfactual)                                |
| `fault_id`                  | Fault ID injected, or null for normal episodes                                  |
| `filament_material`         | e.g. PLA, PETG, ABS, ASA, TPU                                                   |
| `filament_brand_colour`     | Brand + colour string                                                           |
| `filament_spool_mass_start` | Spool mass at episode start (grams), if measured                                 |
| `bed_surface`               | Build-plate type (textured PEI, smooth PEI, engineering plate, …)                |
| `chamber_door_state`        | Open / closed during this episode                                               |
| `ambient_temp_c`            | Ambient temperature at episode start                                            |
| `ambient_humidity_pct`      | Ambient humidity at episode start (if measured)                                 |
| `slicer_profile`            | Slicer profile / preset name used                                               |
| `nozzle_diameter_mm`        | e.g. 0.4 mm                                                                     |
| `print_speed_pct`           | Slicer print-speed override at start (100 = nominal)                            |

Fault-specific metadata fields will be added once the fault catalogue is filled in.

### Vision-specific metadata per episode

If a vision stream is recorded for the episode, in addition to the default metadata log:

- Identifier and model of each camera used in the episode (built-in chamber camera, any external add-on)
- Calibration reference: intrinsics + (if applicable) extrinsics relative to the printer frame
- Time-synchronisation reference between the vision stream(s) and the telemetry stream
- Per-stream framerate actually achieved
- Per-stream dropped-frame count
- Per-stream file path / handle
- Per-stream timestamp file path / handle
- Vision recording start and end timestamps (relative to telemetry start)
- Lighting condition at recording time (chamber lighting on/off, ambient on/off)

---

## Counterfactual Experiment

### Fault Selection

Select **5–6 faults** from the catalogue for which the injection moment can be precisely timed and logged — i.e. faults injected at a controllable instant during the print rather than being present from heat-up.

### Protocol

For each selected fault:

1. **Recreate initial conditions as exactly as possible.** Same g-code, same filament reel (and spool mass within a logged tolerance), same bed surface, same ambient conditions, same slicer profile. Log all initial-condition metadata.
2. **Record one clean baseline run** — the full print with no fault injected, under those initial conditions. This is the counterfactual reference.
3. **Record 3–5 fault runs** — same print repeated under the same initial conditions, with the fault event injected at a **randomly sampled timestep** (the same sampled timestep for each of the 3–5 runs) within the print.
4. **Select the best run.** For each fault run, compute the KL divergence between the signal distribution of the pre-event segment and the corresponding segment of the baseline run. The run with the **lowest KL divergence** pre-event is kept as the counterfactual ground-truth pair. Discard the others.

### Metadata to Record per Episode

In addition to the standard telemetry stream, log:

- **`cf_fault_id`** — fault ID injected (null for the baseline run)
- **`cf_injection_timestep`** — sample index at which the event was injected (null for baseline)
- **`cf_injection_time_s`** — wall-clock time of injection relative to the print's `start_print` event (null for baseline)
- **`cf_baseline_episode_id`** — ID of the paired baseline episode

### Output

Each selected counterfactual pair consists of:

- one baseline episode (full clean print)
- one fault episode (same initial conditions, fault injected at a logged timestep, pre-event segment closely matching the baseline)

---

## Recording Schedule

Printer: Bambu Lab P2S - Start time: **\_\_** - End time: **\_\_**

Filament loadout for the block - AMS slot 1: ** | AMS slot 2: ** | AMS slot 3: ** | AMS slot 4: **

Fix and verify nominal operation before moving to the next block.

| Block | Condition       | Fault IDs          | Print Job   | Episodes       | Total | Done? |
| ----- | --------------- | ------------------ | ----------- | -------------- | ----- | ----- |
| 1     | Normal          | -                  | TBD         | TBD            | TBD   | [ ]   |
| 2     | Counterfactual† | TBD (5–6 selected) | TBD         | TBD pairs      | TBD   | [ ]   |
| 3     | Fault           | TBD                | TBD         | TBD            | TBD   | [ ]   |
| 4     | Fault           | TBD                | TBD         | TBD            | TBD   | [ ]   |

**Printer subtotal: TBD episodes**

† **Counterfactual pairs** — N pairs = N baseline episodes + N best-selected fault episodes. See the [Counterfactual Experiment](#counterfactual-experiment) section above for the selection protocol.

---

## Bambu Lab P2S: Signal List

**STATUS: TBD** — exact MQTT topic / payload schema and native publish rates need to be confirmed against the P2S firmware.

The Bambu Lab printers stream telemetry over MQTT on the local network. Expected signal categories (final field names + units to confirm):

### Setpoint (Intent) — what the firmware was commanded to do

- Hotend temperature setpoint
- Bed temperature setpoint
- Chamber temperature setpoint (if controllable)
- Fan speed setpoints (part cooling, aux cooling, chamber)
- Commanded XYZ position (from current g-code line)
- Commanded extrusion / feed rate
- Print speed (mm/s) and speed-override percentage
- AMS active spool / commanded filament swap

### Effort / Feedback (Outcome) — what the printer is actually doing

- Hotend actual temperature
- Bed actual temperature
- Chamber actual temperature
- Fan actual RPMs (if exposed)
- Motor currents (X, Y, Z, extruder) if exposed by the firmware
- Print progress (current layer index, % complete, time remaining, current g-code line)
- Extrusion / filament-feed actual rate (if exposed)
- AMS actual active spool, jam flags, humidity per spool, remaining mass per spool

### Context / Health — printer state and environment

- Ambient temperature (if a sensor is fitted)
- Ambient humidity (if a sensor is fitted)
- Power consumption (if exposed)
- Wi-Fi / network status
- HMS (error / warning codes) — event-driven
- Firmware version
- Build plate detected type (if the printer reports it)

### Status / Mode

- `print_state` (idle / printing / paused / cancelling / finished / error)
- `print_phase` (added by the recording script — see [Phases](#phases-per-print) above)
- `timestamp`

### I/O

- LiDAR first-layer adhesion map (if available on P2S — TBD)
- Chamber camera stream identifier

### Episode Metadata (added by the recording script, constant per episode)

| Column Name              | Source              | Example                       |
| ------------------------ | ------------------- | ----------------------------- |
| `episode_id`             | Auto-generated      | `p2s_benchy_normal_pla_042`   |
| `ctx_printer_type`       | `--printer` flag    | `bambu_lab_p2s`               |
| `ctx_print_job`          | `--job` flag        | `benchy_v1`                   |
| `ctx_fault_label`        | `--condition` flag  | `normal`                      |
| `ctx_filament_material`  | `--filament` flag   | `pla`                         |
| `ctx_recording_date`     | Auto from clock     | `2026-02-20`                  |

### Bambu Lab P2S Grand Total: TBD telemetry columns + TBD metadata columns per timestep at TBD Hz

---

## End of Day Checklist

- [ ] All fault conditions have been reversed (bed cleaned, belts tightened, calibration restored, AMS unloaded if needed, slicer settings reset)
- [ ] Telemetry + vision recordings are saved and backed up
- [ ] Episode counts match the schedule above
- [ ] Per-episode vision artifacts (frames, LiDAR scans if applicable) are present and align with the telemetry stream
- [ ] G-code (.3mf) file for each print job is committed alongside the dataset
- [ ] Filament spool mass and any consumed materials are recorded
- [ ] Start / end times are recorded on this sheet
- [ ] Any anomalies or issues are noted below

**Notes:**

---
