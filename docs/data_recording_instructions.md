# FactoryBench Lab Data Recording Instructions

## Overview

You will run each robot through repetitive industrial tasks while a laptop records sensor data. Each complete cycle of a task is one **episode**. You will record **normal** episodes and **faulty** episodes according to the schedule below.

**Goal**: record data (time series sensoric + metadata). 3 robots, 2-3 tasks each, 3 types of experiment each. -> 15 experiments (screwing task only possible for UR3) amounting in around 12 000 episodes in total, 15 seconds each on average.

**Robots:** UR3, ABB IRB 2600, KUKA KR10
**Tasks:** Pick-and-Place, Stacking, Peg-in-Hole
**Faults:** 33 types (described below)
**Experiments:** 3 different experiments

---

## General Principles

- **Policies are intentionally simple.** The policy quality is not the subject of research; the state understanding is. The tasks are quite simple and we just want the policy to succeed the task a majority of the times without abnormal behaviour (around 95% of times). A naive scripted policy that reliably completes the task (under the noisy environment) is sufficient.
- **Phases must be recorded.** Every logged step must carry a `task_phase` label so downstream models can condition on or segment over task structure. You can code up the "skills" around those phases, or use a different granularity, but the phases of the tasks must be recorded each timestep.
- **Record everything.** All available sensor signals should be logged. Feature selection and pruning will be done at the analysis stage, not here.
- **Early data is useful even if trivial.** A static robot or a trivially successful episode is still valuable for pipeline validation, schema testing, and anomaly injection debugging.

---

## Before You Start: Checklist

- [ ] Recording laptop is powered on and connected to the robot's network
- [ ] Recording script is tested and receiving data (run a 10-second test recording)
- [ ] 3 objects of known weights are on the table, each labeled with its weight in kg
  - **Light:** **\_** kg
  - **Medium:** **\_** kg
  - **Heavy:** **\_** kg
- [ ] Pick position is marked with tape on the table
- [ ] Place position is marked with tape on the table
- [ ] Threaded hole is in place
- [ ] Peg-in-Hole fixture/slot is in place
- [ ] Tool flange wrench is nearby (correct size for this robot)
- [ ] Electrical tape (for gripper wear fault) is nearby
- [ ] Robot programs are loaded and tested:
  - [ ] Pick-and-place program works
  - [ ] Screwing program works (3 layers)
  - [ ] Peg-in-Hole program works

---

## Safety Rules

1. **NEVER leave the robot unattended during fault blocks.** Faults can cause unexpected behavior.
2. **Use reduced speed (50% or lower) during ALL fault injection blocks.**
3. **Keep the emergency stop button within reach at all times.**
4. **If any fault causes the robot to behave unpredictably, hit E-stop immediately and tighten/fix the fault before continuing.**
5. **The loose flange fault should be 1/4 turn MAXIMUM.** More than that risks the tool falling off.
6. **Verify the robot is back to normal operation after fixing each fault before starting the next block.**

---

## The 3 Tasks

### Task 1: Pick-and-Place

1. Robot starts at home position
2. Moves to pick position, grips object
3. Moves to place position, releases object
4. Returns to home position

Note: To enable automatic data collection, robot arm should also return object to pick position, record that part as a seperate pick-and-place episode to double data generation.

#### Phases

| Phase index | Phase name   | Description                                       |
| ----------- | ------------ | ------------------------------------------------- |
| 0           | `above_pick` | EEF moves to a waypoint directly above the object |
| 1           | `descend`    | EEF lowers toward the object                      |
| 2           | `settle`     | Brief hold to let physics settle before grasping  |
| 3           | `close`      | Gripper closes around the object                  |
| 4           | `lift`       | EEF lifts the grasped object off the surface      |
| 5           | `move_xy`    | Lateral transfer toward the bin                   |
| 6           | `lower`      | EEF lowers into the bin                           |
| 7           | `open`       | Gripper opens, object released                    |
| 8           | `retract`    | EEF moves away from the bin                       |
| 9           | `return`     | EEF returns to home configuration                 |

### Task 2: Screwing

1. Robot starts at home position, with screw pre-gripped
2. Moves to threaded hole position
3. Screws the bolt into the hole
4. Drops the screw
5. Returns to home position after each screwing phase
6. Moves to screw position
7. Loosen the screw
8. Returns to home position with screw in gripper

#### Phases

| Phase index | Phase name  | Description                                            |
| ----------- | ----------- | ------------------------------------------------------ |
| 0           | `approach`  | EEF moves to a pre-contact waypoint above the fastener |
| 1           | `descend`   | EEF lowers to engage the fastener                      |
| 2           | `screw`     | Continuous downward force + wrist rotation to thread   |
| 3           | `disengage` | Gripper opens / tool lifts off                         |
| 4           | `retract`   | EEF returns to a safe height                           |
| 5           | `descend`   | EEF lowers to engage the fastener                      |
| 6           | `loosen`    | Loosen screw                                           |
| 7           | `engage`    | Gripper closes                                         |
| 8           | `retract`   | EEF returns to a home position                         |

### Task 3: Peg in Hole

1. Robot starts at home position with object gripped
2. Moves to fixture/slot, inserts object precisely
3. Drop the object and go to home position
4. Move to object position
5. Retrieves object from fixture
6. Returns to home position

#### Phases

| Phase index | Phase name     | Description                                       |
| ----------- | -------------- | ------------------------------------------------- |
| 0           | `above_hole`   | EEF moves above the hole and aligns               |
| 1           | `insert`       | EEF pushes peg downward into the hole             |
| 2           | `disengage`    | Gripper opens                                     |
| 3           | `retract`      | EEF returns to home position                      |
| 4           | `above_object` | EEF moves above the object and aligns with object |
| 5           | `engage`       | Gripper closes                                    |
| 6           | `retract`      | EEF returns to home position                      |

---

## Fault Types

There are 33 fault types across two task families. Each entry gives the fault ID, the `root_cause` key used in data files, a plain-language explanation, and the injection procedure.

**CRITICAL RULE:** After EVERY fault block, FIX the fault completely before moving to the next block. Verify the robot is back to normal operation before starting the next block.

### Metadata per Episode

Every episode (regardless of condition) logs the following default metadata fields alongside the signal stream:

| Field                        | Description                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------- |
| `episode_id`                 | Unique identifier for the episode                                               |
| `robot_model`                | Robot model name (e.g. UR3, KUKA KR10)                                          |
| `task`                       | Task name (pick-and-place, screwing, peg-in-hole)                               |
| `condition`                  | Episode condition (normal, fault, optimization, trajectory-opt, counterfactual) |
| `fault_id`                   | Fault ID injected, or null for normal/optimization episodes                     |
| `weight_of_box`              | Mass of the transported object in kg (pick-and-place only)                      |
| `shape_of_box`               | Shape/dimensions of the transported object (pick-and-place only)                |
| `position_of_box`            | Initial XYZ position of the object on the table, if measurable                  |
| `gripper_model`              | Gripper model and type (e.g. vacuum, two-finger)                                |
| `payload_mass_configured`    | Payload mass value set in PolyScope (kg)                                        |
| `payload_cog_configured`     | Payload CoG offset set in PolyScope (x, y, z in mm)                             |
| `tcp_offset_configured`      | TCP position offset set in PolyScope (x, y, z in mm)                            |
| `tcp_orientation_configured` | TCP orientation set in PolyScope (rx, ry, rz in radians)                        |

Fault-specific metadata fields are listed in the **Metadata** column of each fault table below.

---

### Screwing Task Faults (IDs 1–5)

| ID  | root_cause                 | Explanation                                                                                                                                         | How to Inject                                                                                                                                             | Automation                              | Metadata                           |
| --- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- | ---------------------------------- |
| 1   | `damaged_screw_thread`     | The screw thread is physically damaged, preventing proper engagement during tightening.                                                             | Pre-damage the screw thread with a file or by cross-threading before placing it in the feeder. Run the tightening cycle and observe the torque signature. | Low (screw might fall down, not insert) | damage_description, damage_picture |
| 2   | `extra_assembly_component` | An unexpected object (e.g. a washer) is present in the assembly location, altering contact geometry and tightening dynamics.                        | Place an additional washer or shim in the assembly location before the tightening cycle.                                                                  | Low (screw might fall down, not insert) | object_description, object_picture |
| 3   | `missing_screw`            | The robot attempts tightening but no screw is present, often due to a pick failure or skipped step.                                                 | Remove the screw from the feeder so the robot tightens an empty hole.                                                                                     | High                                    | -                                  |
| 4   | `damaged_plate_thread`     | The threaded hole in the plate is damaged, preventing proper screw engagement.                                                                      | Pre-damage the threaded plate hole with an oversized tap or by cross-threading a screw manually.                                                          | Low (screw might fall down, not insert) | damage_description, damage_picture |
| 5   | `loosening_phase`          | The robot performs a screw-loosening (counterclockwise) rotation instead of tightening: a normal operational phase that produces a distinct signal. | Program the robot to execute a counterclockwise loosening rotation instead of tightening. No hardware modification required.                              | High (it is already encoded in task)    | -                                  |

---

### Pick-and-Place Task Faults (IDs 6–15, 29–31)

#### Mechanical / Hardware Faults

| ID  | root_cause                      | Explanation                                                                                                                                                                                 | How to Inject                                                                                                                                                                                                                                                              | Automation                                                                                                     | Metadata                                    |
| --- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| 6   | `additional_axis_friction`      | Increased mechanical friction at one joint or gearbox requires higher motor torque to maintain the commanded trajectory. Simulates dirt or age-related gear wear.                           | Apply a controlled amount of friction compound (e.g. dry chalk or fine abrasive paste) inside a target joint housing, or restrict the joint with a calibrated friction brake fixture. Rcord in metadata details of injection. Record weight and specific joint in metadata | Medium (requires manual physical setup per joint and material, then run a few episodes)                        | joint_affected, material_description        |
| 10  | `additional_axis_payload`       | A physical weight attached to one robot link increases its effective inertia and gravity loading across all joints.                                                                         | Attach a calibrated dead weight (e.g. 0.5–2 kg) directly to one of the robot links using a mechanical fixture, then run the standard program.                                                                                                                              | Medium (requires manual attachment of dead weight, but then multiple episodes can be recorded automatically)   | link_affected, added_mass, picture_of_robot |
| 13  | `cable_routed_on_robot`         | A cable fixed along the robot arm with cable ties introduces pose-dependent friction and gravity loading as it bends and twists during motion.                                              | Route a cable along the arm using cable ties at multiple link points, then run the full motion program across varying poses.                                                                                                                                               | High (set up once, then record all episodes)                                                                   | picture_of_robot                            |
| 15  | `unstable_mounting_platform`    | The robot's base is unstable due to released wheel brakes or uneven foam beneath it, introducing low-frequency vibrations during motion.                                                    | Release the wheel brakes of the robot trolley, or place calibrated foam pads of varying thickness under the base mounting plate.                                                                                                                                           | High (set up once, then record multiple episodes)                                                              | picture_of_robot, description_of_experiment |
| 26  | `magnetic_encoder_misalignment` | The absolute magnetic encoder read head is physically displaced from the magnetic ring (due to shock, loose mounting, or external magnetic interference) causing invalid position decoding. | Loosen the encoder read head mounting screws on a target joint by a controlled amount (0.5–1 mm gap increase) using feeler gauges, then power on and run.                                                                                                                  | Medium (requires hardware access inside joint cover, but then multiple episodes can be recorded automatically) | gap_added_mm, joint_affected                |
| 27  | `motor_phase_fault`             | One or more motor phase wires are partially or fully disconnected at the joint connector, reducing available torque and causing asymmetric current draw.                                    | Partially unplug one motor phase connector at a target joint (leave it seated but not latched) to introduce intermittent phase contact, then run the program.                                                                                                              | High (set up once then run all episodes)                                                                       | joints_affected                             |

#### Gripper Faults

| ID  | root_cause                      | Explanation                                                                                                                              | How to Inject                                                                                                                                                                         | Automation                                                       | Metadata            |
| --- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------- |
| 8   | `gripper_activation_failure`    | The vacuum gripper fails to activate and never picks up the can. The robot completes the motion but carries no payload.                  | Disconnect the vacuum line or disable the solenoid valve command in the gripper control I/O, then run a pick-and-place cycle. Pick random time (amongst relevant phases) for failure. | High (disable solenoid via I/O command in script)                | time_of_malfunction |
| 9   | `gripper_release_during_motion` | The vacuum gripper releases the can mid-trajectory at a random point during transport, causing an abrupt payload loss.                   | Inject a timed solenoid OFF pulse into the gripper control line mid-trajectory via a programmable relay or script-triggered digital output.                                           | High (timed digital output pulse fully scriptable)               | time_of_malfunction |
| 14  | `invalid_gripping_position`     | The can is gripped at an unusual or offset position due to a timing error in the pick sequence, causing asymmetric payload distribution. | Introduce a `sleep()` call in the URScript before the gripper activation command, so the gripper closes after the arm has begun lifting, gripping the can off-center.                 | High (sleep() injection is a pure software change in the script) | -                   |

#### Payload / Configuration Faults

| ID  | root_cause                          | Explanation                                                                                                                                                                  | How to Inject                                                                                                                                                       | Automation                                                                              | Metadata               |
| --- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ---------------------- |
| 7   | `motor_miscommutation`              | The motor commutation phase is offset in firmware, reducing torque-forming efficiency without affecting the commanded path. Simulates electrical or magnetic component wear. | Use the UR service interface or a custom firmware patch to introduce a fixed phase offset (e.g. 10–30°) in the motor commutation table of one joint.                | High (requires firmware-level service interface access once, but then run all episodes) | offset, joint_affected |
| 12  | `unexpected_payload_weight`         | The transported box has an unexpected weight deviating from the nominal value, simulating a damaged box or upstream filling error.                                           | Replace the nominal box with boxes of known different weights (e.g. empty or overfilled) without updating the payload configuration in PolyScope.                   | Medium (requires manually swapping cans between blocks of episode with different cans)  | -                      |
| 22  | `tcp_frame_misconfiguration`        | The Tool Center Point (TCP) frame or mounting orientation is configured incorrectly, causing the robot to detect unexpected gravitational torques.                           | Enter an incorrect TCP offset (e.g. shift X by 100 mm) or wrong mounting angle in Installation > TCP in PolyScope, then run the program.                            | High (pure software parameter change in PolyScope)                                      | offset                 |
| 23  | `zero_payload_misconfiguration`     | The payload mass is left at 0 kg while a tool or workpiece is physically attached, causing the collision detection model to trigger on normal inertial forces.               | Ensure the payload is set to 0 kg in PolyScope while a tool (e.g. gripper, ~300 g) is physically mounted, then run a pick-and-place cycle.                          | High (pure software parameter change in PolyScope)                                      | -                      |
| 28  | `mild_payload_cog_misconfiguration` | The payload center of gravity is specified at an incorrect offset from the tool flange, causing wrong gravity compensation torques and protective stops.                     | Set the CoG offset in the payload configuration at an offset from the actual tool CoG, then execute a multi-pose trajectory. Offset mild to not trigger safety stop | High (pure software parameter change in PolyScope)                                      | offset                 |

#### Collision Faults

| ID  | root_cause                   | Explanation                                                                                                                                                              | How to Inject                                                                                                                                                       | Automation                                                                                                                                                         | Metadata                                          |
| --- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------- |
| 11  | `collision_foam_object`      | The robot contacts a soft foam block in the workspace, producing a brief TCP force spike without triggering a protective stop.                                           | Place a foam cube directly in the programmed TCP trajectory. The robot will push through it with only a brief force spike.                                          | Low-medium (requires manually placing foam block in trajectory, depending on trajectory and displacement in an episode)                                            | picture*of_setup *(once per set up not episode)\_ |
| 29  | `collision_hanging_cable`    | A cable or wire hanging across the workspace contacts the arm during motion, producing a sustained asymmetric drag force rather than a sharp impact.                     | Suspend a loose cable across the robot trajectory at arm height so it drags along the robot link or tool flange during the motion cycle.                            | Low (requires manually suspending cable in workspace, depending on trajectory and displacement in an episode)                                                      | picture*of_setup *(once per set up not episode)\_ |
| 30  | `collision_cardboard_object` | The robot strikes a cardboard carton placed in its path. Cardboard provides moderate resistance: the robot may displace it or trigger a protective stop if it is braced. | Place a cardboard box (free-standing or lightly braced) in the robot trajectory. Brace the box against a wall to increase resistance and provoke a protective stop. | Low (requires manually placing cardboard box in trajectory, depending on trajectory and displacement in an episode)                                                | picture*of_setup *(once per set up not episode)\_ |
| 31  | `collision_rigid_object`     | The robot collides with a hard, immovable object such as a metal fixture, invariably triggering an immediate protective stop.                                            | Place a rigid metal block or clamp a solid fixture in the programmed TCP path. The collision will trigger an immediate protective stop.                             | High (requires manually placing rigid obstacle in trajectory, but robot should perform safety stop = no displacement. Then all episodes can be ran on this set up) | picture*of_setup *(once per set up not episode)\_ |

#### External Disturbance Faults

| ID  | root_cause                 | Explanation                                                                                                                                                          | How to Inject                                                                                                                                                                   | Automation                                               | Metadata                                          |
| --- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------- |
| 25  | `external_arm_disturbance` | A continuous external force pulls or pushes the robot arm at the TCP during motion (e.g. a spring anchored to the bench frame), causing persistent torque anomalies. | Anchor a spring or elastic band between a fixed point in the workspace (e.g. bench frame) and the tool flange (no contact along the arm links) and run the standard trajectory. | Low (requires manual rig setup with spring/elastic band) | picture*of_setup *(once per set up not episode)\_ |

---

### Peg-in-Hole Task Faults (IDs 32–36)

| ID  | root_cause                   | Explanation                                                                                                                                                                                | How to Inject                                                                                                                                                       | Automation                                                                                                          | Metadata                           |
| --- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| 32  | `peg_insertion_misalignment` | The peg approaches the hole at an angular or lateral offset, causing the tip to jam against the hole rim rather than enter cleanly, producing elevated contact forces and tracking errors. | Introduce a small angular offset (2–5°) or lateral offset (3–8 mm) in the insertion approach waypoint so the peg contacts the hole rim instead of entering cleanly. | High (pure waypoint offset change in robot program)                                                                 | offset                             |
| 33  | `hole_obstruction`           | A foreign object or debris is lodged inside the insertion hole, preventing full peg insertion and generating abnormal force buildup as the peg contacts the obstruction.                   | Place a small object (e.g. a metal chip, rubber disc, or folded paper) inside the hole before the insertion cycle.                                                  | Medium/High (requires manually placing object in hole per block, but should be a one time set up)                   | object_description, object_picture |
| 34  | `incorrect_insertion_depth`  | The insertion motion terminates at an incorrect Z depth due to a misconfigured waypoint, leaving the peg partially inserted or causing it to over-push against the hole bottom.            | Modify the insertion endpoint waypoint Z offset by +/- 10–20 mm from the nominal value in the robot program.                                                        | High (pure waypoint Z offset change in robot program)                                                               | offset                             |
| 35  | `peg_surface_contamination`  | Contamination or increased surface roughness on the peg raises insertion friction, producing stick-slip force patterns and higher-than-nominal contact force throughout the stroke.        | Apply a thin layer of dry chalk powder, fine abrasive paste, or low-tack adhesive to the peg surface before the insertion cycle.                                    | Medium/High (requires manual physical contamination of peg per block, but might be able to do in a one time set up) | material_description               |
| 36  | `fixture_displacement`       | The insertion fixture has shifted laterally from its nominal position due to vibration or improper clamping, creating a mismatch between the programmed approach and the actual hole.      | Deliberately shift the hole fixture by a controlled lateral offset (5–15 mm) from its nominal position without updating the robot program waypoints.                | Low/Medium (requires manually shifting fixture per block of episodes)                                               | offset                             |

---

### General / Software Faults (IDs 16–24)

These faults are not task-specific and can be injected during any motion cycle.
| ID | root_cause | Explanation | How to Inject | Automation | Metadata |
| --- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------- | ----------- |
| 16 | `gearbox_overload` | The joint gearbox operates near its shear torque limit due to excessive acceleration, deceleration, or mechanical obstruction. | Set joint acceleration and deceleration to maximum allowed values with a near-maximum payload and run repeated cycles. | High (pure software parameter change in robot program) | - |
| 19 | `joint_position_limit_violation` | A robot joint moves outside its configured soft or hard position limits, triggering a safety stop. | Program a waypoint slightly beyond the configured soft joint limit, or shift a joint limit in the safety configuration to encroach on the normal trajectory. | High (pure software waypoint or safety config change) | experiment_description |
| 21 | `singularity_passage` | The programmed trajectory passes through or very close to a kinematic singularity (wrist or shoulder), causing the controller to halt linear motion. | Program a MoveL move whose straight-line path passes through the wrist or shoulder singularity, then execute at moderate speed. | High (pure software waypoint change in robot program) | data on trajectory and singularity state (format is flexible) |
| 24 | `safety_plane_violation` | The robot trajectory crosses a configured safety plane or zone boundary, triggering an immediate protective stop. | Configure a safety plane in PolyScope that bisects the normal robot workspace, then run the existing program without modifying its waypoints. | High (pure software safety configuration change in PolyScope) | data on trajectory and safety plane (format is flexible) |

---

## Trajectory Optimization Episodes

Trajectory optimization episodes run the nominal task (no fault injected) with a randomized trajectory instead of the fixed programmed path. Each of the 150 episodes uses a different randomly sampled sequence of waypoints that still completes the task goal, varying approach angles, intermediate poses, joint configurations, and motion speeds. The purpose is to build a dataset of diverse "correct" executions so that downstream models can learn which trajectory variations are efficient and safe, enabling trajectory optimization and benchmarking of motion planners.

In addition to the standard 137-column signal recording, each trajectory optimization episode records three extra metadata fields:

- **duration** — wall-clock time from motion start to task completion (seconds)
- **success** — binary flag (1 = task completed successfully, 0 = failed or aborted)
- **energy** — total electrical energy consumed by all joints during the episode (joules), computed by integrating joint power over the episode duration

These fields are stored as episode-level metadata alongside the time-series signals, not as additional columns in the per-sample signal stream.

---

## Counterfactual Experiment

### Fault Selection

Select **5–6 faults** for which the injection moment can be precisely timed and logged: i.e. faults that are injected at a controllable instant during the episode rather than being present from the start.

### Protocol

For each selected fault:

1. **Recreate initial conditions as exactly as possible.** For pick-and-place: use the exact same object placed at the exact same marked position. For peg-in-hole: reset the fixture and peg to the same starting pose. For screwing: use the same hole and approach. Log all initial condition metadata (object ID, position, orientation, payload weight).

2. **Record one clean baseline run**: the full episode with no fault or event injected, under those initial conditions. This is the counterfactual reference.

3. **Record 3–5 fault runs** each is the same episode repeated under the same initial conditions, but with the fault event injected at a **randomly sampled timestep** (same for each of the 3-5 runs) within the episode.

4. **Select the best run.** For each fault run, compute the KL divergence between the signal distribution of the pre-event segment and the corresponding segment of the baseline run. The run with the **lowest KL divergence** pre-event (i.e. the one whose pre-fault trajectory most closely matched the baseline) is kept as the counterfactual ground truth pair. Discard the others.

### Metadata to Record per Episode

In addition to the standard signal stream, log the following episode-level fields:

- **`cf_fault_id`** — fault ID injected (null for the baseline run)
- **`cf_injection_timestep`** — sample index at which the event was injected (null for baseline)
- **`cf_injection_time_s`** — wall-clock time of injection relative to episode start (null for baseline)
- **`cf_baseline_episode_id`** — ID of the paired baseline episode

### Output

Each selected counterfactual pair consists of:

- one baseline episode (full clean run)
- one fault episode (same initial conditions, fault injected at a logged timestep, pre-event segment closely matching the baseline)

---

## Priorities

### Robot Priority

| Priority | Robots                  | Rationale                                                                                                                                                               |
| -------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| High     | UR3 > KUKA KR10         | UR3 is the most standard cobot in research settings. KUKA KR10 makes the dataset more attractive as most labs do not have access to it, increasing the paper's novelty. |
| Medium   | ABB IRB 2600 > ABB YuMi | Important for cross-embodiment generalization across different kinematic designs and manufacturers.                                                                     |

### Fault Priority

#### High Priority

| IDs       | Fault Group                     | Reason                                                                                                                                 |
| --------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| 32–36     | Peg-in-hole                     | Entirely new task domain with no existing open anomaly dataset. High novelty.                                                          |
| 21–24, 28 | Software / configuration faults | Software-only injection, fully reproducible, and not present in any public dataset. High value for studying operator misconfiguration. |
| 31        | Rigid collision                 | Natural extension of existing collision data but with guaranteed protective stop :useful for safety-critical benchmarking.             |

#### Medium Priority

| IDs          | Fault Group                                                                 | Covered by |
| ------------ | --------------------------------------------------------------------------- | ---------- |
| 1–4          | Screwing faults                                                             | AURSAD     |
| 6–15, 29, 30 | Pick-and-place faults (friction, payload, gripper, collision, entanglement) | voraus-AD  |

#### Lower Priority

| IDs    | Fault Group                     | Reason                                                                                 |
| ------ | ------------------------------- | -------------------------------------------------------------------------------------- |
| 25     | External arm disturbance        | Requires physical rig setup.                                                           |
| 26, 27 | Encoder & motor hardware faults | High diagnostic value but require hardware access (opening joint covers). Higher risk. |
| 16     | Gearbox overload                | Mechanically induced fault requiring careful setup to avoid damage.                    |
| 19     | Limit violations                | Useful edge case but lower frequency in real operations.                               |

### Experiment Priority

| Priority | Experiment type              | Reason                                                                                                      |
| -------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------- |
| High     | Optimization, Counterfactual | Unique contribution not covered by existing datasets. Enables trajectory optimization and causal reasoning. |
| Medium   | Anomaly injection            | Extends existing datasets (voraus-AD, AURSAD) with new tasks and robots.                                    |

---

## Recording Schedule

Robot: \***\*\_\_\*\*** - Start time: **\_\_** - End time: **\_\_**

Payload weights for this robot - Light: ** kg | Medium: ** kg | Heavy: \_\_ kg

Fix and verify normal operation before moving to the next block. This schedule is per robot, if possible (enough people working on it) handle robots in parallel.

### Pick-and-Place

| Block | Condition        | Fault IDs                                    | Payload            | Episodes       | Total | Done? |
| ----- | ---------------- | -------------------------------------------- | ------------------ | -------------- | ----- | ----- |
| 1     | Normal           | -                                            | Light/Medium/Heavy | 200 each       | 600   | [ ]   |
| 2     | Optimization     | 22, 28                                       | Light/Medium/Heavy | 20 each        | 120   | [ ]   |
| 3     | Trajectory Opt.† | -                                            | Light/Medium/Heavy | 150 each       | 450   | [ ]   |
| 4     | Counterfactual‡  | TBD (5–6 selected)                           | Medium             | 60 pairs total | 120   | [ ]   |
| 5     | Fault            | 8, 9, 12, 14                                 | Light/Medium/Heavy | 20 each        | 240   | [ ]   |
| 6     | Fault            | 6, 7, 10, 11, 13, 15, 16, 19, 21, 23, 24, 25 | Light/Medium/Heavy | 20 each        | 720   | [ ]   |
| 7     | Fault            | 26, 27, 29, 30, 31                           | Light/Medium/Heavy | 20 each        | 300   | [ ]   |

**Pick-and-place subtotal: 600 normal + 570 optimization + 1260 fault + 120 counterfactual = 2550 episodes**

---

### Screwing

| Block | Condition        | Fault IDs                                    | Payload | Episodes       | Total | Done? |
| ----- | ---------------- | -------------------------------------------- | ------- | -------------- | ----- | ----- |
| 8     | Normal           | -                                            | -       | 200            | 200   | [ ]   |
| 9     | Optimization     | 22, 28                                       | -       | 20 each        | 40    | [ ]   |
| 10    | Trajectory Opt.† | -                                            | -       | 150            | 150   | [ ]   |
| 11    | Counterfactual‡  | TBD (5–6 selected)                           | -       | 60 pairs total | 120   | [ ]   |
| 12    | Fault            | 1, 2, 3, 4                                   | -       | 20 each        | 80    | [ ]   |
| 13    | Fault            | 6, 7, 10, 11, 13, 15, 16, 19, 21, 23, 24, 25 | -       | 20 each        | 240   | [ ]   |
| 14    | Fault            | 26, 27, 29, 30, 31                           | -       | 20 each        | 100   | [ ]   |

**Screwing subtotal: 200 normal + 190 optimization + 420 fault + 120 counterfactual = 930 episodes**

---

### Peg-in-hole

| Block | Condition        | Fault IDs                                    | Payload | Episodes       | Total | Done? |
| ----- | ---------------- | -------------------------------------------- | ------- | -------------- | ----- | ----- |
| 15    | Normal           | -                                            | -       | 200            | 200   | [ ]   |
| 16    | Optimization     | 22, 28                                       | -       | 20 each        | 40    | [ ]   |
| 17    | Trajectory Opt.† | -                                            | -       | 150            | 150   | [ ]   |
| 18    | Counterfactual‡  | TBD (5–6 selected)                           | -       | 60 pairs total | 120   | [ ]   |
| 19    | Fault            | 6, 7, 10, 11, 13, 15, 16, 19, 21, 23, 24, 25 | -       | 20 each        | 240   | [ ]   |
| 20    | Fault            | 26, 27, 29, 30, 31                           | -       | 20 each        | 100   | [ ]   |
| 21    | Fault            | 32, 33, 34, 35, 36                           | -       | 20 each        | 100   | [ ]   |

**Peg-in-hole subtotal: 200 normal + 190 optimization + 440 fault + 120 counterfactual = 950 episodes**

---

‡ 60 pairs = 60 baseline episodes + 60 best-selected fault episodes. See [Counterfactual Experiment](#counterfactual-experiment) for the selection protocol. Recording effort is ~180–300 additional candidate runs per task (discarded, not counted in totals).

**Grand total: 1000 normal + 910 optimization + 2120 fault + 360 counterfactual = 4390 episodes per robot (3460 without screwing task)**

4380+3460+3460 = 12 220 episodes in total

---

## UR3: Complete Signal List (137 columns)

The recording script captures ALL of these signals at 500Hz. You do not need to configure anything -- the script handles this automatically.

### Setpoint (Intent) -- What the controller commands

| #   | ur_rtde Method        | Count  | FactoryNet Column Names                                                                                                                           |
| --- | --------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `getTargetQ()`        | 6      | `setpoint_pos_0`, `setpoint_pos_1`, `setpoint_pos_2`, `setpoint_pos_3`, `setpoint_pos_4`, `setpoint_pos_5`                                        |
| 2   | `getTargetQd()`       | 6      | `setpoint_vel_0`, `setpoint_vel_1`, `setpoint_vel_2`, `setpoint_vel_3`, `setpoint_vel_4`, `setpoint_vel_5`                                        |
| 3   | `getTargetQdd()`      | 6      | `setpoint_acc_0`, `setpoint_acc_1`, `setpoint_acc_2`, `setpoint_acc_3`, `setpoint_acc_4`, `setpoint_acc_5`                                        |
| 4   | `getTargetCurrent()`  | 6      | `setpoint_current_0` ... `setpoint_current_5`                                                                                                     |
| 5   | `getTargetMoment()`   | 6      | `setpoint_torque_0` ... `setpoint_torque_5`                                                                                                       |
| 6   | `getTargetTCPPose()`  | 6      | `setpoint_tcp_x`, `setpoint_tcp_y`, `setpoint_tcp_z`, `setpoint_tcp_rx`, `setpoint_tcp_ry`, `setpoint_tcp_rz`                                     |
| 7   | `getTargetTCPSpeed()` | 6      | `setpoint_tcp_speed_x`, `setpoint_tcp_speed_y`, `setpoint_tcp_speed_z`, `setpoint_tcp_speed_rx`, `setpoint_tcp_speed_ry`, `setpoint_tcp_speed_rz` |
|     |                       | **42** |                                                                                                                                                   |

### Effort / Feedback (Outcome) -- What the robot actually does

| #   | ur_rtde Method               | Count  | FactoryNet Column Names                                                                                                               |
| --- | ---------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| 8   | `getActualQ()`               | 6      | `actual_pos_0` ... `actual_pos_5`                                                                                                     |
| 9   | `getActualQd()`              | 6      | `actual_vel_0` ... `actual_vel_5`                                                                                                     |
| 10  | `getActualCurrent()`         | 6      | `effort_current_0` ... `effort_current_5`                                                                                             |
| 11  | `getActualCurrentAsTorque()` | 6      | `effort_torque_0` ... `effort_torque_5`                                                                                               |
| 12  | `getJointControlOutput()`    | 6      | `effort_control_output_0` ... `effort_control_output_5`                                                                               |
| 13  | `getActualTCPPose()`         | 6      | `actual_tcp_x`, `actual_tcp_y`, `actual_tcp_z`, `actual_tcp_rx`, `actual_tcp_ry`, `actual_tcp_rz`                                     |
| 14  | `getActualTCPSpeed()`        | 6      | `actual_tcp_speed_x` ... `actual_tcp_speed_rz`                                                                                        |
| 15  | `getActualTCPForce()`        | 6      | `actual_tcp_force_x`, `actual_tcp_force_y`, `actual_tcp_force_z`, `actual_tcp_force_rx`, `actual_tcp_force_ry`, `actual_tcp_force_rz` |
|     |                              | **48** |                                                                                                                                       |

### Context / Health -- Machine state and environment

| #   | ur_rtde Method                 | Count  | FactoryNet Column Names                                                                              |
| --- | ------------------------------ | ------ | ---------------------------------------------------------------------------------------------------- |
| 16  | `getJointTemperatures()`       | 6      | `ctx_joint_temp_0` ... `ctx_joint_temp_5`                                                            |
| 17  | `getActualJointVoltage()`      | 6      | `ctx_joint_voltage_0` ... `ctx_joint_voltage_5`                                                      |
| 18  | `getJointMode()`               | 6      | `ctx_joint_mode_0` ... `ctx_joint_mode_5`                                                            |
| 19  | `getActualToolAccelerometer()` | 3      | `ctx_tool_accel_x`, `ctx_tool_accel_y`, `ctx_tool_accel_z`                                           |
| 20  | `getFtRawWrench()`             | 6      | `ctx_ft_raw_fx`, `ctx_ft_raw_fy`, `ctx_ft_raw_fz`, `ctx_ft_raw_tx`, `ctx_ft_raw_ty`, `ctx_ft_raw_tz` |
| 21  | `getActualMainVoltage()`       | 1      | `ctx_main_voltage`                                                                                   |
| 22  | `getActualRobotVoltage()`      | 1      | `ctx_robot_voltage`                                                                                  |
| 23  | `getActualRobotCurrent()`      | 1      | `ctx_robot_current`                                                                                  |
| 24  | `getActualMomentum()`          | 1      | `ctx_momentum`                                                                                       |
| 25  | `getSpeedScaling()`            | 1      | `ctx_speed_scaling`                                                                                  |
| 26  | `getSpeedScalingCombined()`    | 1      | `ctx_speed_scaling_combined`                                                                         |
| 27  | `getActualExecutionTime()`     | 1      | `ctx_execution_time`                                                                                 |
|     |                                | **33** |                                                                                                      |

### Status / Mode

| #   | ur_rtde Method          | Count | FactoryNet Column Names  |
| --- | ----------------------- | ----- | ------------------------ |
| 28  | `getTimestamp()`        | 1     | `timestamp`              |
| 29  | `getRobotMode()`        | 1     | `ctx_robot_mode`         |
| 30  | `getRobotStatus()`      | 1     | `ctx_robot_status`       |
| 31  | `getSafetyMode()`       | 1     | `ctx_safety_mode`        |
| 32  | `getSafetyStatusBits()` | 1     | `ctx_safety_status_bits` |
| 33  | `getRuntimeState()`     | 1     | `ctx_runtime_state`      |
| 34  | `getPayload()`          | 1     | `ctx_payload_mass`       |
|     |                         | **7** |                          |

### I/O

| #   | ur_rtde Method                 | Count | FactoryNet Column Names |
| --- | ------------------------------ | ----- | ----------------------- |
| 35  | `getActualDigitalInputBits()`  | 1     | `io_digital_inputs`     |
| 36  | `getActualDigitalOutputBits()` | 1     | `io_digital_outputs`    |
| 37  | `getStandardAnalogInput0()`    | 1     | `io_analog_input_0`     |
| 38  | `getStandardAnalogInput1()`    | 1     | `io_analog_input_1`     |
| 39  | `getStandardAnalogOutput0()`   | 1     | `io_analog_output_0`    |
| 40  | `getStandardAnalogOutput1()`   | 1     | `io_analog_output_1`    |
|     |                                | **6** |                         |

### Episode Metadata (added by the recording script, constant per episode)

| Column Name          | Source                 | Example                               |
| -------------------- | ---------------------- | ------------------------------------- |
| `episode_id`         | Auto-generated         | `ur3_pick_and_place_normal_0.5kg_042` |
| `ctx_robot_type`     | `--robot` flag         | `ur3`                                 |
| `ctx_task`           | `--task` flag          | `pick_and_place`                      |
| `ctx_fault_label`    | `--condition` flag     | `normal`                              |
| `ctx_payload_kg`     | `--payload_kg` flag    | `0.5`                                 |
| `ctx_recording_date` | Auto from system clock | `2026-02-20`                          |

### UR3 Grand Total: 137 sensor columns + 6 metadata columns = 143 columns per timestep at 500Hz

---

## ABB IRB 2600: Signal List

**STATUS: WAITING FOR JONAS**

The ABB IRC5 controller signal availability depends on the firmware version and installed software options.

**Signals we need to confirm:**

| Signal Category               | What we need | ABB signal name (to be confirmed)       |
| ----------------------------- | ------------ | --------------------------------------- |
| Commanded joint positions     | 6 values     | `rax_1..6` target or similar            |
| Commanded joint velocities    | 6 values     | TBD                                     |
| Commanded joint accelerations | 6 values     | TBD (may need to compute from velocity) |
| Commanded TCP pose            | 6 values     | TBD                                     |
| Actual joint positions        | 6 values     | `rax_1..6` actual or similar            |
| Actual joint velocities       | 6 values     | TBD                                     |
| Motor torque per axis         | 6 values     | `MOC/MotorTorque` or similar            |
| Motor current per axis        | 6 values     | TBD                                     |
| TCP force/torque              | 6 values     | Only if force sensor installed          |
| Motor temperature per axis    | 6 values     | TBD                                     |
| Supply voltage                | 1 value      | TBD                                     |
| Robot mode / status           | varies       | TBD                                     |

---

## KUKA KR10: Signal List

**STATUS: WAITING FOR JONAS**

The KUKA KRC4 controller has system variables accessible through RSI and KRL.

**Signals we need to confirm:**

| Signal Category               | What we need | KUKA system variable (to be confirmed)    |
| ----------------------------- | ------------ | ----------------------------------------- |
| Commanded joint positions     | 6 values     | `$AXIS_CMD[1..6]` or `$AXIS_ACT_CMD`      |
| Commanded joint velocities    | 6 values     | `$VEL_AXIS_CMD[1..6]` or similar          |
| Commanded joint accelerations | 6 values     | TBD (may need to compute)                 |
| Commanded TCP pose            | 6 values     | `$POS_ACT_CMD` or similar                 |
| Actual joint positions        | 6 values     | `$AXIS_ACT[1..6]`                         |
| Actual joint velocities       | 6 values     | `$VEL_AXIS_ACT[1..6]`                     |
| Gear torque per axis          | 6 values     | `$TORQUE_ACT[1..6]` or `$TORQUE_AXIS_ACT` |
| Motor current per axis        | 6 values     | `$MOT_CUR[1..6]` or `$CURR_ACT`           |
| Motor temperature per axis    | 6 values     | `$TEMPERATURE_MOTOR[1..6]`                |
| TCP force/torque              | 6 values     | Only if force sensor installed            |
| Supply voltage                | 1 value      | TBD                                       |
| Robot mode / status           | varies       | `$MODE_OP`, `$PRO_STATE`                  |

---

## End of Day Checklist

- [ ] All fault conditions have been reversed (flange tight, TCP reset, tape removed, correct object, centered grip)
- [ ] All recording files are saved and backed up
- [ ] Episode counts match the schedule above
- [ ] Start/end times are recorded on this sheet
- [ ] Any anomalies or issues are noted below

**Notes:**

---

---

---

---
