## UR3e Time-Series Schema

UR3e episodes are stored as CSV files (e.g. `ur3e_episode_001.csv`). Each row is one time sample.

- **Robot**: UR3e (6 revolute joints)
- **Joint index convention**: `{axis} ∈ {0,1,2,3,4,5}` corresponds to joints 0–5 in the UR RTDE interface.
- **Time index**: `timestamp_ms` (integer, milliseconds since episode start)

### Signal Groups

- **INTENT**: motion commands from the trajectory planner/controller.
- **CONTEXT**: dynamic operating conditions measured during the episode.
- **OUTCOME**: measured motion and external sensor feedback.

---

## Columns

### 1. Time

| Column         | Type    | Unit | Group  | Description                                  |
|----------------|---------|------|--------|----------------------------------------------|
| `timestamp_ms` | int64   | ms   | index  | Time since episode start in milliseconds.    |

---

### 2. INTENT – commanded motion

#### 2.1 Joint-level commands

| Pattern                   | Example            | Type   | Unit   | Source                 | Description                                                |
|---------------------------|--------------------|--------|--------|------------------------|------------------------------------------------------------|
| `setpoint_pos_{axis}`     | `setpoint_pos_0`   | float  | rad    | RTDE `target_q`        | Commanded joint angle for joint `{axis}`.                 |
| `setpoint_speed_{axis}`   | `setpoint_speed_0` | float  | rad/s  | RTDE `target_qd`       | Commanded joint angular velocity for joint `{axis}`.      |
| `setpoint_acc_{axis}`     | `setpoint_acc_0`   | float  | rad/s² | derived from `target_qd` | Commanded joint angular acceleration for joint `{axis}`. |

#### 2.2 TCP (tool) pose commands

`setpoint_tcp_*` encodes the commanded pose of the tool center point in the base frame:

| Column            | Type   | Unit | Description                                  |
|-------------------|--------|------|----------------------------------------------|
| `setpoint_tcp_0`  | float  | m    | Commanded X position of TCP in base frame.   |
| `setpoint_tcp_1`  | float  | m    | Commanded Y position of TCP in base frame.   |
| `setpoint_tcp_2`  | float  | m    | Commanded Z position of TCP in base frame.   |
| `setpoint_tcp_3`  | float  | rad  | Commanded rotation around X (Rx).            |
| `setpoint_tcp_4`  | float  | rad  | Commanded rotation around Y (Ry).            |
| `setpoint_tcp_5`  | float  | rad  | Commanded rotation around Z (Rz).            |

#### 2.3 Gripper command

| Column            | Type  | Unit  | Description                                               |
|-------------------|-------|-------|-----------------------------------------------------------|
| `gripper_command` | int   | {0,1} | Digital command to end-effector: `0=open`, `1=close`.    |

---

### 3. CONTEXT – dynamic operating conditions

| Column                 | Type  | Unit | Source                      | Description                                        |
|------------------------|-------|------|-----------------------------|----------------------------------------------------|
| `joint_temp_{axis}`    | float | °C   | RTDE `joint_temperatures`   | Estimated motor temperature of joint `{axis}`.     |
| `main_voltage`         | float | V    | RTDE `actual_main_voltage`  | DC bus / main supply voltage of controller.        |
| `safety_mode`          | int   | code | RTDE `safety_status_bits`   | Encoded safety state. `0` = normal; others = safe/reduced/stop modes depending on mapping. |

Note: exact safety mode code mapping should be documented separately if needed.

---

### 4. OUTCOME – measured motion and sensor feedback

#### 4.1 Joint feedback

| Pattern                   | Example              | Type  | Unit  | Source               | Description                                           |
|---------------------------|----------------------|-------|-------|----------------------|-------------------------------------------------------|
| `feedback_pos_{axis}`     | `feedback_pos_0`     | float | rad   | RTDE `actual_q`      | Measured joint angle for joint `{axis}`.             |
| `feedback_speed_{axis}`   | `feedback_speed_0`   | float | rad/s | RTDE `actual_qd`     | Measured joint angular velocity for joint `{axis}`.  |
| `effort_current_{axis}`   | `effort_current_0`   | float | A     | RTDE `actual_current`| Joint motor current for joint `{axis}`.              |

#### 4.2 System protection state

| Column                 | Type  | Unit    | Source              | Description                                        |
|------------------------|-------|---------|---------------------|----------------------------------------------------|
| `protective_stop_state`| int   | {0,1}   | RTDE `protective_stop` | `0` = no protective stop, `1` = protective stop active. |

#### 4.3 Vibration (external accelerometer)

| Column        | Type  | Unit | Description                                                       |
|---------------|-------|------|-------------------------------------------------------------------|
| `vibration_0` | float | g    | Acceleration along X axis at the tool (external accelerometer).  |
| `vibration_1` | float | g    | Acceleration along Y axis at the tool.                           |
| `vibration_2` | float | g    | Acceleration along Z axis at the tool.                           |

#### 4.4 Acoustic emission (external microphone)

| Column       | Type  | Unit | Description                                      |
|--------------|-------|------|--------------------------------------------------|
| `acoustic_0` | float | dB   | Sound level from contact microphone near tool.  |

#### 4.5 Contact forces and torques

Indexing convention for force/torque vectors:

- `*_0` → Fx (N)  
- `*_1` → Fy (N)  
- `*_2` → Fz (N)  
- `*_3` → Tx (Nm)  
- `*_4` → Ty (Nm)  
- `*_5` → Tz (Nm)  

There are two versions:

| Pattern                    | Example                 | Type  | Unit (per index) | Source                    | Description                                          |
|---------------------------|-------------------------|-------|------------------|---------------------------|------------------------------------------------------|
| `est_contact_force_{i}`   | `est_contact_force_0`   | float | N / Nm           | RTDE `actual_TCP_force`   | Controller-estimated TCP wrench at the tool.        |
| `true_force_{i}`          | `true_force_0`          | float | N / Nm           | external 6-axis FT sensor | Measured TCP wrench at the tool (ground truth).     |

- For indices 0–2: values are in newtons (N).  
- For indices 3–5: values are in newton-meters (Nm).

---

## Episode-level Metadata (separate from CSV)

Per-episode fields such as `episode_id`, `machine_id`, `payload_mass_kg`, `material_id`,
`fault_label`, `task_description`, and `video_path` should be stored in a separate metadata
file or table, not repeated per row.