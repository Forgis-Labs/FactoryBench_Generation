import pandas as pd
import numpy as np
import random
from typing import Dict, List, Optional

class SyntheticEpisodeGenerator:
    def __init__(self, robot_model="ABB_IRB2600"):
        self.robot_model = robot_model
        # Basic specs for IRB 2600 (approximate if extraction fails)
        self.specs = {
            "max_speed": 3.0, # rad/s
            "max_acc": 5.0,   # rad/s^2
            "temp_limit": 50.0 # deg C
        }
        
    def generate_episode(self, duration_sec=10, hz=100, scenario="normal") -> pd.DataFrame:
        """
        Generates a synthetic episode dataframe.
        Scenario: 'normal', 'temp_anomaly_joint_1', 'velocity_spike_joint_3', 'drift_joint_0'
        """
        n_steps = int(duration_sec * hz)
        timestamps = np.linspace(0, duration_sec * 1000, n_steps).astype(int) # ms
        
        data = {
            "timestamp_ms": timestamps
        }
        
        # 0. Base Trajectory (Sine waves for motion)
        t = np.linspace(0, duration_sec, n_steps)
        
        for axis in range(6):
            # Planned motion
            freq = 0.2 + 0.1 * axis
            amp = 0.5 - 0.05 * axis
            phase = random.uniform(0, 2*np.pi)
            
            pos = amp * np.sin(2 * np.pi * freq * t + phase)
            vel = amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t + phase)
            acc = -amp * (2 * np.pi * freq)**2 * np.sin(2 * np.pi * freq * t + phase)
            
            data[f"setpoint_pos_{axis}"] = pos
            data[f"setpoint_speed_{axis}"] = vel
            data[f"setpoint_acc_{axis}"] = acc
            
            # Feedback (Ideal + Noise)
            noise_pos = np.random.normal(0, 0.001, n_steps)
            data[f"feedback_pos_{axis}"] = pos + noise_pos
            
            noise_vel = np.random.normal(0, 0.01, n_steps)
            data[f"feedback_speed_{axis}"] = vel + noise_vel
            
            # Dynamics (Simplified)
            # Motor current ~ Acceleration + Velocity (friction) + Gravity
            current = 0.5 * abs(acc) + 0.2 * abs(vel) + 0.1 * np.sin(pos) # simplified gravity
            data[f"effort_current_{axis}"] = current + np.random.normal(0, 0.05, n_steps)
            
            # Temperature model: rises with current^2
            # Temp[t] = Temp[t-1] + k1 * current^2 - k2 * (Temp - Ambient)
            temps = np.zeros(n_steps)
            temps[0] = 40.0 + random.uniform(-2, 2)
            ambient = 25.0
            
            for i in range(1, n_steps):
                d_temp = 0.01 * (data[f"effort_current_{axis}"][i]**2) - 0.005 * (temps[i-1] - ambient)
                temps[i] = temps[i-1] + d_temp
                
            data[f"joint_temp_{axis}"] = temps

        # 1. Inject Anomalies based on Scenario
        anomaly_info = {
            "present": False,
            "type": None,
            "joint": None,
            "time_range": None
        }
        
        if scenario != "normal":
            anomaly_info["present"] = True
            start_idx = int(0.3 * n_steps)
            end_idx = int(0.7 * n_steps)
            anomaly_info["time_range"] = (timestamps[start_idx]/1000, timestamps[end_idx]/1000)
            
            if "temp_anomaly" in scenario:
                 # e.g. 'temp_anomaly_joint_1'
                 try:
                    target_joint = int(scenario.split("_")[-1])
                 except: target_joint = 0
                 
                 anomaly_info["type"] = "Overheating"
                 anomaly_info["joint"] = f"joint_{target_joint+1}" # Human readable usually 1-indexed
                 
                 # Add drift to temperature
                 drift = np.linspace(0, 30, end_idx - start_idx)
                 data[f"joint_temp_{target_joint}"][start_idx:end_idx] += drift
                 # Sustain high temp after
                 data[f"joint_temp_{target_joint}"][end_idx:] += 30
                 
            elif "velocity_spike" in scenario:
                 try:
                    target_joint = int(scenario.split("_")[-1])
                 except: target_joint = 2
                 
                 anomaly_info["type"] = "Velocity Spike"
                 anomaly_info["joint"] = f"joint_{target_joint+1}"
                 
                 # Add spike
                 spike_width = 50 # steps
                 mid = (start_idx + end_idx) // 2
                 data[f"feedback_speed_{target_joint}"][mid-spike_width:mid+spike_width] += 5.0 # rad/s burst
                 
            elif "drift" in scenario:
                 try:
                    target_joint = int(scenario.split("_")[-1])
                 except: target_joint = 0
                 
                 anomaly_info["type"] = "Position Drift"
                 anomaly_info["joint"] = f"joint_{target_joint+1}"
                 
                 drift = np.linspace(0, 0.5, n_steps - start_idx)
                 data[f"feedback_pos_{target_joint}"][start_idx:] += drift

        # 2. Fill other columns (Context, Outcome)
        data["main_voltage"] = np.random.normal(400, 2, n_steps)
        data["safety_mode"] = [1]*n_steps # 1=Running
        data["protective_stop_state"] = [0]*n_steps
        
        # Tool / Gripper
        data["setpoint_tcp_0"] = np.zeros(n_steps) # Placeholders
        data["setpoint_tcp_1"] = np.zeros(n_steps)
        data["setpoint_tcp_2"] = np.zeros(n_steps)
        data["gripper_command"] = [0] * n_steps
        
        # Sensors
        data["vibration_0"] = np.random.normal(0, 0.1, n_steps)
        if scenario == "vibration_anomaly":
             data["vibration_0"][start_idx:] += np.random.normal(0, 2.0, n_steps-start_idx)
             anomaly_info["type"] = "Excessive Vibration"
             anomaly_info["joint"] = "Tool"

        df = pd.DataFrame(data)
        return df, anomaly_info

if __name__ == "__main__":
    gen = SyntheticEpisodeGenerator()
    df, meta = gen.generate_episode(scenario="temp_anomaly_joint_1")
    print(f"Generated episode with shape {df.shape}")
    print(f"Metadata: {meta}")
    print(df.head())
