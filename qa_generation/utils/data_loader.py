import pandas as pd
import numpy as np
import random
import os
from typing import Tuple, Dict, Any

class ABBDataLoader:
    def __init__(self, csv_path="datasets/ABB.csv"):
        self.csv_path = csv_path
        self.df = None
        self._load_data()
        
    def _load_data(self):
        if not os.path.exists(self.csv_path):
            # Fallback for running from different CWD
            if os.path.exists(os.path.join("..", self.csv_path)):
                self.csv_path = os.path.join("..", self.csv_path)
            elif os.path.exists(os.path.join("FactoryBench", self.csv_path)):
                 self.csv_path = os.path.join("FactoryBench", self.csv_path)
                 
        try:
            self.df = pd.read_csv(self.csv_path)
            # Ensure timestamp is sorted just in case
            if "timestamp_ms" in self.df.columns:
                self.df = self.df.sort_values("timestamp_ms")
            print(f"Loaded {len(self.df)} samples from {self.csv_path}")
        except Exception as e:
            print(f"Error loading {self.csv_path}: {e}")
            self.df = pd.DataFrame() # Empty fallback

    def get_episode(self, duration_sec=10, scenario="normal") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Returns a slice of real data modified to fit the scenario.
        """
        if self.df is None or self.df.empty:
            raise ValueError("No data loaded.")
            
        # 1. Get a random slice of real data
        # Assuming 100Hz approx, but we use timestamp
        duration_ms = duration_sec * 1000
        total_duration = self.df["timestamp_ms"].max() - self.df["timestamp_ms"].min()
        
        if total_duration < duration_ms:
            # If dataset is too short, return whole thing repeated or just whole thing
            base_slice = self.df.copy()
        else:
            # Random start time
            max_start = self.df["timestamp_ms"].max() - duration_ms
            start_ms = random.uniform(self.df["timestamp_ms"].min(), max_start)
            end_ms = start_ms + duration_ms
            
            # Extract slice
            base_slice = self.df[(self.df["timestamp_ms"] >= start_ms) & (self.df["timestamp_ms"] <= end_ms)].copy()
            
            # Reset timestamp to start at 0 for the episode
            base_slice["timestamp_ms"] = base_slice["timestamp_ms"] - base_slice["timestamp_ms"].iloc[0]

        # 2. Inject Anomalies (Hybrid Approach)
        meta = {
            "present": False,
            "type": None,
            "joint": None,
            "time_range": None
        }
        
        n_samples = len(base_slice)
        if n_samples == 0: return base_slice, meta
        
        if scenario != "normal":
            meta["present"] = True
            
            # Define anomaly window (middle 30%)
            start_idx = int(0.35 * n_samples)
            end_idx = int(0.65 * n_samples)
            meta["time_range"] = (
                base_slice.iloc[start_idx]["timestamp_ms"]/1000, 
                base_slice.iloc[end_idx]["timestamp_ms"]/1000
            )

            if "temp_anomaly" in scenario:
                    # e.g. 'temp_anomaly_joint_1'
                    try:
                        target_joint = int(scenario.split("_")[-1])
                    except: 
                        target_joint = 0
                    
                    j_col = f"joint_temp_{target_joint}"
                    if j_col in base_slice.columns:
                        meta["type"] = "Overheating"
                        meta["joint"] = f"joint_{target_joint+1}"
                        
                        # Add drift: +0.5C per sample in window until +15C
                        drift = np.linspace(0, 15, end_idx - start_idx)
                        base_slice.iloc[start_idx:end_idx, base_slice.columns.get_loc(j_col)] += drift
                        # Sustain
                        base_slice.iloc[end_idx:, base_slice.columns.get_loc(j_col)] += 15
                    
            elif "velocity_spike" in scenario:
                    try:
                        target_joint = int(scenario.split("_")[-1])
                    except: 
                        target_joint = 2
                    
                    j_col = f"feedback_speed_{target_joint}"
                    if j_col in base_slice.columns:
                        meta["type"] = "Velocity Spike"
                        meta["joint"] = f"joint_{target_joint+1}"
                        
                        # Simple spike
                        spike_val = 5.0 # rad/s
                        mid = (start_idx + end_idx) // 2
                        width = max(1, int(n_samples * 0.05))
                        base_slice.iloc[mid-width:mid+width, base_slice.columns.get_loc(j_col)] += spike_val
                        
            elif "drift" in scenario:
                    # Sensor drift
                    try:
                        target_joint = int(scenario.split("_")[-1])
                    except: 
                        target_joint = 0
                    
                    j_col = f"feedback_pos_{target_joint}"
                    if j_col in base_slice.columns:
                        meta["type"] = "Position Drift"
                        meta["joint"] = f"joint_{target_joint+1}"
                        
                        drift = np.linspace(0, 0.2, n_samples - start_idx)
                        base_slice.iloc[start_idx:, base_slice.columns.get_loc(j_col)] += drift

        return base_slice, meta

if __name__ == "__main__":
    # Test
    loader = ABBDataLoader()
    if loader.df is not None:
        df, meta = loader.get_episode(scenario="temp_anomaly_joint_1")
        print(f"Slice shape: {df.shape}")
        print(f"Meta: {meta}")
        print(df[["timestamp_ms", "joint_temp_1"]].head())
