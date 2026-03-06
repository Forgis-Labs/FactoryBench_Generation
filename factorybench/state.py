"""In-memory state management for active benchmark runs."""
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, date
from threading import Lock
from pathlib import Path
import json


@dataclass
class RunProgress:
    """Progress state for an active run."""
    run_id: str
    total_samples: int
    processed_samples: int = 0
    current_cost: float = 0.0
    should_stop: bool = False
    status: str = "running"  # running, stopped, completed, failed
    error: Optional[str] = None
    

class RunStateManager:
    """Thread-safe manager for active run states."""
    
    def __init__(self):
        self._active_runs: Dict[str, RunProgress] = {}
        self._lock = Lock()
    
    def start_run(self, run_id: str, total_samples: int) -> RunProgress:
        """Initialize progress tracking for a new run."""
        with self._lock:
            progress = RunProgress(run_id=run_id, total_samples=total_samples)
            self._active_runs[run_id] = progress
            return progress
    
    def get_progress(self, run_id: str) -> Optional[RunProgress]:
        """Get current progress for a run."""
        with self._lock:
            return self._active_runs.get(run_id)
    
    def update_progress(self, run_id: str, processed_samples: int, current_cost: float):
        """Update run progress."""
        with self._lock:
            if run_id in self._active_runs:
                self._active_runs[run_id].processed_samples = processed_samples
                self._active_runs[run_id].current_cost = current_cost
    
    def request_stop(self, run_id: str) -> bool:
        """Request a run to stop. Returns True if run exists and is running."""
        with self._lock:
            if run_id in self._active_runs:
                progress = self._active_runs[run_id]
                if progress.status == "running":
                    progress.should_stop = True
                    return True
            return False
    
    def should_stop(self, run_id: str) -> bool:
        """Check if run should stop."""
        with self._lock:
            if run_id in self._active_runs:
                return self._active_runs[run_id].should_stop
            return False
    
    def complete_run(self, run_id: str, status: str = "completed", error: Optional[str] = None):
        """Mark run as complete."""
        with self._lock:
            if run_id in self._active_runs:
                self._active_runs[run_id].status = status
                self._active_runs[run_id].error = error
                # Note: Daily costs are now calculated from run files, not stored in memory
    
    def cleanup_run(self, run_id: str):
        """Remove run from active tracking (after completion)."""
        with self._lock:
            self._active_runs.pop(run_id, None)
    
    def get_daily_cost(self, day: Optional[date] = None) -> float:
        """Get total cost for a specific day by reading from run files (default: today)."""
        if day is None:
            day = date.today()
        
        # Import here to avoid circular dependency
        from .config import RUN_DIR
        
        total_cost = 0.0
        run_dir = Path(RUN_DIR)
        
        if not run_dir.exists():
            return 0.0
        
        # Read all run files and sum costs for the specified day
        for run_file in run_dir.glob("*.json"):
            try:
                with run_file.open("r", encoding="utf-8") as f:
                    run_data = json.load(f)
                
                # Parse the run's date from started_at
                started_at = run_data.get("started_at")
                if started_at:
                    run_date = datetime.fromisoformat(started_at).date()
                    if run_date == day:
                        # Add this run's cost from aggregate
                        aggregate = run_data.get("aggregate", {})
                        cost = aggregate.get("cost_total", 0.0)
                        
                        # If no aggregate cost yet, check if it's an active run
                        if cost == 0.0:
                            run_id = run_data.get("run_id")
                            if run_id and run_id in self._active_runs:
                                cost = self._active_runs[run_id].current_cost
                        
                        total_cost += cost
            except (json.JSONDecodeError, ValueError, KeyError):
                # Skip malformed files
                continue
        
        return total_cost
    
    def get_active_runs(self) -> Dict[str, RunProgress]:
        """Get all active runs."""
        with self._lock:
            return dict(self._active_runs)


# Global singleton instance
run_state = RunStateManager()
