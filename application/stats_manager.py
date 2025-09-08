"""
Real-time statistics manager for job processing
Handles FPS calculation, violation tracking, and persistence
"""
import time
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import deque
import threading

@dataclass
class JobStats:
    job_id: str
    frames: int = 0
    fps: float = 0.0
    violations: int = 0
    progress: float = 0.0
    timestamp: str = ""
    
    def to_dict(self):
        return asdict(self)

@dataclass 
class ViolationEvent:
    job_id: str
    plate: str
    confidence: float = 0.0
    bbox: Optional[List[float]] = None
    frame_thumb: Optional[str] = None
    timestamp: str = ""
    frame_number: int = 0
    
    def to_dict(self):
        return asdict(self)

class StatsManager:
    """Thread-safe statistics manager with rolling FPS calculation"""
    
    def __init__(self):
        self.stats: Dict[str, JobStats] = {}
        self.violations: Dict[str, List[ViolationEvent]] = {}
        self.fps_windows: Dict[str, deque] = {}
        self.lock = threading.Lock()
        self.fps_window_size = 10  # Rolling window for FPS calculation
        
    def create_job_stats(self, job_id: str) -> JobStats:
        """Initialize stats for a new job"""
        with self.lock:
            stats = JobStats(
                job_id=job_id,
                timestamp=time.strftime('%Y-%m-%dT%H:%M:%S')
            )
            self.stats[job_id] = stats
            self.violations[job_id] = []
            self.fps_windows[job_id] = deque(maxlen=self.fps_window_size)
            logging.info(f"Created stats for job {job_id}")
            return stats
    
    def update_frame_processed(self, job_id: str, total_frames: int = None):
        """Update frame count and calculate rolling FPS"""
        with self.lock:
            if job_id not in self.stats:
                return
                
            stats = self.stats[job_id]
            stats.frames += 1
            
            # Update FPS calculation
            current_time = time.time()
            fps_window = self.fps_windows[job_id]
            fps_window.append(current_time)
            
            if len(fps_window) >= 2:
                time_diff = fps_window[-1] - fps_window[0]
                if time_diff > 0:
                    stats.fps = (len(fps_window) - 1) / time_diff
            
            # Update progress if total frames known
            if total_frames and total_frames > 0:
                stats.progress = min((stats.frames / total_frames) * 100, 100.0)
            
            stats.timestamp = time.strftime('%Y-%m-%dT%H:%M:%S')
            
    def add_violation(self, job_id: str, violation: ViolationEvent):
        """Add a new violation event"""
        with self.lock:
            if job_id not in self.violations:
                self.violations[job_id] = []
                
            violation.timestamp = time.strftime('%Y-%m-%dT%H:%M:%S')
            self.violations[job_id].append(violation)
            
            # Update violation count in stats
            if job_id in self.stats:
                self.stats[job_id].violations = len(self.violations[job_id])
                self.stats[job_id].timestamp = violation.timestamp
                
            # Keep only last 50 violations per job
            if len(self.violations[job_id]) > 50:
                self.violations[job_id] = self.violations[job_id][-50:]
                
            logging.info(f"Added violation for job {job_id}: {violation.plate}")
    
    def get_stats(self, job_id: str) -> Optional[JobStats]:
        """Get current stats for a job"""
        with self.lock:
            return self.stats.get(job_id)
    
    def get_violations(self, job_id: str, limit: int = 20) -> List[ViolationEvent]:
        """Get recent violations for a job"""
        with self.lock:
            violations = self.violations.get(job_id, [])
            return violations[-limit:] if violations else []
    
    def cleanup_job(self, job_id: str):
        """Clean up stats for completed job"""
        with self.lock:
            self.stats.pop(job_id, None)
            self.violations.pop(job_id, None)
            self.fps_windows.pop(job_id, None)
            logging.info(f"Cleaned up stats for job {job_id}")

# Global stats manager instance
stats_manager = StatsManager()
