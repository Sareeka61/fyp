import uuid
import threading
import time
import logging
from queue import Queue
from enum import Enum
from typing import Dict, Any, Optional
import cv2
import os

class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

class Job:
    """Represents a video processing job with background processing capabilities"""

    def __init__(self, video_path: str, job_id: Optional[str] = None):
        self.job_id = job_id or str(uuid.uuid4())
        self.video_path = video_path
        self.status = JobStatus.PENDING
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.error_message: Optional[str] = None

        # Queues for real-time processing
        self.frame_queue = Queue(maxsize=30)  # Buffer for processed frames
        self.event_queue = Queue()  # Queue for events (violations, progress, etc.)

        # Processing thread
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Job metadata
        self.total_frames = 0
        self.processed_frames = 0
        self.violations_count = 0
        self.progress = 0.0
        self.results = []  # Store plate detection results
        self.frame_snapshots = []  # Store frame snapshots for display

        # Output paths
        from application.config import UPLOAD_FOLDER_PATH
        self.output_dir = os.path.join(UPLOAD_FOLDER_PATH, f"job_{self.job_id}")
        os.makedirs(self.output_dir, exist_ok=True)

        logging.info(f"Created job {self.job_id} for video: {video_path}")

    def start_processing(self, processor_func, *args, **kwargs):
        """Start background processing of the job"""
        if self.status != JobStatus.PENDING:
            logging.warning(f"Cannot start job {self.job_id}: status is {self.status.value}")
            return

        self.status = JobStatus.PROCESSING
        self.started_at = time.time()

        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._process_video,
            args=(processor_func, args, kwargs),
            daemon=True
        )
        self.processing_thread.start()

        logging.info(f"Started processing job {self.job_id}")

    def _process_video(self, processor_func, args, kwargs):
        """Background video processing function"""
        try:
            # Call the processor function with job context
            result = processor_func(self, *args, **kwargs)

            if not self.stop_event.is_set():
                self.status = JobStatus.COMPLETED
                self.completed_at = time.time()
                self.progress = 100.0
                logging.info(f"Job {self.job_id} completed successfully")

                # Send completion event
                self.event_queue.put({
                    'type': 'completion',
                    'job_id': self.job_id,
                    'total_frames': self.total_frames,
                    'violations_count': self.violations_count,
                    'completed_at': self.completed_at
                })

        except Exception as e:
            self.status = JobStatus.ERROR
            self.error_message = str(e)
            self.completed_at = time.time()
            logging.error(f"Job {self.job_id} failed: {e}")

            # Send error event
            self.event_queue.put({
                'type': 'error',
                'job_id': self.job_id,
                'error': str(e),
                'completed_at': self.completed_at
            })

    def stop_processing(self):
        """Stop the job processing"""
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_event.set()
            self.processing_thread.join(timeout=5.0)

        self.status = JobStatus.CANCELLED
        self.completed_at = time.time()
        logging.info(f"Job {self.job_id} cancelled")

    def update_progress(self, processed_frames: int, total_frames: int):
        """Update job progress"""
        self.processed_frames = processed_frames
        self.total_frames = total_frames
        self.progress = min((processed_frames / total_frames) * 100 if total_frames > 0 else 0, 100.0)

        # Send progress event
        self.event_queue.put({
            'type': 'progress',
            'job_id': self.job_id,
            'processed_frames': processed_frames,
            'total_frames': total_frames,
            'progress': self.progress
        })

    def add_violation(self, violation_data: Dict[str, Any]):
        """Add a violation event"""
        self.violations_count += 1

        # Send violation event
        event_data = {
            'type': 'violation',
            'job_id': self.job_id,
            'violation_count': self.violations_count,
            **violation_data
        }
        self.event_queue.put(event_data)

    def add_frame(self, frame_data: Dict[str, Any]):
        """Add a processed frame to the queue"""
        try:
            self.frame_queue.put(frame_data, timeout=1.0)
        except:
            # Queue full, remove oldest frame
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put(frame_data)
            except:
                pass  # Queue empty, just skip

    def get_status_dict(self) -> Dict[str, Any]:
        """Get job status as dictionary"""
        return {
            'job_id': self.job_id,
            'status': self.status.value,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'progress': self.progress,
            'total_frames': self.total_frames,
            'processed_frames': self.processed_frames,
            'violations_count': self.violations_count,
            'error_message': self.error_message,
            'video_path': self.video_path
        }

    def cleanup(self):
        """Clean up job resources"""
        self.stop_processing()

        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break

        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except:
                break

        logging.info(f"Cleaned up job {self.job_id}")

class JobManager:
    """Manages multiple video processing jobs"""

    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        from application.config import MAX_CONCURRENT_JOBS
        self.max_concurrent_jobs = MAX_CONCURRENT_JOBS  # Limit concurrent processing
        self.active_jobs = 0
        self.lock = threading.Lock()

    def create_job(self, video_path: str) -> Job:
        """Create a new job"""
        job = Job(video_path)
        with self.lock:
            self.jobs[job.job_id] = job
        logging.info(f"Created job {job.job_id}")
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self.jobs.get(job_id)

    def start_job(self, job_id: str, processor_func, *args, **kwargs):
        """Start a job if under concurrent limit"""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        with self.lock:
            if self.active_jobs >= self.max_concurrent_jobs:
                raise RuntimeError("Maximum concurrent jobs reached")

            self.active_jobs += 1

        try:
            job.start_processing(processor_func, *args, **kwargs)
        except Exception as e:
            with self.lock:
                self.active_jobs -= 1
            raise e

    def stop_job(self, job_id: str):
        """Stop a job"""
        job = self.get_job(job_id)
        if job:
            job.stop_processing()
            with self.lock:
                if job.status == JobStatus.PROCESSING:
                    self.active_jobs -= 1

    def cleanup_completed_jobs(self, max_age_hours: int = 24):
        """Clean up old completed jobs"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        jobs_to_remove = []
        with self.lock:
            for job_id, job in self.jobs.items():
                if (job.status in [JobStatus.COMPLETED, JobStatus.ERROR, JobStatus.CANCELLED] and
                    job.completed_at and (current_time - job.completed_at) > max_age_seconds):
                    jobs_to_remove.append(job_id)

            for job_id in jobs_to_remove:
                job = self.jobs.pop(job_id)
                job.cleanup()
                logging.info(f"Cleaned up old job {job_id}")

    def get_active_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Get all active jobs"""
        return {
            job_id: job.get_status_dict()
            for job_id, job in self.jobs.items()
            if job.status in [JobStatus.PENDING, JobStatus.PROCESSING]
        }

# Global job manager instance
job_manager = JobManager()
