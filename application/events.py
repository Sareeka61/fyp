import json
import logging
import time
from flask import Response
from typing import Generator
import queue
from application.job_manager import JobStatus

def generate_sse_events(job):
    """
    Generate Server-Sent Events for real-time job updates
    Yields SSE formatted events for progress, violations, and completion
    """
    logging.info(f"Starting SSE stream for job {job.job_id}")

    # Send initial job status
    yield f"data: {json.dumps({'type': 'status', 'status': job.status.value, 'progress': job.progress})}\n\n"

    while not job.stop_event.is_set() and job.status in [JobStatus.PROCESSING, JobStatus.PENDING]:
        try:
            # Get event from queue with timeout
            event = job.event_queue.get(timeout=1.0)

            if event:
                # Format as SSE
                sse_data = json.dumps(event)
                yield f"data: {sse_data}\n\n"

        except queue.Empty:
            # Send heartbeat to keep connection alive
            yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': time.time()})}\n\n"
            continue
        except Exception as e:
            logging.error(f"Error in SSE stream for job {job.job_id}: {e}")
            break

    # Send final completion event
    if job.status == 'completed':
        yield f"data: {json.dumps({'type': 'completed', 'job_id': job.job_id, 'total_violations': job.violations_count})}\n\n"
    elif job.status == 'error':
        yield f"data: {json.dumps({'type': 'error', 'job_id': job.job_id, 'error': job.error_message})}\n\n"

    logging.info(f"Ended SSE stream for job {job.job_id}")

def create_sse_response(job):
    """
    Create Flask Response for Server-Sent Events
    """
    return Response(
        generate_sse_events(job),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Cache-Control'
        }
    )

def send_job_event(job, event_type: str, data: dict = None):
    """
    Send a custom event to the job's event queue
    """
    if data is None:
        data = {}

    event = {
        'type': event_type,
        'timestamp': time.time(),
        **data
    }

    try:
        job.event_queue.put(event, timeout=1.0)
    except queue.Full:
        logging.warning(f"Event queue full for job {job.job_id}, dropping event: {event_type}")

def send_progress_update(job, processed_frames: int, total_frames: int):
    """
    Send progress update event
    """
    send_job_event(job, 'progress', {
        'processed_frames': processed_frames,
        'total_frames': total_frames,
        'progress': (processed_frames / total_frames) * 100 if total_frames > 0 else 0
    })

def send_violation_event(job, violation_data: dict):
    """
    Send violation detection event
    """
    send_job_event(job, 'violation', violation_data)

def send_completion_event(job):
    """
    Send job completion event
    """
    send_job_event(job, 'completion', {
        'job_id': job.job_id,
        'total_frames': job.total_frames,
        'violations_count': job.violations_count,
        'completed_at': time.time()
    })

def send_error_event(job, error_message: str):
    """
    Send job error event
    """
    send_job_event(job, 'error', {
        'job_id': job.job_id,
        'error': error_message,
        'error_at': time.time()
    })

def send_status_update(job):
    """
    Send job status update event
    """
    send_job_event(job, 'status', {
        'status': job.status.value,
        'progress': job.progress
    })


