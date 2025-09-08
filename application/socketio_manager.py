"""
Flask-SocketIO manager for real-time communication
Handles job events, progress updates, and violation notifications
"""
import logging
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask import request
from typing import Dict, Any
import time

class SocketIOManager:
    """Manages WebSocket connections and real-time events"""

    socketio = SocketIO(cors_allowed_origins="*", async_mode="eventlet")
    
    def __init__(self, app=None):
        self.socketio = None
        self.connected_clients: Dict[str, set] = {}  # job_id -> set of session_ids
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize SocketIO with Flask app"""
        self.socketio = SocketIO(
            app,
            cors_allowed_origins="*",
            async_mode='threading',
            logger=True,
            engineio_logger=False
        )
        
        # Register event handlers
        self.socketio.on_event('connect', self.handle_connect)
        self.socketio.on_event('disconnect', self.handle_disconnect)
        self.socketio.on_event('join_job', self.handle_join_job)
        self.socketio.on_event('leave_job', self.handle_leave_job)
        
        logging.info("SocketIO initialized")
    
    def handle_connect(self):
        """Handle client connection"""
        session_id = request.sid
        logging.info(f"Client connected: {session_id}")
        emit('connected', {'status': 'connected', 'session_id': session_id})
    
    def handle_disconnect(self):
        """Handle client disconnection"""
        session_id = request.sid
        logging.info(f"Client disconnected: {session_id}")
        
        # Remove from all job rooms
        for job_id, clients in self.connected_clients.items():
            clients.discard(session_id)
    
    def handle_join_job(self, data):
        """Handle client joining a job room"""
        job_id = data.get('job_id')
        session_id = request.sid
        
        if not job_id:
            emit('error', {'message': 'job_id required'})
            return
            
        join_room(job_id)
        
        if job_id not in self.connected_clients:
            self.connected_clients[job_id] = set()
        self.connected_clients[job_id].add(session_id)
        
        logging.info(f"Client {session_id} joined job {job_id}")
        emit('joined_job', {'job_id': job_id, 'status': 'joined'})
    
    def handle_leave_job(self, data):
        """Handle client leaving a job room"""
        job_id = data.get('job_id')
        session_id = request.sid
        
        if not job_id:
            return
            
        leave_room(job_id)
        
        if job_id in self.connected_clients:
            self.connected_clients[job_id].discard(session_id)
            
        logging.info(f"Client {session_id} left job {job_id}")
        emit('left_job', {'job_id': job_id, 'status': 'left'})
    
    def emit_job_started(self, job_id: str):
        """Emit job started event"""
        if not self.socketio:
            return
            
        self.socketio.emit('job_started', {
            'job_id': job_id,
            'timestamp': time.time()
        }, room=job_id)
        
        logging.info(f"Emitted job_started for {job_id}")
    
    def emit_job_stats(self, job_id: str, stats: Dict[str, Any]):
        """Emit job statistics update"""
        if not self.socketio:
            logging.warning(f"SocketIO not initialized, cannot emit job_stats for {job_id}")
            return
            
        event_data = {
            'job_id': job_id,
            'timestamp': time.time(),
            **stats
        }
        
        self.socketio.emit('job_stats', event_data, room=job_id)
        
        # Enhanced logging for debugging
        connected_count = len(self.connected_clients.get(job_id, set()))
        logging.info(f"Emitted job_stats for {job_id} to {connected_count} clients: progress={stats.get('progress', 0)}%, frames={stats.get('frames', 0)}, violations={stats.get('violations', 0)}")
    
    def emit_job_preview(self, job_id: str, jpeg_b64: str):
        """Emit job preview frame"""
        if not self.socketio:
            return
            
        self.socketio.emit('job_preview', {
            'job_id': job_id,
            'jpeg_b64': jpeg_b64,
            'timestamp': time.time()
        }, room=job_id)
    
    def emit_violation(self, job_id: str, violation_data: Dict[str, Any]):
        """Emit violation detection event"""
        if not self.socketio:
            logging.warning(f"SocketIO not initialized, cannot emit violation for {job_id}")
            return
            
        event_data = {
            'job_id': job_id,
            'timestamp': time.time(),
            **violation_data
        }
        
        self.socketio.emit('violation', event_data, room=job_id)
        
        connected_count = len(self.connected_clients.get(job_id, set()))
        logging.info(f"Emitted violation for {job_id} to {connected_count} clients: plate={violation_data.get('plate', 'Unknown')}, confidence={violation_data.get('confidence', 'N/A')}")
    
    def emit_job_completed(self, job_id: str, totals: Dict[str, Any]):
        """Emit job completion event"""
        if not self.socketio:
            logging.warning(f"SocketIO not initialized, cannot emit job_completed for {job_id}")
            return
            
        event_data = {
            'job_id': job_id,
            'timestamp': time.time(),
            **totals
        }
        
        self.socketio.emit('job_completed', event_data, room=job_id)
        
        connected_count = len(self.connected_clients.get(job_id, set()))
        logging.info(f"Emitted job_completed for {job_id} to {connected_count} clients: total_frames={totals.get('frames', 0)}, violations={totals.get('violations', 0)}")
    
    def emit_job_failed(self, job_id: str, error: str):
        """Emit job failure event"""
        if not self.socketio:
            return
            
        self.socketio.emit('job_failed', {
            'job_id': job_id,
            'error': error,
            'timestamp': time.time()
        }, room=job_id)
        
        logging.error(f"Emitted job_failed for {job_id}: {error}")

# Global SocketIO manager instance
socketio_manager = SocketIOManager()
