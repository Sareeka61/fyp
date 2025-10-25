import flask
from flask import Flask, request, jsonify, Response, render_template, send_from_directory
from flask_cors import CORS
import os
import logging
import time
from datetime import datetime
import sys
import json
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from application import config
from application.image_processing import process_file
from application.enhanced_image_processing import process_video_for_job_enhanced as process_video_for_job
from application.job_manager import job_manager
from application.streaming import create_mjpeg_response
from application.events import create_sse_response
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s'
)

# Ensure upload folder exists
try:
    os.makedirs(config.UPLOAD_FOLDER_PATH, exist_ok=True)
    logging.info(f"Upload folder ready: {config.UPLOAD_FOLDER_PATH}")
except OSError as e:
    logging.error(f"Could not create upload folder '{config.UPLOAD_FOLDER_PATH}': {e}", exc_info=True)

# Global model variables - will be loaded synchronously at startup
plate_detection_model = None
char_seg_model = None
char_recog_model = None
device = "cpu"
ocr_font_path = None
models_loaded = False

def load_models_synchronously():
    """Load models synchronously at startup."""
    global plate_detection_model, char_seg_model, char_recog_model, device, ocr_font_path, models_loaded
    try:
        from application.model_loader import load_models
        logging.info("Starting synchronous model loading...")
        plate_detection_model, char_seg_model, char_recog_model, device, ocr_font_path = load_models()
        models_loaded = all([plate_detection_model, char_seg_model, char_recog_model])
        if models_loaded:
            logging.info("All models loaded successfully.")
        else:
            logging.error("Some models failed to load. Application cannot start without models.")
            raise RuntimeError("Model loading failed - cannot start application")
    except Exception as e:
        logging.error(f"Model loading failed: {e}", exc_info=True)
        models_loaded = False
        raise

app = Flask(__name__)
# CORS for everything under /api (and streams/SSE are same-origin via Vite proxy)
CORS(app, resources={r"/*": {"origins": "*", "supports_credentials": True}})

app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER_PATH
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.secret_key = config.FLASK_SECRET_KEY

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = config.SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = config.SQLALCHEMY_TRACK_MODIFICATIONS

from flask_login import LoginManager, login_required, current_user

from flask_migrate import Migrate
from application.user_model import db

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize Flask-Migrate
migrate = Migrate(app, db)

# Add rate limiting for authentication endpoints
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

@login_manager.unauthorized_handler
def unauthorized_callback():
    if flask.request.path.startswith('/api/'):
        return flask.jsonify({'error': 'Unauthorized'}), 401
    else:
        return flask.redirect(login_manager.login_view)

# -------------------- Health & Root --------------------

@app.route('/', methods=['GET'])
def root():
    """Root endpoint that returns API status."""
    return jsonify({
        'status': 'online',
        'models_loaded': models_loaded,
        'version': '1.0'
    })

@app.route('/api/health', methods=['GET'])
def api_health():
    """Simple health check for frontend proxy."""
    return jsonify({'ok': True, 'models_loaded': models_loaded}), 200

@app.route('/api/welcome', methods=['GET'])
def welcome():
    """Returns a welcome message and logs the request."""
    logging.info(f"Request received: {request.method} {request.path}")
    return jsonify({'message': 'Welcome to the Flask API Service!'})

# -------------------- Static uploads (if needed by reports/previews) --------------------

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# -------------------- Upload --------------------

@app.route('/api/upload', methods=['POST'])
@login_required
def upload_file_route():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate extension
        from application import config
        allowed_exts = config.ALLOWED_EXTENSIONS
        ext = '.' + file.filename.rsplit('.', 1)[-1].lower()
        if ext not in allowed_exts:
            return jsonify({
                'error': f'Invalid file type {ext}, allowed: {", ".join(sorted(allowed_exts))}'
            }), 400

        # Ensure upload folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # Save file
        safe_filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(file_path)

        # Create job
        job = job_manager.create_job(file_path)

        # Save AnalysisJob to database
        from application.user_model import AnalysisJob
        analysis_job = AnalysisJob(
            job_id=job.job_id,
            filename=safe_filename,
            file_path=file_path,
            status='pending'
        )

        # Associate with user if logged in
        analysis_job.user_id = current_user.id

        db.session.add(analysis_job)
        db.session.commit()

        return jsonify({
            'job_id': job.job_id,
            'type': 'video',   # if you also support images, detect mimetype here
            'preview_url': f'/api/jobs/{job.job_id}/stream.mjpg',
            'message': 'Video uploaded successfully'
        }), 201

    except RequestEntityTooLarge:
        return jsonify({
            'error': 'File too large',
            'limit_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
        }), 413
    except Exception as e:
        logging.error('Upload failed: %s', e, exc_info=True)
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500
# -------------------- Job status / control --------------------

@app.route('/api/process/status/<job_id>', methods=['GET'])
def get_process_status(job_id):
    """Get processing status for a job"""
    logging.info(f"INFO: /api/process/status/{job_id} called")

    # First try to get from database
    from application.user_model import AnalysisJob
    analysis_job = AnalysisJob.query.filter_by(job_id=job_id).first()
    if analysis_job:
        return jsonify({
            'job_id': analysis_job.job_id,
            'status': analysis_job.status,
            'created_at': analysis_job.created_at.isoformat() if analysis_job.created_at else None,
            'started_at': analysis_job.started_at.isoformat() if analysis_job.started_at else None,
            'completed_at': analysis_job.completed_at.isoformat() if analysis_job.completed_at else None,
            'progress': min(100.0, (analysis_job.processed_frames / analysis_job.total_frames * 100) if analysis_job.total_frames and analysis_job.total_frames > 0 else 0),
            'total_frames': analysis_job.total_frames or 0,
            'processed_frames': analysis_job.processed_frames or 0,
            'violations_count': analysis_job.violations_count or 0,
            'fps': 0.0,  # Will be calculated from job manager if available
            'error_message': None,
            'video_path': analysis_job.file_path
        })

    # Fallback to job manager
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    status_dict = job.get_status_dict()
    logging.info(f"INFO: Job {job_id} status: {status_dict['status']}")
    return jsonify(status_dict)

@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get job status (legacy endpoint)"""
    return get_process_status(job_id)

@app.route('/api/process/start', methods=['POST'])
def start_processing():
    """Start processing a job - creates job with queued status and triggers background processing"""
    data = request.get_json()
    if not data or 'job_id' not in data:
        return jsonify({'error': 'job_id required'}), 400

    job_id = data['job_id']
    logging.info(f"INFO: /api/process/start called for job {job_id}")

    # Validate models are loaded
    if not models_loaded:
        logging.error("Cannot start job: Models not loaded")
        return jsonify({'error': 'AI models not loaded. Please restart the application.'}), 503

    if plate_detection_model is None or char_seg_model is None or char_recog_model is None:
        logging.error("Cannot start job: One or more models are None")
        return jsonify({'error': 'AI models not available. Please check model files.'}), 503

    job = job_manager.get_job(job_id)
    if not job:
        logging.error(f"Job {job_id} not found")
        return jsonify({'error': 'Job not found'}), 404

    if job.status != job.status.QUEUED:
        logging.warning(f"Job {job_id} status is {job.status.value}, cannot start")
        return jsonify({'error': f'Job status is {job.status.value}, cannot start'}), 400

    try:
        from application.image_processing import process_video_for_job
        logging.info(f"INFO: Starting background processing for job {job_id}")

        # Start background processing
        threading.Thread(
            target=run_model,
            args=(job_id, {
                'plate_model': plate_detection_model,
                'seg_model': char_seg_model,
                'recog_model': char_recog_model,
                'device': device,
                'ocr_font_path': ocr_font_path
            }),
            daemon=True
        ).start()

        logging.info(f"INFO: Background processing thread started for job {job_id}")
        return jsonify({'job_id': job_id}), 202

    except Exception as e:
        logging.error(f"INFO: Error starting job {job_id}: {e}", exc_info=True)
        return jsonify({'error': f'Failed to start job: {e}'}), 500

def run_model(job_id, params):
    """Background runner for model processing"""
    logging.info(f"INFO: Background runner started for job {job_id}")
    ctx = app.app_context()
    ctx.push()
    try:
        job = job_manager.get_job(job_id)
        if not job:
            logging.error(f"INFO: Job {job_id} not found in background runner")
            return

        # Get AnalysisJob from database
        from application.user_model import AnalysisJob
        analysis_job = AnalysisJob.query.filter_by(job_id=job_id).first()

        # Set status to running
        job.status = job.status.RUNNING
        job.started_at = time.time()
        logging.info(f"INFO: Job {job_id} status set to running")

        # Update database status to running
        if analysis_job:
            analysis_job.status = 'running'
            analysis_job.started_at = datetime.now()
            db.session.commit()

        # Resolve absolute path
        video_path = os.path.abspath(job.video_path)
        if not os.path.exists(video_path):
            job.status = job.status.FAILED
            job.error_message = f"Video file not found: {video_path}"
            job.completed_at = time.time()
            logging.error(f"INFO: Job {job_id} failed - video file not found: {video_path}")
            # Update database for failure
            if analysis_job:
                analysis_job.status = 'failed'
                analysis_job.completed_at = datetime.now()
                db.session.commit()
            return

        logging.info(f"INFO: Processing video at absolute path: {video_path}")

        # Call the processor
        from application.image_processing import process_video_for_job
        result = process_video_for_job(
            job,
            params['plate_model'],
            params['seg_model'],
            params['recog_model'],
            params['device'],
            params['ocr_font_path']
        )

        # Success
        job.status = job.status.SUCCEEDED
        job.completed_at = time.time()
        logging.info(f"INFO: Job {job_id} completed successfully")

    except Exception as e:
        job.status = job.status.FAILED
        job.error_message = str(e)
        job.completed_at = time.time()
        logging.error(f"INFO: Job {job_id} failed with exception: {e}", exc_info=True)
        # Update database for failure
        if analysis_job:
            analysis_job.status = 'failed'
            analysis_job.completed_at = datetime.now()
            db.session.commit()
    finally:
        ctx.pop()


# -------------------- Live MJPEG stream --------------------

@app.route('/api/jobs/<job_id>/stream.mjpg', methods=['GET'])
def stream_job(job_id):
    """MJPEG stream for live preview"""
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    status_value = getattr(job.status, "value", job.status)
    if status_value != 'running':
        return jsonify({'error': 'Job not running'}), 400

    return create_mjpeg_response(job)

# -------------------- SSE events (preferred path is /api/...) --------------------

@app.route('/jobs/<job_id>/events')
@app.route('/api/jobs/<job_id>/events')
def job_events(job_id):
    """Server-Sent Events endpoint for real-time job updates (typed for the frontend)."""
    def sse(data: dict) -> str:
        # Minimal SSE format: one JSON message per event
        return f"data: {json.dumps(data)}\n\n"

    def generate():
        job = job_manager.get_job(job_id)
        if not job:
            yield sse({'type': 'error', 'error': 'Job not found'})
            return

        last_processed = -1
        last_total = -1
        last_status = None
        last_violation_count = 0

        while True:
            status_dict = job.get_status_dict() if hasattr(job, 'get_status_dict') else job.get_status()
            # Safe defaults
            status = status_dict.get('status')
            processed_frames = status_dict.get('processed_frames', 0)
            total_frames = status_dict.get('total_frames', 0)
            fps = status_dict.get('fps', 0.0)
            progress = status_dict.get('progress', 0.0)
            violations_count = status_dict.get('violations_count', 0)

            # 1) Status event (only when it changes)
            if status and status != last_status:
                yield sse({'type': 'status', 'status': status})
                last_status = status

            # 2) Progress event (only when something changes)
            if (processed_frames != last_processed) or (total_frames != last_total):
                payload = {
                    'type': 'progress',
                    'progress': progress,
                    'processed_frames': processed_frames,
                    'total_frames': total_frames,
                    'fps': fps
                }
                yield sse(payload)
                last_processed = processed_frames
                last_total = total_frames

            # 3) Violation events (emit any new ones since last tick)
            # Try to pull a proper list, otherwise derive from results[] that have violation=True
            violation_events = []
            if hasattr(job, 'violation_events') and isinstance(job.violation_events, list):
                violation_events = job.violation_events
            elif getattr(job, 'results', None):
                # derive best-effort violation items
                violation_events = [
                    {
                        'plate_text': r.get('final_text') or r.get('plate_text') or 'Unknown',
                        'confidence': r.get('confidence', 0.0),
                        'frame_number': r.get('frame_number', 0),
                        'violation_time': r.get('violation_time_epoch') or r.get('violation_time', 0),
                        'track_id': r.get('track_id')
                    }
                    for r in job.results if r.get('violation')
                ]

            if len(violation_events) > last_violation_count:
                # emit only the new ones
                new_items = violation_events[last_violation_count:]
                for ev in new_items:
                    yield sse({
                        'type': 'violation',
                        'job_id': job_id,
                        'plate_text': ev.get('plate_text', 'Unknown'),
                        'confidence': ev.get('confidence', 0.0),
                        'frame_number': ev.get('frame_number', 0),
                        'violation_time': ev.get('violation_time', time.time()),
                        'track_id': ev.get('track_id'),
                        # keep a running count so the UI badge looks right
                        'violation_count': last_violation_count + 1
                    })
                last_violation_count = len(violation_events)

            # 4) Completion
            if status in ('completed', 'error', 'cancelled'):
                yield sse({'type': 'completion', 'status': status})
                break

            time.sleep(0.5)

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        }
    )

# -------------------- Optional live page --------------------

@app.route('/jobs/<job_id>/live')
def job_live_view(job_id):
    """Serve the job live view page (optional)"""
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return render_template('job_live.html', job_id=job_id)

# -------------------- Results & Reports --------------------

@app.route('/api/process/result/<job_id>', methods=['GET'])
def get_process_result(job_id):
    """Get processing results for a job - returns 400 if not succeeded"""
    logging.info(f"INFO: /api/process/result/{job_id} called")

    # First try to get from database
    from application.user_model import AnalysisJob, Violation, PlateDetection
    analysis_job = AnalysisJob.query.filter_by(job_id=job_id).first()
    if analysis_job:
        # Check status - only return results if succeeded
        if analysis_job.status != 'succeeded':
            return jsonify({'error': f'Job status is {analysis_job.status}, results not available'}), 400

        # Get violations from database
        violations = Violation.query.filter_by(job_id=analysis_job.id).all()
        violations_data = [{
            'final_text': v.plate_text,
            'violation_time': v.violation_time.isoformat() if v.violation_time else None,
            'violation_time_formatted': v.violation_time.strftime('%H:%M:%S') if v.violation_time else None,
            'confidence': v.confidence,
            'frame_number': v.frame_number,
            'bbox': [v.bbox_x1, v.bbox_y1, v.bbox_x2, v.bbox_y2] if v.bbox_x1 is not None else None,
            'violation': True,
            'original_plate': v.original_plate,
            'deskewed_plate': v.deskewed_plate,
            'digital_plate': v.digital_plate,
            'characters': json.loads(v.characters) if v.characters else []
        } for v in violations]

        # Get all plate detections from database
        plate_detections = PlateDetection.query.filter_by(job_id=analysis_job.id).all()
        detections_data = [{
            'final_text': pd.plate_text,
            'confidence': pd.confidence,
            'frame_number': pd.frame_number,
            'bbox': [pd.bbox_x1, pd.bbox_y1, pd.bbox_x2, pd.bbox_y2] if pd.bbox_x1 is not None else None,
            'violation': pd.is_violation,
            'original_plate': pd.original_plate,
            'deskewed_plate': pd.deskewed_plate,
            'digital_plate': pd.digital_plate,
            'characters': json.loads(pd.characters) if pd.characters else []
        } for pd in plate_detections]

        # Calculate duration
        duration = 0
        if analysis_job.completed_at and analysis_job.started_at:
            duration = (analysis_job.completed_at - analysis_job.started_at).total_seconds()

        results_data = {
            'job_id': job_id,
            'status': analysis_job.status,
            'total_plates': len(plate_detections),
            'violations_count': len(violations),
            'processed_frames': analysis_job.processed_frames or 0,
            'total_frames': analysis_job.total_frames or 0,
            'started_at': analysis_job.started_at.isoformat() if analysis_job.started_at else None,
            'completed_at': analysis_job.completed_at.isoformat() if analysis_job.completed_at else None,
            'results': detections_data,
            'violations': violations_data,
            'frame_snapshots': [],  # Will be populated from job manager if available
            'average_fps': (analysis_job.processed_frames / duration) if duration and analysis_job.processed_frames else 0.0
        }

        # Try to get frame snapshots from job manager if job is still active
        job = job_manager.get_job(job_id)
        if job and hasattr(job, 'frame_snapshots'):
            results_data['frame_snapshots'] = job.frame_snapshots

        logging.info(f"INFO: Returning results for succeeded job {job_id}")
        return jsonify(results_data)

    # Fallback to job manager if not in database
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    # Check status - only return results if succeeded
    if job.status.value != 'succeeded':
        return jsonify({'error': f'Job status is {job.status.value}, results not available'}), 400

    # Avoid div-by-zero
    duration = 0
    if job.completed_at and job.started_at and (job.completed_at - job.started_at) > 0:
        duration = (job.completed_at - job.started_at)

    results_data = {
        'job_id': job.job_id,
        'status': getattr(job.status, "value", job.status),
        'total_plates': len(job.results),
        'violations_count': job.violations_count,
        'processed_frames': job.processed_frames,
        'total_frames': job.total_frames,
        'started_at': job.started_at,
        'completed_at': job.completed_at,
        'results': job.results,
        'violations': [r for r in job.results if r.get('violation', False)],
        'frame_snapshots': getattr(job, 'frame_snapshots', []),
        'average_fps': (job.processed_frames / duration) if duration else 0.0
    }

    logging.info(f"INFO: Returning results for succeeded job {job_id}")
    return jsonify(results_data)

@app.route('/api/jobs/<job_id>/results', methods=['GET'])
def get_job_results(job_id):
    """Get detailed job results (legacy endpoint)"""
    return get_process_result(job_id)

@app.route('/api/jobs/<job_id>/report/<format_type>', methods=['GET'])
def download_report(job_id, format_type):
    """Download job report in specified format"""
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    try:
        if format_type.lower() == 'csv':
            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)

            # Headers
            writer.writerow(['Frame', 'Plate Text', 'Confidence', 'Violation', 'Timestamp', 'Coordinates'])

            # Data
            for result in job.results:
                writer.writerow([
                    result.get('frame_number', ''),
                    result.get('final_text', ''),
                    result.get('confidence', ''),
                    'Yes' if result.get('violation', False) else 'No',
                    result.get('violation_time', ''),
                    str(result.get('plate_coordinates', ''))
                ])

            output.seek(0)
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename=traffic-analysis-{job_id}.csv'}
            )

        else:
            return jsonify({'error': 'Unsupported format'}), 400

    except Exception as e:
        logging.error(f"Error generating {format_type} report for job {job_id}: {e}", exc_info=True)
        return jsonify({'error': f'Failed to generate {format_type} report'}), 500

@app.route('/api/jobs/<job_id>/stats', methods=['GET'])
def api_job_stats(job_id):
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    # Prefer a dict method if present
    d = job.get_status_dict() if hasattr(job, 'get_status_dict') else job.get_status()
    return jsonify({
        'job_id': job_id,
        'frames': d.get('processed_frames', 0),
        'fps': d.get('fps', 0.0),
        'violations': d.get('violations_count', 0),
        'progress': d.get('progress', 0.0),
        'timestamp': int(time.time())
    })


@app.route('/api/jobs/<job_id>/violations', methods=['GET'])
def api_job_violations(job_id):
    from application.user_model import AnalysisJob, Violation

    limit = int(request.args.get('limit', 20))

    # First try to get from database
    analysis_job = AnalysisJob.query.filter_by(job_id=job_id).first()
    if analysis_job:
        violations = Violation.query.filter_by(job_id=analysis_job.id).order_by(Violation.violation_time.desc()).limit(limit).all()
        out = [{
            'job_id': job_id,
            'plate': v.plate_text,
            'confidence': v.confidence,
            'bbox': [v.bbox_x1, v.bbox_y1, v.bbox_x2, v.bbox_y2] if v.bbox_x1 is not None else None,
            'frame_thumb': None,  # Could be populated from violation images if stored
            'timestamp': v.violation_time.timestamp() if v.violation_time else int(time.time()),
            'frame_number': v.frame_number
        } for v in violations]
        return jsonify(out)

    # Fallback to job manager if not in database
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify([])

    if hasattr(job, 'violation_events') and isinstance(job.violation_events, list):
        events = job.violation_events
        out = [{
            'job_id': job_id,
            'plate': ev.get('plate_text', 'Unknown'),
            'confidence': ev.get('confidence', 0.0),
            'bbox': ev.get('bbox'),
            'frame_thumb': ev.get('frame_thumb'),
            'timestamp': ev.get('violation_time') or ev.get('timestamp') or int(time.time()),
            'frame_number': ev.get('frame_number', 0)
        } for ev in events[-limit:]]
        return jsonify(out)

    # Fallback: build from results[]
    if getattr(job, 'results', None):
        vres = [r for r in job.results if r.get('violation')]
        out = [{
            'job_id': job_id,
            'plate': r.get('final_text') or r.get('plate_text') or 'Unknown',
            'confidence': r.get('confidence', 0.0),
            'bbox': r.get('plate_coordinates'),
            'frame_thumb': r.get('original_plate'),
            'timestamp': r.get('violation_time') or r.get('timestamp') or int(time.time()),
            'frame_number': r.get('frame_number', 0)
        } for r in vres[-limit:]]
        return jsonify(out)

    return jsonify([])


# -------------------- Authentication --------------------

from application.user_model import db, User
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

@login_manager.user_loader
def load_user(user_id):
    # Updated to use Session.get() as Query.get() is deprecated in SQLAlchemy 2.0
    from sqlalchemy.orm import Session
    session = Session(db.engine)
    return session.get(User, int(user_id))

# Initialize database tables
with app.app_context():
    try:
        db.create_all()
        logging.info("Database tables created successfully.")
    except Exception as e:
        logging.error(f"Failed to create database tables: {e}", exc_info=True)
        raise

@app.route('/api/signup', methods=['POST'])
@limiter.limit("5 per minute")
def signup():
    try:
        data = request.get_json()
        logging.info(f"Signup request data: {data}")
        if not data:
            return jsonify({'success': False, 'error': 'Invalid JSON data'}), 400

        name = data.get('name') or data.get('username')
        email = data.get('email')
        password = data.get('password')
        confirm_password = data.get('confirm_password')

        if not name or not email or not password:
            return jsonify({'success': False, 'error': 'Name, email, and password are required'}), 400

        if confirm_password and password != confirm_password:
            return jsonify({'success': False, 'error': 'Passwords do not match'}), 400

        # Password complexity validation
        if len(password) < 8:
            return jsonify({'success': False, 'error': 'Password must be at least 8 characters long'}), 400
        if not any(char.isupper() for char in password):
            return jsonify({'success': False, 'error': 'Password must contain at least one uppercase letter'}), 400
        if not any(char.islower() for char in password):
            return jsonify({'success': False, 'error': 'Password must contain at least one lowercase letter'}), 400
        if not any(char.isdigit() for char in password):
            return jsonify({'success': False, 'error': 'Password must contain at least one number'}), 400

        # Check for existing users
        if User.query.filter_by(username=name).first():
            return jsonify({'success': False, 'error': 'Username already exists'}), 400

        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'error': 'Email already exists'}), 400

        # Create user
        user = User(username=name, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)
        logging.info(f"User registered successfully: name={name}, email={email}")
        return jsonify({'success': True, 'message': 'User registered successfully'}), 201

    except Exception as e:
        db.session.rollback()
        logging.error(f"Signup failed: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'Internal server error during signup'}), 500

@app.route('/api/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        return jsonify({'error': 'Invalid username or password'}), 401

    login_user(user)
    return jsonify({'message': 'Login successful', 'user': {'id': user.id, 'username': user.username, 'email': user.email}}), 200

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logout successful'}), 200

@app.route('/api/user', methods=['GET'])
@login_required
def get_user():
    return jsonify({'user': {'id': current_user.id, 'username': current_user.username, 'email': current_user.email}}), 200

@app.route('/api/jobs/history', methods=['GET'])
@login_required
def get_job_history():
    """Get user's analysis job history"""
    from application.user_model import AnalysisJob

    # Get pagination parameters
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))

    # Query user's jobs ordered by creation date (newest first)
    jobs_query = AnalysisJob.query.filter_by(user_id=current_user.id).order_by(AnalysisJob.created_at.desc())

    # Paginate
    jobs = jobs_query.paginate(page=page, per_page=per_page, error_out=False)

    job_history = []
    for job in jobs.items:
        # Calculate duration if completed
        duration = None
        if job.completed_at and job.started_at:
            duration = (job.completed_at - job.started_at).total_seconds()

        job_history.append({
            'job_id': job.job_id,
            'filename': job.filename,
            'status': job.status,
            'total_frames': job.total_frames,
            'processed_frames': job.processed_frames,
            'violations_count': job.violations_count,
            'created_at': job.created_at.isoformat() if job.created_at else None,
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'duration_seconds': duration,
            'progress': min(100.0, (job.processed_frames / job.total_frames * 100) if job.total_frames and job.total_frames > 0 else 0)
        })

    return jsonify({
        'jobs': job_history,
        'pagination': {
            'page': jobs.page,
            'per_page': jobs.per_page,
            'total': jobs.total,
            'pages': jobs.pages,
            'has_next': jobs.has_next,
            'has_prev': jobs.has_prev
        }
    }), 200


# -------------------- Main --------------------

if __name__ == '__main__':
    logging.info("----- Starting Flask API Server -----")
    logging.info(f"Flask Secret Key: {'Set' if config.FLASK_SECRET_KEY != 'your_very_secret_key_change_me' else '!!! Using Default !!!'}")
    logging.info(f"Max Upload Size: {config.MAX_CONTENT_LENGTH / (1024*1024):.1f} MB")
    logging.info(f"Allowed Extensions: {', '.join(config.ALLOWED_EXTENSIONS)}")

    # Load models synchronously before starting the server
    try:
        load_models_synchronously()
        logging.info("All models loaded successfully - starting server")
    except Exception as e:
        logging.error(f"Failed to load models - aborting startup: {e}")
        exit(1)

    # threaded=True helps SSE/MJPEG alongside processing threads
    app.run(host='0.0.0.0', port=5003, debug=True, threaded=True)
