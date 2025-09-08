import flask
from flask import Flask, request, jsonify, Response, render_template, send_from_directory
from flask_cors import CORS
import os
import logging
import time
import sys
import json
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from application import config
from application.model_loader import load_models
from application.image_processing import process_file
from application.job_manager import job_manager
from application.streaming import create_mjpeg_response
from application.events import create_sse_response

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

logging.info("----- Initializing Application - Loading Models -----")
try:
    plate_detection_model, char_seg_model, char_recog_model, device, ocr_font_path = load_models()
    models_loaded = all([plate_detection_model, char_seg_model, char_recog_model])
    if not models_loaded:
        logging.error("One or more models failed to load. Application might not function correctly.")
except Exception as load_err:
    logging.error(f"A critical error occurred during model loading: {load_err}", exc_info=True)
    plate_detection_model, char_seg_model, char_recog_model, device, ocr_font_path = None, None, None, "cpu", None
    models_loaded = False

app = Flask(__name__)
# CORS for everything under /api (and streams/SSE are same-origin via Vite proxy)
CORS(app, resources={r"/*": {"origins": "*"}})

app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER_PATH
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.secret_key = config.FLASK_SECRET_KEY

from flask_login import LoginManager

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Add rate limiting for authentication endpoints
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
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

# -------------------- Static uploads (if needed by reports/previews) --------------------

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# -------------------- Upload --------------------

@app.route('/api/upload', methods=['POST'])
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

@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get job status"""
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify(job.get_status_dict())

@app.route('/api/jobs/<job_id>/start', methods=['POST'])
def start_job(job_id):
    """Start processing a job"""
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if job.status != job.status.PENDING:
        return jsonify({'error': f'Job status is {job.status.value}, cannot start'}), 400

    try:
        from application.image_processing import process_video_for_job

        job_manager.start_job(
            job_id,
            process_video_for_job,
            plate_detection_model,
            char_seg_model,
            char_recog_model,
            device,
            ocr_font_path
        )

        return jsonify({'message': 'Job started successfully'}), 200

    except Exception as e:
        logging.error(f"Error starting job {job_id}: {e}", exc_info=True)
        return jsonify({'error': f'Failed to start job: {e}'}), 500


# -------------------- Live MJPEG stream --------------------

@app.route('/api/jobs/<job_id>/stream.mjpg', methods=['GET'])
def stream_job(job_id):
    """MJPEG stream for live preview"""
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    status_value = getattr(job.status, "value", job.status)
    if status_value != 'processing':
        return jsonify({'error': 'Job not processing'}), 400

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

@app.route('/api/jobs/<job_id>/results', methods=['GET'])
def get_job_results(job_id):
    """Get detailed job results"""
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

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

    return jsonify(results_data)

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
    limit = int(request.args.get('limit', 20))
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

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    # Updated to use Session.get() as Query.get() is deprecated in SQLAlchemy 2.0
    from sqlalchemy.orm import Session
    session = Session(db.engine)
    return session.get(User, int(user_id))

# Initialize database
app.config['SQLALCHEMY_DATABASE_URI'] = config.SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = config.SQLALCHEMY_TRACK_MODIFICATIONS
db.init_app(app)

with app.app_context():
    db.create_all()

@app.route('/api/signup', methods=['POST'])
@limiter.limit("5 per minute")
def signup():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'error': 'Username, email, and password are required'}), 400

    # Password complexity validation
    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters long'}), 400
    if not any(char.isupper() for char in password):
        return jsonify({'error': 'Password must contain at least one uppercase letter'}), 400
    if not any(char.islower() for char in password):
        return jsonify({'error': 'Password must contain at least one lowercase letter'}), 400
    if not any(char.isdigit() for char in password):
        return jsonify({'error': 'Password must contain at least one number'}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already exists'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already exists'}), 400

    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    login_user(user)
    return jsonify({'message': 'User created successfully', 'user': {'id': user.id, 'username': user.username, 'email': user.email}}), 201

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


# -------------------- Main --------------------

if __name__ == '__main__':
    logging.info("----- Starting Flask API Server -----")
    logging.info(f"Flask Secret Key: {'Set' if config.FLASK_SECRET_KEY != 'your_very_secret_key_change_me' else '!!! Using Default !!!'}")
    logging.info(f"Max Upload Size: {config.MAX_CONTENT_LENGTH / (1024*1024):.1f} MB")
    logging.info(f"Allowed Extensions: {', '.join(config.ALLOWED_EXTENSIONS)}")
    logging.info(f"Models Loaded: {models_loaded}")
    if not models_loaded:
        logging.warning("Running with one or more models missing!")

    # threaded=True helps SSE/MJPEG alongside processing threads
    app.run(host='0.0.0.0', port=5003, debug=True, threaded=True)
