import flask
from flask import Flask, render_template, request, redirect, url_for, flash
import os
import logging
import tempfile
import time
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from application import config
from application.model_loader import load_models

from application.image_processing import process_file
from application.job_manager import job_manager
from application.streaming import create_mjpeg_response
from application.events import create_sse_response, send_status_update

from application.utils import to_base64


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')

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

logging.info("Model Load")


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER_PATH
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.secret_key = config.FLASK_SECRET_KEY

app.jinja_env.filters['to_base64'] = to_base64
app.jinja_env.globals.update(zip=zip)


@app.route('/', methods=['GET', 'POST'])
def upload_file_route():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request.', 'error')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No file selected.', 'warning')
            return redirect(request.url)

        if file:
            original_filename = file.filename
            _, file_extension = os.path.splitext(original_filename)
            file_extension = file_extension.lower()

            if file_extension not in config.ALLOWED_EXTENSIONS:
                allowed_str = ", ".join(config.ALLOWED_EXTENSIONS)
                flash(f'Unsupported file type: "{file_extension}". Allowed types: {allowed_str}', 'error')
                logging.warning(f"Upload rejected: Unsupported file type '{file_extension}' from file '{original_filename}'")
                return redirect(request.url)

            temp_path = None
            fd = None
            try:
                fd, temp_path = tempfile.mkstemp(suffix=file_extension, dir=app.config['UPLOAD_FOLDER'], text=False)
                file.save(temp_path)
                logging.info(f"File '{original_filename}' saved temporarily to '{temp_path}'")

                if fd is not None:
                    os.close(fd)
                    fd = None

                if not models_loaded:
                     flash('Models are not loaded correctly. Cannot process file.', 'error')
                     logging.error("Processing aborted: Models not loaded.")
                     return redirect(url_for('upload_file_route'))

                video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
                if file_extension in video_extensions:
                    # Create job for live processing
                    job = job_manager.create_job(temp_path)
                    logging.info(f"Created job {job.job_id} for video file '{original_filename}'")

                    # Redirect to live preview page
                    return redirect(url_for('job_live_view', job_id=job.job_id))
                else:
                    # Process images synchronously
                    start_process_time = time.time()
                    results = process_file(
                        temp_path,
                        plate_detection_model,
                        char_seg_model,
                        char_recog_model,
                        device,
                        ocr_font_path
                     )
                    end_process_time = time.time()
                    logging.info(f"Processing '{original_filename}' completed in {end_process_time - start_process_time:.3f} seconds. Found {len(results)} plates.")

                    return render_template('results.html', results=results, filename=original_filename)

            except Exception as e:
                logging.error(f"Error processing uploaded file '{original_filename}': {e}", exc_info=True)
                flash(f'An error occurred during processing: {str(e)}', 'error')
                return redirect(url_for('upload_file_route'))

            finally:
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError as close_err:
                         logging.warning(f"Warning: Could not close temp file descriptor {fd}: {close_err}")

    return render_template('upload.html')


@app.route('/jobs', methods=['POST'])
def create_job():
    """Create a new video processing job"""
    if 'file' not in request.files:
        return {'error': 'No file part in the request'}, 400

    file = request.files['file']
    if file.filename == '':
        return {'error': 'No file selected'}, 400

    if file:
        original_filename = file.filename
        _, file_extension = os.path.splitext(original_filename)
        file_extension = file_extension.lower()

        if file_extension not in config.ALLOWED_EXTENSIONS:
            return {'error': f'Unsupported file type: {file_extension}'}, 400

        temp_path = None
        fd = None
        try:
            fd, temp_path = tempfile.mkstemp(suffix=file_extension, dir=app.config['UPLOAD_FOLDER'], text=False)
            file.save(temp_path)
            logging.info(f"File '{original_filename}' saved temporarily to '{temp_path}'")

            if fd is not None:
                os.close(fd)
                fd = None

            # Create job
            job = job_manager.create_job(temp_path)

            return {
                'job_id': job.job_id,
                'status': job.status.value,
                'message': 'Job created successfully',
                'live_preview_url': f'/jobs/{job.job_id}/live'
            }, 201

        except Exception as e:
            logging.error(f"Error creating job for file '{original_filename}': {e}", exc_info=True)
            return {'error': str(e)}, 500

        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError as close_err:
                     logging.warning(f"Warning: Could not close temp file descriptor {fd}: {close_err}")


@app.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get job status"""
    job = job_manager.get_job(job_id)
    if not job:
        return {'error': 'Job not found'}, 404

    return job.get_status_dict()


@app.route('/jobs/<job_id>/stream.mjpg', methods=['GET'])
def stream_job(job_id):
    """MJPEG stream for live preview"""
    job = job_manager.get_job(job_id)
    if not job:
        return {'error': 'Job not found'}, 404

    if job.status != job.status.PROCESSING:
        return {'error': 'Job not processing'}, 400

    return create_mjpeg_response(job)


@app.route('/jobs/<job_id>/start', methods=['POST'])
def start_job(job_id):
    """Start processing a job"""
    job = job_manager.get_job(job_id)
    if not job:
        return {'error': 'Job not found'}, 404

    if job.status != job.status.PENDING:
        return {'error': f'Job status is {job.status.value}, cannot start'}, 400

    try:
        # Import here to avoid circular imports
        from application.image_processing import process_video_for_job

        job_manager.start_job(job_id, process_video_for_job,
                            plate_detection_model, char_seg_model, char_recog_model, device, ocr_font_path)

        return {'message': 'Job started successfully'}, 200

    except Exception as e:
        logging.error(f"Error starting job {job_id}: {e}", exc_info=True)
        return {'error': str(e)}, 500


@app.route('/jobs/<job_id>/events')
def job_events(job_id):
    """Server-Sent Events endpoint for real-time job updates"""
    job = job_manager.get_job(job_id)
    if not job:
        return {'error': 'Job not found'}, 404

    return create_sse_response(job)


@app.route('/jobs/<job_id>/live')
def job_live_view(job_id):
    """Live preview page for job monitoring"""
    job = job_manager.get_job(job_id)
    if not job:
        return {'error': 'Job not found'}, 404

    return render_template('job_live.html', job_id=job_id)


@app.route('/jobs/<job_id>/results')
def job_results(job_id):
    """Display results for a completed job"""
    job = job_manager.get_job(job_id)
    if not job:
        flash('Job not found.', 'error')
        return redirect(url_for('upload_file_route'))

    if job.status != job.status.COMPLETED:
        flash('Job is not completed yet.', 'warning')
        return redirect(url_for('job_live_view', job_id=job_id))

    # Get filename from video path
    filename = os.path.basename(job.video_path)

    # Get frame snapshots from job object
    frame_snapshots = getattr(job, 'frame_snapshots', [])
    logging.info(f"Job {job_id} results: {len(job.results)} detections, {len(frame_snapshots)} snapshots")
    if frame_snapshots:
        logging.info(f"Sample snapshot: {frame_snapshots[0]}")
    return render_template('results.html', results=job.results, filename=filename,
                         job_id=job_id, frame_snapshots=frame_snapshots)


@app.route('/jobs/<job_id>/snapshots/<filename>')
def serve_snapshot(job_id, filename):
    """Serve snapshot images for a job"""
    job = job_manager.get_job(job_id)
    if not job:
        logging.error(f"Job {job_id} not found for snapshot request")
        return {'error': 'Job not found'}, 404

    if not hasattr(job, 'output_dir') or not job.output_dir:
        logging.error(f"No output directory for job {job_id}")
        return {'error': 'No output directory for job'}, 404

    snapshots_dir = os.path.join(job.output_dir, 'snapshots')
    snapshot_path = os.path.join(snapshots_dir, filename)

    logging.info(f"Serving snapshot: {snapshot_path}")

    if not os.path.exists(snapshot_path):
        logging.error(f"Snapshot not found: {snapshot_path}")
        # Try to check if the file exists with a different case (Windows insensitive)
        # or if the file is in a different directory level
        # For debugging, list files in snapshots_dir
        try:
            files = os.listdir(snapshots_dir)
            logging.error(f"Files in snapshots directory: {files}")
        except Exception as e:
            logging.error(f"Error listing snapshots directory: {e}")
        return {'error': 'Snapshot not found'}, 404

    from flask import send_file
    return send_file(snapshot_path, mimetype='image/jpeg')


if __name__ == '__main__':
    logging.info("----- Starting Flask Application Web Server -----")
    logging.info(f"Flask Secret Key: {'Set' if config.FLASK_SECRET_KEY != 'your_very_secret_key_change_me' else '!!! Using Default !!!'}")
    logging.info(f"Max Upload Size: {config.MAX_CONTENT_LENGTH / (1024*1024):.1f} MB")
    logging.info(f"Allowed Extensions: {', '.join(config.ALLOWED_EXTENSIONS)}")
    logging.info(f"Models Loaded: {models_loaded}")
    if not models_loaded:
        logging.warning("Running with one or more models missing!")

    app.run(host='0.0.0.0', port=5001, debug=True)