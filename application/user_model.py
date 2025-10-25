from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    analysis_jobs = db.relationship('AnalysisJob', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class AnalysisJob(db.Model):
    __tablename__ = 'analysis_jobs'

    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.String(100), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    status = db.Column(db.String(50), default='pending')  # pending, processing, completed, error, cancelled
    total_frames = db.Column(db.Integer, default=0)
    processed_frames = db.Column(db.Integer, default=0)
    violations_count = db.Column(db.Integer, default=0)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    violations = db.relationship('Violation', backref='job', lazy=True, cascade='all, delete-orphan')
    plate_detections = db.relationship('PlateDetection', backref='job', lazy=True, cascade='all, delete-orphan')
    csv_reports = db.relationship('CsvReport', backref='job', lazy=True, cascade='all, delete-orphan')

class Violation(db.Model):
    __tablename__ = 'violations'

    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('analysis_jobs.id'), nullable=False)
    plate_text = db.Column(db.String(20), nullable=False)
    violation_time = db.Column(db.DateTime, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    frame_number = db.Column(db.Integer, nullable=False)
    bbox_x1 = db.Column(db.Integer)
    bbox_y1 = db.Column(db.Integer)
    bbox_x2 = db.Column(db.Integer)
    bbox_y2 = db.Column(db.Integer)
    violation_image_path = db.Column(db.String(500))  # Path to violation snapshot image
    original_plate = db.Column(db.Text)  # Base64 image data
    deskewed_plate = db.Column(db.Text)  # Base64 image data
    digital_plate = db.Column(db.Text)   # Base64 image data
    characters = db.Column(db.Text)      # JSON string of character data
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PlateDetection(db.Model):
    __tablename__ = 'plate_detections'

    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('analysis_jobs.id'), nullable=False)
    plate_text = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    frame_number = db.Column(db.Integer, nullable=False)
    bbox_x1 = db.Column(db.Integer)
    bbox_y1 = db.Column(db.Integer)
    bbox_x2 = db.Column(db.Integer)
    bbox_y2 = db.Column(db.Integer)
    is_violation = db.Column(db.Boolean, default=False)
    original_plate = db.Column(db.Text)  # Base64 image data
    deskewed_plate = db.Column(db.Text)  # Base64 image data
    digital_plate = db.Column(db.Text)   # Base64 image data
    characters = db.Column(db.Text)      # JSON string of character data
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class CsvReport(db.Model):
    __tablename__ = 'csv_reports'

    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('analysis_jobs.id'), nullable=False)
    report_type = db.Column(db.String(50), nullable=False)  # 'violations', 'detections', etc.
    file_path = db.Column(db.String(500), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
