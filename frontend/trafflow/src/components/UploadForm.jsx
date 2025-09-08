import React, { useState, useRef } from 'react';
import { API_BASE, WS_BASE } from '../config';

const UploadForm = ({ onUploadComplete }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
        setError(null);
      }
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file && validateFile(file)) {
      setSelectedFile(file);
      setError(null);
    }
  };

  const validateFile = (file) => {
    const allowedTypes = [
      'image/png', 'image/jpg', 'image/jpeg', 'image/webp',
      'video/mp4', 'video/avi', 'video/mov', 'video/quicktime', 'video/x-msvideo'
    ];
    
    if (!allowedTypes.includes(file.type)) {
      setError('Please select a valid image (PNG, JPG, JPEG, WebP) or video (MP4, AVI, MOV) file.');
      return false;
    }
    
    const maxSize = 500 * 1024 * 1024; // 500MB
    if (file.size > maxSize) {
      setError('File size must be less than 500MB.');
      return false;
    }
    
    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setError('Please select a file to upload.');
      return;
    }

    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

      try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });      

      if (!response.ok) {
        let errorMessage = 'Upload failed';
        try {
          const errorData = await response.json();
          errorMessage = errorData.error || errorMessage;
        } catch {
          // Response is not JSON or empty
          errorMessage = `Upload failed with status ${response.status}`;
        }
        throw new Error(errorMessage);
      }

      const result = await response.json();
      onUploadComplete(result);
    } catch (err) {
      console.error('Upload error:', err);
      setError(err.message || 'Failed to upload file. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileIcon = (file) => {
    if (file.type.startsWith('image/')) {
      return (
        <svg width="48" height="48" className="text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
      );
    } else {
      return (
        <svg width="48" height="48" className="text-purple" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
        </svg>
      );
    }
  };

  return (
    <div className="container">
      <div className="row justify-content-center">
        <div className="col-lg-8">
          {/* Hero Section */}
          <div className="text-center mb-5">
            <div className="card shadow-custom">
              <div className="card-body p-5">
                <div className="gradient-bg text-white rounded-3 d-inline-flex align-items-center justify-content-center mb-4" style={{width: '64px', height: '64px'}}>
                  <svg width="32" height="32" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                </div>
                <h1 className="display-5 fw-bold text-dark mb-4">
                  Intelligent Traffic Analysis
                </h1>
                <p className="lead text-muted mb-4">
                  Upload your traffic video or image for AI-powered license plate recognition and violation detection
                </p>
                
                <div className="row g-3 text-muted">
                  <div className="col-md-4">
                    <div className="d-flex align-items-center justify-content-center">
                      <svg width="20" height="20" className="text-success me-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      <span className="small">Real-time Processing</span>
                    </div>
                  </div>
                  <div className="col-md-4">
                    <div className="d-flex align-items-center justify-content-center">
                      <svg width="20" height="20" className="text-success me-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      <span className="small">Nepali License Plates</span>
                    </div>
                  </div>
                  <div className="col-md-4">
                    <div className="d-flex align-items-center justify-content-center">
                      <svg width="20" height="20" className="text-success me-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      <span className="small">Traffic Violations</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Upload Form */}
          <div className="card shadow-custom">
            <div className="card-body p-4">
              <form onSubmit={handleSubmit}>
                {/* Drag and Drop Area */}
                <div
                  className={`position-relative border-2 border-dashed rounded-3 p-5 text-center ${
                    dragActive
                      ? 'border-primary bg-primary bg-opacity-10'
                      : selectedFile
                      ? 'border-success bg-success bg-opacity-10'
                      : 'border-secondary hover-lift'
                  }`}
                  style={{transition: 'all 0.2s ease-in-out'}}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".png,.jpg,.jpeg,.webp,.mp4,.avi,.mov,.mkv"
                    onChange={handleFileSelect}
                    className="position-absolute top-0 start-0 w-100 h-100 opacity-0"
                    style={{cursor: 'pointer'}}
                  />
                  
                  {selectedFile ? (
                    <div>
                      <div className="d-flex justify-content-center mb-3">
                        {getFileIcon(selectedFile)}
                      </div>
                      <div className="mb-3">
                        <p className="h5 fw-semibold text-dark">{selectedFile.name}</p>
                        <p className="text-muted small">
                          {formatFileSize(selectedFile.size)} • {selectedFile.type.split('/')[0].toUpperCase()}
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={() => {
                          setSelectedFile(null);
                          if (fileInputRef.current) fileInputRef.current.value = '';
                        }}
                        className="btn btn-outline-danger btn-sm"
                      >
                        Remove file
                      </button>
                    </div>
                  ) : (
                    <div>
                      <div className="d-flex justify-content-center mb-3">
                        <svg width="64" height="64" className="text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                        </svg>
                      </div>
                      <div>
                        <p className="h5 fw-semibold text-dark mb-2">
                          {dragActive ? 'Drop your file here' : 'Drag & drop your file here'}
                        </p>
                        <p className="text-muted mb-3">or click to browse</p>
                        <div className="d-flex flex-wrap justify-content-center gap-2">
                          <span className="badge bg-light text-dark">PNG</span>
                          <span className="badge bg-light text-dark">JPG</span>
                          <span className="badge bg-light text-dark">MP4</span>
                          <span className="badge bg-light text-dark">AVI</span>
                          <span className="badge bg-light text-dark">MOV</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {error && (
                  <div className="alert alert-danger d-flex align-items-center mt-3" role="alert">
                    <svg width="20" height="20" className="me-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <div>{error}</div>
                  </div>
                )}

                <button
                  type="submit"
                  disabled={isUploading || !selectedFile}
                  className={`btn btn-lg w-100 mt-4 d-flex align-items-center justify-content-center hover-lift ${
                    isUploading || !selectedFile
                      ? 'btn-secondary'
                      : 'gradient-bg text-white shadow-custom'
                  }`}
                >
                  {isUploading ? (
                    <>
                      <div className="spinner-border spinner-border-sm me-2" role="status">
                        <span className="visually-hidden">Loading...</span>
                      </div>
                      <span>Uploading...</span>
                    </>
                  ) : (
                    <>
                      <svg width="24" height="24" className="me-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      <span>Start Analysis</span>
                    </>
                  )}
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UploadForm;
