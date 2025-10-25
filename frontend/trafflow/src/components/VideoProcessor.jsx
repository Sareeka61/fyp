import React, { useState, useEffect, useRef } from 'react';
import { API_BASE, WS_BASE } from '../config';

const VideoProcessor = ({ jobData, onProcessingComplete }) => {
  const [status, setStatus] = useState('queued');
  const [progress, setProgress] = useState(0);
  const [stats, setStats] = useState({
    framesProcessed: 0,
    totalFrames: 0,
    violationsCount: 0,
    currentFps: 0
  });

  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [streamUrl, setStreamUrl] = useState('');

  const pollingIntervalRef = useRef(null);
  const streamImgRef = useRef(null);

  useEffect(() => {
    if (jobData?.job_id) {
      // Start polling for status updates
      startPolling();
      setStreamUrl(`${API_BASE}/jobs/${jobData.job_id}/stream.mjpg?${Date.now()}`);

      return () => {
        stopPolling();
      };
    }

    return () => {
      stopPolling();
    };
  }, [jobData?.job_id]);

  const startPolling = () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }
    pollingIntervalRef.current = setInterval(pollStatus, 2000); // Poll every 2 seconds
  };

  const stopPolling = () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  };

  const pollStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/process/status/${jobData.job_id}`, {
        credentials: 'include',
      });
      if (response.ok) {
        const data = await response.json();
        updateStatus(data);
      } else {
        console.error('Failed to poll status:', response.status);
      }
    } catch (err) {
      console.error('Error polling status:', err);
    }
  };

  const updateStatus = (data) => {
    setStatus(data.status);
    setProgress(data.progress || 0);
    setStats({
      framesProcessed: data.processed_frames || 0,
      totalFrames: data.total_frames || 0,
      violationsCount: data.violations_count || 0,
      currentFps: data.fps || 0
    });

    if (data.status === 'running') {
      setIsProcessing(true);
    } else if (data.status === 'succeeded') {
      setIsProcessing(false);
      stopPolling();
      fetchResults();
    } else if (data.status === 'failed') {
      setIsProcessing(false);
      setError(data.error_message || 'Processing failed');
      stopPolling();
    }
  };

  const fetchResults = async () => {
    try {
      const response = await fetch(`${API_BASE}/process/result/${jobData.job_id}`, {
        credentials: 'include',
      });
      if (response.ok) {
        const results = await response.json();
        onProcessingComplete(results);
      } else {
        console.error('Failed to fetch results:', response.status);
        onProcessingComplete({ status: 'succeeded', job_id: jobData.job_id }); // fallback with job_id
      }
    } catch (err) {
      console.error('Error fetching results:', err);
      onProcessingComplete({ status: 'succeeded', job_id: jobData.job_id }); // fallback with job_id
    }
  };

  const startProcessing = async () => {
    try {
      setError(null);
      const response = await fetch(`${API_BASE}/process/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ job_id: jobData.job_id }),
        credentials: 'include',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to start processing');
      }

      setStatus('running');
      setIsProcessing(true);
    } catch (err) {
      setError(err.message);
      console.error('Error starting processing:', err);
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'running': return 'bg-primary';
      case 'succeeded': return 'bg-success';
      case 'failed': return 'bg-danger';
      default: return 'bg-secondary';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'running': return 'Processing Video';
      case 'succeeded': return 'Processing Complete';
      case 'failed': return 'Processing Error';
      default: return 'Ready to Start';
    }
  };

  return (
    <div className="container-fluid">
      {/* Header Card */}
      <div className="card shadow-custom mb-4">
        <div className="card-body">
          <div className="d-flex align-items-center justify-content-between mb-4">
            <div className="d-flex align-items-center">
              <div className={`rounded-circle me-3 ${getStatusColor()} ${status === 'processing' ? 'pulse' : ''}`} style={{width: '16px', height: '16px'}}></div>
              <div>
                <h2 className="h3 fw-bold text-dark mb-1">{getStatusText()}</h2>
                <p className="text-muted mb-0">Job ID: {jobData.job_id}</p>
              </div>
            </div>
            
            {!isProcessing && (status === 'queued' || status === 'pending') && (
              <button
                onClick={startProcessing}
                className="btn gradient-bg text-white d-flex align-items-center hover-lift"
              >
                <svg width="20" height="20" className="me-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1m4 0h1m-6 4h.01M19 10a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>Start Processing</span>
              </button>
            )}
          </div>

          {/* Progress Bar */}
          <div className="mb-4">
            <div className="d-flex justify-content-between small text-muted mb-2">
              <span>Progress</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <div className="progress" style={{height: '12px'}}>
              <div 
                className="progress-bar gradient-bg progress-bar-striped progress-bar-animated"
                role="progressbar"
                style={{ width: `${progress}%` }}
                aria-valuenow={progress}
                aria-valuemin="0"
                aria-valuemax="100"
              ></div>
            </div>
          </div>

          {error && (
            <div className="alert alert-danger d-flex align-items-center" role="alert">
              <svg width="20" height="20" className="me-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div>Error: {error}</div>
            </div>
          )}
        </div>
      </div>

      <div className="row g-4">
        {/* Video Stream */}
        <div className="col-lg-8">
          <div className="card shadow-custom overflow-hidden">
            <div className="card-header bg-dark text-white">
              <h3 className="card-title mb-0 d-flex align-items-center">
                <svg width="20" height="20" className="me-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <span>Live Processing Stream</span>
              </h3>
            </div>
            
            <div className="position-relative bg-dark d-flex align-items-center justify-content-center" style={{aspectRatio: '16/9'}}>
              {isProcessing && status === 'running' ? (
                <img
                  ref={streamImgRef}
                  src={streamUrl}
                  alt="Live Processing Stream"
                  className="w-100 h-100"
                  style={{objectFit: 'contain'}}
                  onError={() => setError('Stream connection failed')}
                />
              ) : (
                <div className="text-center text-muted">
                  <svg width="64" height="64" className="mb-3 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                  <p className="h5">
                    {status === 'queued' || status === 'pending' ? 'Click "Start Processing" to begin' : 'Processing not active'}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Stats and Violations Panel */}
        <div className="col-lg-4">
          {/* Stats Cards */}
          <div className="card shadow-custom mb-4">
            <div className="card-body">
              <h3 className="card-title d-flex align-items-center mb-4">
                <svg width="20" height="20" className="me-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <span>Live Statistics</span>
              </h3>
              
              <div className="d-grid gap-3">
                <StatCard 
                  label="Frames Processed"
                  value={stats.framesProcessed}
                  total={stats.totalFrames}
                  icon="📊"
                />
                <StatCard 
                  label="Processing FPS"
                  value={stats.currentFps}
                  unit="fps"
                  icon="⚡"
                />
                <StatCard 
                  label="Violations Detected"
                  value={stats.violationsCount}
                  highlight={stats.violationsCount > 0}
                  icon="🚨"
                />
              </div>
            </div>
          </div>

          {/* Violations List */}
          <div className="card shadow-custom">
            <div className="card-body">
              <h3 className="card-title d-flex align-items-center mb-4">
                <svg width="20" height="20" className="text-danger me-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
                <span>Recent Violations</span>
              </h3>
              
              <div style={{maxHeight: '320px', overflowY: 'auto'}}>
                {stats.violationsCount === 0 ? (
                  <div className="text-center py-5">
                    <div className="text-muted mb-3">
                      <svg width="48" height="48" className="mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <p className="text-muted">No violations detected yet</p>
                  </div>
                ) : (
                  <div className="text-center py-5">
                    <div className="text-muted mb-3">
                      <svg width="48" height="48" className="mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                      </svg>
                    </div>
                    <p className="text-muted">{stats.violationsCount} violations detected</p>
                    <p className="small text-muted">Detailed results available after processing completes</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ label, value, total, unit, highlight, icon }) => (
  <div className={`card border-2 ${
    highlight ? 'border-danger bg-danger-subtle' : 'border-light bg-light'
  }`}>
    <div className="card-body p-3">
      <div className="d-flex align-items-center justify-content-between mb-2">
        <span className="small text-muted">{label}</span>
        <span style={{fontSize: '1.2rem'}}>{icon}</span>
      </div>
      <p className={`h4 fw-bold mb-0 ${highlight ? 'text-danger' : 'text-dark'}`}>
        {value}{total ? `/${total}` : ''}{unit ? ` ${unit}` : ''}
      </p>
    </div>
  </div>
);

export default VideoProcessor;
