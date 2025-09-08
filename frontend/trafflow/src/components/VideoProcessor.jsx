import React, { useState, useEffect, useRef } from 'react';
import { API_BASE, WS_BASE } from '../config';

const VideoProcessor = ({ jobData, onProcessingComplete }) => {
  const [status, setStatus] = useState('pending');
  const [progress, setProgress] = useState(0);
  const [stats, setStats] = useState({
    framesProcessed: 0,
    totalFrames: 0,
    violationsCount: 0,
    currentFps: 0
  });
  const [violations, setViolations] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [streamUrl, setStreamUrl] = useState('');
  
  const eventSourceRef = useRef(null);
  const streamImgRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  useEffect(() => {
    if (jobData?.job_id) {
      // First check job status
      checkJobStatus();
      // Only connect to events if job is still processing
      const timer = setTimeout(() => {
        if (status === 'processing') {
          connectToEvents();
        }
      }, 1000);
      setStreamUrl(`${API_BASE}/jobs/${jobData.job_id}/stream.mjpg?${Date.now()}`);
      
      return () => {
        clearTimeout(timer);
        if (eventSourceRef.current) {
          eventSourceRef.current.close();
          eventSourceRef.current = null;
        }
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }
      };
    }

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    };
  }, [jobData?.job_id]);

  useEffect(() => {
    let timer;
    if (status === 'processing') {
      timer = setTimeout(() => {
        connectToEvents();
      }, 1000);
    }
    return () => {
      if (timer) clearTimeout(timer);
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    };
  }, [status, jobData?.job_id]);

  const connectToEvents = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    eventSourceRef.current = new EventSource(`${API_BASE}/jobs/${jobData.job_id}/events`);

    eventSourceRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleEvent(data);
      } catch (err) {
        console.error('Error parsing event data:', err);
      }
    };

    eventSourceRef.current.onerror = (error) => {
      console.error('EventSource error:', error);
      setError('Connection to processing server lost');
      setIsProcessing(false);

      // Attempt to reconnect after 3 seconds
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      reconnectTimeoutRef.current = setTimeout(() => {
        if (jobData?.job_id) {
          connectToEvents();
          setError(null);
          setIsProcessing(true);
        }
      }, 3000);
    };
  };

  const handleEvent = (data) => {
    switch (data.type) {
      case 'status':
        setStatus(data.status);
        if (data.status === 'processing') {
          setIsProcessing(true);
        } else if (data.status === 'completed') {
          setIsProcessing(false);
          setTimeout(() => onProcessingComplete(data), 2000);
        } else if (data.status === 'error') {
          setIsProcessing(false);
          setError(data.error || 'Processing failed');
        }
        break;
        
      case 'progress':
        setProgress(data.progress || 0);
        setStats(prev => ({
          ...prev,
          framesProcessed: data.processed_frames || 0,
          totalFrames: data.total_frames || 0,
          currentFps: data.fps || 0
        }));
        break;
        
      case 'violation':
        setViolations(prev => [{
          plateText: data.plate_text || 'Unknown',
          time: new Date(data.violation_time * 1000).toLocaleString(),
          confidence: data.confidence || 0,
          trackId: data.track_id || 'N/A',
          frameNumber: data.frame_number || 0
        }, ...prev].slice(0, 10));
        
        setStats(prev => ({
          ...prev,
          violationsCount: data.violation_count || 0
        }));
        break;
        
      case 'completion':
        setStatus('completed');
        setProgress(100);
        setIsProcessing(false);
        // Fetch detailed results before calling onProcessingComplete
        fetch(`${API_BASE}/jobs/${jobData.job_id}/results`)
          .then(res => res.json())
          .then(results => {
            onProcessingComplete(results);
          })
          .catch(err => {
            console.error('Failed to fetch job results on completion:', err);
            onProcessingComplete({ status: 'completed' }); // fallback
          });
        break;
        
      case 'error':
        setStatus('error');
        setError(data.error || 'Unknown error occurred');
        setIsProcessing(false);
        break;
    }
  };

  const checkJobStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/jobs/${jobData.job_id}`);
      if (response.ok) {
        const jobStatus = await response.json();
        setStatus(jobStatus.status);
        setProgress(jobStatus.progress || 0);
        setStats({
          framesProcessed: jobStatus.processed_frames || 0,
          totalFrames: jobStatus.total_frames || 0,
          violationsCount: jobStatus.violations_count || 0,
          currentFps: 0
        });
        
        if (jobStatus.status === 'completed') {
          setIsProcessing(false);
          // Fetch detailed results
          const resultsResponse = await fetch(`${API_BASE}/jobs/${jobData.job_id}/results`);
          if (resultsResponse.ok) {
            const results = await resultsResponse.json();
            onProcessingComplete(results);
          }
        } else if (jobStatus.status === 'processing') {
          setIsProcessing(true);
          // Connect to events for real-time updates
          connectToEvents();
        }
      }
    } catch (err) {
      console.error('Error checking job status:', err);
    }
  };

  const startProcessing = async () => {
    try {
      setError(null);
      const response = await fetch(`${API_BASE}/jobs/${jobData.job_id}/start`, { 
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to start processing');
      }

      setStatus('processing');
      setIsProcessing(true);
    } catch (err) {
      setError(err.message);
      console.error('Error starting processing:', err);
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'processing': return 'bg-primary';
      case 'completed': return 'bg-success';
      case 'error': return 'bg-danger';
      default: return 'bg-secondary';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'processing': return 'Processing Video';
      case 'completed': return 'Processing Complete';
      case 'error': return 'Processing Error';
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
            
            {!isProcessing && status === 'pending' && (
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
              {isProcessing ? (
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
                    {status === 'pending' ? 'Click "Start Processing" to begin' : 'Processing not active'}
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
                {violations.length === 0 ? (
                  <div className="text-center py-5">
                    <div className="text-muted mb-3">
                      <svg width="48" height="48" className="mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <p className="text-muted">No violations detected yet</p>
                  </div>
                ) : (
                  <div className="d-grid gap-3">
                    {violations.map((violation, idx) => (
                      <div 
                        key={idx}
                        className="border border-danger-subtle bg-danger-subtle rounded-3 p-3 hover-lift"
                      >
                        <div className="d-flex align-items-center justify-content-between mb-2">
                          <span className="font-monospace fw-bold text-danger h5 mb-0">
                            {violation.plateText}
                          </span>
                          <span className="badge bg-danger">
                            #{stats.violationsCount - idx}
                          </span>
                        </div>
                        <div className="small text-muted">
                          <div>🕒 {violation.time}</div>
                          <div>📊 Confidence: {(violation.confidence * 100).toFixed(1)}%</div>
                          <div>🎯 Frame: {violation.frameNumber}</div>
                        </div>
                      </div>
                    ))}
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
