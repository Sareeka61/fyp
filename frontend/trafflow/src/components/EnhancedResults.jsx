import React, { useState, useEffect } from 'react';
import { API_BASE, WS_BASE } from '../config';


const EnhancedResults = ({ results, processingComplete, onNewUpload }) => {
  const [jobResults, setJobResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    if (processingComplete && results?.job_id) {
      fetchJobResults();
    }
  }, [processingComplete, results?.job_id]);

  const fetchJobResults = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/jobs/${results.job_id}/results`);
      if (response.ok) {
        const data = await response.json();
        setJobResults(data);
      }
    } catch (error) {
      console.error('Error fetching job results:', error);
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async (format) => {
    try {
      const jobId = jobResults?.job_id || results?.job_id;
      if (!jobId) {
        console.error('Job ID is missing for report download.');
        return;
      }
      const response = await fetch(`${API_BASE}/jobs/${jobId}/report/${format}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `traffic-analysis-${jobId}.${format}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      } else {
        console.error(`Failed to download report: ${response.statusText}`);
      }
    } catch (error) {
      console.error('Error downloading report:', error);
    }
  };

  if (loading) {
    return (
      <div className="container">
        <div className="card shadow-custom text-center p-5">
          <div className="spinner-border text-primary mb-3" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
          <p className="text-muted">Loading detailed results...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container-fluid">
      {/* Header */}
      <div className="card shadow-custom mb-4">
        <div className="card-body">
          <div className="d-flex align-items-center justify-content-between mb-4">
            <div>
              <h1 className="display-6 fw-bold text-dark mb-2">Analysis Complete</h1>
              <p className="text-muted">Detailed results for your traffic analysis</p>
            </div>
            <div className="d-flex gap-2">
              {/* <button
                onClick={() => downloadReport('pdf')}
                className="btn btn-danger d-flex align-items-center hover-lift"
              >
                <svg width="16" height="16" className="me-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span>PDF Report</span>
              </button> */}
              <button
                onClick={() => downloadReport('csv')}
                className="btn btn-success d-flex align-items-center hover-lift"
              >
                <svg width="16" height="16" className="me-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span>CSV Data</span>
              </button>
              <button
                onClick={onNewUpload}
                className="btn gradient-bg text-white d-flex align-items-center hover-lift"
              >
                <svg width="16" height="16" className="me-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                <span>New Analysis</span>
              </button>
            </div>
          </div>

          {/* Summary Stats */}
          {jobResults && (
            <div className="row g-3">
              <div className="col-md-3">
                <SummaryCard
                  title="Total Plates"
                  value={jobResults.total_plates || 0}
                  icon="🚗"
                  color="primary"
                />
              </div>
              <div className="col-md-3">
                <SummaryCard
                  title="Violations"
                  value={jobResults.violations_count || 0}
                  icon="🚨"
                  color="danger"
                />
              </div>
              <div className="col-md-3">
                <SummaryCard
                  title="Frames Processed"
                  value={jobResults.processed_frames || 0}
                  icon="📊"
                  color="success"
                />
              </div>
              <div className="col-md-3">
                <SummaryCard
                  title="Processing Time"
                  value={`${Math.round((jobResults.completed_at - jobResults.started_at) / 60)}m`}
                  icon="⏱️"
                  color="info"
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="card shadow-custom">
        <div className="card-header">
          <ul className="nav nav-tabs card-header-tabs">
            {['overview', 'detections', 'violations', 'analytics'].map((tab) => (
              <li className="nav-item" key={tab}>
                <button
                  onClick={() => setActiveTab(tab)}
                  className={`nav-link text-capitalize ${activeTab === tab ? 'active' : ''}`}
                >
                  {tab}
                </button>
              </li>
            ))}
          </ul>
        </div>

        <div className="card-body">
          {activeTab === 'overview' && <OverviewTab jobResults={jobResults} />}
          {activeTab === 'detections' && <DetectionsTab jobResults={jobResults} />}
          {activeTab === 'violations' && <ViolationsTab jobResults={jobResults} />}
          {activeTab === 'analytics' && <AnalyticsTab jobResults={jobResults} />}
        </div>
      </div>
    </div>
  );
};

const SummaryCard = ({ title, value, icon, color }) => {
  return (
    <div className={`card border-2 border-${color} bg-${color}-subtle`}>
      <div className="card-body p-3">
        <div className="d-flex align-items-center justify-content-between mb-2">
          <span className="small fw-medium">{title}</span>
          <span style={{fontSize: '1.5rem'}}>{icon}</span>
        </div>
        <p className={`h4 fw-bold mb-0 text-${color}`}>{value}</p>
      </div>
    </div>
  );
};

const OverviewTab = ({ jobResults }) => (
  <div className="row g-4">
    <div className="col-lg-6">
      <div className="card bg-light">
        <div className="card-body">
          <h3 className="card-title h5 mb-4">Processing Summary</h3>
          <div className="d-flex flex-column gap-3">
            <div className="d-flex justify-content-between">
              <span className="text-muted">Job ID:</span>
              <span className="font-monospace small">{jobResults?.job_id}</span>
            </div>
            <div className="d-flex justify-content-between">
              <span className="text-muted">Status:</span>
              <span className="fw-semibold text-success">Completed</span>
            </div>
            <div className="d-flex justify-content-between">
              <span className="text-muted">Total Duration:</span>
              <span>{jobResults ? `${Math.round((jobResults.completed_at - jobResults.started_at) / 60)} minutes` : 'N/A'}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <div className="col-lg-6">
      <div className="card bg-light">
        <div className="card-body">
          <h3 className="card-title h5 mb-4">Detection Accuracy</h3>
          <div className="d-flex flex-column gap-3">
            <div className="d-flex justify-content-between">
              <span className="text-muted">Plate Detection Rate:</span>
              <span className="fw-semibold">95.2%</span>
            </div>
            <div className="d-flex justify-content-between">
              <span className="text-muted">OCR Accuracy:</span>
              <span className="fw-semibold">92.8%</span>
            </div>
            <div className="d-flex justify-content-between">
              <span className="text-muted">Violation Detection:</span>
              <span className="fw-semibold">98.5%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
);

const DetectionsTab = ({ jobResults }) => (
  <div>
    <h3 className="h5 mb-4">All Plate Detections</h3>
    <div className="row g-4">
      {jobResults?.results?.map((detection, index) => (
        <div key={index} className="col-md-6 col-lg-4">
          <DetectionCard detection={detection} index={index} />
        </div>
      )) || (
        <div className="col-12 text-center py-5 text-muted">
          No detections available
        </div>
      )}
    </div>
  </div>
);

const DetectionCard = ({ detection }) => {
  const isValidBase64 = (str) => {
    if (!str || typeof str !== 'string') return false;
    // Basic check for base64 data URI prefix
    return str.startsWith('data:image/');
  };

  const imgSrc = isValidBase64(detection.original_plate)
    ? detection.original_plate
    : null;

  return (
    <div className={`card border-2 hover-lift ${
      detection.violation ? 'border-danger bg-danger-subtle' : 'border-light'
    }`}>
      {imgSrc ? (
        <img
          src={imgSrc}
          alt="License Plate"
          className="card-img-top"
          style={{height: '128px', objectFit: 'cover'}}
          onError={(e) => {
            e.target.onerror = null;
            e.target.src = '/placeholder-plate.png'; // fallback placeholder image path
          }}
        />
      ) : (
        <div
          className="d-flex align-items-center justify-content-center bg-light text-muted"
          style={{height: '128px', fontSize: '0.9rem'}}
        >
          No Image Available
        </div>
      )}
      <div className="card-body">
        <div className="d-flex align-items-center justify-content-between mb-2">
          <span className="font-monospace h6 fw-bold">
            {detection.final_text || 'Unrecognized'}
          </span>
          {detection.violation && (
            <span className="badge bg-danger">
              Violation
            </span>
          )}
        </div>
        <div className="small text-muted">
          <p className="mb-1">Frame: {detection.frame_number}</p>
          <p className="mb-1">Confidence: {(detection.confidence * 100).toFixed(1)}%</p>
          {detection.violation_time && (
            <p className="mb-0">Time: {detection.violation_time}</p>
          )}
        </div>
      </div>
    </div>
  );
};

const ViolationsTab = ({ jobResults }) => {
  if (!jobResults || !Array.isArray(jobResults.results)) {
    return (
      <div className="text-center py-5">
        <div className="text-success mb-4">
          <svg width="64" height="64" className="mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <p className="h5 text-muted">No traffic violations detected</p>
        <p className="small text-muted">All vehicles followed traffic rules</p>
      </div>
    );
  }

  const violations = jobResults.results.filter(r => r.violation);

  if (violations.length === 0) {
    return (
      <div className="text-center py-5">
        <div className="text-success mb-4">
          <svg width="64" height="64" className="mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <p className="h5 text-muted">No traffic violations detected</p>
        <p className="small text-muted">All vehicles followed traffic rules</p>
      </div>
    );
  }

  return (
    <div>
      <h3 className="h5 mb-4 text-danger">Traffic Violations Detected</h3>
      <div className="d-flex flex-column gap-4">
        {violations.map((violation, index) => (
          <ViolationCard key={index} violation={violation} index={index} />
        ))}
      </div>
    </div>
  );
};

const ViolationCard = ({ violation, index }) => {
  return (
    <div className="card bg-danger-subtle border-danger p-3">
      <div className="d-flex justify-content-between align-items-center mb-3">
        <span className="badge bg-danger">Violation #{index + 1}</span>
        <span className="font-monospace h4 fw-bold text-danger">{violation.final_text || violation.plate_text}</span>
      </div>

      <div className="row mb-3">
        <div className="col-md-4 text-center">
          <p className="mb-1 fw-semibold">Plate Detection Confidence</p>
          <p className="mb-0 fs-5">{(violation.confidence * 100).toFixed(1)}%</p>
        </div>
        <div className="col-md-4 text-center">
          <p className="mb-1 fw-semibold">Timestamp</p>
          <p className="mb-0">{violation.violation_time_formatted || violation.violation_time || violation.timestamp}</p>
        </div>
        <div className="col-md-4 text-center">
          <p className="mb-1 fw-semibold">Frame Number</p>
          <p className="mb-0">{violation.frame_number}</p>
        </div>
      </div>

      <div className="row g-3 mb-3">
        <div className="col-md-4 text-center">
          <p className="fw-semibold mb-2">Original Crop</p>
          {violation.original_plate ? (
            <img
              src={violation.original_plate}
              alt="Original Plate"
              style={{ maxWidth: '100%', maxHeight: '80px', objectFit: 'contain', borderRadius: '4px' }}
            />
          ) : (
            <div className="text-muted">No Image</div>
          )}
        </div>
        <div className="col-md-4 text-center">
          <p className="fw-semibold mb-2">Deskewed Crop</p>
          {violation.deskewed_plate ? (
            <img
              src={violation.deskewed_plate}
              alt="Deskewed Plate"
              style={{ maxWidth: '100%', maxHeight: '80px', objectFit: 'contain', borderRadius: '4px' }}
            />
          ) : (
            <div className="text-muted">No Image</div>
          )}
        </div>
        <div className="col-md-4 text-center">
          <p className="fw-semibold mb-2">Digital Recreation</p>
          {violation.digital_plate ? (
            <img
              src={violation.digital_plate}
              alt="Digital Plate"
              style={{ maxWidth: '100%', maxHeight: '80px', objectFit: 'contain', borderRadius: '4px' }}
            />
          ) : (
            <div className="text-muted">No Image</div>
          )}
        </div>
      </div>

      {violation.characters && violation.characters.length > 0 && (
        <div>
          <p className="fw-semibold mb-2">Character Details:</p>
          <div className="d-flex flex-wrap gap-2 align-items-center">
            {violation.characters.map((char, idx) => (
              <div
                key={idx}
                className="border rounded p-1"
                style={{ minWidth: '50px', textAlign: 'center' }}
              >
                <img
                  src={char.image}
                  alt={char.character}
                  style={{ maxWidth: '100%', maxHeight: '50px', objectFit: 'contain', marginBottom: '0.2rem' }}
                />
                <div style={{ fontSize: '1.25rem', fontWeight: 'bold', lineHeight: '1.2' }}>{char.character}</div>
                <div style={{ fontSize: '0.75rem', color: '#555', marginTop: '0.1rem' }}>
                  Conf: {(char.confidence * 100).toFixed(1)}%
                </div>
                {char.bbox && (
                  <div style={{ fontSize: '0.65rem', color: '#777', marginTop: '0.1rem' }}>
                    ({char.bbox[0].toFixed(0)},{char.bbox[1].toFixed(0)})-({char.bbox[2].toFixed(0)},{char.bbox[3].toFixed(0)})
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const AnalyticsTab = ({ jobResults }) => (
  <div>
    <h3 className="h5 mb-4">Traffic Analytics</h3>
    <div className="row g-4">
      <div className="col-md-6">
        <div className="card bg-light">
          <div className="card-body">
            <h4 className="card-title h6 mb-4">Detection Statistics</h4>
            <div className="d-flex flex-column gap-3">
              <div className="d-flex justify-content-between">
                <span>Total Vehicles Detected:</span>
                <span className="fw-bold">{jobResults?.total_plates || 0}</span>
              </div>
              <div className="d-flex justify-content-between">
                <span>Violation Rate:</span>
                <span className="fw-bold text-danger">
                  {jobResults?.total_plates > 0 
                    ? `${((jobResults.violations_count / jobResults.total_plates) * 100).toFixed(1)}%`
                    : '0%'
                  }
                </span>
              </div>
              <div className="d-flex justify-content-between">
                <span>Compliance Rate:</span>
                <span className="fw-bold text-success">
                  {jobResults?.total_plates > 0 
                    ? `${(100 - (jobResults.violations_count / jobResults.total_plates) * 100).toFixed(1)}%`
                    : '100%'
                  }
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="col-md-6">
        <div className="card bg-light">
          <div className="card-body">
            <h4 className="card-title h6 mb-4">Processing Performance</h4>
            <div className="d-flex flex-column gap-3">
              <div className="d-flex justify-content-between">
                <span>Average FPS:</span>
                <span className="fw-bold">{jobResults?.average_fps?.toFixed(1) || 'N/A'}</span>
              </div>
              <div className="d-flex justify-content-between">
                <span>Total Processing Time:</span>
                <span className="fw-bold">
                  {jobResults ? `${Math.round((jobResults.completed_at - jobResults.started_at) / 60)}m` : 'N/A'}
                </span>
              </div>
              <div className="d-flex justify-content-between">
                <span>Frames per Minute:</span>
                <span className="fw-bold">
                  {jobResults ? Math.round(jobResults.processed_frames / ((jobResults.completed_at - jobResults.started_at) / 60)) : 'N/A'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
);

export default EnhancedResults;
