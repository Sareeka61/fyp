import React, { useEffect, useState, useRef } from 'react';
import { API_BASE, WS_BASE } from '../config';

const Results = ({ results, type }) => {
  const [status, setStatus] = useState('pending');
  const [progress, setProgress] = useState(0);
  const [stats, setStats] = useState({
    framesProcessed: 0,
    totalFrames: 0,
    violationsCount: 0,
    currentFps: 0
  });
  const [violations, setViolations] = useState([]);

  const eventSourceRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  
  const connectToEvents = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    eventSourceRef.current = new EventSource(`${API_BASE}/jobs/${results.job_id}/events`);

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

      // Attempt to reconnect after 3 seconds
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      reconnectTimeoutRef.current = setTimeout(() => {
        if (results?.job_id) {
          connectToEvents();
        }
      }, 3000);
    };
  };

  useEffect(() => {
    if (type === 'video' && results?.job_id) {
      connectToEvents();
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
  }, [results?.job_id]);

  const handleEvent = (data) => {
    switch (data.type) {
      case 'status':
        setStatus(data.status);
        break;
      case 'progress':
        setProgress(data.progress);
        setStats(prev => ({
          ...prev,
          framesProcessed: data.processed_frames,
          totalFrames: data.total_frames,
          currentFps: ((data.processed_frames / data.total_frames) * 100).toFixed(1)
        }));
        break;
      case 'violation':
        setViolations(prev => [{
          plateText: data.plate_text,
          time: new Date(data.violation_time * 1000).toLocaleString(),
          ...data
        }, ...prev].slice(0, 10));
        setStats(prev => ({
          ...prev,
          violationsCount: data.violation_count
        }));
        break;
    }
  };

  if (!results) return null;

  return (
    <div className="bg-white p-8 rounded-xl shadow-lg mt-8 max-w-7xl mx-auto">
      <h2 className="text-3xl font-bold mb-6 text-gray-800 border-b pb-4">
        Analysis Results
      </h2>
      
      {type === 'video' && (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-xl border border-blue-100">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className={`h-4 w-4 rounded-full ${
                  status === 'processing' ? 'bg-blue-500 animate-pulse' :
                  status === 'completed' ? 'bg-green-500' :
                  status === 'error' ? 'bg-red-500' :
                  'bg-gray-500'
                }`}/>
                <h3 className="text-xl font-semibold text-blue-800">
                  {status === 'processing' ? 'Processing Video' :
                   status === 'completed' ? 'Processing Complete' :
                   status === 'error' ? 'Processing Error' :
                   'Waiting to Start'}
                </h3>
              </div>
              <span className="font-mono bg-blue-50 px-2 py-1 rounded text-sm">
                Job: {results.job_id}
              </span>
            </div>

            <div className="mt-4">
              <div className="relative h-2 bg-blue-100 rounded-full overflow-hidden">
                <div 
                  className="absolute h-full bg-blue-500 transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-right text-sm text-gray-600 mt-1">{Math.round(progress)}%</p>
            </div>
          </div>

          {/* Video Preview */}
          <div className="bg-black rounded-xl overflow-hidden aspect-video">
            {status === 'processing' && (
              <img
              src={`${API_BASE}/jobs/${results.job_id}/stream.mjpg?${Date.now()}`}
                alt="Live Processing"
                className="w-full h-full object-contain"
              />
            )}
          </div>

          {/* Stats Panel */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard 
              label="Frames Processed"
              value={stats.framesProcessed}
              total={stats.totalFrames}
            />
            <StatCard 
              label="Processing FPS"
              value={stats.currentFps}
              unit="fps"
            />
            <StatCard 
              label="Violations"
              value={stats.violationsCount}
              highlight={stats.violationsCount > 0}
            />
            <StatCard 
              label="Progress"
              value={Math.round(progress)}
              unit="%"
            />
          </div>

          {/* Violations List */}
          <div className="bg-white rounded-xl shadow p-6">
            <h3 className="text-lg font-semibold mb-4">Recent Violations</h3>
            {violations.length === 0 ? (
              <p className="text-gray-500 text-center py-4">No violations detected yet</p>
            ) : (
              <div className="space-y-3">
                {violations.map((violation, idx) => (
                  <div 
                    key={idx}
                    className="flex items-center justify-between p-3 bg-red-50 rounded-lg border border-red-200"
                  >
                    <div>
                      <p className="font-mono font-bold text-red-700">{violation.plateText}</p>
                      <p className="text-sm text-gray-600">{violation.time}</p>
                    </div>
                    <span className="text-xs bg-red-100 text-red-800 px-2 py-1 rounded-full">
                      Violation #{stats.violationsCount - idx}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {type === 'image' && results.results && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {results.results.map((result, index) => (
            <div 
              key={index} 
              className={`rounded-xl overflow-hidden transition-all duration-300 hover:shadow-xl
                ${result.violation_detected 
                  ? 'border-2 border-red-300 bg-red-50' 
                  : 'border border-gray-200 bg-white'}`}
            >
              <div className="relative">
                <img 
                  src={result.original_plate} 
                  alt="License Plate" 
                  className="w-full h-48 object-cover"
                />
                {result.violation_detected && (
                  <div className="absolute top-4 right-4">
                    <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-red-100 text-red-800">
                      <span className="h-2 w-2 rounded-full bg-red-500 mr-2"></span>
                      Violation
                    </span>
                  </div>
                )}
              </div>
              <div className="p-4">
                <p className="font-mono text-2xl text-center font-bold tracking-wider text-gray-700">
                  {result.final_text || 'N/A'}
                </p>
                {result.confidence && (
                  <p className="text-sm text-gray-500 text-center mt-2">
                    Confidence: {(result.confidence * 100).toFixed(1)}%
                  </p>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {type === 'image' && results.results && results.results.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-500 text-lg">No results found</p>
        </div>
      )}
    </div>
  );
};

const StatCard = ({ label, value, total, unit, highlight }) => (
  <div className={`p-4 rounded-lg border ${
    highlight ? 'border-red-200 bg-red-50' : 'border-gray-200 bg-white'
  }`}>
    <p className="text-sm text-gray-600">{label}</p>
    <p className={`text-2xl font-bold ${highlight ? 'text-red-600' : 'text-gray-800'}`}>
      {value}{total ? `/${total}` : ''}{unit ? ` ${unit}` : ''}
    </p>
  </div>
);

export default Results;
