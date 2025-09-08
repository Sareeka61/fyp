/**
 * Custom hook for real-time job processing updates via SocketIO
 * DISABLED: Using SSE instead for real-time updates
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import type { JobStats, ViolationEvent, UseJobStreamReturn, SocketEvents } from '../types/job';
import { API_BASE, WS_BASE } from '../config';

// DISABLED: Using SSE instead
// export const useJobStream = (): UseJobStreamReturn => {
const useJobStream = (): UseJobStreamReturn => {
  const [stats, setStats] = useState<JobStats | null>(null);
  const [violations, setViolations] = useState<ViolationEvent[]>([]);
  const [preview, setPreview] = useState<string | null>(null);
  const [status, setStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const socketRef = useRef<Socket | null>(null);
  const currentJobIdRef = useRef<string | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback((jobId: string) => {
    if (socketRef.current && currentJobIdRef.current === jobId) {
      return; // Already connected to this job
    }

    // Disconnect from previous job if any
    if (socketRef.current) {
      disconnect();
    }

    currentJobIdRef.current = jobId;
    setStatus('connecting');
    setError(null);

    try {
      // Create socket connection
      socketRef.current = io(WS_BASE, {
        transports: ['websocket', 'polling'],
        timeout: 5000,
        forceNew: true
      });

      const socket = socketRef.current;

      // Connection handlers
      socket.on('connect', () => {
        console.log('SocketIO connected');
        setStatus('connected');
        setError(null);
        
        // Join the job room
        socket.emit('join_job', { job_id: jobId });
      });

      socket.on('disconnect', (reason) => {
        console.log('SocketIO disconnected:', reason);
        setStatus('disconnected');
        setIsProcessing(false);
        
        // Auto-reconnect after 2 seconds if not intentional
        if (reason !== 'io client disconnect' && currentJobIdRef.current) {
          reconnectTimeoutRef.current = setTimeout(() => {
            if (currentJobIdRef.current) {
              connect(currentJobIdRef.current);
            }
          }, 2000);
        }
      });

      socket.on('connect_error', (err) => {
        console.error('SocketIO connection error:', err);
        setStatus('error');
        setError(`Connection failed: ${err.message}`);
      });

      // Job event handlers
      socket.on('joined_job', (data: SocketEvents['joined_job']) => {
        console.log('Joined job room:', data.job_id);
      });

      socket.on('job_started', (data: SocketEvents['job_started']) => {
        console.log('Job started:', data.job_id);
        setIsProcessing(true);
        setError(null);
      });

      socket.on('job_stats', (data: SocketEvents['job_stats']) => {
        console.log('Job stats update:', data);
        setStats({
          job_id: data.job_id,
          frames: data.frames,
          fps: data.fps,
          violations: data.violations,
          progress: data.progress,
          timestamp: data.timestamp
        });
      });

      socket.on('job_preview', (data: SocketEvents['job_preview']) => {
        setPreview(data.jpeg_b64);
      });

      socket.on('violation', (data: SocketEvents['violation']) => {
        console.log('New violation:', data);
        const violation: ViolationEvent = {
          job_id: data.job_id,
          plate: data.plate,
          confidence: data.confidence,
          bbox: data.bbox,
          frame_thumb: data.frame_thumb,
          timestamp: data.timestamp,
          frame_number: data.frame_number
        };
        
        setViolations(prev => [violation, ...prev].slice(0, 20)); // Keep last 20
      });

      socket.on('job_completed', (data: SocketEvents['job_completed']) => {
        console.log('Job completed:', data.job_id);
        setIsProcessing(false);
        setStats(prevStats => prevStats ? { ...prevStats, progress: 100 } : null);
      });

      socket.on('job_failed', (data: SocketEvents['job_failed']) => {
        console.error('Job failed:', data.error);
        setIsProcessing(false);
        setError(data.error);
        setStatus('error');
      });

      socket.on('error', (data: SocketEvents['error']) => {
        console.error('Socket error:', data.message);
        setError(data.message);
      });

    } catch (err) {
      console.error('Failed to create socket connection:', err);
      setStatus('error');
      setError('Failed to establish connection');
    }
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (socketRef.current) {
      if (currentJobIdRef.current) {
        socketRef.current.emit('leave_job', { job_id: currentJobIdRef.current });
      }
      socketRef.current.disconnect();
      socketRef.current = null;
    }

    currentJobIdRef.current = null;
    setStatus('disconnected');
    setIsProcessing(false);
    setStats(null);
    setViolations([]);
    setPreview(null);
    setError(null);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  // Fallback polling if socket fails
  useEffect(() => {
    if (status === 'error' && currentJobIdRef.current) {
      const jobId = currentJobIdRef.current;
      
      const pollStats = async () => {
        try {
          const response = await fetch(`/api/jobs/${jobId}/stats`);
          if (response.ok) {
            const statsData = await response.json();
            setStats(statsData);
            setError(null);
          }
        } catch (err) {
          console.error('Polling error:', err);
        }
      };

      const pollViolations = async () => {
        try {
          const response = await fetch(`/api/jobs/${jobId}/violations?limit=20`);
          if (response.ok) {
            const violationsData = await response.json();
            setViolations(violationsData);
          }
        } catch (err) {
          console.error('Violations polling error:', err);
        }
      };

      // Poll every 1 second as fallback
      const statsInterval = setInterval(pollStats, 1000);
      const violationsInterval = setInterval(pollViolations, 2000);

      return () => {
        clearInterval(statsInterval);
        clearInterval(violationsInterval);
      };
    }
  }, [status]);

  return {
    stats,
    violations,
    preview,
    status,
    isProcessing,
    error,
    connect,
    disconnect
  };
};

// DISABLED: Using SSE instead
// };
