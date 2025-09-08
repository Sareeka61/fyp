/**
 * TypeScript types for job processing and real-time updates
 */

export interface JobStats {
  job_id: string;
  frames: number;
  fps: number;
  violations: number;
  progress: number;
  timestamp: string;
}

export interface ViolationEvent {
  job_id: string;
  plate: string;
  confidence?: number;
  bbox?: [number, number, number, number];
  frame_thumb?: string;
  timestamp: string;
  frame_number: number;
}

export interface JobStatus {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'error' | 'cancelled';
  created_at: number;
  started_at?: number;
  completed_at?: number;
  progress: number;
  total_frames: number;
  processed_frames: number;
  violations_count: number;
  error_message?: string;
  video_path: string;
}

export interface SocketEvents {
  // Outgoing events
  join_job: { job_id: string };
  leave_job: { job_id: string };
  
  // Incoming events
  connected: { status: string; session_id: string };
  joined_job: { job_id: string; status: string };
  left_job: { job_id: string; status: string };
  job_started: { job_id: string; timestamp: number };
  job_stats: JobStats & { timestamp: number };
  job_preview: { job_id: string; jpeg_b64: string; timestamp: number };
  violation: ViolationEvent & { timestamp: number };
  job_completed: JobStats & { timestamp: number };
  job_failed: { job_id: string; error: string; timestamp: number };
  error: { message: string };
}

export interface UseJobStreamReturn {
  stats: JobStats | null;
  violations: ViolationEvent[];
  preview: string | null;
  status: 'connecting' | 'connected' | 'disconnected' | 'error';
  isProcessing: boolean;
  error: string | null;
  connect: (jobId: string) => void;
  disconnect: () => void;
}
