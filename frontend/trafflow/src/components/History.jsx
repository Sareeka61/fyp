import React, { useState, useEffect } from 'react';
import { Card, Table, Badge, Button, Pagination, Spinner, Alert } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import { API_BASE } from '../config';

const History = () => {
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [pagination, setPagination] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const navigate = useNavigate();

  useEffect(() => {
    fetchJobHistory(currentPage);
  }, [currentPage]);

  const fetchJobHistory = async (page = 1) => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch(`${API_BASE}/jobs/history?page=${page}&per_page=10`, {
        credentials: 'include',
      });
      if (response.ok) {
        const data = await response.json();
        setJobs(data.jobs);
        setPagination(data.pagination);
      } else if (response.status === 401) {
        setError('Please log in to view your history');
      } else {
        setError('Failed to load job history');
      }
    } catch (err) {
      setError('Network error while loading history');
      console.error('Error fetching job history:', err);
    } finally {
      setLoading(false);
    }
  };

  const getStatusBadge = (status) => {
    const variants = {
      'completed': 'success',
      'processing': 'primary',
      'pending': 'warning',
      'error': 'danger',
      'cancelled': 'secondary'
    };
    return <Badge bg={variants[status] || 'secondary'}>{status}</Badge>;
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  };

  const formatDuration = (seconds) => {
    if (!seconds) return 'N/A';
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  };

  const handleViewResults = () => {
    // For now, redirect to the main app and trigger a new analysis view
    // In a full implementation, we'd have a dedicated results page
    window.location.href = '/';
  };

  const handlePageChange = (page) => {
    setCurrentPage(page);
  };

  if (loading) {
    return (
      <div className="text-center py-5">
        <Spinner animation="border" role="status">
          <span className="visually-hidden">Loading history...</span>
        </Spinner>
        <p className="mt-3 text-muted">Loading your analysis history...</p>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="danger" className="text-center">
        {error}
      </Alert>
    );
  }

  return (
    <div className="container-fluid">
      <div className="d-flex justify-content-between align-items-center mb-4">
        <div>
          <h1 className="display-6 fw-bold text-dark mb-2">Analysis History</h1>
          <p className="text-muted">View your past traffic analysis jobs</p>
        </div>
      </div>

      {jobs.length === 0 ? (
        <Card className="text-center py-5">
          <Card.Body>
            <div className="text-success mb-4">
              <svg width="64" height="64" className="mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
            </div>
            <h3 className="h5 text-muted">No Analysis Jobs Yet</h3>
            <p className="text-muted">Start your first traffic analysis to see it here</p>
            <Button variant="primary" onClick={() => navigate('/')}>
              Start New Analysis
            </Button>
          </Card.Body>
        </Card>
      ) : (
        <>
          <Card>
            <Card.Body className="p-0">
              <div className="table-responsive">
                <Table hover className="mb-0">
                  <thead className="table-light">
                    <tr>
                      <th className="border-0 fw-semibold">Job ID</th>
                      <th className="border-0 fw-semibold">Filename</th>
                      <th className="border-0 fw-semibold">Status</th>
                      <th className="border-0 fw-semibold">Violations</th>
                      <th className="border-0 fw-semibold">Progress</th>
                      <th className="border-0 fw-semibold">Duration</th>
                      <th className="border-0 fw-semibold">Created</th>
                      <th className="border-0 fw-semibold">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {jobs.map((job) => (
                      <tr key={job.job_id}>
                        <td className="font-monospace small">{job.job_id}</td>
                        <td className="text-truncate" style={{ maxWidth: '200px' }} title={job.filename}>
                          {job.filename}
                        </td>
                        <td>{getStatusBadge(job.status)}</td>
                        <td>
                          {job.violations_count > 0 ? (
                            <Badge bg="danger">{job.violations_count}</Badge>
                          ) : (
                            <span className="text-muted">0</span>
                          )}
                        </td>
                        <td>
                          <div className="d-flex align-items-center">
                            <div className="progress flex-grow-1 me-2" style={{ height: '6px', minWidth: '60px' }}>
                              <div
                                className="progress-bar"
                                style={{ width: `${job.progress}%` }}
                              />
                            </div>
                            <small className="text-muted">{Math.round(job.progress)}%</small>
                          </div>
                        </td>
                        <td>{formatDuration(job.duration_seconds)}</td>
                        <td className="small text-muted">{formatDate(job.created_at)}</td>
                        <td>
                          {job.status === 'completed' && (
                            <Button
                              variant="outline-primary"
                              size="sm"
                              onClick={() => handleViewResults(job.job_id)}
                            >
                              View Results
                            </Button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </div>
            </Card.Body>
          </Card>

          {pagination && pagination.pages > 1 && (
            <div className="d-flex justify-content-center mt-4">
              <Pagination>
                <Pagination.Prev
                  disabled={!pagination.has_prev}
                  onClick={() => handlePageChange(currentPage - 1)}
                />
                {Array.from({ length: pagination.pages }, (_, i) => i + 1).map((page) => (
                  <Pagination.Item
                    key={page}
                    active={page === currentPage}
                    onClick={() => handlePageChange(page)}
                  >
                    {page}
                  </Pagination.Item>
                ))}
                <Pagination.Next
                  disabled={!pagination.has_next}
                  onClick={() => handlePageChange(currentPage + 1)}
                />
              </Pagination>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default History;
