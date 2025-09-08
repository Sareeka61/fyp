import React from 'react';
import { Link } from 'react-router-dom';
import { Container } from 'react-bootstrap';

const LandingPage = () => {
  return (
    <div className="min-vh-100 bg-white">
      {/* Navbar */}
      <nav className="navbar navbar-expand-lg navbar-light bg-white shadow-sm fixed-top">
        <Container>
          <a className="navbar-brand fw-bold" href="#" style={{ color: '#287e74' }}>
            TrafFlow
          </a>
          <div className="collapse navbar-collapse justify-content-end">
            <ul className="navbar-nav">
              <li className="nav-item me-3">
                <Link to="/login" className="btn btn-outline-primary" style={{ borderColor: '#287e74', color: '#287e74' }}>
                  Login
                </Link>
              </li>
              <li className="nav-item">
                <Link
                  to="/signup"
                  className="btn btn-primary"
                  style={{
                    backgroundColor: '#287e74',
                    borderColor: '#287e74',
                    color: 'white',
                  }}
                  onMouseEnter={e => {
                    e.currentTarget.style.backgroundColor = 'white';
                    e.currentTarget.style.color = '#287e74';
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.backgroundColor = '#287e74';
                    e.currentTarget.style.color = 'white';
                  }}
                >
                  Sign Up
                </Link>
              </li>
            </ul>
          </div>
        </Container>
      </nav>

      {/* Hero Section */}
      <Container className="d-flex flex-column justify-content-center align-items-center text-center min-vh-100 pt-5">
        <h1 className="display-4 fw-bold mb-3" style={{ color: '#287e74' }}>
          Nepali Number Plate Detection System
        </h1>
        <p className="lead mb-4" style={{ color: '#287e74' }}>
          Trafflow – Intelligent Traffic Violation Detection
        </p>
        <div>
          <Link to="/login" className="btn btn-primary btn-lg" style={{ backgroundColor: '#287e74', borderColor: '#287e74' }}>
            Get Started
          </Link>
        </div>
      </Container>
    </div>
  );
};

export default LandingPage;
