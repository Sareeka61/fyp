import React from 'react';
import { useAuth } from '../contexts/AuthContext';
import { Button, Dropdown } from 'react-bootstrap';

const Header = ({ onNewUpload, showNewUpload }) => {
  const { user, logout } = useAuth();

  const handleLogout = async () => {
    await logout();
    // After logout, switch to login view
    if (typeof window !== 'undefined' && window.location) {
      // Reload page to reset app state or alternatively use a callback to reset authView
      window.location.reload();
    }
  };

  return (
    <header className="gradient-bg text-white shadow-lg">
      <div className="container py-4">
        <div className="d-flex align-items-center justify-content-between">
          <div className="d-flex align-items-center">
            <div className="bg-white bg-opacity-25 p-3 rounded-3 me-3">
              <svg width="32" height="32" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
              </svg>
            </div>
            <div>
              <h1 className="h2 fw-bold mb-0">TrafFlow</h1>
              <p className="text-white-50 small mb-0">Traffic Recognition & Flow Analysis</p>
            </div>
          </div>

          <div className="d-flex align-items-center">
            {user && (
              <div className="me-3">
                <Dropdown>
                  <Dropdown.Toggle variant="outline-light" id="user-dropdown">
                    Welcome, {user.username}
                  </Dropdown.Toggle>
                  <Dropdown.Menu>
                    <Dropdown.Item onClick={handleLogout}>
                      Logout
                    </Dropdown.Item>
                  </Dropdown.Menu>
                </Dropdown>
              </div>
            )}

            {showNewUpload && (
              <button
                onClick={onNewUpload}
                className="btn btn-outline-light d-flex align-items-center hover-lift"
              >
                <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24" className="me-2">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                <span>New Analysis</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
