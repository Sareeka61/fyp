import React from 'react';
import { useNavigate } from 'react-router-dom';
import SignupForm from '../components/SignupForm';
import { Container } from 'react-bootstrap';

const SignupPage = () => {
  const navigate = useNavigate();

  const handleSwitchToLogin = () => {
    navigate('/login');
  };

  const handleSignupSuccess = () => {
    navigate('/login');
  };

  return (
    <div className="min-vh-100 d-flex align-items-center justify-content-center bg-light">
      <Container style={{ maxWidth: '450px' }}>
        <SignupForm
          onSwitchToLogin={handleSwitchToLogin}
          onSignupSuccess={handleSignupSuccess}
        />
      </Container>
    </div>
  );
};

export default SignupPage;
