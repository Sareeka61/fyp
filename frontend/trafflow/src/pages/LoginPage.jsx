import React from 'react';
import { useNavigate } from 'react-router-dom';
import LoginForm from '../components/LoginForm';
import { Container } from 'react-bootstrap';

const LoginPage = () => {
  const navigate = useNavigate();

  const handleSwitchToSignup = () => {
    navigate('/signup');
  };

  return (
    <div className="min-vh-100 d-flex align-items-center justify-content-center bg-light">
      <Container style={{ maxWidth: '450px' }}>
        <LoginForm onSwitchToSignup={handleSwitchToSignup} />
      </Container>
    </div>
  );
};

export default LoginPage;
