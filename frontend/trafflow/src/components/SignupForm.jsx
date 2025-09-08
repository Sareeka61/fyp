import React, { useState } from 'react';
import { Form, Button, Alert, Card, InputGroup, ProgressBar } from 'react-bootstrap';
import { useAuth } from '../contexts/AuthContext';
import { Eye, EyeSlash, CheckCircle, XCircle } from 'react-bootstrap-icons';

const SignupForm = ({ onSwitchToLogin, onSignupSuccess }) => {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [touched, setTouched] = useState({
    username: false,
    email: false,
    password: false,
    confirmPassword: false
  });
  const { signup } = useAuth();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
    // Clear error when user starts typing
    if (error) setError('');
  };

  const handleBlur = (field) => {
    setTouched({ ...touched, [field]: true });
  };

  const validateEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const getPasswordStrength = (password) => {
    let strength = 0;
    if (password.length >= 8) strength += 25;
    if (/[A-Z]/.test(password)) strength += 25;
    if (/[a-z]/.test(password)) strength += 25;
    if (/[0-9]/.test(password)) strength += 25;
    return strength;
  };

  const getPasswordStrengthText = (strength) => {
    if (strength < 25) return 'Very Weak';
    if (strength < 50) return 'Weak';
    if (strength < 75) return 'Fair';
    if (strength < 100) return 'Good';
    return 'Strong';
  };

  const getPasswordStrengthColor = (strength) => {
    if (strength < 25) return 'danger';
    if (strength < 50) return 'warning';
    if (strength < 75) return 'info';
    if (strength < 100) return 'primary';
    return 'success';
  };

  const getFieldError = (field) => {
    if (!touched[field]) return '';
    switch (field) {
      case 'username':
        if (!formData.username.trim()) return 'Username is required';
        if (formData.username.length < 3) return 'Username must be at least 3 characters';
        break;
      case 'email':
        if (!formData.email.trim()) return 'Email is required';
        if (!validateEmail(formData.email)) return 'Please enter a valid email address';
        break;
      case 'password':
        if (!formData.password) return 'Password is required';
        if (formData.password.length < 8) return 'Password must be at least 8 characters';
        break;
      case 'confirmPassword':
        if (!formData.confirmPassword) return 'Please confirm your password';
        if (formData.password !== formData.confirmPassword) return 'Passwords do not match';
        break;
      default:
        return '';
    }
    return '';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    // Mark all fields as touched for validation
    setTouched({
      username: true,
      email: true,
      password: true,
      confirmPassword: true
    });

    const errors = ['username', 'email', 'password', 'confirmPassword']
      .map(field => getFieldError(field))
      .filter(error => error);

    if (errors.length > 0) {
      setError(errors[0]);
      return;
    }

    setLoading(true);

    const result = await signup(formData.username.trim(), formData.email.trim(), formData.password);

    if (result.success) {
      // Clear form fields after successful signup
      setFormData({
        username: '',
        email: '',
        password: '',
        confirmPassword: ''
      });
      setTouched({
        username: false,
        email: false,
        password: false,
        confirmPassword: false
      });
      if (onSignupSuccess) {
        onSignupSuccess();
      }
    } else {
      setError(result.error);
    }

    setLoading(false);
  };

  const passwordStrength = getPasswordStrength(formData.password);

  return (
    <Card className="mx-auto shadow-sm" style={{ maxWidth: '500px' }}>
      <Card.Body className="p-4">
        <Card.Title className="text-center mb-4" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
          Create Account
        </Card.Title>

        {error && <Alert variant="danger" className="mb-3">{error}</Alert>}

        <Form onSubmit={handleSubmit}>
          <Form.Group className="mb-3">
            <Form.Label>Username</Form.Label>
            <Form.Control
              type="text"
              name="username"
              value={formData.username}
              onChange={handleChange}
              onBlur={() => handleBlur('username')}
              isInvalid={!!getFieldError('username')}
              required
              placeholder="Choose a username"
              className="form-control-lg"
            />
            <Form.Control.Feedback type="invalid">
              {getFieldError('username')}
            </Form.Control.Feedback>
          </Form.Group>

          <Form.Group className="mb-3">
            <Form.Label>Email</Form.Label>
            <Form.Control
              type="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              onBlur={() => handleBlur('email')}
              isInvalid={!!getFieldError('email')}
              required
              placeholder="Enter your email"
              className="form-control-lg"
            />
            <Form.Control.Feedback type="invalid">
              {getFieldError('email')}
            </Form.Control.Feedback>
          </Form.Group>

          <Form.Group className="mb-4">
            <Form.Label>Password</Form.Label>
            <InputGroup>
              <Form.Control
                type={showPassword ? 'text' : 'password'}
                name="password"
                value={formData.password}
                onChange={handleChange}
                onBlur={() => handleBlur('password')}
                isInvalid={!!getFieldError('password')}
                required
                placeholder="Create a password"
                className="form-control-lg"
              />
              <Button
                variant="outline-secondary"
                onClick={() => setShowPassword(!showPassword)}
                className="border-start-0"
              >
                {showPassword ? <EyeSlash /> : <Eye />}
              </Button>
            </InputGroup>
            {formData.password && (
              <div className="mt-2">
                <small className="text-muted">Password strength: {getPasswordStrengthText(passwordStrength)}</small>
                <ProgressBar
                  now={passwordStrength}
                  variant={getPasswordStrengthColor(passwordStrength)}
                  className="mt-1"
                  style={{ height: '6px' }}
                />
              </div>
            )}
            <Form.Control.Feedback type="invalid">
              {getFieldError('password')}
            </Form.Control.Feedback>
          </Form.Group>

          <Form.Group className="mb-4">
            <Form.Label>Confirm Password</Form.Label>
            <InputGroup>
              <Form.Control
                type={showConfirmPassword ? 'text' : 'password'}
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleChange}
                onBlur={() => handleBlur('confirmPassword')}
                isInvalid={!!getFieldError('confirmPassword')}
                required
                placeholder="Confirm your password"
                className="form-control-lg"
              />
              <Button
                variant="outline-secondary"
                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                className="border-start-0"
              >
                {showConfirmPassword ? <EyeSlash /> : <Eye />}
              </Button>
              {formData.confirmPassword && (
                <InputGroup.Text className="bg-transparent border-start-0">
                  {formData.password === formData.confirmPassword ? (
                    <CheckCircle className="text-success" />
                  ) : (
                    <XCircle className="text-danger" />
                  )}
                </InputGroup.Text>
              )}
            </InputGroup>
            <Form.Control.Feedback type="invalid">
              {getFieldError('confirmPassword')}
            </Form.Control.Feedback>
          </Form.Group>

          <Button
            variant="primary"
            type="submit"
            className="w-100 mb-3"
            disabled={loading}
            size="lg"
          >
            {loading ? 'Creating account...' : 'Sign Up'}
          </Button>
        </Form>

        <div className="text-center">
          <Button
            variant="link"
            onClick={onSwitchToLogin}
            className="p-0 text-decoration-none"
          >
            Already have an account? <span className="text-primary">Log in</span>
          </Button>
        </div>
      </Card.Body>
    </Card>
  );
};

export default SignupForm;
