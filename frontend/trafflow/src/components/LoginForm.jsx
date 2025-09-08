import React, { useState } from 'react';
import { Form, Button, Alert, Card, InputGroup } from 'react-bootstrap';
import { useAuth } from '../contexts/AuthContext';
import { Eye, EyeSlash } from 'react-bootstrap-icons';

const LoginForm = ({ onSwitchToSignup }) => {
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [touched, setTouched] = useState({ username: false, password: false });
  const { login } = useAuth();

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

  const getFieldError = (field) => {
    if (!touched[field]) return '';
    if (field === 'username' && !formData.username.trim()) return 'Username is required';
    if (field === 'password' && !formData.password) return 'Password is required';
    return '';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    // Capture current form data locally
    const currentUsername = formData.username.trim();
    const currentPassword = formData.password;

    // Mark all fields as touched for validation
    setTouched({ username: true, password: true });

    // Clear form fields immediately on submit click
    setFormData({
      username: '',
      password: ''
    });
    setTouched({ username: false, password: false });

    if (!currentUsername || !currentPassword) {
      setLoading(false);
      return;
    }

    const result = await login(currentUsername, currentPassword);

    if (result.success) {
      // Clear form fields after successful login
      setFormData({
        username: '',
        password: ''
      });
      setTouched({ username: false, password: false });
    } else {
      setError(result.error);
    }

    setLoading(false);
  };

  return (
    <Card className="mx-auto shadow-sm" style={{ maxWidth: '450px' }}>
      <Card.Body className="p-4">
        <Card.Title className="text-center mb-4" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
          Welcome Back
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
              placeholder="Enter your username"
              className="form-control-lg"
            />
            <Form.Control.Feedback type="invalid">
              {getFieldError('username')}
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
                placeholder="Enter your password"
                className="form-control-lg"
              />
              <Button
                variant="outline-secondary"
                onClick={() => setShowPassword(!showPassword)}
                className="border-start-0"
              >
                {showPassword ? <EyeSlash /> : <Eye />}
              </Button>
              <Form.Control.Feedback type="invalid">
                {getFieldError('password')}
              </Form.Control.Feedback>
            </InputGroup>
          </Form.Group>

          <Button
            variant="primary"
            type="submit"
            className="w-100 mb-3"
            disabled={loading}
            size="lg"
          >
            {loading ? 'Logging in...' : 'Login'}
          </Button>
        </Form>

        <div className="text-center">
          <Button
            variant="link"
            onClick={onSwitchToSignup}
            className="p-0 text-decoration-none"
          >
            Don't have an account? <span className="text-primary">Sign up</span>
          </Button>
        </div>
      </Card.Body>
    </Card>
  );
};

export default LoginForm;
