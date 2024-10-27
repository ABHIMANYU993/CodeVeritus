import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import Video from '../../components/Video';
import './Login.css';

const Login = ({videoSrc}) => {
  const [activeTab, setActiveTab] = useState('user');
  const [isSignUp, setIsSignUp] = useState(false);
  const [credentials, setCredentials] = useState({ username: '', password: '', email: '' });
  const [errorMessage, setErrorMessage] = useState(''); // Error message state
  const [successMessage, setSuccessMessage] = useState(''); // Success message state
  const navigate = useNavigate();

  // Handle input changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setCredentials({ ...credentials, [name]: value });
  };

  // Handle login submission
  const handleLogin = async (e) => {
    e.preventDefault();
    setErrorMessage(''); // Clear any previous messages
    setSuccessMessage('');

    try {
      const response = await axios.post('http://localhost:5000/api/login', {
        role: activeTab,
        username: credentials.username,
        password: credentials.password,
      });

      if (response.status === 200) {
        // Save the username to localStorage
        localStorage.setItem('username', credentials.username);
        activeTab === 'user' ? navigate('/companyinfo') : navigate('/results');
      }
    } catch (error) {
      setErrorMessage(error.response?.data?.message || 'Invalid username or password');
    }
  };

  // Handle signup submission
  const handleSignUp = async (e) => {
    e.preventDefault();
    setErrorMessage('');
    setSuccessMessage(''); // Clear any previous messages

    try {
      const response = await axios.post('http://localhost:5000/api/signup', {
        role: activeTab,
        username: credentials.username,
        email: credentials.email,
        password: credentials.password,
      });
      setSuccessMessage(response.data.message); // Display success message in the form
      setIsSignUp(false); // Switch back to login form after successful signup
    } catch (error) {
      setErrorMessage(error.response?.data?.message || 'Sign-up failed');
    }
  };

  // Navigate to home page
  const goToHome = () => navigate('/');

  return (
    <>
      <Video videoSrc="3130182-uhd_3840_2160_30fps.mp4" />
      <div className="login-container">
        <div className="content-section">
          <h2>Welcome to Our Platform</h2>
          <p>Join us to explore personalized tools designed to help you detect and prevent code plagiarism.</p>
          <p>Log in or sign up to get started and access features that enhance coding integrity and academic success.</p>
          <button onClick={goToHome} className="btn btn-secondary">Back</button>
        </div>

        <div className="form-section">
          <div className="tabs">
            <button
              className={`tab ${activeTab === 'user' ? 'active' : ''}`}
              onClick={() => setActiveTab('user')}
            >
              User
            </button>
            <button
              className={`tab ${activeTab === 'admin' ? 'active' : ''}`}
              onClick={() => setActiveTab('admin')}
            >
              Admin
            </button>
          </div>

          {!isSignUp ? (
            <form className="login-form" onSubmit={handleLogin}>
              <h2>{activeTab === 'user' ? 'User Login' : 'Admin Login'}</h2>

              <div className="form-group">
                <label>Username</label>
                <input
                  type="text"
                  name="username"
                  value={credentials.username}
                  onChange={handleInputChange}
                  required
                />
              </div>

              <div className="form-group">
                <label>Password</label>
                <input
                  type="password"
                  name="password"
                  value={credentials.password}
                  onChange={handleInputChange}
                  required
                />
              </div>

              {/* Error message */}
              {errorMessage && <p className="error-message">{errorMessage}</p>}

              <button type="submit" className="btn btn-primary">Login</button>

              <p className="signup-option">
                Don't have an account?{' '}
                <span className="signup-link" onClick={() => setIsSignUp(true)}>
                  Sign Up
                </span>
              </p>
            </form>
          ) : (
            <form className="login-form" onSubmit={handleSignUp}>
              <h2>{activeTab === 'user' ? 'User Sign Up' : 'Admin Sign Up'}</h2>

              <div className="form-group">
                <label>Username</label>
                <input
                  type="text"
                  name="username"
                  value={credentials.username}
                  onChange={handleInputChange}
                  required
                />
              </div>

              <div className="form-group">
                <label>Email</label>
                <input
                  type="email"
                  name="email"
                  value={credentials.email}
                  onChange={handleInputChange}
                  required
                />
              </div>

              <div className="form-group">
                <label>Password</label>
                <input
                  type="password"
                  name="password"
                  value={credentials.password}
                  onChange={handleInputChange}
                  required
                />
              </div>

              {/* Error message */}
              {errorMessage && <p className="error-message">{errorMessage}</p>}

              {/* Success message */}
              {successMessage && <p className="success-message">{successMessage}</p>}

              <button type="submit" className="btn btn-primary">Sign Up</button>

              <p className="signup-option">
                Already have an account?{' '}
                <span className="signup-link" onClick={() => setIsSignUp(false)}>
                  Login
                </span>
              </p>
            </form>
          )}
        </div>
      </div>
    </>
  );
};

export default Login;
