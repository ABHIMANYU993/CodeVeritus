import React from 'react';
import { useNavigate } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowRight } from '@fortawesome/free-solid-svg-icons';
const Intro = () => {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate('/login');
  };

  return (
    <section className="hero-section mt-5">
      <div className="container">
        <h1 className="display-4 fw-bold text-center">Welcome to CodeVeritus</h1>
        <p className="lead mt-3">Crush Code Plagiarism with Confidence</p>
        <p className="mt-3">
          At CodeVeritus, we make code integrity effortless. Whether you’re an educator, recruiter, or developer, our tools ensure every submission is authentic and plagiarism-free.
        </p>
        <button
          className="btn btn-lg btn-custom mt-4"
          onClick={handleGetStarted}
        >
          Get Started <FontAwesomeIcon icon={faArrowRight} className="ms-2" />
        </button>
      </div>
    </section>
  );
};

export default Intro;
