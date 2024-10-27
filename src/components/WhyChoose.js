import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCode, faBrain, faTasks } from '@fortawesome/free-solid-svg-icons'; 

const WhyChoose = () => {
  return (
    <section className="py-5 bg-light"   style={{
      background: 'linear-gradient(135deg, #2c3e52, #fd746a)',
      color: 'black',
    }}>
      <div className="container">
        <h2 className="text-center fw-bold mb-4">Why Choose CodeVeritus?</h2>
        <div className="row text-center">
          <div className="col-md-4 feature-box">
            <FontAwesomeIcon 
              icon={faCode} 
              size="3x" 
              className="mb-3" 
              style={{ color: '#e74c3c' }} 
            />
            <h4>12+ Languages Supported</h4>
            <p>Including Python, JavaScript, and C++</p>
          </div>
          <div className="col-md-4 feature-box">
            <FontAwesomeIcon 
              icon={faBrain} 
              size="3x" 
              className="mb-3" 
              style={{ color: '#e74c3c' }} 
            />
            <h4>Advanced Detection</h4>
            <p>Harnessing powerful algorithms for smarter decisions</p>
          </div>
          <div className="col-md-4 feature-box">
            <FontAwesomeIcon 
              icon={faTasks} 
              size="3x" 
              className="mb-3" 
              style={{ color: '#e74c3c' }}
            />
            <h4>Batch Submissions</h4>
            <p>Process them all in one go!</p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default WhyChoose;
