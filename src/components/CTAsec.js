import React from 'react';
import { useNavigate } from 'react-router-dom'; 
const CTAsec = () => {
  const navigate = useNavigate(); 
  const handleUploadClick = () => {
    navigate('/login'); 
  };

  return (
    <section
      className="text-center py-5"
      style={{
        background: 'linear-gradient(135deg, #2c3e52, #fd746a)',
        color: 'white',
      }}
    >
      <h2 className="fw-bold">Own the Future with CodeVeritus</h2>
      <p className="lead mt-3">
        The future of programming belongs to original thinkers—and with CodeVeritus, that future starts today.
      </p>
      <button
        className="btn btn-lg btn-light mt-3"
        onClick={handleUploadClick}
      >
        Upload Your Code
      </button>
    </section>
  );
};

export default CTAsec;
