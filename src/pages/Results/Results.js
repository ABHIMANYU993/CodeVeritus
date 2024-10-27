import React, { useEffect, useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { Link } from "react-router-dom";
import axios from "axios";
import { FaArrowLeft } from "react-icons/fa"; 
import "./Results.css"; 

const ResultsPage = () => {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchResults = async () => {
      try {
        const response = await axios.get("http://localhost:5000/api/backend");
        setResults(response.data);
      } catch (err) {
        console.error("Error fetching results:", err);
        setError("Failed to load results. Please try again.");
      } finally {
        setLoading(false);
      }
    };
    fetchResults();
  }, []);

  if (loading) {
    return <div className="loading-text animate__animated animate__fadeIn">Loading results...</div>;
  }

  if (error) {
    return (
      <div className="alert alert-danger text-center my-5 animate__animated animate__shakeX">
        {error}
      </div>
    );
  }

  return (
    <div className="results-page">
      <div className="container py-5 my-5 animate__animated animate__fadeIn">
        <h2 className="page-title text-center">Code Submitted by Users</h2>
        <p className="text-center subtitle">
          Review the predictions generated from user code submissions below.
        </p>

        <div className="results-container">
          {results.length > 0 ? (
            results.map((result, index) => (
              <div key={index} className="result-card animate__animated animate__fadeInUp">
                <div className="card-header">
                  <h5>Submitted by: {result.userId}</h5>
                </div>
                <div className="card-body">
                  <pre className="code-block">{result.codes.join('\n')}</pre>
                  <p className="prediction-text">
                    <strong>Prediction:</strong> {result.prediction}
                  </p>
                </div>
              </div>
            ))
          ) : (
            <div className="alert alert-warning text-center animate__animated animate__bounceIn">
              No results to display. Please submit some code for analysis.
            </div>
          )}
        </div>

        <div className="text-center mt-4">
          <Link to="/" className="btn btn-primary back-button">
            <FaArrowLeft className="me-2" /> Go Back
          </Link>
        </div>
      </div>
    </div>
  );
};

export default ResultsPage;
