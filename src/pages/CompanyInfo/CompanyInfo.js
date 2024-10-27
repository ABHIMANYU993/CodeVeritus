import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './CompanyInfo.css';
const CompanyInfo = () => {
  const navigate = useNavigate(); 

  const [startTime, setStartTime] = useState("");
  const [endTime, setEndTime] = useState("");

  const formatTime = (date) => {
    let hours = date.getHours();
    let minutes = date.getMinutes();
    const ampm = hours >= 12 ? 'PM' : 'AM';
    hours = hours % 12 || 12;
    const minutesStr = minutes < 10 ? '0' + minutes : minutes;
    return `${hours}:${minutesStr} ${ampm}`;
  };

 
  useEffect(() => {
    const now = new Date();
    const start = formatTime(now);

    const end = new Date(now.getTime() + 60 * 60 * 1000); 
    const endFormatted = formatTime(end);

    setStartTime(start);
    setEndTime(endFormatted);
  }, []);

  const handleProceed = () => {
    navigate('/selectusers');
  };

  return (
    <div className="company-container">
      <div className="content-box animate-fade-in">
        <h1 className="heading animate-slide-down">Welcome to CodeVeritus</h1>

        <p className="description">
          This session on our platform is designed to help conduct seamless coding interviews.
          You are allotted the following session time:
        </p>

        <div className="session-details animate-zoom-in">
          <h2>Session Details</h2>
          <p>Duration: <strong>1 hour</strong></p>
          <p>Start Time: <strong>{startTime}</strong></p>
          <p>End Time: <strong>{endTime}</strong></p>
          <p className="note">
            Please complete your tasks within the scheduled time.
          </p>
        </div>

        <button onClick={handleProceed} className="proceed-button animate-bounce">
          Proceed to Code Workspace
        </button>
      </div>
    </div>
  );
};

export default CompanyInfo;
