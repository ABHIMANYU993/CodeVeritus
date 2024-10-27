import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSwatchbook, faRandom, faCode, faComments } from '@fortawesome/free-solid-svg-icons';

const Features = () => {
  return (
    <div className="container my-5">
      <h2 className="text-center fw-bold mb-5" style={{ color: 'rgb(237, 231, 231)' }}>
        How CodeVeritus Takes Detection to the Next Level
      </h2>
      <div className="row g-4">
        <div className="features col-md-6 feature-box">
          <h4>We detect the sneakiest tricks, including:</h4>
          <ul>
            <li>
              <FontAwesomeIcon icon={faSwatchbook} className="me-2" />
              Variable and function name swaps
            </li>
            <li>
              <FontAwesomeIcon icon={faRandom} className="me-2" />
              Restructured code modules
            </li>
            <li>
              <FontAwesomeIcon icon={faCode} className="me-2" />
              Control flow alterations
            </li>
            <li>
              <FontAwesomeIcon icon={faComments} className="me-2" />
              Comment and formatting tweaks
            </li>
          </ul>
        </div>
        <div className="features col-md-6 feature-box">
          <h4>Detection Tiers – Nothing Slips Through the Cracks</h4>
          <ul>
            <li>
              <span style={{ fontWeight: 'bold', color: '#2C2D2D' }}>Tier 1:</span> Catch minor comment edits and formatting changes.
            </li>
            <li>
              <span style={{ fontWeight: 'bold', color: '#2C2D2D' }}>Tier 2:</span> Expose variable or function name swaps with ease.
            </li>
            <li>
              <span style={{ fontWeight: 'bold', color: '#2C2D2D' }}>Tier 3:</span> Detect restructuring of code modules or shifted logic blocks.
            </li>
            <li>
              <span style={{ fontWeight: 'bold', color: '#2C2D2D' }}>Tier 4:</span> Identify tweaked declarations or minor logic alterations.
            </li>
            <li>
              <span style={{ fontWeight: 'bold', color: '#2C2D2D' }}>Tier 5:</span> Spot complex control flow changes like altered loops and conditionals.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default Features;
