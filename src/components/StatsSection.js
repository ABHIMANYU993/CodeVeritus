import React from 'react';
import CountUp from 'react-countup';
import { useInView } from 'react-intersection-observer';

const StatsSection = () => {
  const { ref, inView } = useInView({
    triggerOnce: true,
    threshold: 0.1,
  });

  return (
    <section
      className="stats-section text-white py-5"
      style={{
        background: 'linear-gradient(135deg, #2c3e52, #fd746a)',
        color: 'white',
      }}
    >
      <h2 className="fw-bold text-center mb-3">The Plagiarism Problem is Real—and Growing</h2>
      <p className="lead text-center">2022-2024 Insights You Can’t Ignore</p>
      <div className="row mt-5" ref={ref}>
        <div className="col-md-4 text-center">
          <h3>
            {inView ? (
              <CountUp
                end={5000000}
                duration={2.5}
                separator=","
                formattingFn={(value) => `${(value / 1000000).toFixed(1)}M`}
              />
            ) : (
              '0M'
            )}
          </h3>
          <p>Coding interviews conducted globally</p>
        </div>
        <div className="col-md-4 text-center">
          <h3>
            {inView ? <CountUp end={30} duration={2.5} /> : '0'}%
          </h3>
          <p>of candidates tried to game the system by copy-pasting code</p>
        </div>
        <div className="col-md-4 text-center">
          <h3>
            {inView ? <CountUp end={72} duration={2.5} /> : '0'}%
          </h3>
          <p>of CS students admitted to copying code at least once</p>
        </div>
      </div>
      <p className="mt-4 text-center text-light">
        Traditional tools miss cleverly disguised plagiarism. CodeVeritus has your back.
      </p>
    </section>
  );
};

export default StatsSection;
