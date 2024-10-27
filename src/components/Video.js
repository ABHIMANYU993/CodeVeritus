import React from 'react'

const Video = ({videoSrc}) => {
  return (
    <div className="video-container">
    <video autoPlay muted loop>
      <source src={videoSrc} type="video/mp4" />
      Your browser does not support the video tag.
    </video>
  </div>
  )
}

export default Video