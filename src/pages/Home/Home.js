import React from 'react';
import './Home.css'; // Import CSS if you decide to move styles to a separate file
import Navbar from '../../components/Navbar';
import Intro from '../../components/Intro';
import StatsSection from '../../components/StatsSection';
import Features from '../../components/Features';
import WhyChoose from '../../components/WhyChoose';
import ResSection from '../../components/ResSection';
import CTAsec from '../../components/CTAsec';
import Video from '../../components/Video';
import Footer from '../../components/Footer';
const Home = ({videoSrc}) => {
  return (
    <div id='/'>
       
     <Video videoSrc="3130182-uhd_3840_2160_30fps.mp4" />
     <Navbar/>
     <Intro className="p-8"/>
     <StatsSection/>
     <Features/>   
     <WhyChoose/>  
    <ResSection/>
    <CTAsec/>
     <Footer/>
    </div>
  );
};

export default Home;
