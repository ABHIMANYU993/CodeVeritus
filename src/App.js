
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './pages/Home/Home';
import Login from './pages/Login/Login';
import CompanyInfo from './pages/CompanyInfo/CompanyInfo';
import SelectUsersPage from './pages/Usercode/SelectUsers';
import ResultsPage from './pages/Results/Results';

import 'bootstrap/dist/css/bootstrap.min.css';
 

const App = () => {
  return (
   
    <Router>
      <Routes>
      
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/companyinfo" element={<CompanyInfo />} />
        <Route path="/selectusers" element={<SelectUsersPage />} />
        <Route path="/results" element={<ResultsPage />} />
      </Routes>
    </Router>
  );
};

export default App;