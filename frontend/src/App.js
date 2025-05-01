import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { AboutUs } from "./About_us";
import { LandingPage } from "./Desktop";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/about" element={<AboutUs />} />
      </Routes>
    </Router>
  );
}

export default App;
