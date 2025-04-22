// Desktop.jsx
import React from "react";
import "./style.css";
import logo from "./images/logo.png"; 

export const Desktop = () => {
  return (
    <div className="landing-page">
      <header className="header">
        <img src={logo} alt="Lambda Logo" className="logo" />
        <h1 className="title">Welcome to Lambda</h1>
      </header>

      <main className="main-content warm-theme">
        <div className="intro-box">
          <p className="intro-text">
            Our mission is to enhance ocular disease classification with machine learning.
          </p>
        </div>
      </main>

      <footer className="footer">
        <p className="footer-text">© 2025 Lambda Team. All rights reserved.</p>
      </footer>
    </div>
  );
};