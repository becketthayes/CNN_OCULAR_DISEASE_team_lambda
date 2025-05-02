import React from "react";
import { Link } from "react-router-dom";
import logo from "../images/logo-1.png";

export const Header = ({ activePage }) => {
  return (
    <header className="header">
      <div className="logo-container">
        <Link to="/">
          <img className="logo" alt="Opticare Logo" src={logo} />
        </Link>
      </div>
      
      <nav className="navigation">
        <div className="nav-items">
          {activePage === "home" ? (
            <button className="nav-button active">
              <div className="button-text">Home</div>
            </button>
          ) : (
            <Link to="/" className="nav-button">
              <div className="button-text">Home</div>
            </Link>
          )}
          
          {activePage === "about" ? (
            <div className="nav-link active">
              <div className="link-text">About Us</div>
            </div>
          ) : (
            <Link to="/about" className="nav-link">
              <div className="link-text">About Us</div>
            </Link>
          )}
          
          {activePage === "results" ? (
            <div className="nav-link active">
                <div className="link-text">About your Results</div>
            </div>
            ) : (
            <Link to="/results" className="nav-link">
                <div className="link-text">About your Results</div>
            </Link>
            )}


        </div>
      </nav>
    </header>
  );
};