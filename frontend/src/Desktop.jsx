import React, { useState } from "react";
import logo from "./images/logo-1.png";
import "./style.css";
import { Link } from "react-router-dom";
import { Header } from "./components/Header";

export const LandingPage = () => {
  const [prediction, setPrediction] = useState("Displays results here");

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("image", file);

    try {
      const res = await fetch("http://localhost:5001/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setPrediction(data.prediction || "No result returned.");
    } catch (err) {
      setPrediction("Prediction failed.");
      console.error(err);
    }
  };

  return (
    <div className="page">
      <div className="container">
        <Header activePage="home" />
        
        <main className="main-content">
          <h1 className="page-title">Opticare</h1>
          
          <p className="page-subtitle">
            Upload a fundus image of the eye to get quick diagnosis!
          </p>
          
          <label className="upload-container clickable-upload">
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              hidden
            />
            <p className="upload-text">Click here to upload an eye image</p>
          </label>
          
          <div className="result-container">
            <p className="result-text">{prediction}</p>
          </div>
        </main>
      </div>
    </div>
  );
};