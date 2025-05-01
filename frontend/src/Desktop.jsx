import React, { useState } from "react";
import logo1 from "./images/logo-1.png";
import "./style.css";
import { Link } from "react-router-dom";

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
    <div className="page-wrapper">
      <div className="landing-page">
        <div className="overlap-group">
          <div className="rectangle" />
          <div className="text-wrapper">Opticare</div>

          <div className="items">
            <div className="div">About your Results</div>
            <Link to="/about" className="div">About Us</Link>
            <button className="button">
              <div className="text-wrapper-2">Home</div>
            </button>
          </div>

          <img className="logo" alt="Logo" src={logo1} />
        </div>

        <div className="overlap">
          <div className="text-wrapper-3">{prediction}</div>
        </div>

        <label className="div-wrapper clickable-upload">
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            hidden
          />
          <p className="p">Click here to upload an eye image</p>
        </label>

        <p className="text-wrapper-4">
          Upload a fundus image of the eye to get quick diagnosis!
        </p>
      </div>
    </div>
  );
};
