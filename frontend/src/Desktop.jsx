import React from "react";
import logo1 from "./images/logo-1.png";
import "./style.css";

export const LandingPage = () => {
  return (
    <div className="page-wrapper">
      <div className="landing-page" data-model-id="1:902">
        <div className="overlap-group">
          <div className="rectangle" />

          <div className="text-wrapper">Opticare</div>

          <div className="items">
            <div className="div">About your Results</div>

            <div className="div">About Us</div>

            <button className="button">
              <div className="text-wrapper-2">Home</div>
            </button>
          </div>

          <img className="logo" alt="Logo" src={logo1} />
        </div>

        <div className="overlap">
          <div className="text-wrapper-3">Displays results here</div>
        </div>

        <div className="div-wrapper">
          <p className="p">Drag or drop image of the eye here</p>
        </div>

        <p className="text-wrapper-4">
          Upload a fundus image of the eye to get quick diagnosis!
        </p>
      </div>
    </div>
  );
};
