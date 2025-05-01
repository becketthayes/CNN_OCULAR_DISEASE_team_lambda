import React from "react";
import "./style_au.css";
import logo from "./images/logo-1.png"
import disease_distribution from "./images/distribution-of-diseases-1.png"
import photo_2 from "./images/photo-2.png"

export const AboutUs = () => {
  return (
    <div className="about-us">
      <div className="container">
        <header className="header">
          <div className="logo-container">
            {/* Replace with actual image import */}
            <img className="logo" alt="Logo" src={logo} />
          </div>
          
          <nav className="navigation">
            <div className="nav-items">
              <button className="nav-button primary">
                <div className="button-text">Home</div>
              </button>
              
              <div className="nav-link active">
                <div className="link-text">About Us</div>
              </div>
              
              <div className="nav-link">
                <div className="link-text">About your Results</div>
              </div>
            </div>
          </nav>
        </header>
        
        <main className="main-content">
          <section className="problem-statement-section">
            <h2 className="section-title">Our problem statement</h2>
            <p className="section-text">
              The early detection of ocular diseases is vital to ensure healthy
              eyes in patients across the world. Many people do not have access
              to the medical support necessary to detect these issues, which can
              lead to severe health complications. There is a need for viable
              and readily available methods for identifying ocular diseases.
            </p>
          </section>
          
          <div className="section-divider"></div>
          
          <section className="solution-section">
            <h2 className="section-title">Our solution</h2>
            <p className="section-text">
              Our program addresses the problem by using machine learning models
              to classify ocular diseases, allowing users to upload fundus photos
              of eyes. Our model will be trained by thousands of retinal images in
              order to accurately diagnose a wide range of eye conditions.
              <br /><br />
              The website quickly provides a diagnosis and accuracy percentage by
              identifying abnormalities.
            </p>
          </section>
          
          <div className="section-divider"></div>
          
          <section className="dataset-section">
            <button className="dataset-button">
              <div className="button-text">Link to dataset!</div>
            </button>
            
            <div className="disease-legend">
              <div className="legend-column">
                <p className="legend-text">
                  Normal (N),<br />
                  Diabetes (D),<br />
                  Glaucoma (G),<br />
                  Cataract (C),
                </p>
              </div>
              <div className="legend-column">
                <p className="legend-text">
                  Age related Macular Degeneration (A),<br />
                  Hypertension (H),<br />
                  Pathological Myopia (M),<br />
                  Other diseases/abnormalities (O)
                </p>
              </div>
            </div>
            
            <div className="disease-prevalence">
              <h3 className="subsection-title">Disease Prevalence:</h3>
              <p className="subsection-text">
                Diabetes (D) and Hypertension (H) are among the most common
                conditions.<br />
                Cataracts (C) and Glaucoma (G) also have a significant presence.<br />
                Pathological Myopia (M) and Age-related Macular Degeneration (A)
                are less common.
              </p>
            </div>
            
            <div className="chart-container">
              <img
                className="distribution-chart"
                alt="Distribution of diseases"
                src={disease_distribution}
              />
            </div>
            
            <div className="screenshot-container">
              <img
                className="screenshot"
                alt="Screenshot"
                src={photo_2}
              />
            </div>
            
            <div className="age-trends">
              <h3 className="subsection-title">Age-Related Trends:</h3>
              <p className="subsection-text">
                Diabetes and Hypertension increase in frequency with age.<br />
                Cataracts and Glaucoma are more prevalent in older age groups.<br />
                Other diseases (O) are spread across all age groups but peak in
                middle age.<br />
                Younger age groups (0-20, 21-40) have very few cases, likely
                reflecting a bias in the dataset toward older individuals.
              </p>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
};