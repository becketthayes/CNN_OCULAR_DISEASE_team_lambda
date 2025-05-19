import React from "react";
import "./style.css"; //
import disease_distribution from "./images/distribution-of-diseases-1.png"; //
import photo_2 from "./images/photo-2.png"; //
import { Header } from "./components/Header"; //

export const AboutUs = () => {
  return (
    <div className="page"> {/* */}
      <div className="container"> {/* */}
        <Header activePage="about" /> {/* */}
        
        <main className="main-content"> {/* */}
          <section className="content-section"> {/* */}
            <h2 className="section-title">Our problem statement</h2> {/* */}
            <p className="section-text"> {/* */}
              The early detection of ocular diseases is vital to ensure healthy
              eyes in patients across the world. Many people do not have access
              to the medical support necessary to detect these issues, which can
              lead to severe health complications. There is a need for viable
              and readily available methods for identifying ocular diseases.
            </p> {/* */}
          </section> {/* */}
          
          <div className="section-divider"></div> {/* */}
          
          <section className="content-section"> {/* */}
            <h2 className="section-title">Our solution</h2> {/* */}
            <p className="section-text"> {/* */}
              Our program addresses the problem by using machine learning models
              to classify ocular diseases, allowing users to upload fundus photos
              of eyes. Our model will be trained by thousands of retinal images in
              order to accurately diagnose a wide range of eye conditions.
              <br /><br />
              The website quickly provides a diagnosis and accuracy percentage by
              identifying abnormalities.
            </p> {/* */}
          </section> {/* */}
          
          <div className="section-divider"></div> {/* */}
          
          <section className="content-section dataset-section"> {/* */}
            <a 
              href="https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k" 
              target="_blank" 
              rel="noopener noreferrer"
              className="dataset-link"
            > {/* */}
              <button className="dataset-button"> {/* */}
                <div className="button-text">Link to dataset!</div> {/* */}
              </button> {/* */}
            </a> {/* */}
            
            <div className="disease-legend"> {/* */}
              <div className="legend-column"> {/* */}
                <p className="legend-text"> {/* */}
                  Normal (N),<br />
                  Diabetes (D),<br />
                  Glaucoma (G),<br />
                  Cataract (C),
                </p> {/* */}
              </div> {/* */}
              <div className="legend-column"> {/* */}
                <p className="legend-text"> {/* */}
                  Age related Macular Degeneration (A),<br />
                  Hypertension (H),<br />
                  Pathological Myopia (M),<br />
                  Other diseases/abnormalities (O)
                </p> {/* */}
              </div> {/* */}
            </div> {/* */}
            
            <div className="chart-container"> {/* */}
              <img
                className="distribution-chart"
                alt="Distribution of diseases"
                src={disease_distribution}
              /> {/* */}
            </div> {/* */}

            <div className="info-block"> {/* */}
              <h3 className="info-title">Disease Prevalence:</h3> {/* */}
              <div className="info-text"> {/* */}
                <ul>
                  <li>Diabetes (D) and Hypertension (H) are among the most common conditions.</li> {/* */}
                  <li>Cataracts (C) and Glaucoma (G) also have a significant presence.</li> {/* */}
                  <li>Pathological Myopia (M) and Age-related Macular Degeneration (A) are less common.</li> {/* */}
                </ul>
              </div> {/* */}
            </div> {/* */}
            
            <div className="screenshot-container"> {/* */}
              <img
                className="screenshot"
                alt="Screenshot"
                src={photo_2}
              /> {/* */}
            </div> {/* */}
            
            <div className="info-block"> {/* */}
              <h3 className="info-title">Age-Related Trends:</h3> {/* */}
              <div className="info-text"> {/* */}
                <ul>
                  <li>Diabetes and Hypertension increase in frequency with age.</li> {/* */}
                  <li>Cataracts and Glaucoma are more prevalent in older age groups.</li> {/* */}
                  <li>Other diseases (O) are spread across all age groups but peak in middle age.</li> {/* */}
                  <li>Younger age groups (0-20, 21-40) have very few cases, likely reflecting a bias in the dataset toward older individuals.</li> {/* */}
                </ul>
              </div> {/* */}
            </div> {/* */}
          </section> {/* */}
        </main> {/* */}
      </div> {/* */}
    </div> /* */
  );
};