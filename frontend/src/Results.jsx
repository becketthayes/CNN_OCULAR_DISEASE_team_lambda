import React from "react";
import "./style.css";
import { Header } from "./components/Header";
import eye_diseases from "./images/eye_diseases.png"

export const Results = () => {
  return (
    <div className="page">
      <div className="container">
        <Header activePage="results" />
        
        <main className="main-content">
          <div className="results-header">
            <h1 className="results-title">Your Eye Health Results</h1>
            <p className="results-description">
              Below is a summary of what each diagnosis could mean. If your results show any potential concerns, 
              please consult an eye care specialist for a thorough examination.
            </p>
          </div>

          {/* Single eye scan image gallery */}
          <div className="eye-scan-gallery">
            <img 
              src={eye_diseases}
              alt="Eye scan results" 
              className="eye-scan-image" 
            />
          </div>
          
          {/* Results Cards */}
          <div className="results-container">
            <div className="result-card">
              <h2 className="result-card-title">Normal (N)</h2>
              <div className="result-card-content">
                <p><strong>Result:</strong> No signs of disease detected.</p>
                <p><strong>What it means:</strong> Your eye appears healthy based on the provided image.</p>
                <p><strong>Recommendation:</strong> Continue regular eye check-ups, especially if you experience any new symptoms.</p>
              </div>
            </div>
            
            <div className="result-card">
              <h2 className="result-card-title">Diabetes (D)</h2>
              <div className="result-card-content">
                <p><strong>Result:</strong> Possible diabetic retinopathy or diabetes-related eye issues detected.</p>
                <p><strong>What it means:</strong> High blood sugar can damage blood vessels in the retina, leading to vision problems.</p>
                <p><strong>Treatment/Next Steps:</strong></p>
                <ul>
                  <li>Consult your healthcare provider for diabetes management.</li>
                  <li>Regular eye exams are critical.</li>
                  <li>Early treatment can prevent vision loss.</li>
                </ul>
              </div>
            </div>
            
            <div className="result-card">
              <h2 className="result-card-title">Glaucoma (G)</h2>
              <div className="result-card-content">
                <p><strong>Result:</strong> Signs of glaucoma detected.</p>
                <p><strong>What it means:</strong> Glaucoma causes increased pressure inside the eye, which can damage the optic nerve and lead to vision loss.</p>
                <p><strong>Treatment/Next Steps:</strong></p>
                <ul>
                  <li>Seek an ophthalmologist for evaluation.</li>
                  <li>Treatments may include eye drops, medication, laser treatment, or surgery.</li>
                  <li>Early detection is key to preserving vision.</li>
                </ul>
              </div>
            </div>
            
            <div className="result-card">
              <h2 className="result-card-title">Cataract (C)</h2>
              <div className="result-card-content">
                <p><strong>Result:</strong> Cataract formation detected.</p>
                <p><strong>What it means:</strong> Cataracts are clouding of the eye's natural lens, often related to aging.</p>
                <p><strong>Treatment/Next Steps:</strong></p>
                <ul>
                  <li>Mild cataracts may just need stronger lighting or glasses.</li>
                  <li>Severe cataracts can be treated with a safe and common surgery to replace the lens.</li>
                </ul>
              </div>
            </div>
            
            <div className="result-card">
              <h2 className="result-card-title">Age-related Macular Degeneration (AMD) (A)</h2>
              <div className="result-card-content">
                <p><strong>Result:</strong> Possible signs of AMD detected.</p>
                <p><strong>What it means:</strong> AMD affects the macula, the part of the retina responsible for sharp central vision.</p>
                <p><strong>Treatment/Next Steps:</strong></p>
                <ul>
                  <li>See an eye specialist for detailed assessment.</li>
                  <li>While there's no cure, certain treatments (like injections or laser therapy) may slow progression.</li>
                  <li>Nutritional supplements may also help in some cases.</li>
                </ul>
              </div>
            </div>
            
            <div className="result-card">
              <h2 className="result-card-title">Hypertension (H)</h2>
              <div className="result-card-content">
                <p><strong>Result:</strong> Signs of blood pressure-related eye changes detected.</p>
                <p><strong>What it means:</strong> High blood pressure can damage the blood vessels in the retina (hypertensive retinopathy).</p>
                <p><strong>Treatment/Next Steps:</strong></p>
                <ul>
                  <li>Control blood pressure with lifestyle changes or medication.</li>
                  <li>Regular monitoring with an eye care professional is advised.</li>
                </ul>
              </div>
            </div>
            
            <div className="result-card">
              <h2 className="result-card-title">Pathological Myopia (M)</h2>
              <div className="result-card-content">
                <p><strong>Result:</strong> Signs of severe nearsightedness detected.</p>
                <p><strong>What it means:</strong> Pathological myopia can cause progressive and serious changes in the eye, increasing the risk of retinal detachment or vision loss.</p>
                <p><strong>Treatment/Next Steps:</strong></p>
                <ul>
                  <li>Regular check-ups with an ophthalmologist are important.</li>
                  <li>Some cases may require surgery or specialized treatments.</li>
                  <li>Protective eyewear and visual aids can help manage the condition.</li>
                </ul>
              </div>
            </div>
            
            <div className="result-card">
              <h2 className="result-card-title">Other Diseases/Abnormalities (O)</h2>
              <div className="result-card-content">
                <p><strong>Result:</strong> Other signs of ocular diseases detected.</p>
                <p><strong>What it means:</strong> This category covers other eye conditions not listed above (e.g., infections, tumors, injuries).</p>
                <p><strong>Treatment/Next Steps:</strong></p>
                <ul>
                  <li>Immediate consultation with an eye specialist is recommended for accurate diagnosis and appropriate treatment.</li>
                </ul>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};