import React, { useState, useRef, useEffect } from "react";
import logo from "./images/logo-1.png";
import "./style.css";
import { Link } from "react-router-dom";
import { Header } from "./components/Header";

export const LandingPage = () => {
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState(null);
  const [messages, setMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedImage, setUploadedImage] = useState(null);
  const chatEndRef = useRef(null);

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
  
    // Create URL for image preview
    const imageUrl = URL.createObjectURL(file);
    setUploadedImage(imageUrl);
  
    const formData = new FormData();
    formData.append("image", file);
  
    try {
      const res = await fetch("http://localhost:5001/predict", {
        method: "POST",
        body: formData,
      });
  
      const data = await res.json();
      const { prediction, confidence } = data;
  
      if (prediction) {
        setPrediction(prediction);
        setConfidence(confidence !== undefined ? (confidence * 100).toFixed(2) : null);
      } else {
        setPrediction("No result returned.");
        setConfidence(null);
      }
    } catch (err) {
      setPrediction("Prediction failed.");
      setConfidence(null);
      console.error(err);
    }
  };
  

  const handleSendMessage = async (e) => {
    e.preventDefault();
    
    if (!currentMessage.trim()) return;
    
    // Add user message to chat
    const userMessage = { text: currentMessage, sender: "user" };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setCurrentMessage("");
    setIsLoading(true);
    
    try {
      // Send message to API
      const response = await fetch("http://localhost:5001/api", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: currentMessage })
      });
      
      const data = await response.json();
      
      // Add bot response to chat
      setMessages(prevMessages => [
        ...prevMessages, 
        { text: data.response || "Sorry, I couldn't process that request.", sender: "bot" }
      ]);
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages(prevMessages => [
        ...prevMessages, 
        { text: "Sorry, there was an error processing your request.", sender: "bot" }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-scroll to bottom of chat when new messages are added
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

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
          
          {/* Image Preview Container */}
          {uploadedImage && (
            <img 
              src={uploadedImage} 
              alt="Uploaded eye fundus" 
              className="uploaded-image"
            />
          )}
          
          <div className="result-container">
            <p className="result-text">
              {prediction && prediction !== "Displays results here" ? (
                <>
                  {prediction}
                  {confidence !== null && ` (Confidence: ${confidence}%)`}
                </>
              ) : (
                "Displays results here"
              )}
            </p>
          </div>
          
          {/* Chatbot Section */}
          <div className="chatbot-section">
            <h2 className="chatbot-title">Ask About Your Results</h2>
            
            <div className="chat-container">
              <div className="messages-container">
                {messages.length === 0 ? (
                  <p className="empty-chat-message">Ask a question about your diagnosis or eye health.</p>
                ) : (
                  messages.map((msg, index) => (
                    <div key={index} className={`message ${msg.sender}-message`}>
                      <div className="message-bubble">
                        <p>{msg.text}</p>
                      </div>
                    </div>
                  ))
                )}
                {isLoading && (
                  <div className="message bot-message">
                    <div className="message-bubble loading">
                      <div className="loading-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
              
              <form className="message-input-form" onSubmit={handleSendMessage}>
                <input
                  type="text"
                  className="message-input"
                  value={currentMessage}
                  onChange={(e) => setCurrentMessage(e.target.value)}
                  placeholder="Type your message here..."
                  disabled={isLoading}
                />
                <button 
                  type="submit" 
                  className="send-button"
                  disabled={isLoading || !currentMessage.trim()}
                >
                  Send
                </button>
              </form>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};