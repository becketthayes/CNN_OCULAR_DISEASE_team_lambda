# Opticare (team_lambda GDSC)

A web app that processes fundus images of the eye and uses a custom Convolutional Neural Network (CNN) to classify potential ocular diseases

--- 

## Table of Contents 
- [Overview](#overview)
- [Features](#features)
- [Technical Details](#technical-details)
- [Usage](#usage)
- [Timeline](#timeline)

--- 

## Overview

**Opticare** aims to provide a viable and readily available method for identifying ocular diseases. The web interface allows users to upload fundus images of the eye and receive a quick diagnosis regarding potential ocular diseases. Additionally, users can learn more about eye and vision health by talking to the built-in chatbot which is beneath the image upload.

The image classifier is a custom-made CNN that is trained on thousands of fundus images from this [Kaggle Dataset](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k). It can recognize ocular diseases including glaucoma, cataracts, hypertension, and myopia. 

--- 

## Features

- Ocular disease classifier that provides real-time ML inference
  - Also displays confidence score for the prediction
- Chatbot tailored toward eye health and vision
- "About your Results" page provides relevant information about all of the eye conditions in the dataset

---

## Technical Details

- Backend
  - Trained a custom CNN using TensorFlow to detect ocular diseases
  - Flask backend handles the image uploads, predictions, and chatbot responses
  - Chatbot makes calls to Gemini API
- Frontend
  - React 
  - Makes API calls to the Flask backend to send images and retrieve predictions

---

## Usage

1. Upload Image: Select and upload a fundus image of the eye.

2. Get Results: The app will process the image and display the predicted ocular disease along with a confidence score.

3. Learn More: Use the chatbot to ask questions about eye health or navigate to the "About your Results" page to get more information on the diagnosed conditions.

---

## Timeline

Predicting Ocular Disease

Problem Statement: The early detection of ocular diseases including glaucoma, cataracts, and age-related macular degeneration is vital to ensure healthy eyes in patients across the world. Many people do not have access to the medical support necessary to detect these issues, which can lead to severe health complications. There is a need for viable and readily available methods for identifying ocular diseases.

Project Idea: This project aims to address these challenges by using machine learning models to classify ocular diseases.  Our model will be trained by thousands of retinal images in order to accurately diagnose a wide range of eye conditions.

Proposed Timeline:

- Week 1: EDA Part 1
Clean the dataset (handle missing values and outliers).
Visualize data distribution and detect class imbalances.

- Week 2: EDA Part 2
Explore evaluation metrics (e.g., accuracy, F1-score).
Apply augmentations (e.g., flipping, rotation) to increase dataset diversity if necessary.

- Week 3: Baseline Model Development
Build a simple baseline CNN or use a pre-trained model (like ResNet).
Set up the training pipeline and evaluate initial performance.

- Week 4: Advanced Model Development
Fine-tune pre-trained models for ocular disease prediction.
Experiment with architectures and hyperparameter tuning (learning rate, batch size).

- Week 5: Model Optimization
Perform advanced hyperparameter tuning.
Generate classification reports and confusion matrices.

- Week 6: Deployment Preparation
Export the trained model for deployment.
Develop a simple web or mobile interface (TensorFlow Lite).

- Week 7: Deployment and Testing
Deploy the app and test with unseen images. Debug as needed.

- Week 8: Documentation and Presentation
Document project findings (EDA, models, and results).
Prepare your Mid-Year Showcase presentation.
