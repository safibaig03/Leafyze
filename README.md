# Tomato Leaf Disease Prediction

A web application that uses a deep learning model to detect diseases in tomato leaves. It classifies leaf images into Late Blight, Early Blight, or Healthy, and aims to assist farmers and gardeners in identifying plant diseases quickly and accurately.

## Overview

This project combines TensorFlow, Flask, and PIL to create an end-to-end system for leaf disease classification. A pre-trained Convolutional Neural Network (CNN) model predicts the disease category of tomato plant leaves. The web app provides a user-friendly interface where users can upload an image and receive a prediction in real-time.

## Features

- Upload a tomato leaf image and get an instant prediction.
- Classifies into:
  - Late Blight
  - Early Blight
  - Healthy
- Clean, responsive frontend using HTML/CSS.
- Flask backend with image preprocessing and prediction.
- Uses a .h5 TensorFlow model trained on augmented dataset.

## Tech Stack

- Python
- TensorFlow / Keras
- Flask
- Pillow (PIL)
- HTML / CSS

## Project Structure

├── app.py                 # Flask backend  
├── templates/  
│   └── index.html         # Main UI page  
├── static/  
│   └── style.css          # CSS styling  
├── model/  
│   └── model.h5           # Trained Keras model  
├── utils/  
│   └── preprocess.py      # Preprocessing helper functions  
├── requirements.txt       # Python dependencies  
└── README.md              # Project documentation  

## Getting Started

1. Clone the repository

   git clone https://github.com/safibaig03/tomato-leaf-disease.git  
   cd tomato-leaf-disease

2. Install dependencies

   pip install -r requirements.txt

3. Run the Flask App

   python app.py

Then open http://localhost:5000 in your browser.

## Sample Prediction

You can test the model by uploading a tomato leaf image.  
(Screenshot/preview can be added here.)

## Model Training

- Trained using a dataset of tomato leaf images:
  - Healthy
  - Early Blight
  - Late Blight
- Used data augmentation to generalize better.
- CNN includes Conv2D, MaxPooling, Dropout, and Dense layers with softmax.
- Model Accuracy: ~95% on validation set.

## Acknowledgements

- PlantVillage Dataset (Kaggle)
- TensorFlow and Flask Docs

## Connect with Me

Mirza Safiulla Baig  
Email: safiullabaig786@gmail.com  
LinkedIn: https://linkedin.com/in/safibaig03  
GitHub: https://github.com/safibaig03
