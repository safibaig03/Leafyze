# Tomato Leaf Disease Prediction

### ✨ **[Live Demo]((https://leafyze.streamlit.app))** ✨

A web application that uses a deep learning model to detect diseases in tomato leaves. It classifies leaf images into Late Blight, Early Blight, or Healthy, and aims to assist farmers and gardeners in identifying plant diseases quickly and accurately.

## Overview

This project combines TensorFlow, Streamlit, and PIL to create an end-to-end system for leaf disease classification. A pre-trained Convolutional Neural Network (CNN) model predicts the disease category of tomato plant leaves. The web app provides a user-friendly interface where users can upload an image and receive a prediction in real-time.

## Features

- Upload a tomato leaf image and get an instant prediction.
- Classifies into:
  - Late Blight
  - Early Blight
  - Healthy
- Interactive and responsive UI built with Streamlit.
- Uses a .h5 TensorFlow model trained on an augmented dataset.

## Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- Pillow (PIL)

## Project Structure

A simple and clean structure for a Streamlit application:
├── app.py              # The main Streamlit application script
├── model/
│   └── model.h5        # Trained Keras model
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

## Getting Started

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/safibaig03/tomato-leaf-disease.git](https://github.com/safibaig03/tomato-leaf-disease.git)
    cd tomato-leaf-disease
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit App**
    ```bash
    streamlit run app.py
    ```
    Your browser will automatically open with the running application.

## Model Training

- Trained using the PlantVillage Dataset of tomato leaf images.
- Categories: Healthy, Early Blight, and Late Blight.
- Used data augmentation to improve generalization.
- The CNN includes Conv2D, MaxPooling, Dropout, and Dense layers.
- Achieved ~95% accuracy on the validation set.

## Acknowledgements

- PlantVillage Dataset (Kaggle)
- TensorFlow and Streamlit Docs

## Connect with Me

- **Mirza Safiulla Baig**
- **Email:** safiullabaig786@gmail.com
- **LinkedIn:** https://linkedin.com/in/safibaig03
- **GitHub:** https://github.com/safibaig03
