import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

st.set_page_config(page_title="Leafyze", page_icon="🍃", layout="wide")

@st.cache_resource
def load_model():
    model_path = 'model.h5'
    if not os.path.exists(model_path):
        model_path = os.path.join('model', 'model.h5')
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

model = load_model()
class_names = ['Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_healthy']

st.markdown("""
<div style="text-align:center; margin-bottom:0px;">
    <h1 style='color:rgb(223, 253, 233); margin-bottom:0px;'>🍃 Leaf Disease Classifier</h1>
    <p style='font-size:16px; margin-top:0px;'>Upload a tomato leaf image to see if it’s healthy or diseased.</p>
</div>
""", unsafe_allow_html=True)
st.markdown("<hr style='margin-top:5px; margin-bottom:5px;'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown(
            """
            <div style="padding:15px; border-radius:10px; background-color:rgb(223, 253, 233)">
                <h4 style="color:black; text-align:center;">📷 Uploaded Image</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, width=300, caption="Uploaded Leaf Image")

    with col2:
        st.markdown(
            """
            <div style="padding:15px; border-radius:10px; background-color:rgb(223, 253, 233)">
                <h4 style="color:black; text-align:center;">🔍 Prediction Results</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("")

        with st.spinner('Classifying...'):
            image_array = np.asarray(image)
            image_batch = np.expand_dims(image_array, axis=0)
            predictions = model.predict(image_batch)

            predicted_class_index = np.argmax(predictions)
            confidence = float(np.max(predictions))
            predicted_class_name = class_names[predicted_class_index]

            st.metric(label="Prediction", value=predicted_class_name)
            st.metric(label="Confidence", value=f"{confidence:.2%}")

            if "healthy" in predicted_class_name.lower():
                st.success("🌱 The leaf looks healthy!")
            else:
                st.error("⚠️ The leaf might have a disease.")

else:
    st.info("👆 Please upload an image to get started.")

    st.markdown("---")
    st.subheader("Or Try one of these sample images:")


SAMPLE_IMAGE_PATHS = {
    "Early Blight": os.path.join("assets", "Tomato_Early_Blight.JPG"),
    "Late Blight": os.path.join("assets", "Tomato_Late_Blight.JPG"),
    "Healthy 1": os.path.join("assets", "Tomato_Healthy.JPG"),
    "Healthy 2": os.path.join("assets", "Tomato_Healthy_.JPG"),  
}

scol1, scol2, scol3, scol4 = st.columns(4)
fixed_width = 150 

for col, key in zip([scol1, scol2, scol3, scol4], SAMPLE_IMAGE_PATHS.keys()):
    with col:
        image_path = SAMPLE_IMAGE_PATHS[key]
        st.image(image_path, width=fixed_width)
        if st.button("Predict", key=key):  
            with st.spinner('Classifying...'):
                image = Image.open(image_path).convert('RGB')
                image_array = np.asarray(image)
                image_batch = np.expand_dims(image_array, axis=0)
                predictions = model.predict(image_batch)
                predicted_class_index = np.argmax(predictions)
                confidence = float(np.max(predictions))
                predicted_class_name = class_names[predicted_class_index]

                if "healthy" in predicted_class_name.lower():
                    st.success(f"🌱 {predicted_class_name} ({confidence:.2%})")
                else:
                    st.error(f"⚠️ {predicted_class_name} ({confidence:.2%})")