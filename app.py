from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

def load_model():
    model_path = r'C:\Users\USER\Desktop\code\leaf_disease\tomato-leaf-disease\model.h5'  
    model = tf.keras.models.load_model(model_path, compile=False)
    return model
model = load_model()

class_names = ['Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_healthy']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        image = Image.open(file)
        image = np.asarray(image)
        predictions = model.predict(np.expand_dims(image, axis=0))
        predicted_class = np.argmax(predictions)
        confidence = float(np.max(predictions))

        response = {
            'predicted_class': class_names[predicted_class],
            'confidence': confidence
        }
        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
