from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import os


app = Flask(__name__)

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
    model = tf.keras.models.load_model(model_path, compile=False)
    return model
model = load_model()

class_names = ['Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_healthy']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         file = request.files['file']
#         image = Image.open(file)
#         image = np.asarray(image)
#         predictions = model.predict(np.expand_dims(image, axis=0))
#         predicted_class = np.argmax(predictions)
#         confidence = float(np.max(predictions))

#         response = {
#             'predicted_class': class_names[predicted_class],
#             'confidence': confidence
#         }
#         return jsonify(response)
    
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        
        # 1. Open the image
        image = Image.open(file.stream).convert('RGB') # Convert to RGB
        
        # 2. Resize the image to the size your model expects (e.g., 256x256)
        #    Replace (256, 256) with your model's actual input size!
        target_size = (256, 256) 
        image = image.resize(target_size)
        
        # 3. Convert the image to a NumPy array and normalize it
        image_array = np.asarray(image) / 255.0 # Normalize pixel values to be between 0 and 1
        
        # 4. Run prediction
        predictions = model.predict(np.expand_dims(image_array, axis=0))
        predicted_class = np.argmax(predictions)
        confidence = float(np.max(predictions))

        response = {
            'predicted_class': class_names[predicted_class],
            'confidence': confidence
        }
        return jsonify(response)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

