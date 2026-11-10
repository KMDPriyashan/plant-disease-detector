from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('plant_disease_model.h5')
with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

print("✅ Model and Class Names loaded")

def preprocess_image(data_url):
    # Base64 decode කිරීම
    header, encoded = data_url.split(',', 1)
    img_bytes = base64.b64decode(encoded)
    
    # Image එක open කරලා RGB වලට හරවනවා
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    # 224x224 ට resize කරනවා
    img = img.resize((224, 224))
    
    # Array එකකට හරවලා normalize කරනවා (0-1 අතර)
    img_array = np.array(img) / 255.0
    
    # Batch dimension එකක් එකතු කරනවා (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Plant Disease API is running 🌿"})