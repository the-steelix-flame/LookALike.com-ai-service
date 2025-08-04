from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64
import numpy as np
import cv2
import os

app = Flask(__name__)
# Enable CORS to allow requests from your Vercel frontend
CORS(app)

@app.route('/generate_embedding', methods=['POST'])
def generate_embedding():
    try:
        data = request.get_json()
        image_base64 = data.get('image_base64')

        if not image_base64:
            return jsonify({'error': 'No image_base64 provided'}), 400

        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]

        img_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        embedding_objs = DeepFace.represent(
            img_path=img, 
            model_name='ArcFace',
            detector_backend='retinaface',
            enforce_detection=False
        )

        if not embedding_objs or 'embedding' not in embedding_objs[0]:
            return jsonify({'error': 'Face could not be detected in the image.'}), 400

        embedding = embedding_objs[0]['embedding']
        return jsonify({'embedding': embedding}), 200

    except Exception as e:
        # Log the full error to the server console for debugging
        print(f"An unexpected error occurred: {e}")
        return jsonify({'error': f"An unexpected error occurred in the AI service: {str(e)}"}), 500

if __name__ == '__main__':
    # Get port from environment variable or default to 10000 for Render
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)