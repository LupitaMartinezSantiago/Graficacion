from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
from PIL import Image, ImageDraw
import io
import os
import cv2
import dlib
import base64
import random

app = Flask(__name__)


predictor_path = "shape_predictor_68_face_landmarks.dat"

try:
    predictor = dlib.shape_predictor(predictor_path)
except RuntimeError as e:
    print(f"Error al cargar el predictor: {e}")
    exit(1)  

detector = dlib.get_frontal_face_detector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No se ha subido ninguna imagen.'}), 400
    
    file = request.files['image']
    
    try:
       
        image = Image.open(file)
        image_np = np.array(image)

       
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

       
        faces = detector(gray)

      
        draw = ImageDraw.Draw(image)

        
        for face in faces:
            landmarks = predictor(gray, face)

            puntos_a_dibujar = [
                21, 22,  
                17, 25,
                36, 37, 38,  
                42, 43, 44, 
                30, 
                51,  
                57,  
                48, 54  
            ]

            for i in puntos_a_dibujar:
                x = landmarks.part(i).x
                y = landmarks.part(i).y
              
                draw.line((x - 4, y - 4, x + 4, y + 4), fill=(255, 0, 0), width=3) 
                draw.line((x - 4, y + 4, x + 4, y - 4), fill=(255, 0, 0), width=3)

        
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        buf.seek(0)

       
        image_data = buf.read()
        image_base64 = 'data:image/jpeg;base64,' + base64.b64encode(image_data).decode('utf-8')

        return jsonify({'message': f'Rostros detectados: {len(faces)}', 'image': image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/plot-keyfacial', methods=['GET'])
def plot_keyfacial():
    
    images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    
    try:
      
        selected_image = random.choice(images)
        image = Image.open(selected_image)
        image_np = np.array(image)

        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        draw = ImageDraw.Draw(image)
# Cliclo
        for face in faces:
            landmarks = predictor(gray, face)
            for i in range(0, 68):  
                x = landmarks.part(i).x
                y = landmarks.part(i).y
               
                draw.line((x - 4, y - 4, x + 4, y + 4), fill=(255, 0, 0), width=3)  # Aumento del ancho
                draw.line((x - 4, y + 4, x + 4, y - 4), fill=(255, 0, 0), width=3)  # Aumento del ancho

        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        buf.seek(0)

        return send_file(buf, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
