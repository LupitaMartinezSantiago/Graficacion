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
        # Abrir la imagen con PIL
        image = Image.open(file)
        image_np = np.array(image)

        # Convertir la imagen a escala de grises para detección de rostros
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        faces = detector(gray)

        # Crear un objeto de dibujo para agregar las "X"
        draw = ImageDraw.Draw(image)

        # Dibujar puntos faciales en la imagen
        for face in faces:
            landmarks = predictor(gray, face)

            puntos_a_dibujar = [
                21, 22,  # Extremos de las cejas
                17, 25,
                36, 37, 38,  # Ojo izquierdo (esquinas y centro)
                42, 43, 44,  # Ojo derecho (esquinas y centro)
                30,  # Nariz
                51,  # Labio superior (centro)
                57,  # Labio inferior (centro)
                48, 54  # Labios (lados)
            ]

            for i in puntos_a_dibujar:
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                # Cambiar el ancho a 3 o 4 para hacer los puntos más grandes
                draw.line((x - 4, y - 4, x + 4, y + 4), fill=(255, 0, 0), width=3)  # Aumento del ancho
                draw.line((x - 4, y + 4, x + 4, y - 4), fill=(255, 0, 0), width=3)  # Aumento del ancho

        # Guardar la imagen con los puntos faciales en un buffer
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        buf.seek(0)

        # Convertir la imagen a base64
        image_data = buf.read()
        image_base64 = 'data:image/jpeg;base64,' + base64.b64encode(image_data).decode('utf-8')

        return jsonify({'message': f'Rostros detectados: {len(faces)}', 'image': image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/plot-keyfacial', methods=['GET'])
def plot_keyfacial():
    # Lista de imágenes de ejemplo (agregar tus propias rutas de imágenes)
    images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    
    try:
        # Seleccionar una imagen aleatoria
        selected_image = random.choice(images)
        image = Image.open(selected_image)
        image_np = np.array(image)

        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        draw = ImageDraw.Draw(image)

        for face in faces:
            landmarks = predictor(gray, face)
            for i in range(0, 68):  # Dibuja todos los puntos faciales
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                # Cambiar el ancho a 3 o 4 para hacer los puntos más grandes
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
