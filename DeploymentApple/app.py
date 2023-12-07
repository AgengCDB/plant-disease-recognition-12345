# Flask
# Ini command buat start deployment
# python DeploymentApple/app.py

from flask import Flask, send_file, render_template, request

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
loaded_model = load_model('attempt_008.h5')

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "DeploymentApple/images/" + imagefile.filename
    imagefile.save(image_path)
        
    img = image.load_img(image_path, target_size=(256, 256, 3))  # Make sure to adjust target_size according to your model's input size
    img_array = image.img_to_array(img)
    # img_array = img_array.reshape((img_array.shape[0], img_array.shape[1], img_array.shape[2]))
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = loaded_model.predict(img_array)
    
    labels = ['AlternariaBoltch',
            'AppleScab',
            'BlackRot',
            'CedarAppleRust',
            'Rust']

    predicted_class_index = np.argmax(predictions)
    predicted_class_label = labels[predicted_class_index]
    
    
    return render_template('index.html', prediction=predicted_class_label)


if __name__ == '__main__':
    app.run(port=3000, debug=True)