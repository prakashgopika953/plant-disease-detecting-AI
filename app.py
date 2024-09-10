import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model

app = Flask(__name__)

model = load_model('model\model.keras')
print('Model loaded. Check http://127.0.0.1:5000/')

labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

def get_result(image_path):
    try:
        img = load_img(image_path, target_size=(225, 225))
        x = img_to_array(img)
        x = x.astype('float32') / 255.
        x = np.expand_dims(x, axis=0)
        predictions = model.predict(x)[0]
        return predictions
    except Exception as e:
        return str(e)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    
    f = request.files['file']
    
    if f.filename == '':
        return 'No selected file'
    
    try:
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads')
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
        
        file_path = os.path.join(upload_path, secure_filename(f.filename))
        f.save(file_path)
        
        predictions = get_result(file_path)
        predicted_label = labels[np.argmax(predictions)]
        return str(predicted_label)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
