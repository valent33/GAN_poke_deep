from flask import Flask, send_from_directory
import tensorflow as tf
from flask_wtf import FlaskForm
from wtforms import SubmitField
from flask import render_template, request
import os
import numpy as np
from flask_bootstrap import Bootstrap

class PredictionForm(FlaskForm):
    submit = SubmitField('Predict')

generator = tf.keras.models.load_model('cgenerator_model_final.h5')
app = Flask(__name__, static_url_path='/static')
bootstrap = Bootstrap(app)
app.config['SECRET_KEY'] = 'secret'

@app.route('/', methods=['GET', 'POST'])
def index():
    form = PredictionForm()
    filename = 'default.png'
    if form.validate_on_submit():
        # Get the checkbox values
        inputs = request.form.getlist('input[]')
        # Convert the values to floats
        label = np.zeros(18)
        for i in inputs:
            label[int(i)-1] = 1
        label = np.reshape(label, (1, 18))
        print(label)
        # generate the image using the GAN model
        image = generator.predict([tf.random.normal(shape=(1, 100)), label])
        # save the image to a temporary file
        filename = 'temp.png'
        temp_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # save the image
        tf.keras.preprocessing.image.save_img(temp_file, image[0])
    return render_template('index.html', form=form, filename=filename)

@app.route('/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
    app.run(debug=True)
