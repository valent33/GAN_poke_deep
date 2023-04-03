from flask import Flask, send_from_directory
import tensorflow as tf
from flask_wtf import FlaskForm
from wtforms import SubmitField
from flask import render_template, request
import os

class PredictionForm(FlaskForm):
    submit = SubmitField('Predict')

generator = tf.keras.models.load_model('generator_model_final.h5')
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'

@app.route('/', methods=['GET', 'POST'])
def index():
    form = PredictionForm()
    filename = 'default.png'
    if form.validate_on_submit():
        # generate the image using the GAN model
        image = generator.predict(tf.random.normal(shape=(1, 100)))
        # save the image to a temporary file
        filename = 'temp.png'
        temp_file = os.path.join(app.config['UPLOAD_FOLDER'], filename).replace('\\', '/')
        # save the image
        tf.keras.preprocessing.image.save_img(temp_file, image[0])
    return render_template('index.html', form=form, filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.run(debug=True)
