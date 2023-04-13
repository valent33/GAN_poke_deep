from flask import Flask, send_from_directory, jsonify, request, Response
import tensorflow as tf
from flask_wtf import FlaskForm
from wtforms import SubmitField
from flask import render_template, request
import os
import numpy as np
from flask_bootstrap import Bootstrap
import requests

class PredictionForm(FlaskForm):
    submit = SubmitField('Create your Pokemon')

# class for image and name form
class ImageForm(FlaskForm):
    submit = SubmitField('Register')

generator = tf.keras.models.load_model('cgenerator_model_final.h5')
app = Flask(__name__, static_url_path='/static')
bootstrap = Bootstrap(app)
app.config['SECRET_KEY'] = 'secret'
# index is called once when the page is loaded
@app.route('/', methods=['GET', 'POST'])
def index():
    form = PredictionForm()
    filename = 'temp.png'  # Example filename
    return render_template('index.html', form=form, filename=filename)

# Create a function to download the image with the name of the pokemon fetched from the form
#('/download', {
#                method: 'POST',
#                body: imageContainerElement.src,
#                name: name
#            })

@app.route('/download', methods=['POST'])
def download():
    # Get data from request body
    data = request.json
    name = data.get('name')

    # the image_url is temp.png
    image_url = 'temp.png'
    # catch dynamically the server address
    server_address = request.host_url
    print(server_address)
    # make it a path understandable by the requests library
    image_url = server_address + image_url
    # Download image from URL
    image_response = requests.get(image_url)
    if image_response.status_code == 200:
        # Create a response with image content as file
        response = Response(image_response.content, content_type='image/jpeg')
        response.headers['Content-Disposition'] = f'attachment; filename={name}.jpg'
        return response
    else:
        return 'Image download failed.', 500

# Create a function that transforms a inputs to a pokemon type and returns a list of the types
def label_to_type(labels):
    types = []
    for label in labels:
        if label == '1':
            types.append('Grass')
        elif label == '2':
            types.append('Psychic')
        elif label == '3':
            types.append('Dark')
        elif label == '4':
            types.append('Bug')
        elif label == '5':
            types.append('Steel')
        elif label == '6':
            types.append('Rock')
        elif label == '7':
            types.append('Normal')
        elif label == '8':
            types.append('Fairy')
        elif label == '9':
            types.append('Water')
        elif label == '10':
            types.append('Dragon')
        elif label == '11':
            types.append('Electric')
        elif label == '12':
            types.append('Poison')
        elif label == '13':
            types.append('Fire')
        elif label == '14':
            types.append('Ice')
        elif label == '15':
            types.append('Ground')
        elif label == '16':
            types.append('Ghost')
        elif label == '17':
            types.append('Fighting')
        elif label == '18':
            types.append('Flying')
    return types


# Create a function to process the image on submit without reloading the page
@app.route('/submit', methods=['POST'])
def submit_form():
    form = PredictionForm(request.form)
    filename = 'temp.png'  # Example filename
    if form.validate_on_submit():
        # Get the checkbox values
        inputs = request.form.getlist('input[]')
        # label to type
        types = label_to_type(inputs)
        # print the types
        print(types)
        # Convert the values to floats
        label = np.zeros(18)
        for i in inputs:
            label[int(i)-1] = 1
        label = np.reshape(label, (1, 18))
        # print the numbers of the labels equal to 1
        for i in range(len(label[0])):
            if label[0][i] == 1:
                print(i+1)
        # generate the image using the GAN model
        image = generator.predict([tf.random.normal(shape=(1, 100)), label])
        # save the image to a temporary file
        filename = 'temp.png'
        temp_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # save the image
        tf.keras.preprocessing.image.save_img(temp_file, image[0])
        # Return new image URL as JSON response
        return jsonify({'success': True, 'image_url': filename, 'types': types})


# Create a function to get jsonify the image filename
@app.route('/get_image', methods=['GET', 'POST'])
def get_image():
    filename = 'temp.png'  # Example filename
    return jsonify(filename=filename)

@app.route('/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
    app.run(debug=True)
