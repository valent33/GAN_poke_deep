from flask import Flask, send_from_directory, jsonify, request, Response
import tensorflow as tf
from flask_wtf import FlaskForm
from wtforms import SubmitField
from flask import render_template, request
import os
import numpy as np
from flask_bootstrap import Bootstrap
import requests
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import json

# define custom font CHERL___.TTF for the graph
my_font = dict(
    family="CHEL___.TTF",
    size=18,
    color="#7f7f7f"
)

class PredictionForm(FlaskForm):
    submit = SubmitField('Create your Pokemon')

# class for image and name form
class ImageForm(FlaskForm):
    submit = SubmitField('Register')

generator = tf.keras.models.load_model('cgenerator_model_final.h5')
stats = tf.keras.models.load_model('stats_model.h5')
name_generator = tf.keras.models.load_model('name_gen_type_stat.h5')
app = Flask(__name__, static_url_path='/static')
bootstrap = Bootstrap(app)
app.config['SECRET_KEY'] = 'secret'

# index is called once when the page is loaded
@app.route('/', methods=['GET', 'POST'])
def index():
    form = PredictionForm()
    filename = 'temp.png'  # Example filename
    graphJSON = plot_stats([10, 150, 70, 90, 65, 100])
    name = "Name"
    return render_template('index.html', form=form, filename=filename, graphjson=graphJSON, name=name)

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
        # Convert the values to floats
        label = np.zeros(18)
        for i in inputs:
            label[int(i)-1] = 1
        label = np.reshape(label, (1, 18))
        # generate the image using the GAN model
        image = generator.predict([tf.random.normal(shape=(1, 100)), label])
        # save the image to a temporary file
        filename = 'temp.png'
        temp_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # save the image
        tf.keras.preprocessing.image.save_img(temp_file, image[0], verbose=0)
        
        # stats prediction
        tmp = stats.predict([image, label], verbose=0)
        
        name = make_name(name_generator, label, tmp)

        # stats * standard deviation + mean
        tmp = tmp[0] * np.array([26.03589435, 29.10823332, 29.11150942, 29.54996808, 27.6194045, 27.81956697]) + np.array([68.33193732, 75.51028955, 70.67037331, 70.30717019, 70.31720702, 67.09646906])
        tmp = tmp.astype(int)

        global graphJSON
        graphJSON = plot_stats(tmp)

        # Return new image URL as JSON response
        return jsonify({'success': True, 'image_url': filename, 'types': types, 'graphjson': graphJSON, 'name': name})


# Create a function to get jsonify the image filename
@app.route('/get_image', methods=['GET', 'POST'])
def get_image():
    filename = 'temp.png'  # Example filename
    return jsonify(filename=filename)

@app.route('/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def plot_stats(stats):
    stats = stats + [stats[0]]
    fig = go.Figure(go.Scatterpolar(
                            r=stats,
                            theta=['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'HP'],
                            line={'width':0},
                            fillcolor='rgba(255, 197, 0, 0.7)',
                            fill='toself',
                            name="stats",
                            hovertemplate="%{r:.1f}<extra></extra>",
                            # none marker to hide the points
                            marker=dict(
                                symbol='circle',
                                size=0.1,
                                color='rgba(255, 197, 0, 0)',
                            )

    ))

        # GLOBAL AVERAGE
        # fig.add_trace(go.Scatterpolar(
        #                     r=total['mean'],
        #                     theta=statistics.index, 
        #                     line={'dash':'dash','color':'black'},
        #                     # marker={'size':0.1},
        #                     # hovertemplate="%{r:.1f}<extra></extra>"
        #                 ), row=Nrow, col=Ncol)

    # add the text annotations
    # for i, value in enumerate(df.iloc[0]):
    for i, value in enumerate(stats):
        # hp
        if i == 0:
            x = 0.5
            y = 1.025
        # attack
        if i == 1:
            x = 1.125
            y = 0.75
        # defense
        if i == 2:
            x = 1.125
            y = 0.07
        # sp. atk
        elif i == 3:
            x = 0.5
            y = -0.225
        # sp. def
        elif i == 4:
            x = -0.12
            y = 0.07
        # speed
        elif i == 5:
            x = -0.12
            y = 0.75
        elif i == 6:
            break

        fig.add_annotation(
            x=x,
            y=y,
            text=str(value),
            showarrow=False,
            font=dict(
                family="Cheri Liney",
                color="#3466AF",
                size=15,

            ),
        )

    fig.update_polars(gridshape='linear', 
                    angularaxis=dict(
                        thetaunit="degrees",
                        rotation=90,
                        direction='clockwise'),
                    radialaxis=dict(visible=False,range=[0, 175]))
    fig.update_layout(
        width=290, height=280, margin=dict(l=45, r=45, b=45, t=45, pad=0),
        plot_bgcolor = 'rgba(0,0,0,0)',
        paper_bgcolor = 'rgba(0,0,0,0)',
        font=dict(
            family="Cheri Liney",
            color="#3466AF",
            size=14.5,
        ),
    )
    # fig.show()
    return json.dumps(fig, cls=PlotlyJSONEncoder)


index_to_char = dict( (i, chr(i+96)) for i in range(1,27))
index_to_char[0] = ' '
index_to_char[27] = '.'

def make_name(model, types=np.zeros((1, 18)), stats=np.zeros((1, 6))):
    name = []
    x = np.zeros((1, 13, 28))
    types = np.tile(types, (1, 13, 1))
    # # concatenatre types with x after char_dim
    x = np.concatenate((x, types), axis=2)
    # concatenate stats at the end of x
    stats = np.tile(stats, (1, 13, 1))
    x = np.concatenate((x, stats), axis=2)
    # x = np.expand_dims(x, axis=2)
    
    end = False
    i = 0
    
    while end==False:
        probs = list(model.predict(x, verbose=0)[0,i])
        probs = probs / np.sum(probs)
        index = np.random.choice(range(28), p=probs)
        if i == 13-2:
            character = '.'
            end = True
        else:
            character = index_to_char[index]
        name.append(character)
        x[0, i+1, index] = 1
        i += 1
        if character == '.':
            end = True
    
    # print(''.join(name))
    return ''.join(name)[:-1]

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
    app.run(debug=True)
