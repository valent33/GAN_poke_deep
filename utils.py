import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf

def load_data(directory='images/final/', batch_size=64, image_size=(128, 128)):
    train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        batch_size=batch_size,
        validation_split=0.2,
        subset="both",
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        crop_to_aspect_ratio=True,
        interpolation="bilinear",
        color_mode="rgba",
        shuffle=True,
        seed=905,
    )

    class_names = train_ds.class_names
    
    return train_ds, val_ds, class_names

def plot_image(image, label, labels={}):
    fig = px.imshow(image, width=128, height=128)
    fig.update_layout(
        title=f"Label: {label}",
        width=500,
        height=350,
        margin={'l': 0},
        # remove axis but leave a border
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            visible=False,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            visible=False,
        ),
        # add a solid contour of the image
        shapes=[
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=0,
                y0=0,
                x1=128,
                y1=128,
                line=dict(
                    color="black",
                    width=2,
                ),
            ),
        ],       
    )
    if labels:
        # display all 5 elements of the dictionary with the key: value
        i = 0
        for key, value in labels.items():
            fig.add_annotation(
                x=1,
                y=1-0.25*i,
                xref="paper",
                yref="paper",
                text=f"{key}: {value*100:.0f}%",
                showarrow=False,
                font=dict(
                    family="Courier New, monospace",
                    size=12,
                    color="#ffffff"
                ),
                align="left",
                bordercolor="#c7c7c7",
                borderwidth=1,
                borderpad=2,
                bgcolor="#ff7f0e",
                opacity=0.8,
            )
            # add image from corresponding folder in images/final
            fig.add_layout_image(
                dict(
                    source=f"images/final/{key}/1.png",
                    xref="paper",
                    yref="paper",
                    x=1,
                    y=1.1-0.25*i,
                    sizex=0.3,
                    sizey=0.3,
                    # sizing="stretch",
                    # opacity=0.8,
                    layer="below",
                )
            )
            i += 1

    fig.show()

def plot_n_images(ds, n, class_names):
    for image, label in ds.take(1):
        for i in range(n):
            print(image[i][0][0])
            print(image[i].shape)
            plot_image(image[i], class_names[int(tf.argmax(tf.reshape(label[i], [-1, 1]), axis=0))])

def plot_family(pokemon):
    df = pd.read_csv("PokeDataset.csv")
    poke = df[df["Name"] == pokemon]
    evo1 = df[df["Chain"] == pokemon]
    try:
        evo2 = df[df["Chain"] == evo1["Name"].values[0]]
    except:
        evo2 = pd.DataFrame(columns=df.columns)
    sousevo1 = df[df["Name"] == poke["Chain"].values[0]]
    try:
        sousevo2 = df[df["Name"] == sousevo1["Chain"].values[0]]
    except:
        sousevo2 = pd.DataFrame(columns=df.columns)
    
    path_to_images = "images/final/"
    poke_paths = poke["Name"].apply(lambda x: path_to_images + x + "/1.png").values
    evo1_paths = evo1["Name"].apply(lambda x: path_to_images + x + "/1.png").values
    evo2_paths = evo2["Name"].apply(lambda x: path_to_images + x + "/1.png").values
    sousevo1_paths = sousevo1["Name"].apply(lambda x: path_to_images + x + "/1.png").values
    sousevo2_paths = sousevo2["Name"].apply(lambda x: path_to_images + x + "/1.png").values

    # plot all images from left to right
    fig = go.Figure()
    fig.add_layout_image(
        dict(
                source=poke_paths[0],
                xref="x",
                yref="y",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
            )
    )
    for i in range(len(evo1_paths)):
        fig.add_layout_image(
            dict(
                source=evo1_paths[i],
                xref="x",
                yref="y",
                x=1,
                y=1+i,
                sizex=1,
                sizey=1,
            )
        )
    for i in range(len(evo2_paths)):
        fig.add_layout_image(
            dict(
                source=evo2_paths[i],
                xref="x",
                yref="y",
                x=2,
                y=1+i,
                sizex=1,
                sizey=1,
            )
        )
    for i in range(len(sousevo1_paths)):
        fig.add_layout_image(
            dict(
                source=sousevo1_paths[i],
                xref="x",
                yref="y",
                x=-1,
                y=1+i,
                sizex=1,
                sizey=1,
            )
        )
    for i in range(len(sousevo2_paths)):
        fig.add_layout_image(
            dict(
                source=sousevo2_paths[i],
                xref="x",
                yref="y",
                x=-2,
                y=1+i,
                sizex=1,
                sizey=1,
            )
        )
    fig.update_layout(
        width=500,
        height=200,
        xaxis=dict(
            showticklabels=False,
            ticks='',
            zeroline=False,
            range=[-2, 3],
        ),
        yaxis=dict(
            showticklabels=False,
            ticks='',
            zeroline=False,
            range=[0, 2],
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    if pokemon == "Eevee":
        fig.update_layout(
            height=700,
            width=200,
            yaxis=dict(
                range=[0, 7],
            ),
            xaxis=dict(
                range=[0, 2],
            )
        ),
    fig.show()

# plot_family("Kirlia")



