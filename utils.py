import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf


def load_data(directory='images/final/', batch_size=64, image_size=(128, 128), GAN=False):
    """load data from a directory and return a tensorflow dataset

    Args:
        directory (str, optional): path to the main folder containing possibly subfolders. Defaults to 'images/final/'.
        batch_size (int, optional): batch size. Defaults to 64.
        image_size (tuple, optional): resize parameter. Defaults to (128, 128).
        GAN (bool, optional): if using GAN, not using train/test split. Defaults to False.

    Returns:
        tensorflow dataset, list: one or two tensorflow datasets and a list of class names
    """
    if not GAN:
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
    else:
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory,
            batch_size=batch_size,
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

        return train_ds, class_names


def plot_image(image, label, labels={}, size=128):
    """plot single image with label and additional information

    Args:
        image (np.array): image to plot
        label (str): label to give to the plot
        labels (dict, optional): additionnal information like fake likelihood. Defaults to {}.
        size (int, optional): size of the plot. Defaults to 128.
    """
    fig = px.imshow(image, width=size, height=size)
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
                x1=size,
                y1=size,
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


def plot_n_images(ds, n, class_names, GAN=False, size=128):
    """plot multiple images at once

    Args:
        ds (tensorflow dataset): must be a tensorflow dataset containing images and labels
        n (int): number of images to plot
        class_names (list): list of Pokémon names, only works with pokédex
        GAN (bool, optional): adjust images back to positive values if coming from GAN. Defaults to False.
        size (int, optional): size of the plot (square). Defaults to 128.
    """
    for image, label in ds.take(1):
        if GAN:
            image = image * 127.5 + 127.5
            image = image.numpy().astype("uint8")
        print("image shape : ", image[0].shape)
        for i in range(n):
            plot_image(image[i], class_names[int(
                tf.argmax(tf.reshape(label[i], [-1, 1]), axis=0))], size=size)


def plot_family(pokemon):
    """plot the family tree of a pokemon

    Args:
        pokemon (str): name of the Pokémon to plot. i.e. "Jungko"
    """
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
    poke_paths = poke["Name"].apply(
        lambda x: path_to_images + x + "/1.png").values
    evo1_paths = evo1["Name"].apply(
        lambda x: path_to_images + x + "/1.png").values
    evo2_paths = evo2["Name"].apply(
        lambda x: path_to_images + x + "/1.png").values
    sousevo1_paths = sousevo1["Name"].apply(
        lambda x: path_to_images + x + "/1.png").values
    sousevo2_paths = sousevo2["Name"].apply(
        lambda x: path_to_images + x + "/1.png").values

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


flip_and_rotate = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(
        0.05, fill_mode="constant", fill_value=1),  # 255 for classification, 1 for GAN
    tf.keras.layers.experimental.preprocessing.RandomTranslation(
        0.05, 0.05, fill_mode="constant", fill_value=1),  # experimental
    # maybe add hue chnages ?
])


def preprocess_image(image, GAN=False):
    """remove alpha channel and normalize image

    Args:
        image (tensor): images from tensorflow dataset
        GAN (bool, optional): if using GAN normalize between -1 and 1. Defaults to False.

    Returns:
        tensor: image modifiée
    """
    alpha_channel = image[:, :, :, 3]
    rgb_channels = image[:, :, :, :3]
    alpha_bool = alpha_channel > 0
    alpha_bool = tf.expand_dims(alpha_bool, axis=-1)
    rgb_channels = tf.where(alpha_bool, rgb_channels, 255)

    if not GAN:
        # Normalize the pixel values to be between 0 and 1
        from keras.applications.efficientnet import preprocess_input
        image = preprocess_input(rgb_channels)
    else:
        # Normalize the pixel values to be between -1 and 1
        image = (rgb_channels - 127.5) / 127.5

    return image


def prepare(ds, shuffle=False, augment=False, GAN=False):
    """prepare the dataset by applying the preprocessing function and data augmentation

    Args:
        ds (tensorflow dataset): image dataset
        shuffle (bool, optional): shuffle data ?. Defaults to False.
        augment (bool, optional): augment data with horizontal flip, rotations and translations ?. Defaults to False.
        GAN (bool, optional): propagate GAN to prepocessing function. Defaults to False.

    Returns:
        tensorflow dataset: reday to use dataset
    """
    ds = ds.map(lambda x, y: (preprocess_image(x, GAN=GAN), y),
                num_parallel_calls=tf.data.AUTOTUNE)

    # Use data augmentation only on the training set
    if augment:
        ds = ds.map(lambda x, y: (flip_and_rotate(x, training=True),
                    y), num_parallel_calls=tf.data.AUTOTUNE)

    # Use buffered prefecting on all datasets
    if shuffle:
        ds = ds.shuffle(1000)

    # ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds
