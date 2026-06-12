import numpy as np

from PIL import Image
from tensorflow.keras.datasets import mnist


def load_mnist():

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(
        -1, 28, 28, 1
    ) / 255.0

    X_test = X_test.reshape(
        -1, 28, 28, 1
    ) / 255.0

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


def preprocess_uploaded_image(uploaded_file):

    image = Image.open(uploaded_file)

    image = image.convert("L")

    image = image.resize((28, 28))

    img = np.array(image)

    img = img / 255.0

    img = img.reshape(
        1,
        28,
        28,
        1
    )

    return img, image