from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense
)

from utils.preprocess_cnn import load_mnist


X_train, X_test, y_train, y_test = load_mnist()

model = Sequential([

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    MaxPooling2D(),

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dense(
        10,
        activation="softmax"
    )

])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)

model.save(
    "models/cnn_model.h5"
)