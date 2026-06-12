from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from utils.preprocess_ann import load_and_preprocess


X_train, X_test, y_train, y_test = load_and_preprocess(
    "data/churn.csv"
)

model = Sequential([

    Dense(
        64,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )

])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

model.save(
    "models/ann_model.h5"
)

print("\nANN Model Saved Successfully!")