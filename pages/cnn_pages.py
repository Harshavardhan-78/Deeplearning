import streamlit as st
import numpy as np

from tensorflow.keras.models import load_model

from utils.preprocess_cnn import (
    preprocess_uploaded_image
)


@st.cache_resource
def load_cnn_model():

    return load_model(
        "models/cnn_model.h5"
    )


def show():

    model = load_cnn_model()

    st.title(
        "CNN - Handwritten Digit Recognition"
    )

    uploaded = st.file_uploader(
        "Upload Digit Image",
        type=[
            "png",
            "jpg",
            "jpeg"
        ]
    )

    if uploaded:

        img, image = (
            preprocess_uploaded_image(
                uploaded
            )
        )

        prediction = model.predict(
            img
        )

        digit = np.argmax(
            prediction
        )

        confidence = np.max(
            prediction
        )

        st.image(
            image,
            width=200
        )

        st.success(
            f"Predicted Digit : {digit}"
        )

        st.info(
            f"Confidence : {confidence:.2%}"
        )