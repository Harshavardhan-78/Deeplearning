import streamlit as st
import numpy as np

from tensorflow.keras.models import load_model

from utils.preprocess_ann import (
    preprocess_single_input
)


@st.cache_resource
def load_ann_model():

    return load_model(
        "models/ann_model.h5"
    )


def show():

    model = load_ann_model()

    st.title(
        "ANN - Customer Churn Prediction"
    )

    st.subheader(
        "Customer Information"
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=10,
        value=5
    )

    balance = st.number_input(
        "Balance",
        value=50000.0
    )

    num_products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=1
    )

    has_card = st.selectbox(
        "Has Credit Card",
        [0, 1]
    )

    is_active = st.selectbox(
        "Is Active Member",
        [0, 1]
    )

    salary = st.number_input(
        "Estimated Salary",
        value=50000.0
    )

    geography = st.selectbox(
        "Country",
        [
            "France",
            "Germany",
            "Spain"
        ]
    )

    gender = st.selectbox(
        "Gender",
        [
            "Female",
            "Male"
        ]
    )

    if st.button(
        "Predict Churn"
    ):

        geo_germany = 0
        geo_spain = 0

        if geography == "Germany":
            geo_germany = 1

        elif geography == "Spain":
            geo_spain = 1

        gender_male = (
            1 if gender == "Male"
            else 0
        )

        data = np.array([
            [
                credit_score,
                age,
                tenure,
                balance,
                num_products,
                has_card,
                is_active,
                salary,
                geo_germany,
                geo_spain,
                gender_male
            ]
        ])

        data = preprocess_single_input(
            data
        )

        prediction = model.predict(
            data
        )

        probability = prediction[0][0]

        st.write(
            f"Churn Probability: {probability:.2%}"
        )

        if probability > 0.5:

            st.error(
                "Customer likely to leave"
            )

        else:

            st.success(
                "Customer likely to stay"
            )