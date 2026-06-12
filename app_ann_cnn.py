import streamlit as st

st.set_page_config(
    page_title="Deep Learning Studio",
    layout="wide"
)

st.markdown("""
<style>

.main-title{
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#4CAF50;
}

</style>
""",
unsafe_allow_html=True)

st.markdown(
"""
<div class='main-title'>
Deep Learning Studio
</div>
""",
unsafe_allow_html=True
)

choice = st.sidebar.radio(
    "Choose Model",
    [
        "ANN",
        "CNN"
    ]
)

if choice == "ANN":

    from pages.ann_pages import show

    show()

else:

    from pages.cnn_pages import show

    show()