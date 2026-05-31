import time
import numpy as np
import cv2
import streamlit as st

from PIL import Image
from io import BytesIO

from utils.enhancement import (

    histogram_equalization,

    clahe_enhancement,

    gamma_enhancement,

    handcrafted_zero_dce,

    neural_zero_dce,

    gainnet_enhancement
)


st.set_page_config(
    page_title="Low-Light Enhancement Studio",
    layout="wide"
)

st.title(
    "Real-Time Low-Light Enhancement Studio"
)

st.write(
    "Upload a low-light image and compare enhancement methods."
)


uploaded_file = st.file_uploader(

    "Upload Image",

    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


method = st.selectbox(

    "Select Enhancement Method",

    [

        "Histogram Equalization",

        "CLAHE",

        "Gamma Correction",

        "Zero-DCE Handcrafted",

        "Zero-DCE Neural",

        "GainNet"

    ]
)


if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image = np.array(
        image
    )

    if st.button(
        "Enhance Image"
    ):

        start = time.time()

        if method == "Histogram Equalization":

            output = histogram_equalization(
                image
            )

        elif method == "CLAHE":

            output = clahe_enhancement(
                image
            )

        elif method == "Gamma Correction":

            output = gamma_enhancement(
                image
            )

        elif method == "Zero-DCE Handcrafted":

            output = handcrafted_zero_dce(
                image
            )

        elif method == "GainNet":

            output = gainnet_enhancement(
            image
        )    

        else:

            output = neural_zero_dce(
                image
            )

        runtime = (
            time.time()
            -
            start
        )

        col1,col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Input Image",
                use_container_width=True
            )

        with col2:

            st.image(
                output,
                caption="Enhanced Image",
                use_container_width=True
            )

        brightness = np.mean(

            cv2.cvtColor(
                output,
                cv2.COLOR_RGB2GRAY
            )

        )

        contrast = np.std(

            cv2.cvtColor(
                output,
                cv2.COLOR_RGB2GRAY
            )

        )

        c1,c2,c3 = st.columns(3)

        c1.metric(
            "Brightness",
            round(
                brightness,
                2
            )
        )

        c2.metric(
            "Contrast",
            round(
                contrast,
                2
            )
        )

        c3.metric(
            "Runtime (s)",
            round(
                runtime,
                4
            )
        )

        buffer = BytesIO()

        Image.fromarray(
            output
        ).save(
            buffer,
            format="PNG"
        )

        st.download_button(

            "Download PNG",

            data=buffer.getvalue(),

            file_name="enhanced.png",

            mime="image/png"
        )