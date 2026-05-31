import cv2
import torch
import numpy as np

from models.zero_dce import (
    DCENet,
    enhance
)

from models.gainnet import (
    GainNet,
    enhance as gain_enhance
)


# ---------------------------------------------------
# Histogram Equalization
# ---------------------------------------------------

def histogram_equalization(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    eq = cv2.equalizeHist(
        gray
    )

    eq = cv2.cvtColor(
        eq,
        cv2.COLOR_GRAY2RGB
    )

    return eq


# ---------------------------------------------------
# CLAHE
# ---------------------------------------------------

def clahe_enhancement(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    out = clahe.apply(
        gray
    )

    out = cv2.cvtColor(
        out,
        cv2.COLOR_GRAY2RGB
    )

    return out


# ---------------------------------------------------
# Gamma
# ---------------------------------------------------

def gamma_enhancement(
    image,
    gamma=0.5
):

    img = image.astype(
        np.float32
    ) / 255.0

    out = np.power(
        img,
        gamma
    )

    out = (
        out * 255
    ).astype(np.uint8)

    return out


# ---------------------------------------------------
# Handcrafted Zero DCE
# ---------------------------------------------------

def handcrafted_zero_dce(
    image
):

    img = image.astype(
        np.float32
    ) / 255.0

    gray = img.mean(axis=2)

    blur1 = cv2.GaussianBlur(
        gray,
        (15,15),
        0
    )

    blur2 = cv2.GaussianBlur(
        gray,
        (51,51),
        0
    )

    blur3 = cv2.GaussianBlur(
        gray,
        (101,101),
        0
    )

    heatmap = (
        (1-blur1)
        +
        (1-blur2)
        +
        (1-blur3)
    ) / 3

    heatmap = np.clip(
        heatmap,
        0,
        1
    )

    out = img.copy()

    for i in range(8):

        strength = np.exp(
            -i/3
        )

        A = strength * heatmap

        A_rgb = np.stack(
            [A,A,A],
            axis=-1
        )

        out = (
            out
            +
            A_rgb *
            out *
            (1-out)
        )

        out = np.clip(
            out,
            0,
            1
        )

    out = (
        out * 255
    ).astype(np.uint8)

    return out


# ---------------------------------------------------
# Load Neural ZeroDCE
# ---------------------------------------------------

device = torch.device("cpu")

model = DCENet()

model.load_state_dict(
    torch.load(
        "weights/zerodce.pth",
        map_location=device
    )
)

model.eval()


# ---------------------------------------------------
# Neural ZeroDCE
# ---------------------------------------------------

def neural_zero_dce(
    image
):

    img = image.astype(
        np.float32
    ) / 255.0

    tensor = torch.tensor(
        img
    ).permute(
        2,
        0,
        1
    ).unsqueeze(0)

    with torch.no_grad():

        maps = model(
            tensor
        )

        out = enhance(
            tensor,
            maps
        )

    out = (
        out[0]
        .permute(
            1,
            2,
            0
        )
        .numpy()
    )

    out = np.clip(
        out,
        0,
        1
    )

    out = (
        out * 255
    ).astype(np.uint8)

    return out


gain_model = GainNet()

gain_model.load_state_dict(
    torch.load(
        "weights/gainnet.pth",
        map_location="cpu"
    )
)

gain_model.eval()

def gainnet_enhancement(
    image
):

    original_h = image.shape[0]
    original_w = image.shape[1]

    img = cv2.resize(
        image,
        (256,256)
    )

    img = (
        img.astype(np.float32)
        /255.0
    )

    tensor = torch.tensor(
        img,
        dtype=torch.float32
    )

    tensor = tensor.permute(
        2,
        0,
        1
    )

    tensor = tensor.unsqueeze(0)

    with torch.no_grad():

        gain = gain_model(
            tensor
        )

        enhanced = gain_enhance(
            tensor,
            gain
        )

    enhanced = (
        enhanced[0]
        .permute(1,2,0)
        .numpy()
    )

    enhanced = np.clip(
        enhanced,
        0,
        1
    )

    enhanced = cv2.resize(
        enhanced,
        (
            original_w,
            original_h
        )
    )

    enhanced = (
        enhanced * 255
    ).astype(np.uint8)

    return enhanced