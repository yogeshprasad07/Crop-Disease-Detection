import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib

matplotlib.use("Agg")

# ============================================================
# TOMATO DISEASE AI
# STREAMLIT APPLICATION
# MobileNetV2 + Grad-CAM
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "mobilenetv2_final_best.keras"
)

IMAGE_SIZE = (224, 224)

# ============================================================
# CLASS NAMES
# IMPORTANT:
# Keep this order identical to the model training order.
# ============================================================

CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

DISPLAY_NAMES = {
    "Tomato___Bacterial_spot": "Bacterial Spot",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___healthy": "Healthy",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___Leaf_Mold": "Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite":
        "Spider Mites / Two-spotted Spider Mite",
    "Tomato___Target_Spot": "Target Spot",
    "Tomato___Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "Tomato Yellow Leaf Curl Virus",
}


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tomato Disease AI",
    page_icon="🍅",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 25px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
    }

    .confidence {
        font-size: 28px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


# ============================================================
# FIND MOBILENETV2 BASE MODEL
# ============================================================

def find_mobilenet_base(model):

    for layer in model.layers:

        # Direct MobileNetV2 layer
        if isinstance(
            layer,
            tf.keras.Model
        ):

            name = layer.name.lower()

            if "mobilenetv2" in name:

                return layer

    # Recursive search
    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.Model
        ):

            for sublayer in layer.layers:

                if "mobilenetv2" in sublayer.name.lower():

                    return sublayer

    raise ValueError(
        "MobileNetV2 base model could not be found."
    )


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_target_conv_layer(base_model):

    conv_layers = []

    for layer in base_model.layers:

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):

            conv_layers.append(layer)

    if not conv_layers:

        raise ValueError(
            "No Conv2D layer found inside MobileNetV2."
        )

    return conv_layers[-1]


# ============================================================
# FIND CLASSIFIER LAYERS
# ============================================================

def find_classifier_layers(model):

    gap_layer = None
    dropout_layer = None
    dense_layer = None

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.layers.GlobalAveragePooling2D
        ):

            gap_layer = layer

        elif isinstance(
            layer,
            tf.keras.layers.Dropout
        ):

            dropout_layer = layer

        elif isinstance(
            layer,
            tf.keras.layers.Dense
        ):

            dense_layer = layer

    if gap_layer is None:

        raise ValueError(
            "GlobalAveragePooling2D layer not found."
        )

    if dense_layer is None:

        raise ValueError(
            "Final Dense layer not found."
        )

    return (
        gap_layer,
        dropout_layer,
        dense_layer
    )


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(image):

    image = image.convert("RGB")

    resized = image.resize(
        IMAGE_SIZE
    )

    array = np.asarray(
        resized
    ).astype(
        np.float32
    )

    tensor = tf.convert_to_tensor(
        array
    )

    tensor = tf.expand_dims(
        tensor,
        axis=0
    )

    return tensor


# ============================================================
# PREDICTION
# ============================================================

def predict_image(model, image_tensor):

    probabilities = model(
        image_tensor,
        training=False
    )

    probabilities = probabilities.numpy()[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

    return (
        predicted_index,
        probabilities,
        confidence
    )


# ============================================================
# GRAD-CAM
# ============================================================

def calculate_gradcam(
    model,
    image_tensor,
    predicted_index
):

    # --------------------------------------------------------
    # Find MobileNetV2
    # --------------------------------------------------------

    base_model = find_mobilenet_base(
        model
    )

    # --------------------------------------------------------
    # Find final convolution layer
    # --------------------------------------------------------

    target_layer = find_target_conv_layer(
        base_model
    )

    # --------------------------------------------------------
    # Build Grad-CAM model
    #
    # IMPORTANT:
    # We use the MobileNetV2 model directly.
    # No custom GradCAMModel is used.
    # --------------------------------------------------------

    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            target_layer.output,
            base_model.output
        ]
    )

    # --------------------------------------------------------
    # Get image in the same scale expected by the model
    #
    # Your parent model contains the Rescaling layer.
    # We therefore use 0-255 image values for the parent model.
    #
    # MobileNetV2 internally receives the output of the
    # parent preprocessing pipeline.
    # --------------------------------------------------------

    # Find parent Rescaling layer
    rescaling_layer = None

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.layers.Rescaling
        ):

            rescaling_layer = layer
            break

    if rescaling_layer is not None:

        base_input = rescaling_layer(
            image_tensor
        )

    else:

        base_input = image_tensor / 255.0

    # --------------------------------------------------------
    # Calculate gradients
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        tape.watch(base_input)

        conv_outputs, base_outputs = grad_model(
            base_input,
            training=False
        )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        gap_layer, dropout_layer, dense_layer = (
            find_classifier_layers(model)
        )

        x = gap_layer(
            base_outputs
        )

        if dropout_layer is not None:

            x = dropout_layer(
                x,
                training=False
            )

        predictions = dense_layer(
            x
        )

        class_score = predictions[
            :, predicted_index
        ]

    # --------------------------------------------------------
    # Gradients
    # --------------------------------------------------------

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:

        raise ValueError(
            "Gradients could not be calculated."
        )

    # --------------------------------------------------------
    # Global average pooling of gradients
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # --------------------------------------------------------
    # Weighted feature maps
    # --------------------------------------------------------

    conv_outputs = conv_outputs[0]

    pooled_gradients = pooled_gradients[0]

    heatmap = tf.reduce_sum(
        conv_outputs
        * pooled_gradients,
        axis=-1
    )

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value + tf.keras.backend.epsilon()
    )

    return heatmap.numpy()


# ============================================================
# CREATE GRAD-CAM OVERLAY
# ============================================================

def create_overlay(
    original_image,
    heatmap
):

    import matplotlib.pyplot as plt

    # Resize heatmap to original image
    original_size = original_image.size

    heatmap_image = Image.fromarray(
        np.uint8(
            heatmap * 255
        )
    )

    heatmap_image = heatmap_image.resize(
        original_size,
        Image.Resampling.BILINEAR
    )

    heatmap_array = np.asarray(
        heatmap_image
    )

    # --------------------------------------------------------
    # Apply JET colormap
    # --------------------------------------------------------

    colormap = matplotlib.colormaps["jet"]

    colored_heatmap = colormap(
        heatmap_array / 255.0
    )

    colored_heatmap = np.uint8(
        colored_heatmap[:, :, :3]
        * 255
    )

    colored_heatmap = Image.fromarray(
        colored_heatmap
    )

    # --------------------------------------------------------
    # Blend original + heatmap
    # --------------------------------------------------------

    original_rgb = original_image.convert(
        "RGB"
    )

    overlay = Image.blend(
        original_rgb,
        colored_heatmap,
        alpha=0.45
    )

    return (
        heatmap_image,
        overlay
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🍅 Tomato Disease AI"
    )

    st.markdown(
        "### Model Information"
    )

    st.write(
        "**Model:** MobileNetV2"
    )

    st.write(
        "**Input:** 224 × 224 RGB image"
    )

    st.write(
        "**Classes:** 10"
    )

    st.write(
        "**Explainability:** Grad-CAM"
    )

    st.write(
        "**Framework:** TensorFlow / Keras"
    )

    st.write(
        "**Deployment:** Streamlit"
    )

    st.divider()

    st.info(
        "Upload a tomato leaf image to obtain "
        "the predicted disease and Grad-CAM "
        "visual explanation."
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🍅 Tomato Disease Detection AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    Upload a tomato leaf image and the trained
    MobileNetV2 model will classify the image.
    Grad-CAM is then used to visualize the regions
    that contributed to the prediction.
    """
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

except Exception as e:

    st.error(
        "Model loading failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    original_image = Image.open(
        uploaded_file
    ).convert(
        "RGB"
    )

    st.divider()

    col1, col2 = st.columns(
        2
    )

    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "Uploaded Image"
        )

        st.image(
            original_image,
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    image_tensor = prepare_image(
        original_image
    )

    try:

        (
            predicted_index,
            probabilities,
            confidence
        ) = predict_image(
            model,
            image_tensor
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        display_name = DISPLAY_NAMES.get(
            predicted_class,
            predicted_class
        )

    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)

        st.stop()

    # --------------------------------------------------------
    # PREDICTION RESULT
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Prediction Result"
    )

    result_col1, result_col2 = st.columns(
        2
    )

    with result_col1:

        st.success(
            f"Predicted Disease: {display_name}"
        )

    with result_col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

    # --------------------------------------------------------
    # TOP 3 PREDICTIONS
    # --------------------------------------------------------

    st.subheader(
        "Top 3 Predictions"
    )

    top_indices = np.argsort(
        probabilities
    )[::-1][:3]

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        class_name = CLASS_NAMES[
            index
        ]

        display_name_top = DISPLAY_NAMES.get(
            class_name,
            class_name
        )

        probability = probabilities[
            index
        ]

        st.write(
            f"**{rank}. {display_name_top}** "
            f"— {probability * 100:.2f}%"
        )

        st.progress(
            float(probability)
        )

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🔥 Grad-CAM Visual Explanation"
    )

    st.write(
        "Grad-CAM highlights regions of the leaf "
        "that contributed most strongly to the "
        "model's prediction."
    )

    st.write(
        "Higher-intensity regions indicate stronger "
        "influence on the classification decision."
    )

    try:

        with st.spinner(
            "Generating Grad-CAM..."
        ):

            heatmap = calculate_gradcam(
                model,
                image_tensor,
                predicted_index
            )

            heatmap_image, overlay = (
                create_overlay(
                    original_image,
                    heatmap
                )
            )

        cam_col1, cam_col2 = st.columns(
            2
        )

        with cam_col1:

            st.image(
                heatmap_image,
                caption="Grad-CAM Heatmap",
                use_container_width=True
            )

        with cam_col2:

            st.image(
                overlay,
                caption="Grad-CAM Overlay",
                use_container_width=True
            )

        st.success(
            "Grad-CAM generated successfully."
        )

    except Exception as e:

        st.error(
            "Grad-CAM generation failed."
        )

        st.exception(e)


