import streamlit as st
import numpy as np
import json
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="COVID-19 X-ray Detection (CNN)",
    layout="centered")

st.title("COVID-19 Detection from Chest X-rays")
st.write(
    "Made by Ayush Anand - CNN Mini-Project")
st.write(
    "Upload a chest X-ray image to classify it as **Covid**, "
    "**Normal**, or **Viral Pneumonia**")

@st.cache_resource
def load_artifacts():
    model = load_model("covid_xray_vgg16.keras")
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names
model, class_names = load_artifacts()
IMG_SIZE = 128  

def preprocess_image(image):
    image = image.convert("L")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image).astype(np.float32)
    img_array = img_array / 255.0
    img_array = np.stack([img_array] * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)
    input_tensor = preprocess_image(image)
    predictions = model.predict(input_tensor, verbose=0)[0]
    top_indices = predictions.argsort()[-2:][::-1]
    primary_class = class_names[top_indices[0]]
    primary_conf = predictions[top_indices[0]] * 100
    secondary_class = class_names[top_indices[1]]
    secondary_conf = predictions[top_indices[1]] * 100
    st.subheader("Prediction Result")
    st.success(f"**Primary Prediction:** {primary_class}")
    st.info(f"**Confidence:** {primary_conf:.2f}%")
    st.warning(
        f"**Second Likely Class:** {secondary_class} "
        f"({secondary_conf:.2f}%)")
    if primary_conf < 60:
        st.error(
            "Low confidence prediction. "
            "Further clinical evaluation is recommended.")
else:
    st.warning("Please upload a chest X-ray image to get a prediction.")

st.markdown("---")
st.caption(
    "Model: VGG16 (Transfer Learning) | "
    "Streamlit App by Ayush Anand (IITG Course)")
