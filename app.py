import streamlit as st
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from feature_extractor import extract_features
import torch
import torch.nn as nn
import torchvision.models as models

@st.cache_data
def load_train_features():
    with open("Resnet_train.pkl", "rb") as f:
        return np.array(pickle.load(f))

@st.cache_data
def load_train_images():
    with open("RawPixels_train.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_labels():
    with open("Labels_train.pkl", "rb") as f:
        return pickle.load(f)

# Load cached resources
train_features = load_train_features()
train_images = load_train_images()
y_train = load_labels()

# --- Load Trained Keras Model ---
model = load_model("model.h5")

# --- Load ResNet model from saved weights ---
@st.cache_resource
def load_resnet():
    resnet = models.resnet50(weights=None)  # Do not download weights
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.load_state_dict(torch.load("resnet50_feature_extractor.pth", map_location=torch.device("cpu")))
    resnet.eval()
    return resnet

resnet = load_resnet()

# --- Streamlit App ---
st.set_page_config(page_title="Image Classification & Retrieval", page_icon="🔍")
st.title("🔍 **CIFAR-10 Image Classification & Similar Image Retrieval**")
st.markdown("""
    Welcome to the CIFAR-10 Image Classification and Retrieval system. 
    Upload an image to get its predicted class and find similar images from our dataset.
""")

# --- File Uploader with Instructions ---
uploaded_file = st.file_uploader("Upload a Query Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    query_image = Image.open(uploaded_file).convert("RGB")
    st.image(query_image, caption="Query Image", use_container_width=True)

    # --- Feature Extraction ---
    image_np = np.array(query_image)
    query_feature = np.array(extract_features(image_np, resnet)).reshape(1, -1)

    # --- Classification ---
    pred_class = int(np.argmax(model.predict(query_feature), axis=1)[0])
    st.subheader(f"📌 **Predicted Class: {pred_class}**")

    # --- Filter by Predicted Class ---
    class_indices = [i for i, label in enumerate(y_train) if label == pred_class]
    if not class_indices:
        st.error("No images found for this class in the training set.")
    else:
        filtered_features = train_features[class_indices]
        filtered_images = [train_images[i] for i in class_indices]

        # --- Cosine Similarity ---
        similarities = cosine_similarity(query_feature, filtered_features)[0]
        top5_indices = similarities.argsort()[-5:][::-1]

        # --- Display Similar Images ---
        st.subheader("📸 **Top 5 Most Similar Images (Same Class)**:")
        cols = st.columns(5)
        for i, idx in enumerate(top5_indices):
            with cols[i]:
                st.image(filtered_images[idx], width=120)
                st.caption(f"Label: {pred_class} | Similarity: {similarities[idx]:.2f}")

        # --- Display Similarity Scores ---
        st.write("**Cosine Similarity Scores:**")
        for idx in top5_indices:
            st.write(f"Image {idx+1}: {similarities[idx]:.2f}")
