import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Neural Style Transfer",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 Neural Style Transfer")
st.write("Apply the artistic style of one image to another using CNN-based feature extraction.")

# -------------------------------
# Device (CPU only for Streamlit)
# -------------------------------
device = torch.device("cpu")

# -------------------------------
# Image Loader
# -------------------------------
def load_image(image, max_size=256):
    transform = transforms.Compose([
        transforms.Resize(max_size),
        transforms.ToTensor()
    ])
    image = Image.open(image).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image.to(device)

# -------------------------------
# Display Image
# -------------------------------
def tensor_to_image(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = image.numpy().transpose(1, 2, 0)
    image = np.clip(image, 0, 1)
    return image

# -------------------------------
# Gram Matrix
# -------------------------------
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    gram = torch.mm(features, features.t())
    return gram / (c * h * w)

# -------------------------------
# Feature Extraction
# -------------------------------
def get_features(image, model):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',  # content layer
        '28': 'conv5_1'
    }

    features = {}
    x = image

    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features

# -------------------------------
# Style Transfer Function (FIXED)
# -------------------------------
def run_style_transfer(content, style, steps=50, style_weight=1e5):
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

    for param in vgg.parameters():
        param.requires_grad = False

    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    # ✅ Detach targets (IMPORTANT FIX)
    with torch.no_grad():
        content_target = content_features['conv4_2']
        style_targets = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # Generated image (leaf tensor)
    generated = content.clone().requires_grad_(True)

    optimizer = optim.Adam([generated], lr=0.02)

    for step in range(steps):
        optimizer.zero_grad()

        generated_features = get_features(generated, vgg)

        # Content loss
        content_loss = torch.mean((generated_features['conv4_2'] - content_target) ** 2)

        # Style loss
        style_loss = 0
        for layer in style_targets:
            gen_gram = gram_matrix(generated_features[layer])
            style_gram = style_targets[layer]
            style_loss += torch.mean((gen_gram - style_gram) ** 2)

        total_loss = content_loss + style_weight * style_loss
        total_loss.backward()
        optimizer.step()

    return generated

# -------------------------------
# UI
# -------------------------------
content_file = st.file_uploader("📸 Upload Content Image", type=["jpg", "png", "jpeg"])
style_file = st.file_uploader("🖌️ Upload Style Image", type=["jpg", "png", "jpeg"])

steps = st.slider("🔁 Number of Steps (higher = better, slower)", 20, 100, 50)

if content_file and style_file:
    content_img = load_image(content_file)
    style_img = load_image(style_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(tensor_to_image(content_img), caption="Content Image", width=250)
    with col2:
        st.image(tensor_to_image(style_img), caption="Style Image", width=250)

    if st.button("✨ Generate Stylized Image"):
        with st.spinner("Applying Neural Style Transfer... Please wait ⏳"):
            output = run_style_transfer(content_img, style_img, steps=steps)

        st.success("Style Transfer Complete!")
        st.image(tensor_to_image(output), caption="Stylized Output", use_container_width=True)

else:
    st.info("Upload both content and style images to begin.")
