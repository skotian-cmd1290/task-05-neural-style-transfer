import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Neural Style Transfer",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 Neural Style Transfer")
st.write("Apply the artistic style of one image to the content of another using VGG19.")

# ---------------- Image Loader ----------------
def load_image(image, max_size=400):
    image = Image.open(image).convert("RGB")
    size = min(max(image.size), max_size)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image

# ---------------- Gram Matrix ----------------
def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# ---------------- VGG Model ----------------
class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg19(pretrained=True).features.eval()
        self.layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',
            '28': 'conv5_1'
        }

    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features

# ---------------- Style Transfer ----------------
def run_style_transfer(content, style, steps=200, style_weight=1e6, content_weight=1):
    model = VGGFeatures()
    generated = content.clone().requires_grad_(True)

    optimizer = optim.Adam([generated], lr=0.01)

    style_features = model(style)
    content_features = model(content)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    for step in range(steps):
        gen_features = model(generated)

        content_loss = torch.mean((gen_features['conv4_2'] - content_features['conv4_2']) ** 2)

        style_loss = 0
        for layer in style_grams:
            gen_gram = gram_matrix(gen_features[layer])
            style_gram = style_grams[layer]
            style_loss += torch.mean((gen_gram - style_gram) ** 2)

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            st.write(f"Step {step}/{steps} — Loss: {total_loss.item():.2f}")

    return generated.detach()

# ---------------- UI ----------------
content_file = st.file_uploader("📷 Upload Content Image", type=["jpg", "png"])
style_file = st.file_uploader("🎨 Upload Style Image", type=["jpg", "png"])

steps = st.slider("🔁 Optimization Steps", 50, 300, 200)

if content_file and style_file:
    content = load_image(content_file)
    style = load_image(style_file)

    st.subheader("Input Images")
    col1, col2 = st.columns(2)
    col1.image(Image.open(content_file), caption="Content Image", use_column_width=True)
    col2.image(Image.open(style_file), caption="Style Image", use_column_width=True)

    if st.button("✨ Generate Stylized Image"):
        with st.spinner("Applying Neural Style Transfer..."):
            output = run_style_transfer(content, style, steps=steps)

        output_image = output.squeeze().permute(1, 2, 0).numpy()
        output_image = np.clip(output_image, 0, 1)

        st.subheader("🎉 Stylized Output")
        st.image(output_image, use_column_width=True)
