# 🎨 Neural Style Transfer Web Application

This project implements **Neural Style Transfer (NST)** using deep learning to apply the artistic style of one image to the content of another.  
It is built using **PyTorch**, **VGG19**, and **Streamlit**, and was developed as **Task-05 for the Prodigy Infotech Internship**.

---

## 🚀 Overview

Neural Style Transfer combines:
- **Content representation** from one image
- **Style representation** from another image

Using a pretrained **VGG19 convolutional neural network**, the application optimizes a generated image to preserve content while adopting artistic textures and patterns.

---

## ✨ Features

- Upload a **content image**
- Upload a **style image**
- Apply CNN-based neural style transfer
- Generate a stylized output image
- Interactive and easy-to-use **Streamlit web interface**
- CPU-friendly implementation (no GPU required)

---

## 🧠 Technical Approach

- **Model:** VGG19 (pretrained on ImageNet)
- **Content loss:** Feature reconstruction from deep CNN layers
- **Style loss:** Gram matrix-based texture representation
- **Optimization:** Gradient-based optimization on the generated image
- **Frameworks:** PyTorch, TorchVision

---

## 🛠️ Tech Stack

- Python
- PyTorch
- TorchVision
- Streamlit
- Pillow (PIL)
- NumPy

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
