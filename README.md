# 🩺 Skin Disease Detector

A **Streamlit-based web application** that predicts the type of skin disease from an uploaded image using a **MobileNetV2 deep learning model** trained on the HAM10000 dataset.  
The app also automatically maps medical disease codes to **full names, short descriptions, and precautionary measures**.

---

## 🚀 Features
- Upload skin lesion images and get instant predictions  
- Shows confidence percentage for each prediction  
- Dark theme with a sleek, modern UI and smooth animations  
- Disease code mapping to full disease names, description & precautions  
- Multi-page layout with a Navbar and Footer  

---

## 🧠 Model
- **Base Model**: MobileNetV2 (pretrained on ImageNet)  
- **Custom Layers**: GlobalAveragePooling + Dense (7 output classes)  
- **Accuracy**: ~74%  

---

## 🖥️ Tech Stack
- **Python 3.9+**
- **TensorFlow / Keras**
- **Streamlit**
- **NumPy**
- **Pillow**

---

## 📷 Screenshots
*(Add screenshots here if needed)*

---

## 📦 Installation & Run (Local)
```bash
# Clone this repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
