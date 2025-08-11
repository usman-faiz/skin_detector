# 🩺 Skin Disease Detector

A Streamlit-based web application that predicts the type of skin disease from an uploaded image using a **MobileNetV2** deep learning model trained on the **HAM10000** dataset.  
The app also automatically maps medical disease codes to full names, short descriptions, and precautionary measures.

---

## 🚀 Features
- 📤 Upload skin lesion images and get instant predictions  
- 📊 Shows confidence percentage for each prediction  
- 🖤 Dark theme with a sleek, modern UI and smooth animations  
- 🩹 Disease code mapping to full disease names, description & precautions  
- 📄 Multi-page layout with a Navbar and Footer  
- 🛑 Early Stopping enabled to prevent overfitting  

---

## 🧠 Model
- **Base Model:** MobileNetV2 (pretrained on ImageNet)  
- **Custom Layers:** GlobalAveragePooling + Dense (7 output classes)  
- **Accuracy:** ~95% (trained with EarlyStopping at ~12 epochs)  

---

## 🖥️ Tech Stack
- Python 3.9+  
- TensorFlow / Keras  
- Streamlit  
- NumPy  
- Pillow  

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

streamlit
streamlit-option-menu
tensorflow
pillow
numpy



---

If you want, I can also **add a "Training Details" section** explaining the dataset, epochs, and early stopping settings so that anyone reading your repo knows exactly how the 95% was achieved.  
Do you want me to add that?


# Run the app
streamlit run app.py
