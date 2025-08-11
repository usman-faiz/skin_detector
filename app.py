import streamlit as st
from streamlit_option_menu import option_menu
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
import time
import os
import gdown

# --- PAGE CONFIG ---
st.set_page_config(page_title="Skin Disease Detector", page_icon="🩺", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background: rgba(20,20,20,0.9);
            text-align: center;
            padding: 0.5rem;
            font-size: 14px;
            color: #aaa;
        }
        .card {
            background: #1e1e1e;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            transition: transform 0.3s ease;
        }
        .card:hover {
            transform: scale(1.05);
            box-shadow: 0px 0px 15px #00BFA5;
        }
        .uploaded-img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 60%;
            border-radius: 12px;
            box-shadow: 0px 0px 15px rgba(0, 191, 165, 0.4);
        }
        /* --- Mobile Responsive Tweaks --- */  
    # --- CUSTOM CSS ---

        body {
            background-color: #121212;
            color: white;
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background: rgba(20,20,20,0.9);
            text-align: center;
            padding: 0.5rem;
            font-size: 14px;
            color: #aaa;
        }
        .card {
            background: #1e1e1e;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            transition: transform 0.3s ease;
        }
        .card:hover {
            transform: scale(1.05);
            box-shadow: 0px 0px 15px #00BFA5;
        }
        .uploaded-img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 60%;
            border-radius: 12px;
            box-shadow: 0px 0px 15px rgba(0, 191, 165, 0.4);
        }

        /* --- Mobile Responsive Tweaks --- */
        @media (max-width: 768px) {
            .uploaded-img {
                width: 90% !important;
            }
            .footer {
                font-size: 12px;
                padding: 0.3rem;
            }
            .card {
                padding: 15px;
            }
            .stButton>button {
                width: 100% !important;
                font-size: 16px !important;
            }
            .css-1kyxreq { /* Navbar wrapper */
                flex-wrap: wrap !important;
                justify-content: center !important;
            }
        }
        @media (max-width: 480px) {
            .uploaded-img {
                width: 100% !important;
            }
            .card {
                padding: 10px;
            }
            .footer {
                font-size: 10px;
            }
        }
    </style>
""", unsafe_allow_html=True)



# --- NAVBAR ---
selected = option_menu(
    menu_title=None,
    options=["Home", "About", "Model Info", "Contact"],
    icons=["house", "info-circle", "bar-chart", "envelope"],
    orientation="horizontal",
    styles={
        "container": {"background-color": "rgba(20,20,20,0.85)", "padding": "5px"},
        "icon": {"color": "#00BFA5", "font-size": "20px"},
        "nav-link": {"color": "white", "font-size": "18px", "text-align": "center"},
        "nav-link-selected": {"background-color": "#00BFA5", "color": "black"},
    }
)

# --- Load Labels ---
try:
    with open("labels.json", "r") as f:
        labels = json.load(f)
except FileNotFoundError:
    labels = []
    st.error("❌ labels.json not found. Please make sure it's in the same directory.")

# --- Download Model from Google Drive if not exists ---
MODEL_PATH = "best_resnet18_skin.pth"
FILE_ID = "1QuMuPYjVt5YllUzgZ1aVGTD1AVuLrCUi"  # Your actual file ID
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"


if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(URL, MODEL_PATH, quiet=False)

# --- Load Model ---
@st.cache_resource
def load_model():
    from torchvision import models
    model = models.resnet18(weights=None)
    num_classes = len(labels)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# --- Disease Details ---
disease_info = {
    "akiec": {"name": "Actinic Keratoses & Intraepithelial Carcinoma",
              "description": "Precancerous lesions caused by sun damage.",
              "precautions": "Avoid sun exposure, use sunscreen, consult dermatologist."},
    "bcc": {"name": "Basal Cell Carcinoma",
            "description": "Slow-growing form of skin cancer.",
            "precautions": "Early treatment, avoid tanning beds, follow up with doctor."},
    "bkl": {"name": "Benign Keratosis-like Lesions",
            "description": "Non-cancerous growths that resemble warts.",
            "precautions": "Generally harmless, monitor changes, seek advice if unsure."},
    "df": {"name": "Dermatofibroma",
           "description": "Common benign skin growth often on limbs.",
           "precautions": "No treatment usually required, monitor for changes."},
    "mel": {"name": "Melanoma",
            "description": "Most dangerous form of skin cancer.",
            "precautions": "Early detection critical, regular skin checkups."},
    "nv": {"name": "Melanocytic Nevi (Moles)",
           "description": "Common skin moles, usually harmless.",
           "precautions": "Monitor for changes in size, color, or shape."},
    "vasc": {"name": "Vascular Lesions",
             "description": "Lesions formed from blood vessels (e.g., hemangiomas).",
             "precautions": "Usually harmless, consult doctor if rapid growth."}
}

# --- Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- HOME PAGE ---
if selected == "Home":
    st.title("Skin Disease Detector 🩺")
    st.write("Upload an image of a skin lesion to predict its type.")

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=False, output_format="PNG")
        st.markdown('<style>img {max-width: 60%;}</style>', unsafe_allow_html=True)

        # Preprocess
        img_tensor = transform(img).unsqueeze(0)

        with st.spinner("Predicting..."):
            time.sleep(1)
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_index = torch.argmax(probs).item()
                confidence = probs[pred_index].item() * 100

        if labels:
            predicted_code = labels[pred_index]
        else:
            predicted_code = f"Class {pred_index}"

        disease = disease_info.get(predicted_code, {
            "name": "Unknown",
            "description": "No data available.",
            "precautions": "No data available."
        })

        st.success(f"**Predicted:** {predicted_code} ({disease['name']})\n"
                   f"**Confidence:** {confidence:.2f}%")
        st.info(f"**Description:** {disease['description']}\n"
                f"**Precautions:** {disease['precautions']}")

# --- ABOUT PAGE ---
elif selected == "About":
    st.title("About This App")
    st.write("""
    This application uses **deep learning** (ResNet18) to classify skin lesion images.
    - **Dataset**: HAM10000
    - **Framework**: PyTorch
    """)

# --- MODEL INFO PAGE ---
elif selected == "Model Info":
    st.title("Model Performance")
    st.info("Performance graphs and metrics will be added here.")

# --- CONTACT PAGE ---
elif selected == "Contact":
    st.title("Contact Us")
    st.write("For queries and feedback:")
    st.markdown("📧 **usmanfaiz085@gmail.com**")

# --- FOOTER ---
st.markdown("""
<div class="footer">
    &copy; 2025 Skin Disease Detector | Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)

