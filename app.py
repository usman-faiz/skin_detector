import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Skin Disease Detector", page_icon="🩺", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .navbar {
            background: rgba(20,20,20,0.85);
            padding: 1rem 2rem;
            border-bottom: 1px solid #00BFA5;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .navbar a {
            margin: 0 15px;
            text-decoration: none;
            color: white;
            font-weight: 500;
            transition: color 0.3s ease-in-out;
        }
        .navbar a:hover {
            color: #00BFA5;
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

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_model.h5")

model = load_model()

labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# --- Disease Info Mapping ---
disease_info = {
    "akiec": {
        "name": "Actinic Keratoses & Intraepithelial Carcinoma",
        "desc": "Precancerous skin lesions caused by sun damage.",
        "precaution": "Limit sun exposure, use sunscreen, consult dermatologist regularly."
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "desc": "Common, slow-growing form of skin cancer.",
        "precaution": "Avoid UV exposure, schedule routine checkups, treat early.",
        "critical": True
    },
    "bkl": {
        "name": "Benign Keratosis-Like Lesions",
        "desc": "Non-cancerous skin growth, usually harmless.",
        "precaution": "Monitor changes, keep skin moisturized, consult if growth changes."
    },
    "df": {
        "name": "Dermatofibroma",
        "desc": "Benign fibrous skin nodule.",
        "precaution": "Generally harmless; consult dermatologist if painful or enlarging."
    },
    "mel": {
        "name": "Melanoma",
        "desc": "Serious form of skin cancer originating in pigment cells.",
        "precaution": "Early detection is key, avoid tanning beds, use broad-spectrum sunscreen.",
        "critical": True
    },
    "nv": {
        "name": "Melanocytic Nevi (Moles)",
        "desc": "Common pigmented spots on skin, usually harmless.",
        "precaution": "Watch for irregular shape/color changes; yearly checkup recommended."
    },
    "vasc": {
        "name": "Vascular Lesions",
        "desc": "Includes hemangiomas and angiomas, mostly benign.",
        "precaution": "Monitor bleeding or rapid growth; avoid injury on affected areas."
    }
}

# --- PAGES ---
if selected == "Home":
    st.title("Skin Disease Detector")
    st.write("Upload an image of a skin lesion to predict its type.")

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Preprocess
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Predicting..."):
            time.sleep(1)
            predictions = model.predict(img_array)
            pred_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100

        pred_code = labels[pred_index]
        info = disease_info.get(pred_code, {"name": "Unknown", "desc": "-", "precaution": "-"})
        badge = "⚠️ **High Risk**" if info.get("critical") else ""

        st.success(f"**Predicted: {info['name']} ({pred_code.upper()})** {badge}  \n**Confidence:** {confidence:.2f}%")
        st.write(f"**Description:** {info['desc']}")
        st.write(f"**Precaution:** {info['precaution']}")

    st.subheader("About Skin Diseases")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card"><h3>Melanoma</h3><p>Serious skin cancer often caused by UV exposure.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h3>Nevus</h3><p>Commonly known as moles, usually harmless.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><h3>BCC</h3><p>Basal cell carcinoma, common but slow-growing.</p></div>', unsafe_allow_html=True)

elif selected == "About":
    st.title("About This App")
    st.write("""
    This application uses **deep learning** to classify skin lesion images.
    - **Dataset**: HAM10000
    - **Model**: MobileNetV2
    - **Accuracy**: ~74%
    """)
    st.subheader("Tech Stack")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card"><h3>Python</h3><p>For model training & backend logic.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h3>TensorFlow</h3><p>For deep learning model development.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><h3>Streamlit</h3><p>For building a quick interactive UI.</p></div>', unsafe_allow_html=True)

elif selected == "Model Info":
    st.title("Model Performance")
    st.info("Graphs & performance metrics will be added here (placeholder).")

elif selected == "Contact":
    st.title("Contact Us")
    st.write("For queries and feedback, reach out to:")
    st.markdown("📧 **usmanfaiz085@gmail.com**")
    st.markdown("📞 **+92 302 9802127**")

# --- FOOTER ---
st.markdown("""
<div class="footer">
    &copy; 2025 Skin Disease Detector | Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
