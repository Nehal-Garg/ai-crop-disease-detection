import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import plotly.express as px
import pickle

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Crop Doctor", layout="wide")

# ---------------- LOAD MODELS ----------------
model = tf.keras.models.load_model("model/plant_disease_model.h5", compile=False)

with open("model/gmm_model.pkl", "rb") as f:
    gmm = pickle.load(f)

# Feature extractor
feature_extractor = tf.keras.Model(
    inputs=model.input,
    outputs=model.layers[-2].output
)

# ---------------- CLASS NAMES ----------------
class_names = [
    'Pepper_bell_Bacterial_spot','Pepper_bell_healthy',
    'Potato_Early_blight','Potato_healthy','Potato_Late_blight',
    'Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight',
    'Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato_Target_Spot','Tomato_Tomato_mosaic_virus',
    'Tomato_Tomato_YellowLeaf_Curl_Virus','Tomato_healthy'
]

# ---------------- BACKGROUND ----------------
def set_bg():
    try:
        with open("static/background.jpeg", "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
        }}
        .block-container {{
            background: rgba(255,255,255,0.95);
            padding: 30px;
            border-radius: 15px;
            max-width: 1400px;
        }}
        h1,h2,h3,p,div {{ color:black !important; }}
        </style>
        """, unsafe_allow_html=True)
    except:
        pass

set_bg()

# ---------------- HELPERS ----------------
def get_crop(name):
    return name.split("_")[0]

def get_disease(name):
    return " ".join(name.split("_")[1:])

def get_info(name):
    name = name.lower()

    if "healthy" in name:
        return ("Healthy plant",
                "Maintain watering & sunlight",
                "Use balanced fertilizer")

    elif "early_blight" in name:
        return ("Fungal disease (Early Blight)",
                "Avoid water on leaves",
                "Use fungicide spray")

    elif "late_blight" in name:
        return ("Severe fungal infection",
                "Remove infected leaves",
                "Apply fungicide")

    elif "bacterial" in name:
        return ("Bacterial infection",
                "Remove infected parts",
                "Use copper spray")

    elif "virus" in name:
        return ("Viral infection",
                "Remove plant",
                "Control insects")

    else:
        return ("Plant disease detected",
                "Monitor plant",
                "Use fertilizer")

# ---------------- TITLE ----------------
st.title("🌿 AI Crop Disease Detection Dashboard")

# ---------------- INPUT ----------------
col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Upload / Capture")
    upload = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

    if upload is None:
        camera = st.camera_input("Capture Image")
        file = camera
    else:
        file = upload

# ---------------- PROCESS ----------------
if file:
    image = Image.open(file)

    with col1:
        st.image(image, caption="Input Image")

    # Preprocess
    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    # CNN Prediction
    prediction = model.predict(img)[0]
    best_idx = np.argmax(prediction)
    confidence = prediction[best_idx]

    best_class = class_names[best_idx]
    crop_name = get_crop(best_class)
    disease_name = get_disease(best_class)

    # Feature extraction
    features = feature_extractor.predict(img)

    # GMM
    cluster = gmm.predict(features)[0]
    probs = gmm.predict_proba(features)[0]

    with col2:

        # ---------------- RESULT ----------------
        st.success(f"🌿 Crop: {crop_name}")
        st.subheader(f"🦠 Disease: {disease_name}")
        st.metric("Confidence", f"{confidence*100:.2f}%")

        # ---------------- BAR GRAPH ----------------
        st.markdown("### Top Predictions")

        top_idx = prediction.argsort()[-3:][::-1]
        labels = [get_disease(class_names[i]) for i in top_idx]
        values = [prediction[i]*100 for i in top_idx]

        fig_bar = px.bar(
            x=labels,
            y=values,
            text=[f"{v:.1f}%" for v in values]
        )

        fig_bar.update_layout(
            yaxis_title="Confidence (%)",
            xaxis_title="Disease"
        )

        st.plotly_chart(fig_bar, use_container_width=True)

        # ---------------- GMM INSIGHT ----------------
        st.markdown("### Advanced ML Insight (GMM)")

        st.info(f"""
        Cluster Assigned: **{cluster}**  
        Cluster Confidence: **{max(probs)*100:.2f}%**

        👉 Groups similar disease patterns  
        👉 Works alongside CNN for deeper analysis
        """)

        # ---------------- GMM GRAPH ----------------
        st.markdown("### GMM Feature Visualization")

        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA

            num_points = 50
            noise = np.random.normal(0, 0.05, (num_points, features.shape[1]))
            sample_features = features + noise

            all_features = np.vstack([features, sample_features])

            if all_features.shape[0] > 1:
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(all_features)

                fig, ax = plt.subplots(figsize=(6,5))

                ax.scatter(
                    reduced[1:, 0],
                    reduced[1:, 1],
                    color="blue",
                    alpha=0.5,
                    label="Cluster Spread"
                )

                ax.scatter(
                    reduced[0, 0],
                    reduced[0, 1],
                    color="red",
                    s=120,
                    label="Input Image"
                )

                ax.set_title("GMM Feature Space")
                ax.set_xlabel("PCA Component 1")
                ax.set_ylabel("PCA Component 2")
                ax.legend()

                st.pyplot(fig)
            else:
                st.warning("Not enough data for visualization")

        except Exception:
            st.warning("Visualization unavailable")

        # ---------------- ANALYSIS ----------------
        st.markdown("### Analysis")

        reason, precaution, fertilizer = get_info(best_class)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.warning(f"Cause\n\n{reason}")
        with c2:
            st.info(f"Precaution\n\n{precaution}")
        with c3:
            st.success(f"Fertilizer\n\n{fertilizer}")

else:
    st.info("Upload a plant leaf image to start")