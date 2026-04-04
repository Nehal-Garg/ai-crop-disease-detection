import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import plotly.express as px
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Crop Doctor", layout="wide")

# ---------------- LOAD MODELS ----------------
model = tf.keras.models.load_model("model/plant_disease_model.h5")

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
    with open("static/background.jpeg", "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
    }}
    .block-container {{
        background: rgba(255,255,255,0.94);
        padding: 30px;
        border-radius: 15px;
        max-width: 1400px;
    }}
    h1,h2,h3,p,div {{ color:black !important; }}
    </style>
    """, unsafe_allow_html=True)

set_bg()

# ---------------- TITLE ----------------
st.title("AI Crop Disease Detection Dashboard")

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

# ---------------- HELPERS ----------------
def get_crop(name):
    return name.split("_")[0]

def get_disease(name):
    parts = name.split("_")
    return " ".join(parts[1:]) if len(parts) > 1 else name

def get_info(name):
    name = name.lower()

    if "healthy" in name:
        return ("Healthy plant","Maintain watering & sunlight","Use balanced fertilizer")

    if "blight" in name:
        return ("Fungal infection","Avoid excess moisture","Use fungicide")

    if "bacterial" in name:
        return ("Bacterial infection","Remove infected leaves","Use copper spray")

    if "virus" in name:
        return ("Viral infection","Remove infected plant","Control insects")

    return ("General stress","Check soil & water","Use fertilizer")

# ---------------- PROCESS ----------------
if file:
    image = Image.open(file)

    with col1:
        st.image(image, caption="Input Image", use_container_width=True)

    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    # ---------------- CNN ----------------
    prediction = model.predict(img)[0]
    confidence = np.max(prediction)
    best_idx = np.argmax(prediction)
    best_class = class_names[best_idx]
    crop_name = get_crop(best_class)

    # ---------------- FEATURE EXTRACTION ----------------
    features = feature_extractor.predict(img)

    # ---------------- GMM ----------------
    cluster = gmm.predict(features)[0]
    probs = gmm.predict_proba(features)[0]

    with col2:

        if confidence < 0.75:
            st.error("Upload a proper plant image")
        else:
            # ---------------- RESULT ----------------
            st.success(f"{crop_name}")
            st.metric("Confidence", f"{confidence*100:.2f}%")

            # ---------------- AML INSIGHT ----------------
            st.markdown("### AML Insight (GMM)")
            st.write(f"Cluster: {cluster}")
            st.write(f"Cluster Confidence: {max(probs)*100:.2f}%")

            # ---------------- BAR GRAPH ----------------
            st.markdown("### Crop Confidence")
            top_idx = prediction.argsort()[-5:][::-1]

            crops = [get_crop(class_names[i]) for i in top_idx]
            probs_plot = [prediction[i] for i in top_idx]

            fig_bar = px.bar(
                x=crops,
                y=probs_plot,
                color=crops,
                text=[f"{p*100:.1f}%" for p in probs_plot]
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ---------------- PIE CHART ----------------
            st.markdown("### Disease Distribution")
            top_idx = prediction.argsort()[-3:][::-1]

            diseases = [get_disease(class_names[i]) for i in top_idx]
            values = [prediction[i] for i in top_idx]

            fig_pie = px.pie(names=diseases, values=values)
            st.plotly_chart(fig_pie, use_container_width=True)

            # ---------------- GMM VISUALIZATION ----------------
            st.markdown("### GMM Cluster Visualization")

            num_points = 150
            noise = np.random.normal(0, 0.3, (num_points, features.shape[1]))
            sample_features = features + noise

            pca = PCA(n_components=2)
            reduced = pca.fit_transform(sample_features)

            clusters = gmm.predict(sample_features)

            fig, ax = plt.subplots(figsize=(6,5))
            colors = ['blue', 'orange', 'green']

            for i in np.unique(clusters):
                pts = reduced[clusters == i]

                ax.scatter(
                    pts[:, 0], pts[:, 1],
                    label=f'Cluster {i}',
                    alpha=0.7,
                    color=colors[i % len(colors)],
                    s=40
                )

                if len(pts) > 2:
                    cov = np.cov(pts.T)
                    mean = np.mean(pts, axis=0)

                    vals, vecs = np.linalg.eigh(cov)
                    angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))

                    width, height = 2 * np.sqrt(vals)

                    ellipse = Ellipse(
                        xy=mean,
                        width=width,
                        height=height,
                        angle=angle,
                        edgecolor=colors[i % len(colors)],
                        fc='none',
                        lw=2
                    )
                    ax.add_patch(ellipse)

            ax.set_title("GMM Clustering on CNN Features")
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            ax.legend()

            st.pyplot(fig)

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