import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# NEW IMPORTS (for GMM)
from sklearn.mixture import GaussianMixture
import numpy as np
import pickle

# ---------------- SETTINGS ----------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 10

DATASET_PATH = "dataset/PlantVillage"

# ---------------- DATA GENERATOR ----------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8,1.2]
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

NUM_CLASSES = train_data.num_classes

# ---------------- BASE MODEL ----------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze all layers (Stage 1)
for layer in base_model.layers:
    layer.trainable = False

# ---------------- CUSTOM HEAD ----------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ---------------- COMPILE ----------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------- TRAIN (STAGE 1) ----------------
print("Training Stage 1 (Frozen Base Model)...")

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_STAGE1
)

# ---------------- FINE-TUNE ----------------
print("Fine-tuning...")

for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------- TRAIN (STAGE 2) ----------------
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_STAGE2
)

# ---------------- SAVE MODEL ----------------
model.save("model/plant_disease_model.h5")

print("CNN Training Complete & Model Saved!")

# ============================================================
# 🔥 NEW PART: FEATURE EXTRACTION + GMM (AML INTEGRATION)
# ============================================================

print("\nExtracting features for GMM...")

# Feature extractor model (remove last classification layer)
feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)

features = []
labels = []

# Reset generator to start
train_data.reset()

# Extract features batch by batch
for i in range(len(train_data)):
    x_batch, y_batch = train_data[i]
    
    feat = feature_model.predict(x_batch)
    features.append(feat)
    
    labels.append(np.argmax(y_batch, axis=1))

# Convert to numpy
features = np.vstack(features)
labels = np.hstack(labels)

print("Feature shape:", features.shape)

# ---------------- APPLY GMM ----------------
print("Applying GMM clustering...")

gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(features)

# Save GMM model
with open("model/gmm_model.pkl", "wb") as f:
    pickle.dump(gmm, f)

print("GMM model saved!")

# ---------------- OPTIONAL: Show cluster distribution ----------------
clusters = gmm.predict(features)

unique, counts = np.unique(clusters, return_counts=True)
print("\nCluster Distribution:")
for u, c in zip(unique, counts):
    print(f"Cluster {u}: {c} samples")