import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model
model = tf.keras.models.load_model("model/plant_disease_model.h5")

# Data generator (same as training)
datagen = ImageDataGenerator(rescale=1./255)

val_data = datagen.flow_from_directory(
    "dataset/PlantVillage",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Predictions
y_true = val_data.classes
y_pred = model.predict(val_data)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
report = classification_report(y_true, y_pred_classes, output_dict=True)

df = pd.DataFrame(report).transpose()

# Plot
plt.figure(figsize=(8,5))
sns.heatmap(df.iloc[:-1, :-1], annot=True, cmap="Blues")

plt.title("Evaluation Metrics")
plt.savefig("report_assets/evaluation.png")

print("Evaluation graph saved!")