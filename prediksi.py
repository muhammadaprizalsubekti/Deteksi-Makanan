import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image

# =====================================================
# LOAD MODEL
# =====================================================

MODEL_PATH = "model_makanan_indonesia.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model tidak ditemukan:\n{MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH
)

# =====================================================
# LOAD LABEL
# =====================================================

with open("labels.json", "r") as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())

# =====================================================
# GAMBAR TEST
# =====================================================

IMG_PATH = "test.jpg"

if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(
        f"Gambar tidak ditemukan:\n{IMG_PATH}"
    )

# =====================================================
# PREPROCESSING
# =====================================================

img = image.load_img(
    IMG_PATH,
    target_size=(224,224)
)

img_array = image.img_to_array(
    img
)

img_array = np.expand_dims(
    img_array,
    axis=0
)

img_array = img_array / 255.0

# =====================================================
# PREDIKSI
# =====================================================

pred = model.predict(
    img_array,
    verbose=0
)

kelas = np.argmax(pred)

confidence = float(
    np.max(pred) * 100
)

# =====================================================
# HASIL
# =====================================================

print("\n======================")
print(" HASIL DETEKSI")
print("======================")

print(
    f"Makanan : {class_names[kelas]}"
)

print(
    f"Confidence : {confidence:.2f}%"
)

if confidence < 60:
    print(
        "⚠️ Keyakinan model rendah"
    )
else:
    print(
        "✅ Makanan berhasil dikenali"
    )

# =====================================================
# TOP 3
# =====================================================

print("\nTOP 3 PREDIKSI")

top3 = np.argsort(
    pred[0]
)[-3:][::-1]

for i, idx in enumerate(top3):

    print(
        f"{i+1}. {class_names[idx]} "
        f"({pred[0][idx]*100:.2f}%)"
    )