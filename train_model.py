import os
import json
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "dataset_gambar", "train")
VALID_DIR = os.path.join(BASE_DIR, "dataset_gambar", "valid")

print("TRAIN DIR:", TRAIN_DIR)
print("VALID DIR:", VALID_DIR)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)

valid_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)

with open("labels.json", "w") as f:
    json.dump(train_generator.class_indices, f)

model = models.Sequential([
    layers.Input(shape=(224,224,3)),

    layers.Conv2D(32,(3,3),activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128,(3,3),activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(
        train_generator.num_classes,
        activation="softmax"
    )
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=20
)

model.save("model_makanan_indonesia.h5")

print("MODEL BERHASIL DISIMPAN")