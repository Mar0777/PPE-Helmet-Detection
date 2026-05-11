import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout
)

print("TensorFlow Version:", tf.__version__)

# =========================
# CONFIG
# =========================

IMG_SIZE = (160, 160)

BATCH_SIZE = 32

EPOCHS = 10

# =========================
# DATASET PATHS
# =========================

train_dir = "../dataset_split/train"

val_dir = "../dataset_split/val"

test_dir = "../dataset_split/test"

# =========================
# DATA GENERATORS
# =========================

train_datagen = ImageDataGenerator(
    rescale=1./255
)

val_test_datagen = ImageDataGenerator(
    rescale=1./255
)

# =========================
# LOAD DATASETS
# =========================

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# =========================
# DATASET INFO
# =========================

print("\nClass Indices:")

print(train_generator.class_indices)

images, labels = next(train_generator)

print("\nBatch Image Shape:", images.shape)

print("Batch Labels Shape:", labels.shape)

print("\nFirst 10 Labels:")

print(labels[:10])

# =========================
# CNN MODEL
# =========================

model = Sequential([

    Input(shape=(224, 224, 3)),

    # Block 1
    Conv2D(32, (3, 3), activation='relu'),

    MaxPooling2D(pool_size=(2, 2)),

    BatchNormalization(),

    # Block 2
    Conv2D(64, (3, 3), activation='relu'),

    MaxPooling2D(pool_size=(2, 2)),

    BatchNormalization(),

    # Block 3
    Conv2D(128, (3, 3), activation='relu'),

    MaxPooling2D(pool_size=(2, 2)),

    BatchNormalization(),

    # Fully Connected Layers
    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.5),

    Dense(64, activation='relu'),

    Dropout(0.3),

    # Output Layer
    Dense(1, activation='sigmoid')
])

# =========================
# MODEL SUMMARY
# =========================

model.summary()

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN MODEL
# =========================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)



# =========================
# SAVE MODEL
# =========================

model.save("../saved_models/cnn_scratch.keras")

print("\nCNN model saved successfully.")