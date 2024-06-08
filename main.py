import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import cv2

# Define constants
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 7

# Learning rate schedule
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 20:
        lr *= 1e-2
    elif epoch > 15:
        lr *= 5e-2
    elif epoch > 10:
        lr *= 1e-1
    return lr

# Extract the ZIP file
with zipfile.ZipFile("C:\\Users\\DEBBATI SUDHEER\\PycharmProjects\\coding\\archive(3).zip", 'r') as zip_ref:
    zip_ref.extractall("dataset")

# Define directories for training and testing data
TRAIN_DIR = 'dataset/train'
TEST_DIR = 'dataset/test'

# Data preprocessing with more augmentation options
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical')

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model with a learning rate scheduler
model.compile(optimizer=Adam(learning_rate=lr_schedule(0)),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
lr_scheduler = LearningRateScheduler(lr_schedule)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

# Train the model with learning rate scheduler callback
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=test_generator.samples // BATCH_SIZE,
    callbacks=[lr_scheduler, early_stopping, reduce_lr])

# Save the model
model.save('emotion_model.h5')  # Save in HDF5 format
model.save('emotion_model')     # Save in TensorFlow SavedModel format

# Function to preprocess new images
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
    return img

# Function to make predictions
def predict_emotion(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    predicted_emotion = emotion_labels[class_idx]
    return predicted_emotion

# Example usage
img_path = 'PrivateTest_88305.jpg'
predicted_emotion = predict_emotion(img_path)
print(f'The predicted emotion is: {predicted_emotion}')

















