import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

# Original CNN Model architecture
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the CNN model with a learning rate scheduler
cnn_model.compile(optimizer=Adam(learning_rate=lr_schedule(0)),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Define learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

# Train the CNN model with learning rate scheduler callback
cnn_history = cnn_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=test_generator.samples // BATCH_SIZE,
    callbacks=[lr_scheduler])

# Define the ResNet model
resnet_model = Sequential([
    ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the ResNet model
resnet_model.compile(optimizer=Adam(learning_rate=lr_schedule(0)),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Train the ResNet model
resnet_history = resnet_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=test_generator.samples // BATCH_SIZE,
    callbacks=[lr_scheduler])

# Evaluate both models
cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(test_generator)
resnet_test_loss, resnet_test_accuracy = resnet_model.evaluate(test_generator)
print(f'CNN Test Loss: {cnn_test_loss}, CNN Test Accuracy: {cnn_test_accuracy}')
print(f'ResNet Test Loss: {resnet_test_loss}, ResNet Test Accuracy: {resnet_test_accuracy}')

# Get predictions from both models
cnn_predictions = cnn_model.predict(test_generator)
resnet_predictions = resnet_model.predict(test_generator)

# Average the predictions
ensemble_predictions = (cnn_predictions + resnet_predictions) / 2
ensemble_pred_labels = np.argmax(ensemble_predictions, axis=1)

# Get true labels
y_true = test_generator.classes

# Evaluate the ensemble model
ensemble_accuracy = np.mean(ensemble_pred_labels == y_true)
print(f'Ensemble Test Accuracy: {ensemble_accuracy}')

# Plot Confusion Matrix for ensemble model
cm = confusion_matrix(y_true, ensemble_pred_labels)
cmd = ConfusionMatrixDisplay(cm, display_labels=test_generator.class_indices.keys())
cmd.plot(cmap=plt.cm.Blues)
plt.show()

# Classification Report for ensemble model
report = classification_report(y_true, ensemble_pred_labels, target_names=test_generator.class_indices.keys())
print(report)