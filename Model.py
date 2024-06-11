import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc

# Define constants
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 50  # Increase the number of epochs
NUM_CLASSES = 7  # Number of emotion categories

# Learning rate schedule
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 40:
        lr *= 1e-2
    elif epoch > 30:
        lr *= 5e-2
    elif epoch > 20:
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
    rotation_range=30,  # Increase rotation range
    width_shift_range=0.2,  # Increase width shift range
    height_shift_range=0.2,  # Increase height shift range
    shear_range=0.3,  # Increase shear range
    zoom_range=0.3,  # Increase zoom range
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

# Model architecture with BatchNormalization and L2 regularization
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model with a learning rate scheduler and SGD optimizer
model.compile(optimizer=SGD(learning_rate=lr_schedule(0), momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
lr_scheduler = LearningRateScheduler(lr_schedule)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

# Train the model with learning rate scheduler and other callbacks
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=test_generator.samples // BATCH_SIZE,
    callbacks=[lr_scheduler, early_stopping, reduce_lr]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Plot training & validation accuracy and loss values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Confusion matrix
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=test_generator.class_indices.keys())
cmd.plot(cmap=plt.cm.Blues)
plt.show()

# Classification report
report = classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys())
print(report)

# Plot ROC curve and AUC for a binary classification (example, assuming binary case)
# This is usually more complex for multi-class, so skipped here for simplicity
# from sklearn.metrics import roc_curve, auc
# fpr, tpr, _ = roc_curve(y_true, predictions[:, 1])
# roc_auc = auc(fpr, tpr)
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()

# Visualization of some prediction samples
sample_images, sample_labels = next(test_generator)
sample_preds = model.predict(sample_images)
sample_preds_labels = np.argmax(sample_preds, axis=1)

plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(sample_images[i].reshape(IMG_SIZE[0], IMG_SIZE[1]), cmap='gray')
    plt.title(f"Pred: {sample_preds_labels[i]}, Actual: {np.argmax(sample_labels[i])}")
    plt.axis('off')
plt.show()