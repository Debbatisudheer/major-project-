Facial Emotion Recognition Model Documentation
Overview

This document explains how to build, train, and evaluate a Convolutional Neural Network (CNN) for recognizing facial emotions. The model classifies images into one of seven emotion categories.
Requirements

    Python 3.x: Programming language.
    TensorFlow: Deep learning framework.
    NumPy: Numerical operations.
    Matplotlib: Plotting graphs.
    Scikit-learn: Machine learning tools.

Setup
Define Constants

Define the image size, batch size, number of epochs, and number of emotion categories.

    IMG_SIZE: Size to which all images will be resized. We use (48, 48) to keep the model lightweight and fast.
    BATCH_SIZE: Number of images processed together in one pass. We use 32 to balance between memory use and training speed.
    EPOCHS: Number of complete passes through the training dataset. We use 30 for adequate training.
    NUM_CLASSES: Number of emotion categories. Here, 7 represents emotions like happy, sad, etc.

python

IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 7

Learning Rate Schedule

A function to adjust the learning rate during training to improve performance.

    Learning Rate: Controls the adjustment of the model's weights.
    This function lowers the learning rate at specific epochs to fine-tune the model as it trains.

python

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 40:
        lr *= 1e-2
    elif epoch > 30:
        lr *= 5e-2
    elif epoch > 20:
        lr *= 1e-1
    return lr

Data Preparation
Extract ZIP File

Extract the dataset from a ZIP file to a specified directory.

    ZipFile: The dataset is compressed, so we need to extract it.
    This step prepares the data for further processing.

python

with zipfile.ZipFile("C:\\Users\\DEBBATI SUDHEER\\PycharmProjects\\coding\\archive(3).zip", 'r') as zip_ref:
    zip_ref.extractall("dataset")

Define Directories

Set the directories for training and testing data.

    TRAIN_DIR: Directory where the training images are stored.
    TEST_DIR: Directory where the testing images are stored.

python

TRAIN_DIR = 'dataset/train'
TEST_DIR = 'dataset/test'

Data Augmentation and Generators

Prepare data augmentation for training and rescaling for testing.

    Data Augmentation: Techniques to artificially expand the training dataset by creating modified versions of images. This helps improve the model's robustness and prevents overfitting.
    ImageDataGenerator: Generates batches of tensor image data with real-time data augmentation.

Train Data Augmentation

Applies various transformations to the training images.

    rescale: Normalizes pixel values to the range [0, 1].
    rotation_range: Randomly rotates images by up to 30 degrees.
    width_shift_range: Randomly shifts images horizontally by up to 20% of the width.
    height_shift_range: Randomly shifts images vertically by up to 20% of the height.
    shear_range: Randomly applies shear transformations.
    zoom_range: Randomly zooms in on images.
    horizontal_flip: Randomly flips images horizontally.
    fill_mode: Fills in missing pixels after transformations.

python

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

Test Data Preprocessing

Only rescale the images for testing.

    Rescale: Normalizes pixel values to the range [0, 1] for consistency with training images.

python

test_datagen = ImageDataGenerator(rescale=1./255)

Data Generators

Generate batches of augmented data for training and rescaled data for testing.

    flow_from_directory: Generates batches of images directly from the directories.

python

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

Model Architecture
Define Model

Build a CNN with multiple layers for feature extraction and classification.

    Sequential: Allows building a model layer-by-layer.
    Conv2D: Convolutional layer to extract features from images.
    BatchNormalization: Normalizes the outputs of previous layers to improve training.
    MaxPooling2D: Reduces the spatial dimensions of the feature maps.
    Dropout: Prevents overfitting by randomly setting a fraction of input units to 0 at each update during training.
    Flatten: Converts the 2D matrices to a 1D vector.
    Dense: Fully connected layer for classification.

python

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

Compile Model

Configure the model for training.

    SGD: Stochastic Gradient Descent optimizer with momentum.
    categorical_crossentropy: Loss function for multi-class classification.
    metrics: List of metrics to be evaluated during training and testing.

python

model.compile(optimizer=SGD(learning_rate=lr_schedule(0), momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

Training
Callbacks

Additional functionalities to improve training.

    LearningRateScheduler: Adjusts the learning rate during training.
    EarlyStopping: Stops training when a monitored metric has stopped improving.
    ReduceLROnPlateau: Reduces the learning rate when a metric has stopped improving.

python

lr_scheduler = LearningRateScheduler(lr_schedule)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

Train Model

Train the model using the training data and validate it using the validation data.

    steps_per_epoch: Number of steps (batches of samples) in one epoch.
    validation_steps: Number of steps (batches of samples) to validate.

python

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=test_generator.samples // BATCH_SIZE,
    callbacks=[lr_scheduler, early_stopping, reduce_lr]
)

Evaluation
Evaluate Model

Evaluate the model's performance on the test data.

    test_loss: Loss on the test data.
    test_accuracy: Accuracy on the test data.

python

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

Plot Accuracy and Loss

Visualize the training and validation accuracy and loss.

    history.history: Contains per-epoch loss and accuracy data.

python

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

Confusion Matrix

Visualize the model's predictions compared to actual labels.

    confusion_matrix: Computes a confusion matrix.
    ConfusionMatrixDisplay: Displays the confusion matrix.

python

predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=test_generator.class_indices.keys())
cmd.plot(cmap=plt.cm.Blues)
plt.show()

Classification Report

Detailed report showing the precision, recall, and F1-score for each class.

    classification_report: Generates a classification report.

python

report = classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys())
print(report)

Visualization of Sample Predictions

Visualize some prediction samples with their actual and predicted labels.

    next(test_generator): Fetches a batch of images and labels.
    model.predict: Predicts the class probabilities.

python

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
