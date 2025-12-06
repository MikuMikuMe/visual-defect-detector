Creating a visual defect detector using machine learning involves several steps, including image data acquisition and preprocessing, model selection and training, and evaluation. Below is a complete Python program that lays out a structure for such a project using a convolutional neural network (CNN) with TensorFlow and Keras. This program includes comments and error handling for clarity.

```python
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Set the random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths to the dataset
train_data_dir = 'path/to/train/dataset'
val_data_dir = 'path/to/validation/dataset'

# Parameters
batch_size = 32
img_height = 128
img_width = 128
num_classes = 2  # Defective and non-defective
epochs = 30

def load_data(data_dir):
    """
    Loads the images and labels from the specified directory.
    
    Parameters:
    data_dir (str): The path to the data directory.

    Returns:
    tuple: Tuple containing arrays of images and labels.
    """
    try:
        images = []
        labels = []

        categories = os.listdir(data_dir)
        for label, category in enumerate(categories):
            category_path = os.path.join(data_dir, category)
            for img_filename in os.listdir(category_path):
                img_path = os.path.join(category_path, img_filename)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                labels.append(label)

        images = np.array(images)
        labels = np.array(labels)

        return images, labels

    except Exception as e:
        print(f"An error occurred while loading data: {e}")

# Load data
try:
    train_images, train_labels = load_data(train_data_dir)
    val_images, val_labels = load_data(val_data_dir)
except Exception as e:
    print(f"Could not load data: {e}")

# Normalize pixel values
try:
    train_images = train_images / 255.0
    val_images = val_images / 255.0
except Exception as e:
    print(f"An error occurred during normalization: {e}")

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()

# Model architecture
def build_model():
    """
    Builds the CNN model.

    Returns:
    keras.Model: The created model.
    """
    try:
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
            MaxPooling2D(2, 2),
            Dropout(0.2),

            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.2),

            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.2),

            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    except Exception as e:
        print(f"An error occurred in model creation: {e}")

# Model training
try:
    model = build_model()

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        train_datagen.flow(train_images, train_labels, batch_size=batch_size),
        validation_data=val_datagen.flow(val_images, val_labels),
        epochs=epochs,
        callbacks=[early_stopping]
    )

except Exception as e:
    print(f"An error occurred during model training: {e}")

# Save model
try:
    model.save('visual_defect_detector_model.h5')
    print("Model saved successfully.")
except Exception as e:
    print(f"An error occurred while saving the model: {e}")

# Evaluate model
try:
    loss, accuracy = model.evaluate(val_images, val_labels)
    print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")
except Exception as e:
    print(f"An error occurred during model evaluation: {e}")
```

### Description:
- **Data Loading**: Images and labels are loaded from specified directories. Exception handling ensures that errors in loading data are reported.
- **Data Preparation**: Images are normalized and augmented for training.
- **Model Construction**: A CNN model is built using Keras. The model includes convolutional, pooling, and dropout layers.
- **Training and Validation**: The model is trained using data augmentation, and early stopping is used to avoid overfitting.
- **Model Saving**: The trained model is saved for future use.
- **Evaluation**: The model is evaluated on validation data, and the results are printed.

Ensure that you replace the `'path/to/train/dataset'` and `'path/to/validation/dataset'` with actual paths where your labeled datasets are stored.