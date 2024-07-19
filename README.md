# chest-xray-pneumonia
# Chest X-Ray Pneumonia Detection

This project involves building a machine learning model to detect pneumonia from chest X-ray images. The process includes setting up Kaggle API credentials, downloading and preparing the dataset, constructing and training a Convolutional Neural Network (CNN) using TensorFlow, and evaluating the model's performance.

## Project Workflow

### 1. Set Up Kaggle API
          

Create a `kaggle.json` file with your Kaggle API credentials and configure your environment to use the Kaggle API.
           
"python
import json

kaggle_token = {
    "username": "your_kaggle_username",
    "key": "your_kaggle_api_key"
}

with open('kaggle.json', 'w') as file:
    json.dump(kaggle_token, file)
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!pip install kaggle"

###     2. Download and Unzip Dataset
Download the dataset from Kaggle and unzip it for use in the project.
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
!unzip chest-xray-pneumonia.zip -d /content/dataset


### 3. Prepare the Data
Set up the directories for training, validation, and testing, and define ImageDataGenerator for data augmentation and preprocessing.
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = "/content/dataset/chest_xray"
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    color_mode='grayscale'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    color_mode='grayscale'
)

### 4. Build and Train the Model
Define and compile a Convolutional Neural Network (CNN) and train it using the prepared data.
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=4,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)
### 5. Evaluate the Model
Evaluate the model's performance on the test set and visualize the results using confusion matrix and classification report.

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=False
)

predictions = model.predict(test_generator)
y_pred = np.round(predictions).flatten()
y_true = test_generator.classes

print("Classification Report:")
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Normal', 'Pneumonia'], rotation=45)
plt.yticks(tick_marks, ['Normal', 'Pneumonia'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


### Requirements
Python packages: json, os, numpy, matplotlib, tensorflow, sklearn, keras, kaggle
Notes
Replace the placeholder Kaggle username and key with your own credentials in the kaggle.json file.
Ensure that all paths and parameters are correctly set for your specific environment.
