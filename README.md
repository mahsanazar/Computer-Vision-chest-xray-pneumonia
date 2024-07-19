# Chest X-Ray Pneumonia Detection

This project involves building a machine learning model to detect pneumonia from chest X-ray images. The process includes setting up Kaggle API credentials, downloading and preparing the dataset, constructing and training a Convolutional Neural Network (CNN) using TensorFlow, and evaluating the model's performance.

## Project Workflow

### 1. Set Up Kaggle API

Create a `kaggle.json` file with your Kaggle API credentials and configure your environment to use the Kaggle API.

### 2. Download and Unzip Dataset

Download the dataset from Kaggle and unzip it for use in the project.

### 3. Prepare the Data

Set up the directories for training, validation, and testing, and define `ImageDataGenerator` for data augmentation and preprocessing.

### 4. Build and Train the Model

Define and compile a Convolutional Neural Network (CNN) and train it using the prepared data.

### 5. Evaluate the Model

Evaluate the model's performance on the test set and visualize the results using confusion matrix and classification report.

## Dataset

- **URL**: [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **License**: Other

## Requirements

- Python packages: `json`, `os`, `numpy`, `matplotlib`, `tensorflow`, `sklearn`, `keras`, `kaggle`

## Notes

- Replace the placeholder Kaggle username and key with your own credentials in the `kaggle.json` file.
- Ensure that all paths and parameters are correctly set for your specific environment.

---

Feel free to copy and paste this README into your GitHub repository.
