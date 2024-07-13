# Fruit Ripeness Prediction

## Overview

This project aims to develop a machine learning model to predict the ripeness of various fruits and vegetables using image data. The dataset contains images categorized as either healthy or rotten for 14 different types of fruits and vegetables.

## Dataset

The Fruit and Vegetable Diseases Dataset consists of 28 directories, each representing a combination of healthy and rotten images for 14 different fruits and vegetables. The images are compiled from various reputable sources, including Kaggle and GitHub repositories.

- Number of Classes: 28 (Healthy and Rotten categories for 14 different fruits and vegetables)
- Image Format: JPEG/PNG
- Image Size: Varies (recommended to resize to a standard size for consistent training)

The dataset is available at Kaggle.

## Current Progress

We have implemented and tested various deep learning models to classify the images. Our initial models faced overfitting issues, which we addressed by incorporating regularization techniques and data augmentation. We also implemented ensemble methods to improve the model's performance and generalization.

### Implemented Models

- Convolutional Neural Network (CNN) with Batch Normalization and Dropout
- Data Augmentation with ImageDataGenerator
- Early Stopping to prevent overfitting
- Class Weights to handle class imbalance
- Random Forest
- Multi-Layer Perceptron (MLP)

### Latest Model Performance

- **Stacking Test Accuracy**: 89.59%
- **Confusion Matrix**:
[[1214  148]
[ 157 1411]]

- **Classification Report**:
            precision    recall  f1-score   support

   Healthy       0.89      0.89      0.89      1362
   Rotten        0.91      0.90      0.90      1568

accuracy                           0.90      2930
macro avg       0.90      0.90      0.90      2930
weighted avg       0.90      0.90      0.90      2930

## How to Run the Project
 ```
 Clone the repository
 git clone https://github.com/JustJin28/Fruit-Ripeness-Prediction.git
 
Navigate to the project directory
cd Fruit-Ripeness-Prediction

install the required dependencies
pip install -r requirements.txt

Download the raw dataset from kaggle: 
https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten

Run the flas app
python app.py

Access the web interface:
Open your browser and go to http://127.0.0.1:5000/. Use the button to open the camera app and take a photo of your fruit for prediction. 
```

Future Work

	•	Further optimization of ensemble methods to enhance model performance.
	•	Exploring other model architectures and hyperparameters to improve accuracy and generalization.
	•	Fine-tuning the data augmentation techniques for better training data variability.

Contributing

Feel free to fork the repository, make improvements, and submit a pull request. Your contributions are welcome!

License

This project is licensed under the MIT License.

