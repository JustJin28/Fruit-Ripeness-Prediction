# Fruit Ripeness Prediction

## Overview
This project aims to develop a machine learning model to predict the ripeness of various fruits and vegetables using image data. The dataset contains images categorized as either healthy or rotten for 14 different types of fruits and vegetables.

## Dataset
The Fruit and Vegetable Diseases Dataset consists of 28 directories, each representing a combination of healthy and rotten images for 14 different fruits and vegetables. The images are compiled from various reputable sources, including Kaggle and GitHub repositories.

- **Number of Classes:** 28 (Healthy and Rotten categories for 14 different fruits and vegetables)
- **Image Format:** JPEG/PNG
- **Image Size:** Varies (recommended to resize to a standard size for consistent training)

The dataset is available at [Kaggle](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten).

## Current Progress
We have implemented and tested various deep learning models to classify the images. Our initial models faced overfitting issues, which we are addressing by incorporating regularization techniques and data augmentation. We are also exploring ensemble methods to improve the model's performance and generalization.

### Implemented Models
- **Convolutional Neural Network (CNN) with Batch Normalization and Dropout**
- **Data Augmentation with ImageDataGenerator**
- **Early Stopping to prevent overfitting**
- **Class Weights to handle class imbalance**

### Latest Model Performance
- **Test Accuracy:** 79.36%
- **Confusion Matrix and Classification Report:
Confusion Matrix:
[[650 712]
 [699 869]]

## Ensemble Methods
We are currently implementing ensemble methods to further enhance the model's performance:
- **Averaging Ensemble:** Combining predictions from multiple CNN models.
- **Stacking Ensemble:** Using a meta-model to make final predictions based on the outputs of multiple base models.

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/JustJin28/Fruit-Ripeness-Prediction.git
   
   Navigate to the project directory
   cd Fruit-Ripeness-Prediction
   
   Install the required dependencies
   pip install -r requirements.txt
   
   Download the dataset from Kaggle and place it in the Raw Data directory.
   
   Future Work

    Implementing ensemble methods to enhance model performance.
    Exploring other model architectures and hyperparameters to improve accuracy and generalization.
    Fine-tuning the data augmentation techniques for better training data variability.

Contributing

Feel free to fork the repository, make improvements, and submit a pull request. Your contributions are welcome!

License

This project is licensed under the MIT License.
   

