# utils.py

import os
import cv2
import numpy as np

dataset_path = "Raw Data/Fruit And Vegetable Diseases Dataset"
image_size = (224, 224)

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(label)
    return images, labels

def prepare_data(dataset_path):
    all_images = []
    all_labels = []
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if "Healthy" in category:
            label = 0
        else:
            label = 1
        images, labels = load_images_from_folder(category_path, label)
        all_images.extend(images)
        all_labels.extend(labels)
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    return all_images, all_labels