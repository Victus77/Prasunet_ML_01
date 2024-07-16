import cv2
import numpy as np
import joblib
import os

# Function to preprocess the image
def preprocess_image(img_path, img_size=(64, 64)):
    if not os.path.exists(img_path):
        raise ValueError(f"Image at path {img_path} does not exist.")
    
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, img_size)
        img = img.flatten()
        return img
    else:
        raise ValueError(f"Image at path {img_path} could not be loaded.")

# Path to the saved model
model_path = 'cat_dog_svm_model.pkl'

# Load the saved model
svm_model = joblib.load(model_path)
print("Model loaded successfully.")

# Path to the new image you want to classify
new_image_path = r'PetImages/360_F_97589769_t45CqXyzjz0KXwoBZT9PRaWGHRk5hQqQ.jpg'  # Ensure the correct file name and extension

# Preprocess the new image
new_image = preprocess_image(new_image_path)

# Reshape the image to match the training data
new_image = new_image.reshape(1, -1)

# Make a prediction
prediction = svm_model.predict(new_image)

# Map the prediction to the corresponding label
labels = {1: 'Cat', 0: 'Dog'}
predicted_label = labels[prediction[0]]
print(f'The predicted label for the image is: {predicted_label}')
