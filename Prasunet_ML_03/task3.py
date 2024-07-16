import os
import sys
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module='cv2')

# Redirect stderr to null to suppress OpenCV warnings
class suppress_stderr(object):
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        self.save_fds = [os.dup(2), os.dup(1)]

    def __enter__(self):
        os.dup2(self.null_fds[0], 2)
        os.dup2(self.null_fds[1], 1)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 2)
        os.dup2(self.save_fds[1], 1)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

# Set the path to the dataset
cat_dog_path = r'C:\Users\yadav\OneDrive\Desktop\project for dsa\project python\PetImages'

# Function to load images and labels
def load_images_and_labels(path, img_size=(64, 64)):
    images = []
    labels = []
    for label, subdir in enumerate(['Cat', 'Dog']):
        subdir_path = os.path.join(path, subdir)
        for filename in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, filename)
            with suppress_stderr():
                img = cv2.imread(img_path)
            if img is not None:
                try:
                    img = cv2.resize(img, img_size)
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
    return np.array(images), np.array(labels)

# Load images and labels
print("Loading images and labels...")
images, labels = load_images_and_labels(cat_dog_path)
print(f"Loaded {len(images)} images and {len(labels)} labels.")

# Check if images and labels are loaded correctly
if len(images) == 0 or len(labels) == 0:
    print("No images or labels loaded. Check the dataset path and structure.")
    sys.exit(1)

# Flatten images
images_flattened = images.reshape(images.shape[0], -1)
print(f"Images flattened to shape: {images_flattened.shape}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images_flattened, labels, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")

# Create and train the SVM model
print("Training the SVM model...")
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Make predictions
print("Making predictions...")
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Save the model
joblib.dump(svm_model, 'cat_dog_svm_model.pkl')
print("Model saved as 'cat_dog_svm_model.pkl'")

# Load the model (when needed)
svm_model = joblib.load('cat_dog_svm_model.pkl')