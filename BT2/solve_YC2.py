import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import cv2
import matplotlib.pyplot as plt

def extract_color_histogram_32(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = []
    h = cv2.calcHist([image], [0, 1, 2], None, (32, 32, 32), [0, 256, 0, 256, 0, 256])
    hist.extend(h.flatten())
    return np.array(hist)

def extract_histogram_16(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = []
    h = cv2.calcHist([image], [0, 1, 2], None, (16, 16, 16), [0, 256, 0, 256, 0, 256])
    hist.extend(h.flatten())
    return np.array(hist)

def load_data(data_dir, feature_extractor):
    X = []
    y = []
    image_paths = []
    class_names = ['Cuc', 'Dao', 'Lan', 'Mai', 'Tho']
    
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for image_name in os.listdir(class_dir):
            if image_name.endswith('.jpg'):
                image_path = os.path.join(class_dir, image_name)
                features = feature_extractor(image_path)
                X.append(features)
                y.append(label)
                image_paths.append(image_path)
    
    return np.array(X), np.array(y), image_paths

# Load test data for both methods
X_train_32, y_train, _= load_data('HoaVietNam/train', extract_color_histogram_32)
X_test_32, y_test, image_paths = load_data('HoaVietNam/test', extract_color_histogram_32)
X_train_16, _, _= load_data('HoaVietNam/train', extract_histogram_16)
X_test_16, _, _= load_data('HoaVietNam/test', extract_histogram_16)

# Train and predict using 32-bin histogram
knn_32 = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='braycurtis')
knn_32.fit(X_train_32, y_train)
y_pred_32 = knn_32.predict(X_test_32)

# Train and predict using 16-bin histogram
knn_16 = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='braycurtis')
knn_16.fit(X_train_16, y_train)
y_pred_16 = knn_16.predict(X_test_16)

# Identify misclassified images
misclassified_32_correct_16 = []
correct_32_misclassified_16 = []
misclassified_both = []

for i, (true_label, pred_32, pred_16) in enumerate(zip(y_test, y_pred_32, y_pred_16)):
    if pred_32 != true_label and pred_16 == true_label:
        misclassified_32_correct_16.append(image_paths[i])
    elif pred_32 == true_label and pred_16 != true_label:
        correct_32_misclassified_16.append(image_paths[i])
    elif pred_32 != true_label and pred_16 != true_label:
        misclassified_both.append(image_paths[i])

# Function to display images
def show_images(image_list, title):
    if not image_list:
        print(f"No images found for: {title}")
        return
    fig, axes = plt.subplots(1, min(5, len(image_list)), figsize=(15, 5))
    if len(image_list) == 1:
        axes = [axes]
    for ax, img_path in zip(axes, image_list[:5]):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(os.path.basename(img_path))
    plt.suptitle(title)
    plt.show()

# Display images for each category
show_images(misclassified_32_correct_16, "Misclassified by 32-bin but Correct by 16-bin")
show_images(correct_32_misclassified_16, "Correct by 32-bin but Misclassified by 16-bin")
show_images(misclassified_both, "Misclassified by both 32-bin and 16-bin")
