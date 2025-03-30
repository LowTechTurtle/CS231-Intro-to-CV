import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import cv2

def extract_color_histogram(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = []
    h = cv2.calcHist([image], [0, 1, 2], None, (32, 32, 32), [0, 256, 0, 256, 0, 256])
    hist.extend(h.flatten())
    return np.array(hist)

def load_data(data_dir):
    X = []
    y = []
    class_names = ['Cuc', 'Dao', 'Lan', 'Mai', 'Tho']
    
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for image_name in os.listdir(class_dir):
            if image_name.endswith('.jpg'):
                image_path = os.path.join(class_dir, image_name)
                features = extract_color_histogram(image_path)
                X.append(features)
                y.append(label)
    
    return np.array(X), np.array(y), class_names

X_train, y_train, class_names = load_data('HoaVietNam/train')
X_test, y_test, _ = load_data('HoaVietNam/test')
weights_options = ['uniform', 'distance']
metric_options = ['braycurtis', 'canberra', 'correlation', 'cosine', 'euclidean', 'minkowski']
best_accuracy = 0
best_params = {'weights': None, 'metric': None}

for weights in weights_options:
    for metric in metric_options:
        knn = KNeighborsClassifier(n_neighbors=7, weights=weights, metric=metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params['weights'] = weights
            best_params['metric'] = metric
            
        print(f"weights: {weights}, metric: {metric}, accuracy: {accuracy}")

print("Best weight and metric option::")
print(f"weights: {best_params['weights']}")
print(f"metric: {best_params['metric']}")
print(f"Best accuracy: {best_accuracy}")
