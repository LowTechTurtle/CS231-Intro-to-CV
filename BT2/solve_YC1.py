import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import itertools

def extract_color_histogram(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = []
    h = cv2.calcHist([image], [0, 1, 2], None, (32, 32, 32), [0, 256, 0, 256, 0, 256])
    hist.extend(h.flatten())
    return np.array(hist)

def extract_color_moments(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = []
    
    for channel in range(3):
        channel_data = image[:,:,channel].ravel()
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        skewness = np.mean((channel_data - mean) ** 3) / (std ** 3)
        features.extend([mean, std, skewness])
    return np.array(features)

def extract_dominant_color(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels.astype(np.float32), 1, None, criteria, 10, flags)
    dominant_color = palette[0]
    return dominant_color

# Hàm load dữ liệu
def load_data(data_dir, feature_extractor):
    X = []
    y = []
    labels = {'Cuc': 0, 'Dao': 1, 'Lan': 2, 'Mai': 3, 'Tho': 4}
    
    for flower in labels.keys():
        folder_path = os.path.join(data_dir, flower)
        for img_name in os.listdir(folder_path):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(folder_path, img_name)
                features = feature_extractor(img_path)
                X.append(features)
                y.append(labels[flower])
    
    return np.array(X), np.array(y)

# Tìm tham số tốt nhất với Histogram
def find_best_params_histogram(train_dir, test_dir):
    X_train, y_train = load_data(train_dir, extract_color_histogram)
    X_test, y_test = load_data(test_dir, extract_color_histogram)
    
    weights_options = ['uniform', 'distance']
    metric_options = ['braycurtis', 'canberra', 'correlation', 'cosine', 'euclidean', 'minkowski']
    
    best_accuracy = 0
    best_params = {'weights': None, 'metric': None}
    
    for weights, metric in itertools.product(weights_options, metric_options):
        knn = KNeighborsClassifier(n_neighbors=7, weights=weights, metric=metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params['weights'] = weights
            best_params['metric'] = metric
    
    return best_params, best_accuracy

# Đường dẫn đến thư mục dữ liệu
train_dir = 'HoaVietNam/train'
test_dir = 'HoaVietNam/test'

# 1. Tìm tham số tốt nhất với Histogram
best_params, hist_accuracy = find_best_params_histogram(train_dir, test_dir)
print("Histogram Results:")
print(f"Best parameters: weights={best_params['weights']}, metric={best_params['metric']}")
print(f"Accuracy: {hist_accuracy:.4f}\n")

# 2. Áp dụng cho Moment màu
X_train_moment, y_train_moment = load_data(train_dir, extract_color_moments)
X_test_moment, y_test_moment = load_data(test_dir, extract_color_moments)

knn_moment = KNeighborsClassifier(n_neighbors=7, 
                                weights=best_params['weights'], 
                                metric=best_params['metric'])
knn_moment.fit(X_train_moment, y_train_moment)
y_pred_moment = knn_moment.predict(X_test_moment)
moment_accuracy = accuracy_score(y_test_moment, y_pred_moment)

print("Color Moments Results:")
print(f"Using weights={best_params['weights']}, metric={best_params['metric']}")
print(f"Accuracy: {moment_accuracy:.4f}\n")

# 3. Áp dụng cho Màu chủ đạo
X_train_dominant, y_train_dominant = load_data(train_dir, extract_dominant_color)
X_test_dominant, y_test_dominant = load_data(test_dir, extract_dominant_color)

knn_dominant = KNeighborsClassifier(n_neighbors=7, 
                                  weights=best_params['weights'], 
                                  metric=best_params['metric'])
knn_dominant.fit(X_train_dominant, y_train_dominant)
y_pred_dominant = knn_dominant.predict(X_test_dominant)
dominant_accuracy = accuracy_score(y_test_dominant, y_pred_dominant)

print("Dominant Color Results:")
print(f"Using weights={best_params['weights']}, metric={best_params['metric']}")
print(f"Accuracy: {dominant_accuracy:.4f}")
