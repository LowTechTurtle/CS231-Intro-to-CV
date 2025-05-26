import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Thông số HOG
hog_params = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "feature_vector": True
}

# Đường dẫn thư mục dữ liệu
data_dir = "Dataset/MyPedestrian"
classes = ["Pedestrian", "NonPedestrian"]

X = []  # danh sách đặc trưng
y = []  # nhãn tương ứng

# Đọc và xử lý ảnh
for label, cls in enumerate(classes):
    cls_dir = os.path.join(data_dir, cls)
    for filename in os.listdir(cls_dir):
        filepath = os.path.join(cls_dir, filename)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 256))
        features = hog(image, **hog_params)
        X.append(features)
        y.append(label)  # 0: Pedestrian, 1: NonPedestrian

X = np.array(X)
y = np.array(y)

# Chia tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Mô hình 1: KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("Kết quả với KNeighborsClassifier:")
print(classification_report(y_test, y_pred_knn, target_names=classes))

# Mô hình 2: LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Kết quả với LogisticRegression:")
print(classification_report(y_test, y_pred_lr, target_names=classes))
