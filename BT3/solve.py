import os
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
import itertools

def extract_color_histogram(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = []
    h = cv2.calcHist([image], [0, 1, 2], None, (16, 16, 16), [0, 256, 0, 256, 0, 256])
    hist.extend(h.flatten())
    return np.array(hist)

def load_data(data_dir):
    X = []
    y = []
    labels = {'Cuc': 0, 'Dao': 1, 'Lan': 2, 'Mai': 3, 'Tho': 4}
    
    for flower in labels.keys():
        folder_path = os.path.join(data_dir, flower)
        for img_name in os.listdir(folder_path):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(folder_path, img_name)
                features = extract_color_histogram(img_path)
                X.append(features)
                y.append(labels[flower])
    
    return np.array(X), np.array(y)

train_dir = 'HoaVietNam/train'
test_dir = 'HoaVietNam/test'

train_data, train_labels = load_data(train_dir)
test_data, test_labels = load_data(test_dir)

param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
}

model = LogisticRegression()

scorer = make_scorer(f1_score, average='macro')
grid_search = GridSearchCV(model, param_grid, scoring=scorer)
grid_search.fit(train_data, train_labels)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
predictions = best_model.predict(test_data)
macro_f1 = f1_score(test_labels, predictions, average='macro')

print(f"Best Macro-F1 Score: {macro_f1:.4f} with Parameters: {best_params}")
