from sklearn.neural_network import MLPClassifier
import os
import pickle
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import concurrent.futures
import gc
import random
def getImgData(file: str):
    img_path = os.path.join(input_dir, file)
    img = imread(img_path)
    img = resize(img, (160, 90))
    img = img.flatten()
    curLabel = []
    split = file.split('_')
    curLabel.append(split[2])
    curLabel.append(split[0])
    curLabel.append(split[1])
    curLabel = '_'.join(curLabel)
    return curLabel, img   

mlp = MLPClassifier(verbose=True)

parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100, 100, 100), (125, 125, 125), (150, 150, 150), (200, 200, 200)],
    'activation': ['tanh', 'relu', 'logistic', 'identity'],
    'alpha': [0.0001, 0.05, 1e-5, 1e-3, 1e-4, 1e-6],
    'epsilon': [1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    'beta_1': [0.9, 0.8, 0.7, 0.6, 0.5],
    'beta_2': [0.999, 0.99, 0.9, 0.8, 0.7],
    'early_stopping': [True, False],
    'warm_start': [True, False],
    'learning_rate_init': [0.001, 0.01, 0.1, 0.0001, 0.00001],
    'n_iter_no_change': [10, 20, 30, 40, 50],
}

from sklearn.model_selection import GridSearchCV
input_dir = os.path.join(os.getcwd(), 'carscraper', 'dataset', 'exterior')
data = []
labels = []
files = os.listdir(input_dir)

with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    futures = []
    i = 0
    for file in files[0:1000]:
        futures.append(executor.submit(getImgData, file))
        i += 1
        if i % 5 == 0 or i == len(files):
            print('Loading', i,'/',len(files), 'images')
    i = 0
    for future in concurrent.futures.as_completed(futures):
        temp1, temp2 = future.result()
        data.append(temp2)
        labels.append(temp1)
        i += 1
        if i % 20 == 0:
            print('Preprocessed', i,'/',len(files), 'images')

data = np.array(data)
labels = np.array(labels)

# remove unique cars
unique, indx, counts = np.unique(labels, return_counts=True, return_index=True)

print('Removing unique cars')
labels = np.delete(labels, indx[np.where(counts == 1)])
data = np.delete(data, indx[np.where(counts == 1)], axis=0)
print('Unique cars removed')
print('Splitting')
x_train, x_test, y_train, y_test = train_test_split(data, labels, shuffle=True, test_size=0.1, stratify=labels)

print('Creating GSCV')
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, verbose=True)
clf.fit(data, labels)

# Best parameter set
y_prediction = clf.predict(x_test)
score = accuracy_score(y_prediction, y_test)

# Previous Best
# 63.0% of samples were correctly classified
# Best parameters found:
#  {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (125, 125, 125)}

print('{}% of samples were correctly classified'.format(str(score * 100)))
print('Best parameters found:\n', clf.best_params_)