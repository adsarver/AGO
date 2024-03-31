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
    img = resize(img, (320, 240))
    img = img.flatten()
    curLabel = file[:-8]
    return curLabel, img

mlp = MLPClassifier(max_iter=100, verbose=True)

parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
    'shuffle': [True, False]
}

from sklearn.model_selection import GridSearchCV
input_dir = os.path.join(os.getcwd(), 'dataset', 'exterior')
data = []
labels = []
files = os.listdir(input_dir)

with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    futures = []
    i = 0
    for file in files[0:20]:
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

x_train, x_test, y_train, y_test = train_test_split(data, labels)

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(data, labels)
# Best parameter set
print('Best parameters found:\n', clf.best_params_)