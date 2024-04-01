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

# MAX THREADS -- Change this to the number of workers you want to use
MAX_WORKERS = 16

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

# prepare data
input_dir = os.path.join(os.getcwd(), 'carscraper', 'dataset', 'exterior')

data = []
labels = []
files = os.listdir(input_dir)
x_test = []
y_test = []
i = 0

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = []
    for file in files[:100]:
        futures.append(executor.submit(getImgData, file))
        i += 1
        if i % 100 == 0 or i == len(files):
            print('Loading', i,'/',len(files), 'images')
    i = 0
    for future in concurrent.futures.as_completed(futures):
        temp1, temp2 = future.result()
        data.append(temp2)
        labels.append(temp1)
        i += 1
        if i % 100 == 0 or i == len(files):
            print('Preprocessed', i,'/',len(files), 'images')

# format data
data = np.array(data)
labels = np.array(labels)
print('Data converted')

unique, indx, counts = np.unique(labels, return_counts=True, return_index=True)


# remove unique cars
print('Removing unique cars')
labels = np.delete(labels, indx[np.where(counts == 1)])
data = np.delete(data, indx[np.where(counts == 1)], axis=0)
print('Unique cars removed')
print('Data shape:', data.shape)

test = []
base = []
start = 0
step = 5021
MLP = MLPClassifier(alpha=1e-5, warm_start=True, shuffle=True, verbose=True)

for i in range(start, len(labels), step):
    print('Splitting images from', i, 'to', i+step,'images')
    # train / test split
    x_train, x_test, y_train, y_test = train_test_split(data[i:i+step], labels[i:i+step], shuffle=True, test_size=0.1, stratify=labels[i:i+step])
    print('Data split\nStarting training on', len(x_train), 'images,', len(x_test), 'images in test set')

    # classifier
    print('Training...')
    np.concatenate((test, x_test))
    np.concatenate((base, y_test))
    MLP.fit(x_train, y_train)


print('Training done')


# test performance
y_prediction = MLP.predict(test)

score = accuracy_score(y_prediction, base)

print('{}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(MLP, open('./models/car_model.p', 'wb'))