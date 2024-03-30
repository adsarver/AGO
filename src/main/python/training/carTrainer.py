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

# prepare data
input_dir = os.path.join(os.getcwd(), 'dataset', 'pictures')
categories = ['Make', 'Model', 'Year']
data = []
labels = []
for file in os.listdir(input_dir)[0:9]:
    img_path = os.path.join(input_dir, file)
    img = imread(img_path)
    img = resize(img, (30, 30))
    data.append(img)
    curLabel = file[:-8]
    labels.append(curLabel)

# format data
data = np.array([image.flatten() for image in data])
labels = np.array(labels)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# classifier
MLP = MLPClassifier(verbose=True, alpha=1e-5, random_state=1)
MLP.fit(x_train, y_train)

# test performance
y_prediction = MLP.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(MLP, open('./car_model.p', 'wb'))