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
input_dir = os.path.join(os.getcwd(), 'dataset', 'picturesTest')
categories = ['Make', 'Model', 'Year']

data = np.matrix()
for category_idx, category in enumerate(categories): # for every category, iterate over the files in the directory
    print(f'Category {category}')
    for file in os.listdir(input_dir)[0:9]:
        img_path = os.path.join(input_dir, file)
        img = imread(img_path)
        img = resize(img, (30, 30))
        data.append(img.flatten())
        curLabel = file.split('_')
        data.put(curLabel, img.flatten())
    

print('Imported')
print('Data: ', data.shape)
print('Converted')
# train / test split
x_train, x_test, y_train, y_test = train_test_split(data, test_size=0.2, shuffle=True, stratify=labels)
print('Split')
# train classifier
# classifier = SVC()

# parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
print('Training')
MLP = MLPClassifier(verbose=True, alpha=1e-5, random_state=1)
MLP.fit(x_train, y_train)
print('Trained')

# test performance
y_prediction = MLP.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(MLP, open('./car_model.p', 'wb'))