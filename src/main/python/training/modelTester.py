import os
from pickle import load
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import joblib
import concurrent.futures

input_dir = os.path.join(os.getcwd(), 'dataset', 'exterior')
data = []
labels = []
# os.chdir(os.path.join(os.getcwd(), 'models'))
model = joblib.load('models/car_model.p')
files = os.listdir(input_dir)


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

step = 10000
ministep = 50
k = 0

with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    futures = []
    for i in range(0, len(os.listdir(input_dir)), 10000):
        for j in range(i, i+ministep):
            k += 1
            futures.append(executor.submit(getImgData, files[i]))
            print('Loading image', k)
            
    for future in concurrent.futures.as_completed(futures):
        temp1, temp2 = future.result()
        data.append(temp2)
        labels.append(temp1)


data = np.array(data)

preds = model.predict(data)

total = len(preds)
correct = 0

for i in range(len(preds)):
    print('Predicted:', preds[i], '\nActual:', labels[i])
    if preds[i] == labels[i]:
        correct += 1
    print('\n')
        
print(correct/total)