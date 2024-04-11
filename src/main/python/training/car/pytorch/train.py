import matplotlib
matplotlib.use("Agg")
from torchvision.io import read_image
from classifier import ResNet50 as ResNet
import concurrent.futures
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim import RAdam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
import os
import pandas as pd
from pathlib import Path

tempset = set()
torch.backends.cudnn.benchmark = True

class CustomImageDataset(Dataset):
	classes = 0
	labels = dict()
	def __init__(self, annotations_file, img_dir, transform=None):
		self.img_labels = pd.read_csv(annotations_file)
		self.img_dir = img_dir
		self.transform = transform
		for i in self.img_labels.iterrows():
			tempset.add(i[1][1])
		self.labels = dict(zip(tempset, range(len(tempset))))
		self.classes = len(tempset)
     
	def __len__(self):
		return len(self.img_labels)

	def __getitem__(self, idx):
		tup = tuple(self.img_labels.iloc[idx])
		imgloc = tup[2]
		labelidx = self.labels[tup[1]]
		img_path = os.path.join(self.img_dir, imgloc)
		image = read_image(img_path)
		image = self.transform(image)

		return image, labelidx

	def getLabel(self, idx):
		tup = tuple(self.img_labels.iloc[idx])
		return tup[1]

# MAX THREADS -- Change this to the number of workers you want to use for importing and loading the dataset
MAX_WORKERS = 12
# define training hyperparameters
INIT_LR = 1e-4
BATCH_SIZE = 64
EPOCHS = 50
DECAY = 1e-2
# define the train and val splits
TRAIN_SPLIT = 0.8
VAL_SPLIT = 1 - TRAIN_SPLIT
ANNOTATIONS = 'exterior_sml/$annotations.csv'

# set the device we will be using to train the model
device = torch.device("cuda")
# load the dataset
print("[INFO] loading the dataset...")

def getAnno(file: str):
    curLabel = []
    split = file.split('_')
    curLabel.append(split[2])
    curLabel.append(split[0])
    curLabel.append(split[1])
    curLabel = '_'.join(curLabel)
    return curLabel, file    

input_dir = os.path.join(os.getcwd(), 'exterior_sml', 'exterior')

labels = []
files = os.listdir(input_dir)

with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
	futures = []
	for file in files[:999]:
		futures.append(executor.submit(getAnno, file))
	for future in concurrent.futures.as_completed(futures):
		temp = future.result()
		labels.append((temp[0], temp[1]))


with open(ANNOTATIONS, 'w', newline='') as file:
	labels = np.array(labels)
	df = pd.DataFrame(labels)
	df.to_csv(file)
	print('[INFO] annotations saved to file')

data = CustomImageDataset(ANNOTATIONS, input_dir, 
	transform=transforms.Compose([
	transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

trainData = CustomImageDataset(ANNOTATIONS, input_dir, 
	transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((128, 128)),
	transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

classes = trainData.classes

# divide train and test data
print("[INFO] generating the train/test split...")
numTrainSamp = int(len(trainData) * 0.9)
numTestSamp = int(len(trainData) * 0.1)

while numTrainSamp + numTestSamp != len(trainData):
	numTrainSamp += 1
 
(data, testData) = random_split(trainData,
	[numTrainSamp, numTestSamp],
	generator=torch.Generator().manual_seed(42))

# calculate the train/validation split
print("[INFO] generating the train/validation split of", len(trainData), "items...")
numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
numValSamples = int(len(trainData) * VAL_SPLIT)

while numTrainSamples + numValSamples != len(trainData):
	numTrainSamples += 1

(trainData, valData) = random_split(trainData,
	[numTrainSamples, numValSamples],
	generator=torch.Generator().manual_seed(42))

print("[INFO] EPOCHS set to", EPOCHS)
print("[INFO] BATCH_SIZE set to", BATCH_SIZE)

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True,
	batch_size=BATCH_SIZE, num_workers=MAX_WORKERS, pin_memory=True)
valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE, num_workers=MAX_WORKERS, pin_memory=True)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE, num_workers=MAX_WORKERS, pin_memory=True)
# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset)
valSteps = len(valDataLoader.dataset) 

# initialize the LeNet model
print("[INFO] initializing the ResNet model...")
model = ResNet(classes).to(device)
model = nn.DataParallel(model).to(device)
# initialize our optimizer and loss function
opt = RAdam(model.parameters(), lr=INIT_LR, weight_decay=DECAY)
lossFn = nn.NLLLoss()

# initialize a dictionary to store training history
H = {
	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
}
# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()
# loop over our epochs
looptimes = []
for e in range(0, EPOCHS):
	loopTime = time.time()
	# set the model in training mode
	model.train()
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0
	# initialize the number of correct predictions in the training
	# and validation step
	trainCorrect = 0
	valCorrect = 0
	# loop over the training set
	for (x, y) in trainDataLoader:
		x = x.to(device)
		y = y.to(device)
		# perform a forward pass and calculate the training loss
		pred = model(x)
		m = nn.LogSoftmax(dim=1)
		loss = lossFn(m(pred), y)
		# zero out the gradients, perform the backpropagation step,
		# and update the weights
		opt.zero_grad()
		loss.backward()
		opt.step()
		# add the loss to the total training loss so far and
		# calculate the number of correct predictions
		totalTrainLoss += loss
		trainCorrect += (pred.argmax(1) == y).type(
			torch.float).sum().item()
  
  # switch off autograd for evaluation
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()
		# loop over the validation set
		for (x, y) in valDataLoader:
			# send the input to the device
			(x, y) = (x.to(device), y.to(device))
			# make the predictions and calculate the validation loss
			pred = model(x)
			m = nn.LogSoftmax(dim=1)
			loss = lossFn(m(pred), y)
			totalValLoss += loss
			# calculate the number of correct predictions
			valCorrect += (pred.argmax(1) == y).type(
				torch.float).sum().item()
   
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps
	# calculate the training and validation accuracy
	trainCorrect = trainCorrect / trainSteps
	valCorrect = valCorrect / valSteps
  
	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["train_acc"].append(trainCorrect)
	H["val_loss"].append(avgValLoss.cpu().detach().numpy())
	H["val_acc"].append(valCorrect)
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
	print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
		avgTrainLoss, trainCorrect))
	print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
		avgValLoss, valCorrect))
	print("Loop time: {:.2f}s".format(time.time() - loopTime))
	print("Total time: {:.2f}s".format(time.time() - startTime))
	looptimes.append(time.time() - loopTime)

	print("Estimated time remaining: {:.2f}s".format((sum(looptimes) / len(looptimes)) * (EPOCHS - e)))
# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))
# we can now evaluate the network on the test set
print("[INFO] evaluating network...")
# turn off autograd for testing evaluation
with torch.no_grad():
	# set the model in evaluation mode
	model.eval()
	
	# initialize a list to store our predictions
	preds = []
	testCorrect = 0
	# loop over the test set
	for (x, y) in testDataLoader:
		# send the input to the device
		x = x.to(device)
		y = y.to(device)
  
		# make the predictions and add them to the list
		pred = model(x)
		predicted = torch.argmax(pred.data, 1)
		testCorrect += (predicted == y).sum().item()
  
# generate a classification report
print("[INFO] Accuracy on {:d} classes: {:.2f}%".format(classes, (testCorrect / len(testDataLoader.dataset)) * 100))

modelFile = Path(os.getcwd()).parents[0]
modelFile = os.path.join(modelFile, 'models', 'car_model_p1', 'car_model.pt')
plotFile = Path(os.getcwd()).parents[0]
plotFile = os.path.join(plotFile, 'models', 'car_model_p1', 'plot.png')
i = 1

while os.path.exists(Path(modelFile).parents[0]):
	i += 1 
	modelFile = os.path.join(Path(os.getcwd()).parents[0], 'models', 'car_model_p{:d}'.format(i), 'car_model.pt')
	plotFile = os.path.join(Path(os.getcwd()).parents[0], 'models', 'car_model_p{:d}'.format(i), 'plot.png')

if not os.path.exists(Path(modelFile).parents[1]): os.mkdir(Path(modelFile).parents[1])
os.mkdir(Path(modelFile).parents[0])

plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(plotFile)
# serialize the model to disk
torch.save(model, modelFile)