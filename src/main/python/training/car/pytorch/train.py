import matplotlib
matplotlib.use("Agg")
from torchvision.io import read_image
import concurrent.futures
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import torchvision.models as models
from torchvision.transforms import v2 as transforms
from torch.optim import RAdam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
import pandas as pd
from pathlib import Path
import weighter

class CustomImageDataset(Dataset):
	classes = 0
	labels = dict()
	dist = dict()
	def __init__(self, annotations_file, img_dir):
		self.img_labels = pd.read_csv(annotations_file)
		self.img_dir = img_dir
		tempset = set()
		for i in self.img_labels.iterrows():
			tempset.add(i[1][1])
			if i[1][1] in self.dist:
				self.dist[i[1][1]] += 1
			else:
				self.dist[i[1][1]] = 1
				
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

		return image, labelidx

	def getLabel(self, idx):
		tup = tuple(self.img_labels.iloc[idx])
		return tup[1]

	def getDistributions(self):
		return self.dist

class TrDataset(Dataset):
	classes = -1
	def __init__(self, base_dataset, transformations):
		super(TrDataset, self).__init__()
		self.base = base_dataset
		self.transformations = transformations

	def __len__(self):
		return len(self.base)

	def __getitem__(self, idx):
		x, y = self.base[idx]
		return self.transformations(x), y

	def getLabel(self, idx):
		x, y = self.base[idx]
		return y

	def getLabels(self):
		labels = []
		for x, y in self.base:
			labels.append(y)
		return labels
	
	def getClassCount(self):
		if self.classes == -1:
			tempset = set()
			for x, y in self.base:
				tempset.add(y)
			self.classes = len(tempset)

		return self.classes

# MAX WORKERS -- Change this to the number of workers you want to use for importing and loading the dataset
MAX_WORKERS = 6
# define training hyperparameters
INIT_LR = 1e-5
BATCH_SIZE = 4
EPOCHS = 400
DECAY = 1e-4
IMG_SIZE = 518
# define the train and val splits
TRAIN_SPLIT = 0.7
VAL_SPLIT = 1 - TRAIN_SPLIT
ANNOTATIONS = '$annotations.csv'

# set the device we will be using to train the model
device = torch.device("cuda")
# load the dataset
print("[INFO] loading the dataset...")

def getAnno(file: str):
	curLabel = []
	split = file.split('_')
	try:
		curLabel.append(split[2])
		curLabel.append(split[0])
		curLabel.append(split[1])
		curLabel = '_'.join(curLabel)
	except IndexError:
		print(split)
	return curLabel, file    

input_dir = os.path.join(os.getcwd(), 'exterior')

labels = []
files = os.listdir(input_dir)

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
	futures = []
	for file in files[:99]:
		if file.endswith('.jpg'):
			futures.append(executor.submit(getAnno, file))
	for future in concurrent.futures.as_completed(futures):
		temp = future.result()
		labels.append((temp[0], temp[1]))


with open(ANNOTATIONS, 'w', newline='') as file:
	labels = np.array(labels)
	df = pd.DataFrame(labels)
	df.to_csv(file)
	print('[INFO] Annotations saved to file')

data = CustomImageDataset(ANNOTATIONS, input_dir)
vaTransforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
	transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
trTransforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
	transforms.ToPILImage(),
    transforms.RandomApply(nn.ModuleList([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomVerticalFlip(),
		transforms.RandomAffine(degrees=(30, 70), translate=(0.0, 0.1), scale=(0.5, 0.7)),
		transforms.RandomPerspective(),
		transforms.ColorJitter(brightness=(0.5, 1), contrast=(0.5, 1), saturation=(0.5, 1), hue=(0, 0))
	])),
    transforms.ToTensor(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

classes = data.classes

# divide train and test data
print("[INFO] Generating the train/test split...")
numTrainSamp = int(len(data) * 0.9)
numTestSamp = int(len(data) * 0.1)

while numTrainSamp + numTestSamp != len(data):
	numTrainSamp += 1
 
print("[INFO] Test set size:", numTestSamp)

labels = data.getDistributions()

(data, testData) = random_split(data,
	[numTrainSamp, numTestSamp],
	generator=torch.Generator().manual_seed(42))

# calculate the train/validation split
print("[INFO] Generating the train/validation split")
numTrainSamples = int(len(data) * TRAIN_SPLIT)
numValSamples = int(len(data) * VAL_SPLIT)

while numTrainSamples + numValSamples != len(data):
	numTrainSamples += 1

print("[INFO] Validation set size:", numValSamples, "Train set size:", numTrainSamples)

(data, valData) = random_split(data,
	[numTrainSamples, numValSamples],
	generator=torch.Generator().manual_seed(42))

trainData = TrDataset(data, trTransforms)
valData = TrDataset(valData, vaTransforms)
testData = TrDataset(testData, vaTransforms)

# generating class weights for training dataset
print("[INFO] Generating class weights for training dataset...")
weights = weighter.make_weights(trainData, trainData.getClassCount(), BATCH_SIZE, MAX_WORKERS)
samplerTrain = WeightedRandomSampler(weights=weights.double(), num_samples=len(trainData))

# weights = weighter.make_weights(valData, valData.getClassCount(), BATCH_SIZE, MAX_WORKERS)
# samplerVal = WeightedRandomSampler(weights=weights, num_samples=len(valData))

# with open('ClassDistCurrent.csv', 'w', newline='') as file:
# 	df = pd.DataFrame([labels])
# 	df.to_csv(file, index=False)
# 	print('[INFO] Distributions saved to file')

print("[INFO] EPOCHS set to", EPOCHS)
print("[INFO] BATCH_SIZE set to", BATCH_SIZE)

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, batch_size=BATCH_SIZE, num_workers=MAX_WORKERS, pin_memory=True, sampler=samplerTrain)
valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE, num_workers=MAX_WORKERS, pin_memory=True)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE, num_workers=MAX_WORKERS, pin_memory=True)
# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset)
valSteps = len(valDataLoader.dataset) 

# initialize the LeNet model
print("[INFO] initializing the ImageNet model...")
# model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT).to(device)
model = model = torch.hub.load("facebookresearch/swag", model="vit_h14_in1k").to(device)
model = nn.DataParallel(model)

# initialize our optimizer and loss function
opt = RAdam(model.parameters(recurse=True), lr=INIT_LR, weight_decay=DECAY)
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
		(x, y) = (x.to(device), y.to(device))
		# perform a forward pass and calculate the training loss
		output = model(x)
		loss1 = lossFn(output, y)
		loss = loss1
		# zero out the gradients, perform the backpropagation step,
		# and update the weights
		opt.zero_grad()
		loss.backward()
		opt.step()
		# add the loss to the total training loss so far and
		# calculate the number of correct predictions
		totalTrainLoss += loss
		_, preds = torch.max(output, 1)
		trainCorrect += torch.sum(preds == y)
  
  # switch off autograd for evaluation
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()
		# loop over the validation set
		for (x, y) in valDataLoader:
			# send the input to the device
			(x, y) = (x.to(device), y.to(device))
			# make the predictions and calculate the validation loss
			output = model(x)
			loss = lossFn(output, y)
			_, preds = torch.max(output, 1)
			totalValLoss += loss
			# calculate the number of correct predictions
			valCorrect += torch.sum(preds == y)
   
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps
	# calculate the training and validation accuracy
	trainCorrect = trainCorrect / trainSteps
	valCorrect = valCorrect / valSteps
  
	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["train_acc"].append(trainCorrect.cpu().detach().numpy())
	H["val_loss"].append(avgValLoss.cpu().detach().numpy())
	H["val_acc"].append(valCorrect.cpu().detach().numpy())
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
	print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
		avgTrainLoss, trainCorrect))
	print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(
		avgValLoss, valCorrect))
	print("Loop time: {:.2f}s".format(time.time() - loopTime))
	print("Total time: {:.2f} minutes\n".format((time.time() - startTime) / 60))
	looptimes.append(time.time() - loopTime)

	print("Estimated time remaining: {:.2f} minutes\n".format(((sum(looptimes) / len(looptimes)) * (EPOCHS - e)) / 60))
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
		output = model(x)
		_, preds = torch.max(output, 1)
		# calculate the number of correct predictions
		testCorrect += torch.sum(preds == y.data)
  
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