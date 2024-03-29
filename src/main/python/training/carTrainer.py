from os.path import dirname, join as pjoin
import os
from sklearn.neural_network import MLPClassifier
import scipy.io as sio

data_dir = pjoin(os.getcwd(), 'dataset')
mat_fname = pjoin(data_dir, 'cars_annos.mat')

mat = sio.loadmat(mat_fname)
print(mat)

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

# clf.fit()