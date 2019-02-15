from  cs231n.data_utils import *
from cs231n.algorithms import NearestNeighbour
import numpy as np

Xtr, Ytr, Xte, Yte = load_CIFAR10('/home/mujahid/Documents/my_projects/repo_pythons/cnn/cs231n/datasets/cifar-10-batches-py')

Xtr_rows = Xtr.reshape(Xtr.shape[0],32*32*3)
Xte_rows = Xte.reshape(Xte.shape[0],32*32*3)


nn = NearestNeighbour()

nn.train(Xtr_rows,Ytr)

Yte_predict = nn.predict(Xte_rows)
print('accuracy : %f' % (np.mean(Yte_predict==Yte)))








