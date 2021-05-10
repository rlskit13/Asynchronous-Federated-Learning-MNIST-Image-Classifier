import torch
import os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import torch.optim as optim

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np

Directory = "C:\\Users\\z5278080\\Fritz_FL_demo\\MNIST\\Models\\Server\\Completed_Model"

# NN architecture using PyTorch
# Less parameters
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def CallModel(idx):
	files = os.listdir(Directory)
	models = []
	Error = []
	for file in files:
		models.append(file)
		model_detail = file.split("_")
		Error.append(model_detail[0])
	Model = Net()
	Model.load_state_dict(torch.load(os.path.join(Directory, models[idx])))
	error = Error[idx]
	return Model, error


def LowestLocalError():
	files = os.listdir(Directory)
	Error = []
	models = []
	for file in files:
		models.append(file)
		model_detail = file.split("_")
		Error.append(model_detail[0])
	Lowest_error_model = models[np.argmin(Error)]
	Model = Net()
	Model.load_state_dict(torch.load(os.path.join(Directory, Lowest_error_model)))
	error = Error[np.argmin(Error)]
	return Model, error

def test(complete_model):
	complete_model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			output = complete_model(data)
			test_loss += F.nll_loss(output, target, size_average=False).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(test_loader.dataset)
	test_losses.append(test_loss)
	print('\nTest set: Avg. loss: {}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
	return test_losses



	

	
test_loader = torch.utils.data.DataLoader(
  							torchvision.datasets.MNIST('/files/', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             	])),
  							batch_size=10000, shuffle=True)

test_losses = []




#complete_model = Net()
#model_name = "Completed_model.pth"
#model_path = os.path.join(Directory, "Completed_model")
model_1, error = LowestLocalError()
model_2, error = CallModel(1)
#complete = complete_model.state_dict()


#for key in complete:
#	print(complete[key])
	
	

test(model_1)
test(model_2)
#for epoch in range(1, n_epochs + 1):
#	train(epoch)
#	test()
#fig