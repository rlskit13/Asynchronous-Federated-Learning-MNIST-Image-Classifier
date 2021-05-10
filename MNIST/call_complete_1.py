import torch
import os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import torch.optim as optim

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np

Directory = "C:\\Users\\z5278080\\Fritz_FL_demo\\MNIST\\Models\\Client_1"

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
	print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
	return test_losses



	

	
test_loader = torch.utils.data.DataLoader(
  							torchvision.datasets.MNIST('/files/', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             	])),
  							batch_size=1000, shuffle=True)
	#batch_size = 128
	
	#train_set = ImgDataset(train_x, train_y, transform)
	#val_set = ImgDataset(val_x, val_y, transform)
	#train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	#val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


test_losses = []




complete_model = Net()
model_name = "Completed_model"
model_path = os.path.join(Directory, model_name)
complete_model.load_state_dict(torch.load(os.path.join(model_path, model_name)))

complete = complete_model.state_dict()

for key in complete:
	print(complete[key])
	
	

test(complete_model)
#for epoch in range(1, n_epochs + 1):
#	train(epoch)
#	test()
#fig