import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset,Subset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch.optim as optim

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np



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

def train_val_dataset(dataset, val_split=0.5):
    idx = []
    for i in range(10):
        for x in range(len(dataset.targets)):
            if dataset.targets[x] == i:
                idx.append(x)
    
    
    train_idx, val_idx = train_test_split(idx, test_size = val_split, shuffle=False)
    split_datasets = {}
    #split_datasets = Subset(dataset, idx)


    split_datasets['train_1'] = Subset(dataset, train_idx)
    split_datasets['train_2'] = Subset(dataset, val_idx)
    return split_datasets

def train(epoch):
	network.train()
	batch_idxs = []
	for batch_idx, (data, target) in enumerate(train_loader):
		batch_idxs.append(batch_idx)
		#print(target)
		optimizer.zero_grad()
		output = network(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
			train_losses.append(loss.item())
			train_counter.append((batch_idx*1000) + ((epoch-1)*len(train_loader.dataset)))
			#torch.save(network.state_dict(), '/results/model.pth')
			#torch.save(optimizer.state_dict(), '/results/optimizer.pth')
			#print(train_counter)

def test():
	network.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			output = network(data)
			test_loss += F.nll_loss(output, target, size_average=False).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(test_loader.dataset)
	test_losses.append(test_loss)
	print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


n_epochs = 20
batch_size_train = 1000
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10



random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)



dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                                ]))

datasets = train_val_dataset(dataset)

train_loader = torch.utils.data.DataLoader(
                            datasets['train_1'],
                            #datasets,
                            batch_size=batch_size_train, shuffle=False)



#for x in range(60000):
#	print((datasets[x][1]))


#print(len(datasets['train_1']))
#for x in range(6):
#	print((datasets['train_2'][x][1]))




test_loader = torch.utils.data.DataLoader(
  							torchvision.datasets.MNIST('/files/', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             	])),
  							batch_size=batch_size_test, shuffle=True)
	#batch_size = 128
	
	#train_set = ImgDataset(train_x, train_y, transform)
	#val_set = ImgDataset(val_x, val_y, transform)
	#train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	#val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
	
	
	

test()
for epoch in range(1, n_epochs + 1):
	train(epoch)
	test()
print(test_counter)
	


fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()


#examples = enumerate(train_loader)
#batch_idx, (example_data, example_targets) = next(examples)

#fig = plt.figure()
#for i in range(6):
#	plt.subplot(2,3,i+1)
#	plt.tight_layout()
#	plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#	plt.title("Ground Truth: {}".format(example_targets[i]))
#	plt.xticks([])
#	plt.yticks([])
#plt.show()

#fig