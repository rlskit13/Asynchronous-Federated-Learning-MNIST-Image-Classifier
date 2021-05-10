import socket
import pickle
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import os
import numpy as np
from pathlib import Path


Directory = "C:\\Users\\z5278080\\Fritz_FL_demo\\MNIST\\Models\\Client_1"

#torch.manual_seed(446)
#np.random.seed(446)


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

def train(epoch, model):
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*1000) + ((epoch-1)*len(train_loader.dataset)))

            #torch.save(network.state_dict(), '/results/model.pth')
            #torch.save(optimizer.state_dict(), '/results/optimizer.pth')

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            #print(data)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return test_loss



def recv(soc, buffer_size=4096, recv_timeout=100):
    received_data = b""
    while str(received_data)[-2] != '.':
        try:
            soc.settimeout(recv_timeout)
            received_data += soc.recv(buffer_size)
        except socket.timeout:
            print("A socket.timeout exception occurred because the server did not send any data for {recv_timeout} seconds. There may be an error or the model may be trained successfully.".format(recv_timeout=recv_timeout))
            return None, 0
        except BaseException as e:
            return None, 0
            print("An error occurred while receiving data from the server {msg}.".format(msg=e))

    try:
        received_data = pickle.loads(received_data)
    except BaseException as e:
        print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
        return None, 0

    return received_data, 1

soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
print("Socket Created.\n")

try:
    soc.connect(("localhost", 10000))
    print("Successful Connection to the Server.\n")
except BaseException as e:
    print("Error Connecting to the Server: {msg}".format(msg=e))
    soc.close()
    print("Socket Closed.")

subject = "echo"
NN_instance = None
Client_ID = "Client_1"
test_loss = 10

n_epochs = 1
train_losses = []
train_counter = []
test_losses = []

while True:
    data_client = {"ID": Client_ID, "subject": subject, "data": NN_instance, "test_loss": test_loss}
    data_byte = pickle.dumps(data_client)
    
    print("Sending the Model to the Server.\n")
    soc.sendall(data_byte)

    
    print("Receiving Reply from the Server.")
    received_data, status = recv(soc=soc, 
                                 buffer_size=100000, 
                                 recv_timeout=100)
    if status == 0:
        print("Nothing Received from the Server.")
        break
    else:
        print(received_data, end="\n\n")

    subject = received_data["subject"]
    if subject == "model":
        ID = received_data["ID"]
        NN_instance = received_data["data"]
        time_struct = time.gmtime()
        model_name = "{ID}_{day}_{month}_{year}_{hour}_{minute}_{second}.pth".format(ID=ID, year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
        model_path = os.path.join(Directory, "Global models")
        Path(model_path).mkdir(parents=True, exist_ok=True)
        torch.save(NN_instance.state_dict(), os.path.join(model_path, model_name))
        print("Received model saved in {path}".format(path=model_path))
    
    elif subject == "done":
        print("The server said the model is trained successfully and no need for further updates its parameters.")
        subject = "end"
        NN_instance = None

   

    elif subject == "end":

        print("Received completed model from server.")
        ID = received_data["ID"]
        Completed_NN_instance = received_data["data"]
        time_struct = time.gmtime()
        #model_name = "{ID}_{day}_{month}_{year}_{hour}_{minute}_{second}.pth".format(ID=ID, year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
        model_name = "Completed_model"
        model_path = os.path.join(Directory, model_name)
        Path(model_path).mkdir(parents=True, exist_ok=True)
        torch.save(Completed_NN_instance.state_dict(), os.path.join(model_path, model_name))
        print("Completed model saved in {path}".format(path=model_path))
        break
    else:
        print("Unrecognized message type.")
        break


    subject = "model"




    
    batch_size_train = 1000
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    #random_seed = 1
    #torch.backends.cudnn.enabled = False
    #torch.manual_seed(random_seed)

    train_loader = torch.utils.data.DataLoader(
                            torchvision.datasets.MNIST('/files/', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                                ])),
                            batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                            torchvision.datasets.MNIST('/files/', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                                ])),
                            batch_size=batch_size_test, shuffle=True)




    
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)


    optimizer = optim.SGD(NN_instance.parameters(), lr=learning_rate,
                      momentum=momentum)
    
    
    
    if ID == f"{Client_ID}_initial":
        
        test_loss = test(NN_instance)
        train(n_epochs, NN_instance)
        n_epochs += 1

        test_loss = test(NN_instance)

    else:
    
    #for epoch in range(1, n_epochs + 1):
        train(n_epochs, NN_instance)
        n_epochs += 1
        test_loss = test(NN_instance)

    time_struct = time.gmtime()
    model_name = "{ID}_{day}_{month}_{year}_{hour}_{minute}_{second}.pth".format(ID="Trained", year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
    model_path = os.path.join(Directory, "Trained local models")
    Path(model_path).mkdir(parents=True, exist_ok=True)
    torch.save(NN_instance.state_dict(), os.path.join(model_path, model_name))
    print("Trained local model saved in {path}".format(path=model_path))

#complete_model = Net()


#model_name = "Completed_model"
#model_path = os.path.join(Directory, model_name)
#complete_model.load_state_dict(torch.load(os.path.join(model_path, model_name)))
#print("Completed model loaded: {model_name}.".format(model_name=model_name))

#print(train_counter)
#print(test_counter)
#print(train_losses)
#print(test_losses)

plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.title('Client_1 Test loss.')
plt.show()

soc.close()
print("Socket Closed.\n")