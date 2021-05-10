import socket
import pickle
import threading
import time

import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import torch.optim as optim

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path


#torch.manual_seed(446)
#np.random.seed(446)

Directory = "C:\\Users\\z5278080\\Fritz_FL_demo\\MNIST\\Models\\Server"

#n_epochs = 3
#batch_size_train = 64
#batch_size_test = 1000
#learning_rate = 0.01
#momentum = 0.5
#log_interval = 10

#random_seed = 1
#torch.backends.cudnn.enabled = False
#torch.manual_seed(random_seed)

# Simple data set




# NN architecture using PyTorch
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
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(2)), 2))
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

class SocketThread(threading.Thread):

    def __init__(self, connection, client_info, buffer_size=10000, recv_timeout=5):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout

    def recv(self):
        received_data = b""
        while True:
            try:
                
                data = self.connection.recv(self.buffer_size)
                received_data += data

                if data == b'': # Nothing received from the client.
                    received_data = b""
                    # If still nothing received for a number of seconds specified by the recv_timeout attribute, return with status 0 to close the connection.
                    if (time.time() - self.recv_start_time) > self.recv_timeout:
                        return None, 0 # 0 means the connection is no longer active and it should be closed.

                elif str(data)[-2] == '.':
                    print("All data ({data_len} bytes) Received from {client_info}.".format(client_info=self.client_info, data_len=len(received_data)))

                    if len(received_data) > 0:
                        try:
                            # Decoding the data (bytes).
                            received_data = pickle.loads(received_data)
                            # Returning the decoded data.
                            return received_data, 1

                        except BaseException as e:
                            print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                            return None, 0

                else:
                    # In case data are received from the client, update the recv_start_time to the current time to reset the timeout counter.
                    self.recv_start_time = time.time()

            except BaseException as e:
                print("Error Receiving Data from the Client: {msg}.\n".format(msg=e))
                return None, 0

    def model_averaging(self, globalmodel, localmodel):
        sdG = globalmodel.state_dict()
        sdL = localmodel.state_dict()
        
        for key in sdG:
            sdG[key] = (sdG[key] + sdL[key] ) / 2.
        globalmodel.load_state_dict(sdG)
        print("Model aggregated.")

        return globalmodel

    def w_model_averaging(self, globalmodel, localmodel, weight_g, weight_l):
        sdG = globalmodel.state_dict()
        sdL = localmodel.state_dict()
        
        for key in sdG:
            sdG[key] = ((weight_g * sdG[key]) + (weight_l * sdL[key])) / 2.
        globalmodel.load_state_dict(sdG)
        print("Model aggregated.")

        return globalmodel

    def newest(self, path):
        files = os.listdir(path)
        paths = [os.path.join(path, basename) for basename in files]
        return max(paths, key=os.path.getctime)



    def reply(self, received_data):
        global NN_instance, X, y
        if (type(received_data) is dict):
            if (("data" in received_data.keys()) and ("subject" in received_data.keys())):
                Client_ID = received_data["ID"]
                subject = received_data["subject"]
                model = received_data["data"]
                error = received_data["test_loss"]
                

                print("Replying to the Client.")
                if subject == "echo":
                    model_path = os.path.join(Directory, "Global models")
                    if len(os.listdir(model_path)) == 0:
                        print("Global model folder is empty.")
                        if model is None:
                            NN_instance = Net() 
                            ID = f"{Client_ID}"           
                            data = {"ID": ID, "subject": "model", "data": NN_instance}
                            time_struct = time.gmtime()
                            model_path = os.path.join(Directory, "Global models")
                            model_name = "{ID}_initial_{day}_{month}_{year}_{hour}_{minute}_{second}.pth".format(ID=ID, year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
                            Path(model_path).mkdir(parents=True, exist_ok=True)
                            torch.save(NN_instance.state_dict(), os.path.join(model_path, model_name))
                            print("Model saved in {path}".format(path=model_path))
                    
                    else:
                        print("Global model is available.")
                        NN_instance = Net()
                        LatestGlobalNN = self.newest(model_path)
                        print(LatestGlobalNN)
                        NN_instance.load_state_dict(torch.load(LatestGlobalNN))
                        ID = f"{Client_ID}"

                        data = {"ID": ID, "subject": "model", "data": NN_instance}
                        time_struct = time.gmtime()
                        model_path = os.path.join(Directory, "Global models")
                        model_name = "{ID}_initial_{day}_{month}_{year}_{hour}_{minute}_{second}.pth".format(ID=ID, year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
                        Path(model_path).mkdir(parents=True, exist_ok=True)
                        torch.save(NN_instance.state_dict(), os.path.join(model_path, model_name))
                        print("Model saved in {path}".format(path=model_path))
                        

                        # In case a client sent a model to the server despite that the model error is 0.0. In this case, no need to make changes in the model.
                        if error <= 0.05:
                            data = {"ID": Client_ID, "subject": "done", "data": NN_instance}
                            print("The client asked for the model but it was already trained successfully. There is no need to send the model to the client for retraining.")
                        else:
                            data = {"ID": Client_ID, "subject": "model", "data": NN_instance}

                    try:
                        response = pickle.dumps(data)
                    except BaseException as e:
                        print("Error Encoding the Message: {msg}.\n".format(msg=e))
                elif subject == "model":
                    try:
                        model = received_data["data"]
                        Client_ID = received_data["ID"]
                        error = received_data["test_loss"]
                        #time_struct = time.gmtime()
                        #model_path = os.path.join(Directory, Client_ID)
                        #model_name = "{ID}_{day}_{month}_{year}_{hour}_{minute}_{second}.pth".format(ID=Client_ID, year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
                        #Path(model_path).mkdir(parents=True, exist_ok=True)
                        #torch.save(model.state_dict(), os.path.join(model_path, model_name))
                        #print("Client models saved in {path}".format(path=model_path))
                        #print("Client's Message Subject is {model}.".format(model=model.state_dict()))
                        if model is None:
                            NN_instance = Net()
                            model = NN_instance
                            #SDM = model.state_dict()
                            #NN = NN_instance.state_dict()

                            #for key in SDM:
                            #   print(SDM[key])
                            #   print("\nlocal model\n")
                            #   print(NN[key])
                            #   break
                        else:
                            print(model)

                            self.model_averaging(NN_instance, model)
                            #self.w_model_averaging(NN_instance, model, 0.2, 0.8)
                            test_loss = float(received_data["test_loss"])

                            print("Test loss = {test_loss}.".format(test_loss=float(test_loss)))

                            if test_loss >= 0.05:
                                time_struct = time.gmtime()
                                model_path = os.path.join(Directory, "Global models")
                                model_name = "{ID}_{day}_{month}_{year}_{hour}_{minute}_{second}.pth".format(ID="Global model", year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
                                Path(model_path).mkdir(parents=True, exist_ok=True)
                                torch.save(model.state_dict(), os.path.join(model_path, model_name))
                                print("Model saved in {path}".format(path=model_path))

                                data = {"ID": "Server", "subject": "model", "data": NN_instance}
                                response = pickle.dumps(data)

                            #print("Server's Message Subject is {model}.".format(model=NN_instance.state_dict()))
                            else:
                                time_struct = time.gmtime()
                                model_path = os.path.join(Directory, "Completed_Model")
                                #model_name = "{ID}_{day}_{month}_{year}_{hour}_{minute}_{second}.pth".format(ID="Completed_Model", year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
                                model_name = "{Error}_Completed_model_{day}_{month}_{year}_{hour}_{minute}_{second}.pth".format(Error=error, ID="Global model", year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
                                Path(model_path).mkdir(parents=True, exist_ok=True)
                                torch.save(model.state_dict(), os.path.join(model_path, model_name))
                                print("Completed model saved in {path}".format(path=model_path))
                                data = {"ID": None,"subject": "done", "data": None}
                                print("\n*****The Model is Trained Successfully*****\n\n")
                            
                            

        
                                

                            #complete_model = Net()

                            #complete_model.load_state_dict(torch.load(os.path.join(model_path, model_name)))
                            #print("Completed model loaded: {model_name}.".format(model_name = model_name))

                            

                            #plt.plot(n_epochs, train_losses, color='blue')
                            #plt.scatter(test_counter, test_losses, color='red')
                            #plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
                            #plt.xlabel('Epochs')
                            #plt.ylabel('negative log likelihood loss')
                            #plt.show()

                                soc.close()
                                print("Socket Closed.\n")
                            #complete_model = Net()
                            #model_name = "Completed_model_{day}_{month}_{year}_{hour}_{minute}_{second}.pth".format(ID="Global model", year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
                            #model_path = os.path.join(Directory, "Completed_Model")
                            #complete_model.load_state_dict(torch.load(os.path.join(model_path, model_name)))
                            #data = {"ID": "Completed_Model","subject": "end", "data": NN_instance}
                            #response = pickle.dumps(data)
                            #print("Sent completed model to all clients.")

                    except BaseException as e:
                        print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))



                else:
                    response = pickle.dumps("Response from the Server")
                            
                try:
                    self.connection.sendall(response)
                except BaseException as e:
                    print("Error Sending Data to the Client: {msg}.\n".format(msg=e))

            else:
                print("The received dictionary from the client must have the 'subject' and 'data' keys available. The existing keys are {d_keys}.".format(d_keys=received_data.keys()))
        else:
            print("A dictionary is expected to be received from the client but {d_type} received.".format(d_type=type(received_data)))

    def run(self):
        print("Running a Thread for the Connection with {client_info}.".format(client_info=self.client_info))

        # This while loop allows the server to wait for the client to send data more than once within the same connection.
        
        while True:
            self.recv_start_time = time.time()
            time_struct = time.gmtime()
            date_time = "\nWaiting to Receive Data from {client_info} Starting from {day}/{month}/{year} {hour}_{minute}_{second} GMT".format(year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec, client_info=self.client_info)
            print(date_time)
            received_data, status = self.recv()
            if status == 0:
                self.connection.close()
                print("\nConnection Closed with {client_info} either due to inactivity for {recv_timeout} seconds or due to an error.".format(client_info=self.client_info, recv_timeout=self.recv_timeout), end="\n\n")
                break

            # print(received_data)
            self.reply(received_data)

soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
print("Socket Created.\n")

# Timeout after which the socket will be closed.
# soc.settimeout(5)

soc.bind(("localhost", 10000))
print("Socket Bound to IPv4 Address & Port Number.\n")

soc.listen(1)
print("Socket is Listening for Connections ....\n")

all_data = b""
while True:
    try:
        connection, client_info = soc.accept()
        print("\nNew Connection from {client_info}.".format(client_info=client_info))
        socket_thread = SocketThread(connection=connection,
                                     client_info=client_info, 
                                     buffer_size=100000,
                                     recv_timeout=100)
        socket_thread.start()
    except:
        soc.close()
        print("(Timeout) Socket Closed Because no Connections Received.\n")
        break

#plt.plot(train_counter, train_losses, color='blue')
#plt.scatter(test_counter, test_losses, color='red')
#plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
#plt.xlabel('number of training examples seen')
#plt.ylabel('negative log likelihood loss')
#plt.show()

soc.close()
print("Socket Closed.\n")
