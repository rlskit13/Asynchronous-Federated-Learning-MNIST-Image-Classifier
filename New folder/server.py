import socket
import pickle
import threading
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path


#torch.manual_seed(446)
#np.random.seed(446)

Directory = "C:\\Users\\z5278080\\Fritz_FL_demo\\New folder\\Models\\Server"

# Simple data set
d = 1
n = 200
X = torch.rand(n,d)
y = 4 * torch.sin(np.pi * X) * torch.cos(6*np.pi*X**2)


# NN architecture using PyTorch
n_hidden_1 = 32
n_hidden_2 = 32
d_out = 1
NN_instance = nn.Sequential(
                           nn.Linear(d, n_hidden_1), 
                            nn.Tanh(),
                            nn.Linear(n_hidden_1, n_hidden_2),
                            nn.Tanh(),
                            nn.Linear(n_hidden_2, d_out)
                            )

class SocketThread(threading.Thread):

    def __init__(self, connection, client_info, buffer_size=4096, recv_timeout=5):
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


    def reply(self, received_data):
        global NN_instance, X, y
        if (type(received_data) is dict):
            if (("data" in received_data.keys()) and ("subject" in received_data.keys())):
                ID = received_data["ID"]
                subject = received_data["subject"]
                model = received_data["data"]
                

                print("Replying to the Client.")
                if subject == "echo":
                    if model is None:
                        data = {"ID": "Initial", "subject": "model", "data": NN_instance}
                        time_struct = time.gmtime()
                        model_path = os.path.join(Directory, "Global models")
                        model_name = "{ID}_{day}_{month}_{year}_{hour}_{minute}_{second}.pth".format(ID="Initial", year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
                        Path(model_path).mkdir(parents=True, exist_ok=True)
                        torch.save(NN_instance.state_dict(), os.path.join(model_path, model_name))
                        print("Model saved in {path}".format(path=model_path))
                    else:
                        time_struct = time.gmtime()
                        model_path = os.path.join(Directory, "Global models")
                        model_name = "{ID}_{day}_{month}_{year}_{hour}_{minute}_{second}.pth".format(ID="Server", year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
                        Path(model_path).mkdir(parents=True, exist_ok=True)
                        torch.save(NN_instance.state_dict(), os.path.join(model_path, model_name))
                        print("Model saved in {path}".format(path=model_path))
                        
                        lossfunc = nn.MSELoss()
                        y_hat = model(X)                        
                        error = lossfunc(y_hat, y)
                        

                        # In case a client sent a model to the server despite that the model error is 0.0. In this case, no need to make changes in the model.
                        if error <= 0.05:
                            data = {"subject": "done", "data": NN_instance}
                            print("The client asked for the model but it was already trained successfully. There is no need to send the model to the client for retraining.")
                        else:
                            data = {"subject": "model", "data": NN_instance}

                    try:
                        response = pickle.dumps(data)
                    except BaseException as e:
                        print("Error Encoding the Message: {msg}.\n".format(msg=e))
                elif subject == "model":
                    try:
                        model = received_data["data"]
                        Client_ID = received_data["ID"]
                        time_struct = time.gmtime()
                        model_path = os.path.join(Directory, Client_ID)
                        model_name = "{ID}_{day}_{month}_{year}_{hour}_{minute}_{second}.pth".format(ID=Client_ID, year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
                        Path(model_path).mkdir(parents=True, exist_ok=True)
                        torch.save(NN_instance.state_dict(), os.path.join(model_path, model_name))
                        print("Client models saved in {path}".format(path=model_path))
                        #print("Client's Message Subject is {model}.".format(model=model.state_dict()))
                        if model is None:
                            model = NN_instance
                        else:
                            
                            #optim = torch.optim.SGD(NN_instance.parameters(), lr=step_size)
                            #print('iter,\tloss')
                            #for i in range(n_epochs):
                            #lossfunc = nn.MSELoss()
                            #y_hat = model(X)                        
                            #error = lossfunc(y_hat, y)
                                
    
                            # In case a client sent a model to the server despite that the model error is 0.0. In this case, no need to make changes in the model.
                            #if error <= 0.05:
                                
                            #    data = {"subject": "done", "data": None}
                            #    response = pickle.dumps(data)
                            #    print("The model is trained successfully and no need to send the model to the client for retraining.")
                                
                            #    return

                            self.model_averaging(NN_instance, model)
                            #print("Loss = {error}\n".format(error=error))
                            #X_grid = torch.from_numpy(np.linspace(0,1,50)).float().view(-1, d)
                            #y_hat = NN_instance(X_grid)
                            #plt.scatter(X.numpy(), y.numpy())
                            #plt.plot(X_grid.detach().numpy(), y_hat.detach().numpy(), 'r')
                            #plt.title('plot of $f(x)$ and $\hat{f}(x)$ Global')
                            #plt.xlabel('$x$')
                            #plt.ylabel('$y$')
                            #plt.show()
                            time_struct = time.gmtime()
                            model_path = os.path.join(Directory, "Global models")
                            model_name = "{ID}_{day}_{month}_{year}_{hour}_{minute}_{second}.pth".format(ID="Global model", year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
                            Path(model_path).mkdir(parents=True, exist_ok=True)
                            torch.save(NN_instance.state_dict(), os.path.join(model_path, model_name))
                            print("Model saved in {path}".format(path=model_path))

                        # print(best_model.trained_weights)
                        # print(model.trained_weights)
                        lossfunc = nn.MSELoss()
                        y_hat = NN_instance(X)                        
                        error = lossfunc(y_hat, y)

                        #predictions = pygad.nn.predict(last_layer=model, data_inputs=data_inputs)
                        print("Error: {error}".format(error=error))
                        #print("Weight: {parameters}".format(parameters=NN_instance.parameters()))

                        #error = numpy.sum(numpy.abs(predictions - data_outputs))
                        

                        if error >= 0.05:
                            data = {"ID": "Server", "subject": "model", "data": NN_instance}
                            response = pickle.dumps(data)
                            #print("Server's Message Subject is {model}.".format(model=NN_instance.state_dict()))
                        else:
                            time_struct = time.gmtime()
                            model_path = os.path.join(Directory, "Completed_Model")
                            #model_name = "{ID}_{day}_{month}_{year}_{hour}_{minute}_{second}.pth".format(ID="Completed_Model", year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
                            model_name = "Completed_model"
                            Path(model_path).mkdir(parents=True, exist_ok=True)
                            torch.save(NN_instance.state_dict(), os.path.join(model_path, model_name))
                            print("Completed model saved in {path}".format(path=model_path))
                            
                            data = {"ID": "Completed_Model","subject": "done", "data": NN_instance}
                            response = pickle.dumps(data)
                            print("\n*****The Model is Trained Successfully*****\n\n")
                            

                            complete_model = nn.Sequential(
                                                            nn.Linear(d, n_hidden_1), 
                                                            nn.Tanh(),
                                                            nn.Linear(n_hidden_1, n_hidden_2),
                                                            nn.Tanh(),
                                                            nn.Linear(n_hidden_2, d_out)
                            )

                            complete_model.load_state_dict(torch.load(os.path.join(model_path, model_name)))
                            print("Completed model loaded: {model_name}.".format(model_name = model_name))

                            lossfunc = nn.MSELoss()
                            y_hat = complete_model(X)                        
                            error = lossfunc(y_hat, y)
                            print("Completed model Error: {error}".format(error=error))

                            X_grid = torch.from_numpy(np.linspace(0,1,50)).float().view(-1, d)
                            y_hat = complete_model(X_grid)
                            plt.scatter(X.numpy(), y.numpy())
                            plt.plot(X_grid.detach().numpy(), y_hat.detach().numpy(), 'r')
                            plt.title('plot of $f(x)$ and $\hat{f}(x)$ Server')
                            plt.xlabel('$x$')
                            plt.ylabel('$y$')
                            plt.show()

                            soc.close()
                            print("Socket Closed.\n")

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
                                     buffer_size=4096,
                                     recv_timeout=100)
        socket_thread.start()
    except:
        soc.close()
        print("(Timeout) Socket Closed Because no Connections Received.\n")
        break

X_grid = torch.from_numpy(np.linspace(0,1,50)).float().view(-1, d)
y_hat = NN_instance(X_grid)
plt.scatter(X.numpy(), y.numpy())
plt.plot(X_grid.detach().numpy(), y_hat.detach().numpy(), 'r')
plt.title('plot of $f(x)$ and $\hat{f}(x)$ Server')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()
soc.close()
print("Socket Closed.\n")
