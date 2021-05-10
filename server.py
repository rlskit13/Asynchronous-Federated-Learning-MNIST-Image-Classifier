import socket
import pickle
import time
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

import numpy as np

torch.manual_seed(446)
np.random.seed(446)

d = 2
#n = 50
#X = torch.randn(n,d)
#true_w = torch.tensor([[-1.0], [2.0]])
#y = X @ true_w + torch.randn(n,1) * 0.1

#step_size = 0.05
#n_epochs = 6000
n_hidden_1 = 32
n_hidden_2 = 32
d_out = 1
model = nn.Sequential(
                            nn.Linear(d, n_hidden_1), 
                            nn.Tanh(),
                            nn.Linear(n_hidden_1, n_hidden_2),
                            nn.Tanh(),
                            nn.Linear(n_hidden_2, d_out)
                            )

class SocketThread(threading.Thread):
    def __init__(self, connection, client_info, buffer_size=1024, recv_timeout=5):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout

    def run(self):

        # This while loop allows the server to wait for the client to send data more than once within the same connection.
        while True:
            self.recv_start_time = time.time()
            time_struct = time.gmtime()
            date_time = "\nWaiting to Receive Data from {client_info} Starting from {day}/{month}/{year} {hour}:{minute}:{second} GMT".format(year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec, client_info=self.client_info)
            print(date_time)
            received_data, status = self.recv()
            if status == 0:
                self.connection.close()
                print("\nConnection Closed with {client_info} either due to inactivity for {recv_timeout} seconds or due to an error.".format(client_info=self.client_info, recv_timeout=self.recv_timeout), end="\n\n")
                break

            # print(received_data)
            self.reply(received_data)


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

    def model_averaging(self, model1, model2):        

        sdA = model1.state_dict()
        sdB = model2.state_dict()
        for key in sdA:
            sdA[key] = (sdB[key] + sdA[key]) / 2.

        model = nn.Sequential(nn.Linear(d, n_hidden_1), 
                      nn.Tanh(),
                      nn.Linear(n_hidden_1, n_hidden_2),
                      nn.Tanh(),
                      nn.Linear(n_hidden_2, d_out)
                      )

        model.load_state_dict(sdA)

        return model

    def reply(self, received_data):
        global model
        if (type(received_data) is dict):
            if (("data" in received_data.keys()) and ("subject" in received_data.keys())):
                subject = received_data["subject"]
                print("Client's Message Subject is {subject}.".format(subject=subject))

                print("Replying to the Client.")
                if subject == "echo":
                    if model is None:
                        data = {"subject": "model", "data": model}
                    #else:
                        #predictions = pygad.nn.predict(last_layer=model, data_inputs=data_inputs)
                        #error = numpy.sum(numpy.abs(predictions - data_outputs))
                        # In case a client sent a model to the server despite that the model error is 0.0. In this case, no need to make changes in the model.
                        #if error == 0:
                        #    data = {"subject": "done", "data": None}
                        #    print("The client asked for the model but it was already trained successfully. There is no need to send the model to the client for retraining.")
                        #else:
                        #    data = {"subject": "model", "data": GANN_instance}

                    try:
                        response = pickle.dumps(data)
                    except BaseException as e:
                        print("Error Encoding the Message: {msg}.\n".format(msg=e))
                elif subject == "model":
                    try:
                        model = received_data["data"]
                        #best_model_idx = received_data["best_solution_idx"]

                        #best_model = GANN_instance.population_networks[best_model_idx]
                        #if model is None:
                        #    model = best_model
                        #else:
                        #    predictions = pygad.nn.predict(last_layer=model, data_inputs=data_inputs)
    
                        #    error = numpy.sum(numpy.abs(predictions - data_outputs))
    
                            # In case a client sent a model to the server despite that the model error is 0.0. In this case, no need to make changes in the model.
                        #    if error == 0:
                        #        data = {"subject": "done", "data": None}
                        #        response = pickle.dumps(data)
                        #        print("The model is trained successfully and no need to send the model to the client for retraining.")
                        #        return

                        #self.model_averaging(model, model)
                        data = {"subject": "model", "data": self.model_averaging(model, model)}
                        response = pickle.dumps(data)
                        print("\n*****The Model is Aggregated Successfully*****\n\n")

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

    


soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
print("Socket Created.\n")

try:
    soc.connect(("localhost", 10000))
    print("Successful connection to the server.\n")

except BaseException as e:
    print("Error connecting the server: {msg}".format(msg=e))
    soc.close()
    print("Socket closed.")

soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
print("Socket Created.\n")

status = 1
while status == 1:
    data = {"subject": "model", "data": model}
    data_byte = pickle.dumps(data)

    print("Sending the initial model to the clients.\n")
    soc.sendall(data_byte)
    status = 0
    if status == 0:
        break



soc.close()
print("Socket closed.\n")

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
                                     buffer_size=1024,
                                     recv_timeout=10)
        socket_thread.start()
    except:
        soc.close()
        print("(Timeout) Socket Closed Because no Connections Received.\n")
        break