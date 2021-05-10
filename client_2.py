import socket
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np

torch.manual_seed(446)
np.random.seed(446)

d = 1
n = 200
X = torch.rand(n,d)
y = 4 * torch.sin(np.pi * X) * torch.cos(6*np.pi*X**2)

step_size = 0.05
n_epochs = 6000
#n_hidden_1 = 32
#n_hidden_2 = 32
#d_out = 1
#model = nn.Sequential(
#                            nn.Linear(d, n_hidden_1), 
#                            nn.Tanh(),
#                            nn.Linear(n_hidden_1, n_hidden_2),
#                            nn.Tanh(),
#                            nn.Linear(n_hidden_2, d_out)
#                            )


def recv(soc, buffer_size=1024, recv_timeout=10):
	received_data = b""
	while str(received_data)[-2] != '.':
		try:
			soc.settimeout(recv_timeout)
			received_data += soc.recv(buffer_size)

		except socket.timeout:
			print("A socket.timeout exception occured because the server did not send any data for {recv_timeout} seconds or due to error or model completly trained.".format(recv.recv_timeout))
			return None, 0

		except BaseException as e:
			return None, 0
			print("Error occured while receiving data from the server {msg}.".format(msg=e))

	try:
		received_data = pickle.loads(received_data)

	except BaseException as e:
		print("Error decoding the client's data: {msg}.\n".format(msg=e))
		return None, 0

	return received_data, 1

soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
print("Socket Created.\n")

try:
	soc.connect(("localhost", 10000))
	print("Successful connection to the server.\n")

except BaseException as e:
	print("Error connecting the server: {msg}".format(msg=e))
	soc.close()
	print("Socket closed.")

#soc = socket.socket()
#print("Socket is created.")

#soc.connect(("localhost", 10000))
#print("Connected to the server.")


#msg = "Another message from the client."
#msg = model
#msg = params
#msg = params.numpy()
#msg = pickle.dumps(msg)
#soc.sendall(msg)
#print("Client sent a message to the server.")

status = 1
while status == 1:
	#data = {"subject": subject, "data": GANN_instance, "best_solution_idx": best_sol_idx}
	#data_byte = pickle.dumps(data)

	#print("Sending the model to the server.\n")
	#soc.sendall(data_byte)
	status = 1
	print("Receiving reply from the server.")
	received_data, status = recv(soc=soc, buffer_size=1024, recv_timeout=10)

	if status == 0:
		print("Nothing is received from the server.")
		break
	else:
		print(received_data, end="\n\n")

	subject = received_data["subject"]

	if subject == "model":
		model = received_data["data"]
		loss_func = nn.MSELoss()

		optim = torch.optim.SGD(received_data.parameters(), lr=step_size)
		#print('iter,\tloss')
		for i in range(n_epochs):
			y_hat = received_data(X)
			loss = loss_func(y_hat, y)
			optim.zero_grad()
			loss.backward()
			optim.step()

		msg = {"subject": "model", "data": received_data}
		msg = pickle.dumps(msg)
		soc.sendall(msg)
		print("Client sent a message to the server.")
	
	elif subject == "done":
		print("The server said the model is trained successfully.")
		break
	else:
		print("Unregconized message type.")
		break


soc.close()
print("Socket closed.\n")

#received_data = b''
#while str(received_data)[-2] != '.':
 #   data = soc.recv(1024)
 #   received_data += data

#received_data = pickle.loads(received_data)
#print("Received data from the client: {received_data}".format(received_data=received_data.type()))


    
    #if i % (n_epochs // 10) == 0:
    #    print('{},\t{:.2f}'.format(i, loss.item()))

#for params in received_data.parameters():
#	print(params)

#msg = received_data
#msg = pickle.dumps(msg)
#soc.sendall(msg)
#print("Client sent a message to the server.")

#received_data = b''
#while str(received_data)[-2] != '.':
#    data = soc.recv(1024)
#    received_data += data

#received_data = pickle.loads(received_data)

#print("Received data from the client: {received_data}".format(received_data=received_data))


#soc.close()
#print("Socket is closed.")