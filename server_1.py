import socket
import pickle
import time
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np

torch.manual_seed(446)
np.random.seed(446)

d_in = 3
d_hidden = 4
d_out = 1
model = torch.nn.Sequential(
                            nn.Linear(d_in, d_hidden),
                            nn.Tanh(),
                            nn.Linear(d_hidden, d_out),
                            nn.Sigmoid()
                           )

class SocketThread(threading.Thread):

	def __init__(self, connection, client_info, buffer_size=1024, recv_timeout=5):
		thread.Thread.__init__(self)
		self.connection = connection
		self.client_info = client_info
		self.buffer_size = buffer_size
		self.recv_timeout = recv_timeout

	def recv(self):
		received_data = b""
		while True:
			try:
				data = connection.recv(self.buffer_size)
				received_data += data

				if data == b'': # Received nothing
					received_data = b""
					# If still no data is received after recv_timeout, return status 0 to close connection
					if (time.time() - self.recv_start_time) > self.recv_timeout:
						return None, 0 # 0 means the connection is no longer active and it should be closed

				elif str(data)[-2] == '.':
					print("All data ({data_len} bytes) Received from {client_info}.".format(client_info=self.client_info, data_len=len(received_data)))

					if len(received_data) > 0:
						try:
							# Decoding the data (bytes).
							received_data = pickle.loads(received_data)
							# Return the decoded data
							return received_data, 1

						except BaseException as e:
							print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
							return None, 0

				else:
					# If any case is received from the client, reset the time counter
					self.recv_start_time = time.time()

			except BaseException as e:
				print("Error Receving Data from the Client: {msg}.\n".format(msg=e))
				return None, 0

	def run(self):
		while True:
			self.recv_start_time = time.time()
			time_struct = time.gmtime()
			date_time = "Waiting to Receive Data Starting from {day}/{month}/{year} {hour}:{minute}:{second} GMT".format(year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
			print(date_time)
			received_data, status = self.recv()
			if status == 0:
				self.connection.close()
				print("Connection closed with {client_info} either due to inactivity for {recv_timeout} seconds or due to an error.".format(client_info=self.client_info, recv_timeout=self.recv_timeout), end="\n\n")
				break

			msg = model
			msg = pickle.dumps(msg)
			connection.sendall(msg)
			print("Server sent a message to the client.")

soc = socket.socket()
print("Socket is created.")

soc.bind(("localhost", 10000))
print("Socket is bound to an address & port number.")

soc.listen(1)
print("Waiting for incoming connection...")

while True:
	try:
		connection, client_info = soc.accept()
		print("New Connection from {client_info}.".format(client_info=client_info))
		socket_thread = SocketThread(connection=connection,
									client_info=client_info,
									buffer_size=1024,
									recv_timeout=10
									)
		socket_thread.start()
	except:
		soc.close()
		print("(Timeout) Socket Closed Because no Connection is received.\n")
		break

