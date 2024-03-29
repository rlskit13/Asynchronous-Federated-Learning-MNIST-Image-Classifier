import socket
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import os
import numpy as np
from pathlib import Path


Directory = "C:\\Users\\z5278080\\Fritz_FL_demo\\New folder\\Models\\Client_3"

#torch.manual_seed(446)
#np.random.seed(446)

d = 1
n = 200
X = torch.rand(n,d)
y = 4 * torch.sin(np.pi * X) * torch.cos(6*np.pi*X**2)


step_size = 0.05
n_epochs = 1000


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
Client_ID = "Client_3"
#best_sol_idx = -1

while True:
    data = {"ID": Client_ID, "subject": subject, "data": NN_instance}
    data_byte = pickle.dumps(data)
    
    print("Sending the Model to the Server.\n")
    soc.sendall(data_byte)
    #if subject == "model":
        #print("Client's Message Subject is {model}.".format(model=NN_instance.state_dict()))
    
    print("Receiving Reply from the Server.")
    received_data, status = recv(soc=soc, 
                                 buffer_size=4096, 
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


    #ga_instance = prepare_GA(GANN_instance)

    #ga_instance.run()

    #ga_instance.plot_result()

    subject = "model"
    #best_sol_idx = ga_instance.best_solution()[2]
    optim = torch.optim.SGD(NN_instance.parameters(), lr=step_size)
    #print('iter,\tloss')
    loss_func = nn.MSELoss()
    running_loss = []
    for i in range(n_epochs):
        y_hat = NN_instance(X)
        loss = loss_func(y_hat, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        running_loss.append(loss.detach().numpy())
        
    print("Epoch: {epoch}, Loss: {loss}".format(loss=loss, epoch=n_epochs))
    #print("Weight: {parameters}".format(parameters=NN_instance.parameters()))
    #epochs = range(1,1001)
    time_struct = time.gmtime()
    model_name = "{ID}_{day}_{month}_{year}_{hour}_{minute}_{second}.pth".format(ID="Trained", year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
    model_path = os.path.join(Directory, "Trained local models")
    Path(model_path).mkdir(parents=True, exist_ok=True)
    torch.save(NN_instance.state_dict(), os.path.join(model_path, model_name))
    print("Trained local model saved in {path}".format(path=model_path))
    #plt.plot(epochs, running_loss, 'g', label='Training loss')
    #plt.plot(n_epochs, loss_val, 'b', label='validation loss')
    #plt.title('Training loss_Client_1')
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.legend()
    #plt.show()
    


    #updates = {"subject": subject, "data": NN_instance}
    #data_byte = pickle.dumps(updates)
    
    #print("Sending the Model to the Server.\n")
    #soc.sendall(data_byte)


#predictions = NN_instance(X)
#print(predictions)
n_hidden_1 = 32
n_hidden_2 = 32
d_out = 1
complete_model = nn.Sequential(
                                nn.Linear(d, n_hidden_1), 
                                nn.Tanh(),
                                nn.Linear(n_hidden_1, n_hidden_2),
                                nn.Tanh(),
                                nn.Linear(n_hidden_2, d_out)
                            )


model_name = "Completed_model"
model_path = os.path.join(Directory, model_name)
complete_model.load_state_dict(torch.load(os.path.join(model_path, model_name)))
print("Completed model loaded: {model_name}.".format(model_name=model_name))

lossfunc = nn.MSELoss()
y_hat = complete_model(X)                        
error = lossfunc(y_hat, y)
print("Completed model Error: {error}".format(error=error))

X_grid = torch.from_numpy(np.linspace(0,1,50)).float().view(-1, d)
y_hat = complete_model(X_grid)
plt.scatter(X.numpy(), y.numpy())
plt.plot(X_grid.detach().numpy(), y_hat.detach().numpy(), 'r')
plt.title('plot of $f(x)$ and $\hat{f}(x)$ Client_3')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()


epochs = range(1,1001)
plt.plot(epochs, running_loss, 'g', label='Training loss')
plt.title('Training loss_Client_3')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.show()


soc.close()
print("Socket Closed.\n")