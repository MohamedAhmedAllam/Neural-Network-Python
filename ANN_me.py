import numpy as np

#load dataset scikit MNIST
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(digits.images[3])
plt.show()

#scale the data
from sklearn.preprocessing import StandardScaler
x_scaler = StandardScaler()
x = x_scaler.fit_transform(digits.data)

#Split the dataset
from sklearn.model_selection import train_test_split
y = digits.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

#convert y into logic matrix
def cvt_y_to_vctr(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect

y_train_vctr = cvt_y_to_vctr(y_train)
y_test_vctr = cvt_y_to_vctr(y_test)

nn_structure = [64, 30, 10]

#sigmoid activation function
def g(x):
    return 1/(1+np.exp(-x))
#Sigmoid derivative    
def g_drv(x):
    return g(x)*(1-g(x))
    
#Initialize weight and bias matrices
def init_weights(nn_structure):
    W = {}
    B = {}
    for s in range(1,len(nn_structure)):
        W[s] = np.random.random_sample((nn_structure[s], nn_structure[s-1]))
        B[s] = np.random.random_sample((nn_structure[s], ))
    return W, B

#Set dW and dB to zero    
def init_deltas(nn_structure):
    Dw = {}
    Db = {}
    for s in range(1, len(nn_structure)):
        Dw[s] = np.zeros((nn_structure[s], nn_structure[s-1]))
        Db[s] = np.zeros((nn_structure[s], ))
    return Dw, Db

def feed_forward(x, W, B):
    H = {1: x}
    Z = {}
    for i in range(1 ,len(W)+1):
        Z[i+1] = W[i].dot(H[i]) + B[i]
        H[i+1] = g(Z[i+1])
    return H, Z
    
def delta_output_layer(y, h_out, z_out):
    return -(y-h_out)*g_drv(z_out)

def delta_hidden_layer(delta_lplus1, w_l, z_l):
    return np.dot(np.transpose(w_l), delta_lplus1) * g_drv(z_l)

def train_nn(nn_structure, X, y, iter_num=3000, alpha = 0.25):
    W, B = init_weights(nn_structure)
    m = len(y)
    cnt = 0
    avg_cost_func = []
    while cnt < iter_num:
        Dw, Db = init_deltas(nn_structure)
        avg_cost = 0
        for i in range(m):
            delta = {} 
            H, Z = feed_forward(X[i, :], W, B)
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = delta_output_layer(y[i, :], H[l], Z[l])
                    avg_cost += np.linalg.norm((y[i,:]-H[l]))
                else:
                    if l > 1:
                        delta[l] = delta_hidden_layer(delta[l+1], W[l], Z[l])
                    Dw[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(H[l][:,np.newaxis]))
                    Db[l] += delta[l+1]
            
        for s in range(1,len(nn_structure)):
            W[s] -= alpha*Dw[s]/m
            B[s] -= alpha*Db[s]/m
        avg_cost = avg_cost/m
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, B, avg_cost_func
    
W, B, avg_cost_func = train_nn(nn_structure, x_train, y_train_vctr, 3000)

plt.plot(avg_cost_func)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.show()

def predict_nn(X, W, B):
    m = len(X)
    y_pred = np.zeros((m, ))
    for i in range(m):
        H, Z = feed_forward(X[i,:], W, B)
        y_pred[i] = np.argmax(H[len(H)])
    return y_pred
    
from sklearn.metrics import accuracy_score
y_pred = predict_nn(x_test, W, B)
accuracy_score(y_test, y_pred)*100