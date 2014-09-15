"""
@author: Jonathan Ticknor, L-3 Data Tactics Corporation

ANN model with sigmoid transfer function in hidden layer and
linear transfer function in output layer. Mini-batch gradient descent 
is used to improve computational efficiency. Current network configuration
is for a single hidden layer. Future releases will include additional 
transfer functions (tanh, etc), option for multiple hidden layers, objective 
functions (cross entropy), and a self-updating learning rate.

Requires numpy

"""

import numpy as np

# Sigmoid transfer function
def sigmoid(z):
    g = 1.0/(1.0 + np.exp(-z))
    return g

# Gradient of the sigmoid transfer function
def sigmoidGradient(z):
    g = sigmoid(z)*(1-sigmoid(z))
    return g
    
# Random initialization of network weights
def randInitializeWeights(L_in,L_out):
    W = np.zeros((L_out, 1 + L_in))
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1+ L_in) * 2 * epsilon_init - epsilon_init
    return W


# Train the model
def nnCostFunction(nn_params, input_layer_size,hidden_layer_size,num_labels,X,y,lambda1,frac1):
    
    # Unroll theta values for input and hidden layers
    Theta1 = np.reshape(nn_params[0:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1))
    Theta2 = np.reshape(nn_params[(hidden_layer_size*(input_layer_size+1)):], (num_labels, hidden_layer_size+1))
    
    m = X.shape[0]
    # Choose samples to use in gradient descent
    if frac1 < 0.001:
        frac1 = 0.001
    n = int(np.ceil(frac1*m))
    datasample = np.unique(np.random.randint(0,m,n))
    n = len(datasample)
    X_new = X[datasample,:]    
    
    # Initialize gradient vectors
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    Theta1_filt = Theta1[:,1:]
    Theta2_filt = Theta2[:,1:]
    
    # Compute cost function value
    colones = np.ones((np.size(X,axis=0),1))
    a1 = np.hstack((colones,X))
    z2 = np.dot(Theta1, np.transpose(a1))
    z2 = sigmoid(z2)    
    a2 = np.vstack((np.ones((1,z2.shape[1])),z2))
    z3 = np.dot(Theta2,a2)
    cost = np.sum(np.sum((np.transpose(y)-z3) ** 2,axis=1))
    reg = (lambda1 / (2.0*m)) * (np.sum(Theta1_filt.ravel()**2) + np.sum(Theta2_filt.ravel()**2))
    J = 1.0/m * cost + reg
    
    Delta1 = 0
    Delta2 = 0
    
    for t in range(n):
        # Step 1: Perform a forward pass with current network parameters
        a1 = np.hstack((1,X_new[t,:]))
        a1 = np.reshape(a1,(len(a1),1))
        z2 = np.dot(Theta1,a1)
        a2 = np.vstack((1,sigmoid(z2)))
        z3 = np.dot(Theta2,a2)
        a3 = z3
        
        # Step 2: Calculate error for the sample used in the pass
        yt = y[datasample[t],:]
        yt = np.reshape(yt,(len(yt),1))
      
        d3 = a3 - yt
        
        # Step 3: Perform backpropagation
        d2 = np.dot(np.transpose(Theta2_filt),d3) * sigmoidGradient(z2)
        Delta2 = Delta2 + np.dot(d3,np.transpose(a2))
        Delta1 = Delta1 + np.dot(d2,np.transpose(a1))
    
    # Compute the gradient
    Theta1_grad = (1.0/m) * Delta1
    Theta2_grad = (1.0/m) * Delta2
    
    # Implement Regularization
    Theta1_grad[:,1:] = Theta1_grad[:,1:] + ((1.0*lambda1/m) * Theta1_filt)
    Theta2_grad[:,1:] = Theta2_grad[:,1:] + ((1.0*lambda1/m) * Theta2_filt)
    
    # Flatten the gradient to pass for weight adjustment
    grad = np.hstack([Theta1_grad.flatten(),Theta2_grad.flatten()])       
        
        
    return J, grad

# Function to calculate ANN outputs using the trained model
def nnOut(nn_params, input_layer_size,hidden_layer_size,num_labels,X,y,lambda1):
    Theta1 = np.reshape(nn_params[0:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1))
    Theta2 = np.reshape(nn_params[(hidden_layer_size*(input_layer_size+1)):], (num_labels, hidden_layer_size+1))
    
    # Compute cost function value
    colones = np.ones((np.size(X,axis=0),1))
    a1 = np.hstack((colones,X))
    z2 = np.dot(Theta1, np.transpose(a1))
    z2 = sigmoid(z2)    
    a2 = np.vstack((np.ones((1,z2.shape[1])),z2))
    z3 = np.dot(Theta2,a2)
    return z3

# Gradient descent controller function
def train_ANN(nn_params, input_layer_size,hidden_layer_size,num_labels,X,Y,lambda1,alpha1,iterations,frac1):
    k = 0
    cost = 1e10
    while True:
        cost2, grad = nnCostFunction(nn_params, input_layer_size,hidden_layer_size,num_labels,X,Y,lambda1,frac1)
        if abs(cost-cost2) < 1e-5 or k > iterations:
            break
        else:
            if cost2 < cost:
                cost = cost2            
                nn_params = nn_params - alpha1 * grad
            if k % 10 == 0:
                print 'Iteration ' + str(k) + ' with error: ' + str(cost2)
            k += 1
    
    return nn_params