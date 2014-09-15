"""
Example using KDD Cup 1999 outlier detection data.
Plot detection rate and false positive rate based on percentage
of top outliers used. Outlierness score is the reconstruction error,
i.e. output error, of each data point.

Requires scikit-learn package

"""

import ffANN as ff
import numpy as np
from sklearn.preprocessing import scale
import pylab as pl

# Load the data: Columns represent features, rows represent data points
q = np.loadtxt('kdd_oneper.txt', delimiter = ',')
X = q[:,:-1]        # Remove final column of data (outlier tag)
X = scale(X)        # Scale the data to mean zero, unit SD
Y = X               # Set outputs = inputs (Replicator Neural Network)
outcomes = q[:,-1]  # Column specifying outlier/no outlier

# Specify network parameters
hidden_layer_size = 10
input_layer_size = np.shape(X)[1]
num_labels = np.shape(X)[1]

lambda1 = 0       # Regularization parameter
alpha1 = 10       # Learning rate
iterations = 100  # Specify # of weight updates
frac1 = 0.001   # Fraction of samples for gradient descent (between 0.001 and 1)

# Initialize the network weights
Theta1 = ff.randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = ff.randInitializeWeights(hidden_layer_size, num_labels);
nn_params = np.hstack([Theta1.flatten(),Theta2.flatten()])

# Train ANN model
weights = ff.train_ANN(nn_params, input_layer_size,hidden_layer_size,num_labels,X,Y,lambda1,alpha1,iterations,frac1)

# Calculate ANN outputs with trained network
output = ff.nnOut(nn_params, input_layer_size,hidden_layer_size,num_labels,X,Y,lambda1)


# Calculate reconstruction error for each isntance
errors = np.sum((np.transpose(output)-X)**2, axis=1)
# Sort samples into outliers and non-outliers
hits = np.where(outcomes == 1)[0]
hitsneg = np.where(outcomes == 0)[0]

# Sort data points by error values
vals = np.zeros((len(errors),2))
for i in range(len(errors)):
    vals[i,0] = i
    vals[i,1] = errors[i]
subarray = vals[:,1]
indices = np.argsort(subarray)
result = vals[indices]

plotter = np.zeros((11,2))

# Calculate detection rate & false positive rate based on
# percentage of top outliers (1-10%) 
for i in range(1,11,1):
    per_obs = np.floor(i/100.0 * len(errors))
    
    outlier_obs = result[-int(per_obs):,0]
    hits2 = 0
    for j in range(len(outlier_obs)):
        if outlier_obs[j] in hits:
            hits2 += 1
    per_out = float(hits2)/len(hits)*100.0
    plotter[i/1,0] = 1.0*(len(outlier_obs)-hits2)/len(hitsneg)*100.0
    plotter[i/1,1] = per_out
    
pl.plot(plotter[:,0],plotter[:,1],'o-')
pl.show()