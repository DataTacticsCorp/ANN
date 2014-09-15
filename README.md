ANN
===
@author: Jonathan Ticknor, L-3 Data Tactics Corporation (@JonTicknor)

ANN model with sigmoid transfer function in hidden layer and
linear transfer function in output layer. Mini-batch gradient descent 
is used to improve computational efficiency. Current network configuration
is for a single hidden layer. Future releases will include additional 
transfer functions (tanh, etc), option for multiple hidden layers, objective 
functions (cross entropy), and a self-updating learning rate.

Requires numpy

example artificial neural network (ANN) 

Example using KDD Cup 1999 outlier detection data.
Plot detection rate and false positive rate based on percentage
of top outliers used. Outlierness score is the reconstruction error,
i.e. output error, of each data point.

Requires scikit-learn package
