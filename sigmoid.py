import numpy as np

def sigmoid(z):
    
    output = 0.0
    #########################################
    output = 1 / (1 + np.exp(-z))
    ########################################/
    
    return output
