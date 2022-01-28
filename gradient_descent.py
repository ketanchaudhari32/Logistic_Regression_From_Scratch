from calculate_hypothesis import *
from compute_cost import *
from plot_cost import *

def gradient_descent(X, y, theta, alpha, iterations):
    """
        :param X            : 2D array of our dataset
        :param y            : 1D array of the groundtruth labels of the dataset
        :param theta        : 1D array of the trainable parameters
        :param alpha        : scalar, learning rate
        :param iterations   : scalar, number of gradient descent iterations
    """
    
    m = X.shape[0] # the number of training samples is the number of rows of array X
    cost_vector = np.array([], dtype=np.float32) # empty array to store the cost for every iteration

    # Gradient Descent
    for it in range(iterations):
        
        # initialize temporary theta, as a copy of the existing theta array
        theta_temp = theta.copy()
        
        sigma = np.zeros((len(theta)))
        for i in range(m):
            #########################################
            hypothesis = calculate_hypothesis(X, theta_temp, i)
            ########################################/
            output = y[i]
            #########################################
            sigma[:] = sigma[:] + ((hypothesis - output) * X[i,:])
            ########################################/
        
        # update theta_temp
        #########################################
        theta_temp[:] = theta_temp[:] - (alpha * sigma[:])
        ########################################/
        
        # copy theta_temp to theta
        theta = theta_temp.copy()
        
        # append current iteration's cost to cost_vector
        iteration_cost = compute_cost(X, y, theta)
        cost_vector = np.append(cost_vector, iteration_cost)
    print('Gradient descent finished.')
    
    return theta, cost_vector
