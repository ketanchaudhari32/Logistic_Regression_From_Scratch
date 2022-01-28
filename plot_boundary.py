import numpy as np

def plot_boundary(X, theta, ax1):
    
    min_x1 = 0.0
    max_x1 = 0.0
    x2_on_min_x1 = 0.0
    x2_on_max_x1 = 0.0
    #########################################
    min_x1 = np.min(X[:,1])
    max_x1 = np.max(X[:,1])
    
    x2_on_min_x1 = (- theta[0] - min_x1*theta[1]) / theta[2]
    x2_on_max_x1 = (- theta[0] - max_x1*theta[1]) / theta[2]
    
    ########################################/
    
    x_array = np.array([min_x1, max_x1])
    y_array = np.array([x2_on_min_x1, x2_on_max_x1])
    ax1.plot(x_array, y_array, c='black', label='decision boundary')
    
    # add legend to the subplot
    ax1.legend()
