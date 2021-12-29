"""

Nick Tulshibagwale
12-28-2021

nn_functions

Functions used for creating 3 layer neural network in neural_network_main.

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def sigmoid(z):
    """
    
    Computes sigmoid function:
        g = Sigmoid(z) computes the sigmoid of z.
    
    """
    g = np.divide(1.0,1.0 + np.exp(-z))
    
    return g

def predict(theta_1,theta_2,x):
    """
    
    Predict the label of an input given a trained neural network 
    p = predict(theta_1, theta_2, x) outputs the predicted label of x given the
    trained weights of a neural network (theta_1, theta_2).
    
    """
    m = x.shape[0] # Number of samples
    num_labels = theta_2.shape[0] # The size of theta 2 matrix indicates labels
    prediction = np.zeros((x.shape[0],1)) # prediction vector for each example
    
    x = np.append(np.ones((m,1)),x,axis=1) # add bias to design matrix
    
    # 1st layer -> 2nd Layer (input to hidden layer)
    z_2 = np.matmul(theta_1,np.transpose(x))                
    a_2 = sigmoid(z_2)                          # activations 
    a_2 = np.append(np.ones((1,m)),a_2,axis=0)  # add bias unit
    
    # 2nd layer -> 3rd Layer (hidden layer to output)
    z_3 = np.matmul(theta_2,a_2)
    a_3 = sigmoid(z_3)                          # probabilities per class
    
    # Class prediction (take class with max probability)
    class_prediction = np.argmax(a_3,axis=0) 
    
    return class_prediction


def nn_cost_function(nn_params,input_layer_size,hidden_layer_size,num_labels,
                     x,y,lambda_reg):    
    """
    
    Implements the neural network cost function for a two layer neural network
    which performs classification. Computes the cost for the neural network. 
    The parameters for the neural network are "unrolled" into the vector 
    nn_params and need to be converted back into the weight matrices. 

    """
    # Reshape parameters back into weight matrices
    theta_1 = nn_params[0:hidden_layer_size*(input_layer_size+1)]
    theta_2 = nn_params[hidden_layer_size*
                        (input_layer_size+1):len(nn_params)+1]

    theta_1 = theta_1.reshape(hidden_layer_size,input_layer_size+1,order='F')
    theta_2 = theta_2.reshape(num_labels,hidden_layer_size+1,order='F')
    
    J = 0                                  # Cost
    m = x.shape[0]                         # Number of examples
    
    # Feedforward neural network and return cost in variable J
    
    x = np.append(np.ones((m,1)),x,axis=1) # add bias to design matrix
    
    # 1st layer -> 2nd Layer (input to hidden layer)
    z_2 = np.matmul(theta_1,np.transpose(x))                
    a_2 = sigmoid(z_2)                          # activations 
    a_2 = np.append(np.ones((1,m)),a_2,axis=0)  # add bias unit
    
    # 2nd layer -> 3rd Layer (hidden layer to output)
    z_3 = np.matmul(theta_2,a_2)
    a_3 = sigmoid(z_3)                          # probabilities per class
    
    h = a_3 # each column contains label probability for a given example
    # shape of h [num of classes x num of examples]
    
    num_classes = h.shape[0] # number of classes
    y_matrix = np.zeros((m,num_classes)) # create a label vector for each ex
    
    for i in range(0,y_matrix.shape[0]):
        k = int(y[i])            # training label (i.e. 5 indicates 5th class)
        y_matrix[i,k-1] = 1  # indicate the correct class by putting a 1
        # In other words, each row corresponds to each example 
        # in each row, there exists a 1 in the position corresponding
        # to correct class. (i.e. 5 would have a 1 in the 4th index)
        
    # Compute cost for each training example     
    for i in range(0,y_matrix.shape[0]):
        
        # Each loop will sum the cost for a single example across all class
        J_example = np.matmul(-y_matrix[i,:],np.log(h[:,i]))-\
        np.matmul((1-y_matrix[i,:]),np.log(1-h[:,i]))    
                    
        J = J + J_example # add to total cost

    # Cost without regularization is averaged over all examples
    J = J / m
    
    # Regularization (adding penalty to cost function)
    # Extract the weights, not including the bias term since its not 
    # kosher to regularize the bias unit
    theta_1_no_bias_unit = theta_1[:,1:theta_1.shape[1]+1]
    theta_2_no_bias_unit = theta_2[:,1:theta_2.shape[1]+1]
    
    J_regularization = lambda_reg / (2 * m) * \
        ( sum(sum(theta_1_no_bias_unit**2)) + 
         sum(sum(theta_2_no_bias_unit**2)) )
    
    # Final cost with regularization
    J = J + J_regularization
    print(f"Cost: {J}")
    
    return J
        
    
def gradient(nn_params,input_layer_size,hidden_layer_size,num_labels,
                     x,y,lambda_reg):
    """
    
    Calculates the gradient for the current parameters using the backprop
    algorithm. Return grad as an unrolled vector for optimization function.
    
    """
    
    # Reshape parameters back into weight matrices
    theta_1 = nn_params[0:hidden_layer_size*(input_layer_size+1)]
    theta_2 = nn_params[hidden_layer_size*
                        (input_layer_size+1):len(nn_params)+1]

    theta_1 = theta_1.reshape(hidden_layer_size,input_layer_size+1,order='F')
    theta_2 = theta_2.reshape(num_labels,hidden_layer_size+1,order='F')
    
    J = 0                                  # Cost
    m = x.shape[0]                         # Number of examples
    theta_1_grad = np.zeros(theta_1.shape)   
    theta_2_grad = np.zeros(theta_2.shape)

    x = np.append(np.ones((m,1)),x,axis=1) # add bias to design matrix

    num_classes = theta_2.shape[0] # number of classes
    y_matrix = np.zeros((m,num_classes)) # create a label vector for each ex
    
    for i in range(0,y_matrix.shape[0]):
        k = int(y[i])            # training label (i.e. 5 indicates 5th class)
        y_matrix[i,k-1] = 1  # indicate the correct class by putting a 1
        # In other words, each row corresponds to each example 
        # in each row, there exists a 1 in the position corresponding
        # to correct class. (i.e. 5 would have a 1 in the 4th index)
        
    # Backpropagation Algorithm
    delta_1 = 0
    delta_2 = 0
    
    for t in range(0,m): # loop through all training examples
        
        x_t = x[t,:] # training example with bias
        y_t = np.transpose(y_matrix[t,:]) # vector with label 
        
        # 1st step - Feedforward pass to calculate activations 
        # Only one training example
        
        # 1st layer -> 2nd layer 
        a_1 = x_t # activations from first layer with bias unit
        z_2 = np.matmul(theta_1,np.transpose(a_1))
        a_2 = sigmoid(z_2)
        a_2 = np.append(1.0,a_2)  # add bias unit

        # 2nd layer -> 3rd layer
        z_3 = np.matmul(theta_2,a_2)
        a_3 = sigmoid(z_3)
        
        # 2nd step - For each output unit compute error wrt to label
        del_3 = a_3 - y_t
        
        # 3rd step - Compute error in hidden layer
        del_2 = np.matmul(np.transpose(theta_2),del_3)*a_2*(1-a_2)
        del_2 = del_2[1:len(del_2)] # remove bias unit
        
        # 4th step - Accumulate gradient
        # Obnoxious - but need to convert 1d array to 2d arrays to perform
        # matrix multiplication correctly
        delta_1_example = np.transpose(np.array([del_2]))@np.array([a_1])
        delta_2_example = np.transpose(np.array([del_3]))@np.array([a_2])
        
        delta_1 = delta_1 + delta_1_example
        delta_2 = delta_2 + delta_2_example
        
    theta_1_grad = delta_1 / m
    theta_2_grad = delta_2 / m
    
    # add regularization (don't regularize bias term)
    theta_1_grad[:,1:theta_1_grad.shape[1]] = \
        theta_1_grad[:,1:theta_1_grad.shape[1]]+\
            lambda_reg/m*theta_1[:,1:theta_1.shape[1]]
    theta_2_grad[:,1:theta_2_grad.shape[1]] = \
        theta_2_grad[:,1:theta_2_grad.shape[1]]+\
            lambda_reg/m*theta_2[:,1:theta_2.shape[1]]
    
    # Neural network cost function gradient, used for training - necessary
    # to minimize the cost function by modyfing parameters        
    grad = np.append(theta_1_grad.reshape(theta_1_grad.size,order='F'),
                 theta_2_grad.reshape(theta_2_grad.size,order='F'),axis=0) 
    
    return grad
    

def rand_initialize_weights(l_in,l_out):
    """
    
    Randomly intializes weights of a layer with l_in incoming connections and
    l_out outgoing connections. 
    
    w will be initialized randomly to break symmetry. First column of w 
    corresponds to weights of the bias unit.
    
    """
    epsilon_init = 0.12
    w = np.random.rand(l_out,1+l_in) * 2 * epsilon_init - epsilon_init
    
    return w


def nn_gradient_descent(nn_params,input_layer_size,hidden_layer_size,
                        num_labels,x,y,lambda_reg,alpha,num_iters):
    """
    
    Performs gradient descent to learn parameters / weight matrices between
    layers. Takes num_iters gradient steps with alpha learning rate.
    
    Note: I never tested this algorithm, always used scipy.minimize
    
    """
    m = y.shape[0]
    J_history = np.zeros((num_iters,1))
    optim_nn_params = nn_params
    
    for iteration in range(0,num_iters+1):
        
        # Compute cost
        J_history[iteration] = nn_cost_function(optim_nn_params,
                                input_layer_size,hidden_layer_size,num_labels,
                                x,y,lambda_reg)
        
        grad = gradient(optim_nn_params, input_layer_size, hidden_layer_size,
                        num_labels, x, y, lambda_reg)
        
        # Gradient descent step
        optim_nn_params = optim_nn_params - alpha*grad
        
        print(f"Iteration #: {iteration}")
        print(f"Cost: {J_history[iteration]}")
        print("\n")
        
    return optim_nn_params
              

def display_data(x, example_width=None, figsize=(10, 10)):
    """
    
    Displays 2D data stored in X in a nice grid. Code copied from
    https://github.com/sachin-101/Machine-Learning-by-Andrew-Ng-Implementation
    -in-Python/blob/master/Programming%20Assignments/Exercise%204/utils.py
    
    """
    
    # Compute rows, cols
    if x.ndim == 2:
        m, n = x.shape
    elif x.ndim == 1:
        n = x.size
        m = 1
        x = x[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        # Display Image
        h = ax.imshow(x[i].reshape(example_width, example_width, order='F'),
                      cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')
