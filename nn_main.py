"""

Nick Tulshibagwale
12-28-2021

nn_main

3 layer neural network coded 'from scratch' using Andrew Ng course, translated
from MATLAB. Used to recognized hand written digits. Eventually will be used
for spectrogram testing.
 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from neural_network_functions import *
        
if __name__=='__main__':
    
    " ----- Inputs -----------------------------------------------------------"
    # 3 Layer NN - Define layer sizes
    INPUT_LAYER_SIZE  = 400;  # 20x20 Input Images of Digits
    HIDDEN_LAYER_SIZE = 25;   # 25 hidden units
    NUM_LABELS = 10;          # 10 labels, from 1 to 10 (map "0" to label 10)   
    
    # Hyperparameters
    LAMBDA_REG = 1;  
    
    # Load in Training Data (x) and Labels (y)
    x = np.genfromtxt('x.csv',delimiter=",")
    y = np.genfromtxt('y.csv',delimiter=",")
    
    
    # Weights generated from Andrew Ng course
    theta_1 = np.genfromtxt('theta1.csv',delimiter=",")
    theta_2 = np.genfromtxt('theta2.csv',delimiter=",")
    
    predict(theta_1,theta_2,x)
    
    " ------ Seperate into training and test set -----------------------------"
    # Shuffle data
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    x = x[indices,:]
    y = y[indices]
    
    # Split 20% into test data, and 80% into train data
    x_test = x[4000:5001,:]
    y_test = y[4000:5001]
    
    x_train = x[0:4000,:]
    y_train = y[0:4000]
    
    " ----- Training Model ---------------------------------------------------"
    
    # The initial weight matrices ('breaking the symmetry')
    initial_theta_1 = rand_initialize_weights(INPUT_LAYER_SIZE,
                                              HIDDEN_LAYER_SIZE)
    initial_theta_2 = rand_initialize_weights(HIDDEN_LAYER_SIZE,
                                              NUM_LABELS)      
    initial_nn_params = np.append( # Unroll parameters (need for optimization)
        initial_theta_1.reshape(initial_theta_1.size,order='F'),
        initial_theta_2.reshape(initial_theta_2.size,order='F'),axis=0)
    
    print("__________________________________________________________________")
    print("Optimizing weight matrices...\n")
    # Minimization of cost function ; optimizing weight parameters    
    result = minimize(fun=nn_cost_function,
                      x0=initial_nn_params,
                      args=(INPUT_LAYER_SIZE,HIDDEN_LAYER_SIZE,NUM_LABELS,
                            x_train,y_train,
                            LAMBDA_REG),
                      method = 'TNC',
                      jac = gradient,
                      options={"maxiter": 200},
                      tol=1e-5)
                      
    # Reshape parameters back into weight matrices
    optim_theta_1 = result.x[0:HIDDEN_LAYER_SIZE*(INPUT_LAYER_SIZE+1)]
    optim_theta_2 = result.x[HIDDEN_LAYER_SIZE*
                             (INPUT_LAYER_SIZE+1):len(result.x)+1]

    optim_theta_1 = optim_theta_1.reshape(HIDDEN_LAYER_SIZE,INPUT_LAYER_SIZE+1,
                                    order='F')
    optim_theta_2 = optim_theta_2.reshape(NUM_LABELS,HIDDEN_LAYER_SIZE+1,
                                          order='F')

    print("__________________________________________________________________")
    print("Training completed, parameters optimized on training data")
    
    " ----- Evaluation -------------------------------------------------------"
    # Trained model with inputs -> what are the nn outputs?
    model_predict_train = predict(optim_theta_1,optim_theta_2,x_train)
    model_predict_test = predict(optim_theta_1,optim_theta_2,x_test)
    model_accuracy_train =\
        np.count_nonzero(model_predict_train+1==y_train)/y_train.shape[0]*100
    model_accuracy_test =\
        np.count_nonzero(model_predict_test+1==y_test)/y_test.shape[0]*100
    # +1 because MATLAB indexing was where dataset from
    print(f"Model accuracy on training data: {model_accuracy_train} %")
    print(f"Model accuracy on test data: {model_accuracy_test} %")
    
    # Visualize the activations 
    display_data(optim_theta_1[:,1:]) # get rid of bias unit
    
    print("\n\n")
    print("Pretrained weights from course below.")
    display_data(theta_1[:,1:])
    
    
    
