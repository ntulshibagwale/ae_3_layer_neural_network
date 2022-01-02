"""

Nick Tulshibagwale
12-28-2021

nn_main_ae_data

3 layer neural network coded 'from scratch' using Andrew Ng course, translated
from MATLAB. Originally, used to recognized hand written digits. Attempt to use
for training on acoustic data.
 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from nn_functions import *
from acoustic_emission_dataset import AcousticEmissionDataset

if __name__=='__main__':
    
    " ----- Inputs -----------------------------------------------------------"

    # Load in the raw acoustic event signals and the labels
    DT = 10**-7              # [seconds] ; sample period / time between samples
    SR = 1/DT                # [Hz] ; sampling rate
    LOW = 200*10**3          # [Hz] ; low frequency cutoff
    HIGH = 800*10**3         # [Hz] ; high frequency cutoff
    NUM_BINS = 26            # Number of bins for partial power feature vector
    FFT_UNITS = 1000         # FFT outputs in Hz, this converts to kHz
    SIG_LEN = 1024           # [samples / signal] ;
    FNAME = '210330-1'       # File name ; '210308-1','210316-1','210330-1'
    
    # Hyperparameters
    LAMBDA_REG = 1;           # Regularization / cost penalty
    N_FFT = 256               # Number of samples per frame of stft    
    HOP_LENGTH = N_FFT+1      # Number of samples between frame of stft
    TEST_PERC = 0.2           # Percentage of labeled data used for test
    TRAIN_PERC = 1-TEST_PERC  # Percentage of labeled data used for training
    NUM_ITERATIONS = 100      # Number of iterations for minimize cost
    
    # 3 Layer NN - Define layer sizes
    INPUT_LAYER_SIZE  = 15*4  # Computed from spectrogram output (DOUBLE CHECK)
    HIDDEN_LAYER_SIZE = 25    # 25 hidden units
    NUM_LABELS = 2            # 2 labels: 0 = matrix , 1 = fiber  
    
    
    " ----- Load in labeled AE data ------------------------------------------"
    
    labeled_ae_data = AcousticEmissionDataset(DT,SR,LOW,HIGH,NUM_BINS,
                                      FFT_UNITS,SIG_LEN,FNAME,N_FFT,HOP_LENGTH)
    
    spectrogram_shape = labeled_ae_data[0][0].numpy().shape
    
    x = []
    y = []
    for training_example in labeled_ae_data:
         x.append(training_example[0].numpy().flatten()) # Unroll vector
         y.append(training_example[1].numpy())
         
    x = np.array(x) # each row is one example, columns are unrolled spectrogram
    y = np.array(y) # each row is one example's label
    
    " ----- Seperate into training and test set ------------------------------"
    
    # Shuffle data
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    x = x[indices,:]
    y = y[indices]
    
    # Split 80% into train data, and 20% into test data
    train_indices = [0,int(TRAIN_PERC*x.shape[0])+1] 
    test_indices= [int(TRAIN_PERC*x.shape[0])+1,x.shape[0]+1]
    
    x_train = x[train_indices[0]:train_indices[1],:]
    y_train = y[train_indices[0]:train_indices[1]]
    
    x_test = x[test_indices[0]:test_indices[1],:]
    y_test = y[test_indices[0]:test_indices[1]]
    
    " ----- Training Model ---------------------------------------------------"
    
    # The initial weight matrices ('breaking the symmetry')
    initial_theta_1 = rand_initialize_weights(INPUT_LAYER_SIZE,
                                              HIDDEN_LAYER_SIZE)
    initial_theta_2 = rand_initialize_weights(HIDDEN_LAYER_SIZE,
                                              NUM_LABELS)      
    initial_nn_params = np.append( # Unroll parameters (need for optimization)
        initial_theta_1.reshape(initial_theta_1.size),
        initial_theta_2.reshape(initial_theta_2.size),axis=0)
    
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
                      options={"maxiter": NUM_ITERATIONS},
                      tol=1e-5)
                      
    # Reshape parameters back into weight matrices
    optim_theta_1 = result.x[0:HIDDEN_LAYER_SIZE*(INPUT_LAYER_SIZE+1)]
    optim_theta_2 = result.x[HIDDEN_LAYER_SIZE*
                             (INPUT_LAYER_SIZE+1):len(result.x)+1]

    optim_theta_1 = optim_theta_1.reshape(HIDDEN_LAYER_SIZE,INPUT_LAYER_SIZE+1)
    optim_theta_2 = optim_theta_2.reshape(NUM_LABELS,HIDDEN_LAYER_SIZE+1)
                                          
    print("__________________________________________________________________")
    print("Training completed, parameters optimized on training data")
    
    " ----- Evaluation -------------------------------------------------------"
    
    # How accurate is model on training data?
    model_accuracy_train = \
        compute_accuracy(optim_theta_1,optim_theta_2,x_train,y_train)
    print(f"Model accuracy on training data: {model_accuracy_train} %")

    # How accurate is model on test data? (Data not used in learning)
    model_accuracy_test = \
        compute_accuracy(optim_theta_1,optim_theta_2,x_test,y_test)
    print(f"Model accuracy on test data: {model_accuracy_test} %")
    
    # Visualize the weights between input layer and hidden layer
    display_data(optim_theta_1[:,1:],  # get rid of bias unit
                 spectrogram_shape[0], # height
                 spectrogram_shape[1]) # width
    

    
