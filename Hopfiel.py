# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:39:56 2016

@author: tailaijin
"""

import numpy as np
import matplotlib.pyplot as plt




def enter_memory():
    n = 10; #number of memory
    memory0 =[[0,0,0,0,0,0,0,0],
              [0,0,0,1,1,0,0,0],
              [0,0,1,0,0,1,0,0],
              [0,1,0,0,0,0,1,0],
              [0,1,0,0,0,0,1,0],
              [0,1,0,0,0,0,1,0],
              [0,1,0,0,0,0,1,0],
              [0,1,0,0,0,0,1,0],
              [0,1,0,0,0,0,1,0],
              [0,1,0,0,0,0,1,0],
              [0,1,0,0,0,0,1,0],
              [0,1,0,0,0,0,1,0],
              [0,1,0,0,0,0,1,0],
              [0,0,1,0,0,1,0,0],
              [0,0,0,1,1,0,0,0],
              [0,0,0,0,0,0,0,0]]
              
    memory1 = [[0,0,0,0,0,0,0,0],
               [0,0,0,0,1,0,0,0],
                [0,0,0,1,1,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,0]]
                
    memory2 = [[0,0,0,0,0,0,0,0],
               [0,0,1,1,1,1,0,0],
               [0,1,1,0,0,1,1,0],
               [0,1,0,0,0,0,1,0],
               [0,1,0,0,0,0,1,0],
               [0,1,0,0,0,0,1,0],
               [0,0,1,0,0,1,1,0],
               [0,0,0,0,1,1,0,0],
               [0,0,0,1,1,0,0,0],
               [0,0,1,1,0,0,0,0],
               [0,1,1,0,0,0,0,0],
               [0,1,0,0,0,0,0,0],
               [0,1,0,0,0,0,1,0],
               [0,1,0,0,0,0,1,0],
               [0,1,1,1,1,1,1,0],
               [0,0,0,0,0,0,0,0]]
               
    memory3 = [[0,0,0,0,0,0,0,0],
               [0,0,1,1,1,1,0,0],
               [0,1,0,0,0,0,1,0],
               [0,0,0,0,0,0,1,0],
               [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,1,0,0],
                [0,0,1,1,1,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,1,0],
                [0,1,0,0,0,0,1,0],
                [0,0,1,1,1,1,0,0],
                [0,0,0,0,0,0,0,0]]
                
    memory4 = [[0,0,0,0,0,0,0,0],
               [0,0,0,0,1,1,0,0],
               [0,0,0,1,0,1,0,0],
               [0,0,1,1,0,1,0,0],
               [0,0,1,0,0,1,0,0],
               [0,1,1,0,0,1,0,0],
               [0,1,0,0,0,1,0,0],
               [0,1,0,0,0,1,0,0],
               [0,1,1,1,1,1,1,0],
               [0,0,0,0,0,1,0,0],
               [0,0,0,0,0,1,0,0],
               [0,0,0,0,0,1,0,0],
               [0,0,0,0,0,1,0,0],
               [0,0,0,0,0,1,0,0],
               [0,0,0,0,0,1,0,0],
               [0,0,0,0,0,0,0,0]]

    memory5 = [[0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,0],
               [0,1,0,0,0,0,0,0],
               [0,1,0,0,0,0,0,0],
               [0,1,0,0,0,0,0,0],
               [0,1,0,0,0,0,0,0],
               [0,1,0,1,1,1,0,0],
               [0,1,1,0,0,0,1,0],
               [0,0,0,0,0,0,1,0],
               [0,0,0,0,0,0,1,0],
               [0,0,0,0,0,0,1,0],
               [0,0,0,0,0,0,1,0],
               [0,0,0,0,0,0,1,0],
               [0,1,0,0,0,0,1,0],
               [0,0,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0]]
               
    memory6 = [[0,0,0,0,0,0,0,0],
               [0,0,0,1,1,0,0,0],
               [0,0,1,0,0,1,0,0],
               [0,1,0,0,0,0,1,0],
               [0,1,0,0,0,0,0,0],
               [0,1,0,0,0,0,0,0],
               [0,1,0,0,0,0,0,0],
               [0,1,0,1,1,0,0,0],
               [0,1,1,0,0,1,0,0],
               [0,1,0,0,0,0,1,0],
               [0,1,0,0,0,0,1,0],
               [0,1,0,0,0,0,1,0],
               [0,1,0,0,0,0,1,0],
               [0,0,1,0,0,1,0,0],
               [0,0,0,1,1,0,0,0],
               [0,0,0,0,0,0,0,0]]
               
    memory7 = [[0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,0],
               [0,0,0,0,0,0,1,0],
               [0,0,0,0,0,1,0,0],
               [0,0,0,0,1,0,0,0],
               [0,0,0,0,1,0,0,0],
               [0,0,0,1,1,0,0,0],
               [0,0,0,1,0,0,0,0],
               [0,0,0,1,0,0,0,0],
               [0,0,0,1,0,0,0,0],
               [0,0,0,1,0,0,0,0],
               [0,0,0,1,0,0,0,0],
               [0,0,0,1,0,0,0,0],
               [0,0,0,1,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0]]
               
    memory8 = [[0,0,0,0,0,0,0,0],
               [0,0,0,1,1,0,0,0],
               [0,0,1,0,0,1,0,0],
               [0,1,0,0,0,0,1,0],
               [0,1,0,0,0,0,1,0],
               [0,1,0,0,0,0,1,0],
               [0,0,1,0,0,1,0,0],
               [0,0,0,1,1,0,0,0],
               [0,0,1,0,0,1,0,0],
               [0,1,0,0,0,0,1,0],
               [0,1,0,0,0,0,1,0],
               [0,1,0,0,0,0,1,0],
               [0,1,0,0,0,0,1,0],
               [0,0,1,0,0,1,0,0],
               [0,0,0,1,1,0,0,0],
               [0,0,0,0,0,0,0,0]]
    
    memory9 = [[0,0,0,0,0,0,0,0],
               [0,0,0,1,1,0,0,0],
               [0,0,1,0,0,1,0,0],
               [0,1,0,0,0,0,1,0],
               [0,1,0,0,0,0,1,0],
               [0,1,0,0,0,0,1,0],
               [0,1,0,0,0,0,1,0],
               [0,1,0,0,0,0,1,0],
               [0,0,1,0,0,1,1,0],
               [0,0,0,1,1,0,1,0],
               [0,0,0,0,0,0,1,0],
               [0,0,0,0,0,0,1,0],
               [0,1,0,0,0,0,1,0],
               [0,1,0,0,0,0,1,0],
               [0,0,1,0,0,1,0,0],
               [0,0,0,1,1,0,0,0]]  
    np_memory = np.array([memory0,memory1,memory2,memory3,memory4,memory5,memory6,memory7,memory8, memory9])
    np_memory = np_memory*2 -1
    #need to plot the memory
    return np_memory, n;
    
def initial (np_memory,n):
    weight = 0
    shape = np.shape(np_memory)
    weight = np.dot(np.matrix(np_memory[0,:]).T,np.matrix(np_memory[0])) 
    for i in range(1,shape[0]):
        weight = weight + np.dot(np.matrix(np_memory[i,:]).T,np.matrix(np_memory[i]))
    np.fill_diagonal(weight,0)
    weight = weight/n #from __future__ import division 
    return weight;

'''    
def acf( np_data ):
   #"active function, input is a np_array"
    shape = np.shape(np_data)
    for i in range(0,shape[1]):
        if np_data[:,i] > 1 :
            np_data[:,i] = 1
            continue
        else:   
            if np_data[:,i] < -1:
                np_data[:,i] = -1
    return np_data;
    '''

   
def acf( np_data, tha ):
   #active function, input is a np_array"
    shape = np.shape(np_data)
    for i in range(0,shape[1]):
        if np_data[:,i] > tha :
            np_data[:,i] = 1
            continue
        else:   
            np_data[:,i] = -1
    return np_data;    

'''
def acf( np_data, tha ):
    #"active function, input is a np_array"
    shape = np.shape(np_data)
    for i in range(0,shape[1]):
        np_data[:,i] = math.tanh(np_data[:,i])+tha
    return np_data;
'''
    
def C_test_Data(n, mermory, std):
    # create n's test data based on the memory and noise_level
    np.random.seed(seed=65535)
    test_data = 0
    #mu, sigma = 0, std # mean and standard deviation
    shape = np.shape(mermory)
    size = shape[1]
    vol = shape[0]*shape[1]
    #for i=0 case begin
    noise = np.reshape(np.random.uniform(-1-std, 1+std, vol),shape)
    test_data = mermory + noise
    #for i=0 case end
    for i in range(1,n):
        noise = np.random.uniform(-1.01, 1.01, size)
        test_data = np.vstack((test_data,mermory+noise))
    for i in range(0,shape[0]*n):
        for j in range(0,shape[1]):
            if test_data[i,j]>0:
                test_data[i,j] = 1
            else:
                test_data[i,j] = -1
    #state = 0 #if all things is correct
    return test_data; #state;   
    
    
#match_memory(test_data, weight, memory,eps)
def match_memory(test_data, weight, np_memory, eps, tha):
    shape = np.shape(np_memory)   
    test_shape = np.shape(test_data)
    i,j,k = (0,0,0)
    for i in range (0, test_shape[0]): 
        data = 0
        data = test_data[i,:]
        print('Observation for ' + repr(i+1) +'th Data point')
        for j in range(0, 5):  #max number of iteration is 10
            visualized(np.reshape(data, (16,8)), 'data')
            ###This visualization has been fixed in order to speed up
            distance = np.zeros(shape[0])
            for k in range(0,shape[0]):
                distance[k] = np.linalg.norm(data - np_memory[k])
            distance_min = min(distance)
            print('distance_min= '+repr(distance_min)+'\nNumber of data point: '+repr(i+1)+'\nIteration: '+repr(j) )
            if distance_min < eps: 
                break;
            else:
                weighted_data = 0
                weighted_data = np.dot(weight,np.matrix(data).T).T
                data = 0
                data = acf(weighted_data, tha)
    return 0,0
    #The function of iteration_info will be devloped in the future.

def visualized (data, title): #title is null when input is ''
    fig, ax = plt.subplots()
    ax.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title(title)
    plt.show()
    return 0;
    
def reshape (data):
    shape = np.shape(data)
    length = shape[1]*shape[2]    
    reshaped = 0    
    reshaped =  np.reshape(data,(shape[0],length))
    return reshaped
    
    
#main function
    #enter the memory
    (memory,n) = enter_memory()
    memory = memory[[1,4,8],:]#Choose some memory to verify the program
    shape = np.shape(memory)
    #visualize for the memory data
    for i in range(0,np.shape(memory)[0]):
        print('Memories for ' + repr(i+1) +'th Data point')
        visualized(memory[i,:,:], '')
    #reshape the memory
    reshaped_memory = reshape(memory)   
    #caculate the weight
    weight = initial (reshaped_memory,n)
    #visualized the weight
    visualized(weight, '')
    n = 5 #the test data for each memory
    ###!!!you can change std here!!!!
    std = 0.1 #the range for the noise from unif()
    #create the test data
    test_data = C_test_Data(n, reshaped_memory, std)
    #visualize for the test data
    for i in range(0,n*shape[0]):
        print('Observation for ' + repr(i+1) +'th Data point')
        visualized(np.reshape(test_data[i,:],(shape[1],shape[2])), '')
    #match the partten
    eps = 1 #toleranted max error
    print('Begin the test!!!')
    match_memory(test_data, weight, reshaped_memory,eps, 1.5)