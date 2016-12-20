
# coding: utf-8

# In[1]:

import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
import datetime
import time
from helpers import *


# In[3]:

path_dataset = "data_train.csv"
ratings = load_data(path_dataset)
# print('Shape of ratings matrix:',ratings.shape)


# In[5]:

def preprocess(ratings):
    
    def find_mean_vectors(ratings):
        
        # calculating the user_mean and item_mean vectors:

        # user_mean vector:

        user_mean = np.zeros((ratings.shape[1],1))

        for user_no in range(ratings.shape[1]):

            a = ratings[:,user_no].sum()
            b = np.shape(ratings[:,user_no].nonzero())[1]
            user_mean[user_no,0] = (a/b)

        # item_mean vector:

        item_mean = np.zeros((ratings.shape[0],1))

        for item_no in range(ratings.shape[0]):

            a = ratings[item_no,:].sum()
            b = np.shape(ratings[item_no,:].nonzero())[1]
            item_mean[item_no,0] = (a/b)
        # print('user_mean and item_mean computed!')   
        return user_mean, item_mean

    user_mean, item_mean = find_mean_vectors(ratings)
    
    mask = ratings.copy()
    mask[mask>0] = 1
    mask = scipy.sparse.lil_matrix.todense(mask)
    
    A = ((user_mean@np.ones((1,ratings.shape[0]))).T)
    B = ((item_mean@np.ones((1,ratings.shape[1]))))
    preproc_layer = ( np.multiply(mask,A)
                     + np.multiply(mask,B) ) / 2
    
    ratings_dense = scipy.sparse.lil_matrix.todense(ratings)
    ratings_preproc = (ratings_dense - (preproc_layer))
    retrieve_layer = (A + B)/2
    return ratings_preproc, preproc_layer, retrieve_layer


# In[6]:

ratings_preproc, preproc_layer, retrieve_layer = preprocess(ratings)


# In[7]:

def find_global_mean(ratings):
    global_mean = np.sum(ratings)/len(ratings.nonzero()[0])
    print(global_mean,'global_mu')
    return global_mean


# In[8]:

def find_bias_vectors(ratings):
    
    # calculating the user_bias and item_bias vectors:

    global_mean = np.sum(ratings)/len(ratings.nonzero()[0])
    # print(global_mean,'global_mu')

    # user_bias vector:
    user_bias = np.zeros((ratings.shape[1],1))

    for user_no in range(ratings.shape[1]):

        a = ratings[:,user_no].sum()
        b = np.shape(ratings[:,user_no].nonzero())[1]
        user_bias[user_no,0] = global_mean - (a/b)

    # item_bias vector:
    item_bias = np.zeros((ratings.shape[0],1))

    for item_no in range(ratings.shape[0]):

        a = ratings[item_no,:].sum()
        b = np.shape(ratings[item_no,:].nonzero())[1]
        item_bias[item_no,0] = global_mean - (a/b)
    
    # print('user_bias and item_bias computed!')   
    return global_mean, user_bias, item_bias


# In[129]:

# global_mean, user_bias_stored, item_bias_stored = find_bias_vectors(ratings_preproc)


# In[51]:

# define parameters

#gamma = 0.000000002 #
# gamma = 0.00008 #for no lambdas w/o preproc
gamma =   0.002

num_features = 20   # K in the lecture notes

lambda_user = 0.0
lambda_item = 0.0
lambda_user_bias = 0
lambda_item_bias = 0
num_epochs = 50     # number of full passes through the train set
errors = [0]

# set seed
np.random.seed(7)


# init matrix
# ratings_preproc already dense, need to do this for train = ratings case (without preprocessing)
# ratings_prepoc_dense = scipy.sparse.lil_matrix.todense(ratings_preproc)

train = ratings_preproc
global_mean = find_global_mean(ratings_preproc)

item_features = np.zeros((ratings.shape[0],num_features))
user_features = np.zeros((ratings.shape[1],num_features))

# item_features = np.random.random((ratings.shape[0],num_features))
# user_features = np.random.random((ratings.shape[1],num_features))

user_bias = np.zeros((ratings.shape[1]))
item_bias = np.zeros((ratings.shape[0]))

# user_bias = np.random.random((ratings.shape[1],1))
# item_bias = np.random.random((ratings.shape[0],1))

# find the non-zero ratings indices 
nz_row, nz_col = train.nonzero()
nz_train = list(zip(nz_row, nz_col))


# In[ ]:

real_train_label = np.zeros(len(nz_train))
prediction_train = np.zeros(len(nz_train))
rmse_train = np.zeros(num_epochs)

# Printing training rmse before any update loop

mat_pred = ( global_mean*np.ones((10000,1000)) +
           (user_bias.reshape((1000,1)).dot(np.ones((1,10000)))).T +
           (item_bias.reshape((10000,1)).dot(np.ones((1,1000)))) +
           np.dot(item_features,user_features.T) )

mat_pred_for_mse = (mat_pred + retrieve_layer)

for i in range(len(nz_train)):
    real_train_label[i] = ratings[nz_train[i][0],nz_train[i][1]]
    prediction_train[i] = mat_pred_for_mse[nz_train[i][0],nz_train[i][1]]

rmse = calculate_mse(real_train_label, prediction_train)
print('rmse with initialization: ',rmse)    

for it in range(num_epochs): 
    
    # print('Iteration No',it+1)
    # shuffle the training rating indices
    np.random.shuffle(nz_train)

    # decrease step size
    # gamma /= 1.2
    
    begin = datetime.datetime.now()
    count = 0
    for d,n in nz_train:
        count += 1

        difference = train[d,n] - mat_pred[d,n]
        
        # Updating the W
        gradient1 = -1* (difference) * user_features[n,:]
        item_features[d,:] = item_features[d,:]*(1 - gamma*lambda_item) - gamma * gradient1
        

        # Updating the Z
        gradient2 = -1* (difference) * item_features[d,:]
        user_features[n,:] = user_features[n,:]*(1 - gamma*lambda_user) - gamma * gradient2
        
        # Updating the user_bias vector
        gradient3 = -1* (difference) 
        user_bias[n] = user_bias[n]*(1 - gamma*lambda_user_bias) - gamma * gradient3
        
        # Updating the item_bias vector
        gradient4 = -1* (difference)
        item_bias[d] = item_bias[d]*(1 - gamma*lambda_item_bias) - gamma * gradient4
        
        
        mat_pred[d,:] = (np.dot(user_features,item_features[d,:])
                         + user_bias
                         + item_bias[d]*np.ones((ratings.shape[1])) 
                         + global_mean*np.ones((ratings.shape[1])))
                         
        mat_pred[:,n] = (np.dot(item_features,user_features[n,:])
                         + item_bias
                         + user_bias[n]*np.ones((ratings.shape[0]))
                         + global_mean*np.ones((ratings.shape[0])))

    mat_pred_for_mse = (mat_pred + retrieve_layer)
                         
    #Calculating training rmse
    for i in range(len(nz_train)):
        real_train_label[i] = ratings[nz_train[i][0],nz_train[i][1]]
        prediction_train[i] = mat_pred_for_mse[nz_train[i][0],nz_train[i][1]]
        
    rmse = calculate_mse(real_train_label, prediction_train) 
    rmse_train[it] = rmse

    print("iter: {}, RMSE on training set: {}.".format(it+1, rmse))
    end = datetime.datetime.now()
    execution_time = (end - begin).total_seconds()
    print('Iteration runtime: ',execution_time)

np.save('user_bias.npy', user_bias)
np.save('item_bias.npy', item_bias)
np.save('user_features.npy', user_features)
np.save('item_features.npy', item_features)
np.save('rmse_train.npy',rmse_train)
