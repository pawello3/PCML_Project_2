import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import datetime
import time
from helpers import *

dataset_file_path = "data_train.csv"
ratings = load_data(dataset_file_path)
print('Shape of ratings matrix:', ratings.shape)

def initialize_matrices_random(train, num_features):
    """
        Initialize randomly matrices W and Z of matrix factorization.

        Arguments:
            train: training set (matrix X)
            num_features: number of latent variables in the W*Z^T decomposition

        Returned value(s):
            item_features: matrix W of shape = num_features, num_item
            user_features: matrix Z of shape = num_features, num_user
    """
    
    # W matrix initialization
    item_features = np.random.random((train.shape[0], num_features))
    # Z matrix initialization
    user_features = np.random.random((train.shape[1], num_features))
    
    return item_features, user_features

def initialize_matrices_first_column_mean(train, num_features):
    """
        Initialize randomly matrices W and Z of matrix factorization.
        In matrix W first column is assigned to average rating for that movie.

        Arguments:
            train: training set (matrix X)
            num_features: number of latent variables in the W*Z^T decomposition

        Returned value(s):
            item_features: matrix W of shape = num_features, num_item
            user_features: matrix Z of shape = num_features, num_user
    """

    # W matrix initialization
    item_features = np.random.random((train.shape[0], num_features))
    item_features[:, 0] = train.mean(axis=1).reshape(item_features.shape[0])
    # Z matrix initialization
    user_features = np.random.random((train.shape[1], num_features))

    return item_features, user_features

def initialize_matrices_global_mean(train, num_features):
    """
        Initialize matrices W and Z of matrix factorization such that W*Z^T contains global mean
        at all positions. Therefore all elements of W and Z equal to square root of global_mean/num_features

        Arguments:
            train: training set (matrix X)
            num_features: number of latent variables in the W*Z^T decomposition

        Returned value(s):
            item_features: matrix W of shape = num_features, num_item
            user_features: matrix Z of shape = num_features, num_user
    """
    
    global_mean = np.sum(train) / len(train.nonzero()[0])

    # W matrix initialization
    item_features = np.sqrt(global_mean/num_features) * np.ones((train.shape[0], num_features)) 
    # Z matrix initialization
    user_features = np.sqrt(global_mean/num_features) * np.ones((train.shape[1], num_features))
    
    return item_features, user_features

def initialize_matrices_SVD(train, num_features):
    """
        Initialize matrices W and Z of matrix factorization using SVD decomposition of original matrix X.

        Arguments:
            train: training set (matrix X)
            num_features: number of latent variables in the W*Z^T decomposition

        Returned value(s):
            item_features: matrix W of shape = num_features, num_item
            user_features: matrix Z of shape = num_features, num_user
    """
    
    U, s, V = np.linalg.svd(train, full_matrices=False)
    
    S = np.diag(s)

    U_1 = U[:, 0:num_features]
    S_1 = S[0:num_features, 0:num_features]
    V_1 = V[0:num_features, :]
    
    # W matrix initialization
    item_features = U_1
    # Z matrix initialization
    user_features = (S_1.dot(V_1)).T
    
    return item_features, user_features

def compute_ALS(train_set, train_nonzero_indices, test_set, test_nonzero_indices, num_epochs, cutoff, max_iter_threshold, num_features, item_features, user_features, test_mode):
    # initialize matrices used to compute RMSE
    train_label = np.zeros(len(train_nonzero_indices))
    test_label = np.zeros(len(test_nonzero_indices))
    train_prediction_label = np.zeros(len(train_nonzero_indices))
    test_prediction_label = np.zeros(len(test_nonzero_indices))
    
    # initialize accumulators for RMSE of every iteration
    train_rmse = np.zeros(num_epochs)
    test_rmse = np.zeros(num_epochs)
    if test_mode == True:
        for i in range(len(test_nonzero_indices)):
            test_label[i] = test_set[test_nonzero_indices[i][0], test_nonzero_indices[i][1]]
    
    lambda_user_diag = np.identity(num_features)
    np.fill_diagonal(lambda_user_diag, lambda_user)
    lambda_item_diag = np.identity(num_features)
    np.fill_diagonal(lambda_item_diag, lambda_user)
    
    last_train_rmse = 0
    
    for it in range(num_epochs):
        begin = datetime.datetime.now() # start time measurement
        
        print("Epoch:", it)

        # perform one iteteration of the algorithm
        
        # first fix item features: Z^T = (W^T*W + (lambda_z*I_K)^(-1)*W^T*X)
        user_features = (np.linalg.inv(item_features.T.dot(item_features) + lambda_user_diag).dot(item_features.T.dot(train_set))).T
        # then fix user features: W^T = (Z^T*Z + (lambda_w*I_K)^(-1)*Z^T*X^T)
        item_features = (np.linalg.inv(user_features.T.dot(user_features) + lambda_item_diag).dot(user_features.T.dot(train_set.T))).T

        # calculate training RMSE
        for i in range(len(train_nonzero_indices)):
            train_label[i] = train_set[train_nonzero_indices[i][0], train_nonzero_indices[i][1]]
            train_prediction_label[i] = item_features[train_nonzero_indices[i][0], :].dot(user_features.T[:, train_nonzero_indices[i][1]])
        
        # store train RMSE of current iteration
        train_rmse[it] = calculate_mse(train_label, train_prediction_label)
        
        print("RMSE on training set:", train_rmse[it])
        
        if test_mode == True:
            # calculate test RMSE
            for i in range(len(test_nonzero_indices)):
                test_prediction_label[i] = item_features[test_nonzero_indices[i][0], :].dot(user_features.T[:, test_nonzero_indices[i][1]])

            # store test RMSE of current iteration
            test_rmse[it] = calculate_mse(test_label, test_prediction_label)

            print("RMSE on test set:", test_rmse[it])

        end = datetime.datetime.now() # stop time measurement
        
        # compute the time of the iteration
        execution_time = (end - begin).total_seconds()
        print("Execution time:", execution_time)

        print("*" * 50)
        
        if np.fabs(last_train_rmse - train_rmse[it]) < max_iter_threshold:
            print("ALREADY")
            if cutoff == True:
                break
        else:
            last_train_rmse = train_rmse[it]
    
    return item_features.dot(user_features.T), train_rmse, test_rmse, it

# set random seed
np.random.seed(888)

# define parameters
num_epochs = 50 # number of iterations of ALS
cutoff = True # setting for usage of max_iter_threshold stop condition
max_iter_threshold = 0.00005 # stop condition for ALS algorithm, no visible improvement
split_ratio = 0.9 # ratio between size of training and test set
test_mode = True
if split_ratio == 1.0:
    test_mode = False

def nonzero_indices(matrix):
    nz_row, nz_col = matrix.nonzero()
    return list(zip(nz_row, nz_col))

# find the non-zero ratings indices in the training set
nonzero_indices = nonzero_indices(ratings)

# convert sparse matrix representation to dense matrix representation
ratings_dense = scipy.sparse.lil_matrix.todense(ratings)

# preprocessing
initialize_methods = [initialize_matrices_random, initialize_matrices_first_column_mean, initialize_matrices_global_mean, initialize_matrices_SVD]
item_features, user_features = None, None

for method in range(4):
	init_method_num = method # number of matrices initialization method
	'''temp'''
	if method != 3:
		continue
	'''temp'''
	for features in list([10, 25, 50, 100]):
		num_features = features # number of latent features in matrix factorization
		'''temp'''
		if features == 10:
			continue
		'''temp'''
		for lambda1 in list([10, 20, 35, 50, 60]):
			lambda_item = lambda1 # regularization parameter for item features
			for lambda2 in list([10, 20, 35, 50, 60]):
				lambda_user = lambda2 # regularization parameter for user features
				
				print("Method:", method)
				print("Number of features:", features)
				print("Lambda item:", lambda1)
				print("Lambda user:", lambda2)
				
				if init_method_num == 1:
					cutoff = False
				else:
					cutoff = True
				if init_method_num != 2:
					# initialize matrices W and Z
					item_features, user_features = initialize_methods[init_method_num](ratings_dense, num_features)

				# normalize rows of ratings matrix by substracting mean (bias) rating for each movie
				h = np.nanmean(np.where(ratings_dense != 0, ratings_dense, np.nan), axis = 0)
				for i, j in nonzero_indices:
					ratings_dense[i, j] -= h[j]

				# normalize columns of ratings matrix by substracting mean (bias) rating for each users
				v = np.nanmean(np.where(ratings_dense != 0, ratings_dense, np.nan), axis = 1)
				for i, j in nonzero_indices:
					ratings_dense[i, j] -= v[i]

				if init_method_num == 2:
					# initialize matrices W and Z
					item_features, user_features = initialize_methods[init_method_num](ratings_dense, num_features)

				# split data into training and test sets
				np.random.shuffle(nonzero_indices)

				split_point = int(np.floor(len(nonzero_indices) * split_ratio))
				train_nonzero_indices = nonzero_indices[:split_point]
				test_nonzero_indices = nonzero_indices[split_point:]

				train_set = np.zeros(ratings_dense.shape)
				test_set = np.zeros(ratings_dense.shape)

				for i, j in train_nonzero_indices:
					train_set[i, j] = ratings_dense[i, j]

				for i, j in test_nonzero_indices:
					test_set[i, j] = ratings_dense[i, j]

				# compute the prediction and errors
				prediction, train_rmse, test_rmse, num_iter = compute_ALS(train_set, train_nonzero_indices, test_set, test_nonzero_indices, num_epochs, cutoff, max_iter_threshold, num_features, item_features, user_features, test_mode)
				print("#" * 50)
				print("#" * 50)
				print("#" * 50)
