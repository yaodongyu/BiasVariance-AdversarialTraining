import numpy as np


"""
Frequently Used Functions
"""

ell = lambda z: np.log(1 + np.exp(-z))

def loss(theta, X, y, ep, standard=1):
    if standard:
        los = np.mean(ell(y* (X @ theta)))
    else:
        los = np.mean(ell(y* (X @ theta) - ep * np.linalg.norm(theta)))
    predictor = ((X @ theta)>0).astype(float).reshape(-1, 1)
    error = 1 - np.mean((y==2*predictor-1).astype(float))
    return los, error

def gradient(theta, X, y, ep, n):
    if ep == 0:
        grad = -(1.0/n)*(1/(np.exp(y*(X@theta)) + 1)).T @ (y*X)
    else:
        grad =  -(1.0/n)*(1/(np.exp(y*(X@theta)- ep * np.linalg.norm(theta)) + 1)).T @ (y*X - ep * theta.reshape(1, -1)/np.linalg.norm(theta))
    return grad.T

"""
Data Distribution
"""
class Mixture:
    def __init__(self, d, sigma):
        self.d = d
        self.sigma = sigma

    def generate_sample(self, n=1):
        y = (np.random.rand(n)> 0.5).astype(float).reshape(-1, 1)
        y = 2*y - 1
        means = y * np.ones((n, self.d))/np.sqrt(self.d)
        X = self.sigma * np.random.normal(size=(n, self.d))
        X = X + means
        return X, y


"""
Sample and compute
"""

def sample_theta_lean(d, n, sigma, ep, seed='no'):
    # Initializing Weights and Data
    data = Mixture(d, sigma)
    if seed !='no':
        np.random.seed(seed)
    X, y = data.generate_sample(n)

    # Begin Optimization
    # Initializing beta
    theta = (1e-20)*np.random.normal(size = (d, 1))

    # Optimization Param
    T = 400
    gamma = 1
    wd = 1e-2

    for t in range(T):
        grad = gradient(theta, X, y, ep, n).reshape(theta.shape) +wd*theta
        theta = theta - gamma * grad
    tr_loss, tr_error = loss(theta, X, y, ep)
    return theta, tr_loss, tr_error

def compute_bv(d, n, sigma, ep, X, y, ind, num_training_sets=5):
    # building theta matrix
    Theta = np.zeros((num_training_sets, d))
    tr_loss = np.zeros(num_training_sets)
    tr_error = np.zeros(num_training_sets)
    for i in range(num_training_sets):
        dummy = sample_theta_lean(d, n, sigma, ep, ind*i)
        tr_loss[i], tr_error[i] =  dummy[1], dummy[2]
        Theta[i,:] = dummy[0].reshape(-1)

    # Computing Average Error
    test_error = np.mean(np.apply_along_axis(lambda th: loss(th, X.T, y, ep)[1], 1, Theta))

    num_test_sample = X.shape[1]
    average_loss = np.sum(ell(Theta @ (y*X.T).T))/(num_training_sets * num_test_sample)

    average_variance = -np.sum(np.log(np.exp(-np.sum(ell(Theta @ X), 0)/num_training_sets) + \
                              np.exp(-np.sum(ell(-Theta @ X), 0)/num_training_sets)))/num_test_sample
    average_tr_loss = np.mean(tr_loss)
    average_tr_error = np.mean(tr_error)

    return average_loss - average_variance, average_variance, average_tr_loss, average_tr_error, test_error
