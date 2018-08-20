import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = X.dot(W)

  for i in range(X.shape[0]):
      sum_i = 0
      f[i] -= np.max(f[i])
      for j in range(f[i].shape[0]):
          sum_i += np.exp(f[i][j])
      loss -= np.log(np.exp(f[i][y[i]]) / sum_i)
      for j in range(f[i].shape[0]):
          dW[:, j] += (np.exp(f[i][j]) / sum_i) * X[i]
      dW[:, y[i]] -= X[i]
      
  loss /= X.shape[0]
  loss += reg * np.sum(W * W)
  dW /= X.shape[0]
  dW += reg * W * 2
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = X.dot(W)
  f -= np.matrix(np.max(f, axis = 1)).T
  expf = np.exp(f)
  sum_i = np.sum(expf, axis = 1)
  expf_norm = expf / np.matrix(sum_i).T
  loss = -np.log(np.divide(expf[range(expf.shape[0]), y], sum_i))
  loss = np.mean(loss) + reg * np.sum(W * W)
  expf_norm[range(expf_norm.shape[0]), y] -= 1
  dW = np.dot(X.T, expf_norm) / X.shape[0] + reg * W * 2

  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

