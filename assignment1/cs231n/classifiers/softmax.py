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
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    scores = np.matmul(W.T, X[i])
    scores -= np.max(scores)

    scores_exp_sum = np.sum(np.exp(scores))
    scores_exp = np.exp(scores[y[i]])
    loss -= np.log(scores_exp/scores_exp_sum)
    
    for j in range(num_classes):
      dW[:, j] += np.exp(scores[j]) / scores_exp_sum * X[i]
    dW[:, y[i]] -= X[i]

  loss /= num_train
  dW /= num_train

  loss += reg * np.sum(W * W)
  dW += reg * 2 * W
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
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.matmul(X, W)
  scores -= np.max(scores)

  scores_exp = np.exp(scores)
  scores_exp_sum = np.sum(scores_exp, axis=1)
  correct_scores = scores_exp[range(num_train), y]

  loss = correct_scores/ scores_exp_sum
  loss = -np.sum(np.log(loss)) / num_train + reg * np.sum(W * W)

  softmax = scores_exp / scores_exp_sum.reshape(num_train, 1)
  softmax[range(num_train), y]  = -(scores_exp_sum - correct_scores)/scores_exp_sum
  dW = X.T @ softmax / num_train + 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

