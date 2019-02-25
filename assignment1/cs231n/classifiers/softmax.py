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
  num_train = X.shape[0]
  num_class = W.shape[1]
  num_dim = W.shape[0]

  for i in range(num_train):
    s = X[i] @ W
    # shift values in case of numeric over-flow
    s -= np.max(s)
    y_hat = np.exp(s[y[i]])
    denominator = np.sum(np.exp(s))
    loss += -np.log(y_hat/denominator)
    
    # backpropagation
    dW[np.arange(num_dim),y[i]] += (y_hat/denominator - 1) * X[i] 
    
    for j in range(num_class):
        if j == y[i]:
            continue
        dW[np.arange(num_dim),j] += (np.exp(s[j])/denominator) * X[i]
  
  loss /= num_train
  loss += reg * np.sum(W*W)

  dW /= num_train
  dW += 2 * reg * W  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

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
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = X @ W
  score -= np.max(score, axis = 1, keepdims=True) # shape = (N, C)
  score_y = score[np.arange(num_train), y] # shape = (N,)
  y_hat = np.exp(score_y)

  denominator = np.sum(np.exp(score), axis = 1) # shape=(N,)
  loss += np.sum(-np.log(y_hat/denominator))

  z_hat = np.exp(score) 
  z_hat /= np.expand_dims(denominator, axis=1) # shape = (N,C)
  z_hat[np.arange(num_train), y] -= 1
  dW = X.T @ z_hat

  

  loss /= num_train
  loss += reg * np.sum(W * W)
   
  dW /= num_train
  dW += 2 * reg * W  
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

