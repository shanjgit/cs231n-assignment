from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    
    out = x.reshape(x.shape[0],-1) @ w + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dx = (dout @ w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0],-1).T @ dout
    db = np.sum(dout,axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0,x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    mask_x = np.zeros(x.shape)
    mask_x[x > 0] += 1
    dx = dout * mask_x
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    # print("bn_param: ", bn_param)
    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        mu = np.mean(x, axis = 0) # shape=(D,)
        var = np.mean((x - mu)**2, axis=0) # shape=(D,)
        
        # normaliaze the data
        x_hat = (x - mu) / np.sqrt(var + eps)
        
        # shift and rescale 
        y = gamma * x_hat + beta
        out = y
        
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var
        
        cache = (x_hat, var, mu, x, gamma, beta, eps)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward. 
    cache = (x_hat, var, mu, x, gamma, beta, eps)
    

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    x_hat, var, mu, x, gamma, beta, eps = cache 
    N = x.shape[0]
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0) # shape = (D,)
    # Notics that since dL/dbeta = dL/dout dout/dbeta, where out = gamma * x_hat + beta,
    # beta.shape = (N,D), beta = [[beta1,beta2,...,betaD],[beta1, beta2,...,betaD]...]
    # and (dbeta / dbeta).shape = (N,D,D) = [np.identity(D),np.identity(D),...,np.identity(D)]       
    
    dx_hat = dout * gamma 
    
    # calculate dx according to the computation graph:
    # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    dinverse_var = np.sum(dx_hat * (x - mu),axis=0) #(D,)
    dvar = -0.5 * dinverse_var * (var + eps)**(-1.5) # (D,)
    dx_bar_square = (np.ones(x.shape)*dvar) / N #(N,D)
    dx_bar2 = 2 * dx_bar_square * (x - mu) #(N.D)
    dx_bar1 = dx_hat/np.sqrt(var + eps) # (N,D)
    dx_bar = dx_bar1 + dx_bar2
    dmu = -np.sum(dx_bar,axis=0)
    dx1 = dx_bar
    dx2 = (dmu * np.ones(x.shape))/N
    dx = dx1 + dx2
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    m = dout.shape[0]
    dx, dgamma, dbeta = None, None, None
    x_hat, var, mu, x, gamma, beta, eps = cache 
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0) # shape = (D,)
    
    dx_hat = dout * gamma 
    dx = (1/m) * gamma * ((var + eps)**(-0.5)) * \
    (m * dout - dbeta - ((x - mu)* (var+eps)**(-1)) * np.sum(dout*(x-mu),axis=0))
    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    N, D = x.shape
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    mu = np.mean(x, axis=1, keepdims=True) # shape=(N,)
    var = np.mean((x - mu)**2, axis=1, keepdims=True) # shape=(N,)
        
    # normaliaze the data
    x_hat = (x - mu) / np.sqrt(var + eps) # shape=(N,D)
        
    # shift and rescale the data
    y = gamma * x_hat + beta
    out = y
    cache = (x_hat, var, mu, x, gamma, beta, eps)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    D = dout.shape[1]
    x_hat, var, mu, x, gamma, beta, eps = cache 
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0) # shape = (D,)
    
    dx_hat = dout * gamma 
    
    dinverse_var = np.sum(dx_hat * (x - mu),axis=1, keepdims=True) #(N,,1)
    dvar = -0.5 * dinverse_var * (var + eps)**(-1.5) # (N,1)
    dx_bar_square = (np.ones(x.shape)*dvar) / D #(N,D)
    dx_bar2 = 2 * dx_bar_square * (x - mu) #(N.D)
    dx_bar1 = dx_hat/np.sqrt(var + eps) # (N,D)
    dx_bar = dx_bar1 + dx_bar2
    dmu = -np.sum(dx_bar,axis=1, keepdims=True)
    dx1 = dx_bar
    dx2 = (dmu * np.ones(x.shape))/D
    dx = dx1 + dx2
    
    # dx = (1/D) * gamma * ((var + eps)**(-0.5)) * \
    # (D * dout - dbeta - ((x - mu)* (var+eps)**(-1)) * np.sum(dout*(x-mu),axis=1, keepdims=True))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = np.random.rand(*x.shape) < p
        x = (x * mask) / p
        out = np.copy(x)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = np.copy(x)
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask /dropout_param['p']
        dx = np.copy(dx)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = np.copy(dout)
    return dx


def conv_single_step(input_slice, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    input_slice -- slice of input data of shape (n_C_input,f, f, )
    W -- Weight parameters contained in a window - matrix of shape (n_C_ipnut, HH, WW)
    b -- Bias parameters contained in a window - matrix of shape ()
    
    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """

    ### START CODE HERE ### (≈ 2 lines of code)
    # Element-wise product between a_slice and W. Do not add the bias yet.
    
    s = input_slice * W
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + b
    ### END CODE HERE ###

    return Z


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None

    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    S = conv_param['stride']
    P = conv_param['pad']
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    H_prime = 1 + int((H + 2 * P - HH)/S)
    W_prime = 1 + int((W + 2 * P - WW)/S)
    out = np.zeros((N, F, H_prime, W_prime))

    # zero padding
    x_pad = np.pad(x, ((0,0),(0,0),(P,P),(P,P)), mode='constant', constant_values=(0,))
    # print('x_pad shape: ',x_pad.shape)
    for n in range(N):
        for f in range(F):
            for h_prime in range(H_prime):
                for w_prime in range(W_prime):
                    h_start = h_prime * S
                    h_end = h_prime * S + HH
                    w_start = w_prime * S
                    w_end = w_prime * S + WW
                    
                    input_slice = x_pad[n,:,h_start:h_end,w_start:w_end]
                    # print("input_slice shape: ",input_slice.shape)
                    out[n,f, h_prime, w_prime] = conv_single_step(input_slice, w[f,:,:,:], b[f])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    x, w, b, conv_param  = cache 
    
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    S = conv_param['stride']
    P = conv_param['pad']
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    H_prime = 1 + int((H + 2 * P - HH)/S)
    W_prime = 1 + int((W + 2 * P - WW)/S)
    # zero padding
    x_pad = np.pad(x, ((0,0),(0,0),(P,P),(P,P)), mode='constant', constant_values=(0,))
    
    dx_pad = np.zeros(x_pad.shape)
    dw = np.zeros(w.shape)
    db = np.zeros((F,))
     
    for n in range(N):
        for f in range(F):
            db[f] += np.sum(dout[n,f,:,:])
            for h_prime in range(H_prime):
                for w_prime in range(W_prime):
                     
                    h_start = h_prime * S
                    h_end = h_prime * S + HH
                    w_start = w_prime * S
                    w_end = w_prime * S + WW
                    
                    dw[f,:,:,:] += dout[n,f,h_prime,w_prime] * x_pad[n,:,h_start:h_end,w_start:w_end]
                    dx_pad[n,:,h_start:h_end,w_start:w_end] += dout[n,f,h_prime,w_prime] * w[f,:,:,:]
    dx = dx_pad[:,:,P:P+H, P:P+W]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################

    
    N, C, H, W = x.shape
    pool_H = pool_param['pool_height']
    pool_W = pool_param['pool_width']
    stride = pool_param['stride']
    H_prime = 1 + int((H - pool_H)/stride)
    W_prime = 1 + int((W - pool_W)/stride)
    out = np.zeros((N, C, H_prime, W_prime))
    
    for n in range(N):
        for c in range(C):
            for h_prime in range(H_prime):
                for w_prime in range(W_prime):
                    h_start = h_prime * stride
                    h_end = h_start + pool_H
                    w_start = w_prime * stride
                    w_end = w_start + pool_W
                    
                    input_slice = x[n, c, h_start:h_end, w_start:w_end]
                    
                    # max pooling
                    out[n, c, h_prime, w_prime] = np.max(input_slice)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_H = pool_param['pool_height']
    pool_W = pool_param['pool_width']
    stride = pool_param['stride']
    H_prime = 1 + int((H - pool_H)/stride)
    W_prime = 1 + int((W - pool_W)/stride)
    dx = np.zeros_like(x)
    
    for n in range(N):
        for c in range(C):
            for h_prime in range(H_prime):
                for w_prime in range(W_prime):
                    h_start = h_prime * stride
                    h_end = h_start + pool_H
                    w_start = w_prime * stride
                    w_end = w_start + pool_W
                    
                    input_slice = x[n, c, h_start:h_end, w_start:w_end]
                    mask = (input_slice == np.max(input_slice))
                    dx[n, c, h_start:h_end, w_start:w_end] += dout[n, c, h_prime, w_prime] * mask
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (C,) giving running mean of features
      - running_var Array of shape (C,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    
    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = x.shape
    x = x.reshape(-1,C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = out.reshape(N, C, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return np.copy(out), cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = dout.shape
    dx, dgamma, dbeta = batchnorm_backward(dout.reshape(-1,C), cache)
    dx = dx.reshape(N, C, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    N, C, H, W = x.shape
    assert G <= C and C % G == 0
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    x = x.reshape(N, G, C // G, H, W)
    mu = np.mean(x, axis=(2,3,4), keepdims=True) # shape=(N,G,1,1,1)
    var = np.mean((x - mu)**2, axis=(2,3,4), keepdims=True) # shape=(N,G,1,1,1)
        
    # normaliaze the data
    x_hat = ((x - mu) / np.sqrt(var + eps)).reshape(N, C, H, W) 
        
    # shift and rescale the data
    y = gamma.reshape(1,C,1,1) * x_hat + beta.reshape(1,C,1,1)
    out = np.copy(y)
    cache = (x_hat, var, mu, x, gamma, beta, eps)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
      
    N, C, H, W = dout.shape
    x_hat, var, mu, x, gamma, beta, eps = cache 
    _, G, g, _, _ = x.shape
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    dgamma = np.sum(dout * x_hat, axis=(0,2,3), keepdims=True) # shape = (1,C,1,1)
    dbeta = np.sum(dout, axis=(0,2,3), keepdims=True) # shape = (1,C,1,1)
    
    dx_hat = dout * gamma.reshape(1,C,1,1) # shape = (N, C, H, W)
    
    dinverse_var = np.sum(dx_hat.reshape(N, G, g, H, W) * (x - mu),axis=(2,3,4), keepdims=True)
    #(N,G,1,1,1)
    
    dvar = -0.5 * dinverse_var * (var + eps)**(-1.5) # (N,G,1,1,1)
    dx_bar_square = (np.ones(x.shape)*dvar) / (g*H*W) #(N,G,g,H,W)
    dx_bar2 = 2 * dx_bar_square * (x - mu) #(N.G,g,H,W)
    dx_bar1 = dx_hat.reshape(N, G, g, H, W)/np.sqrt(var + eps) # (N,G,g,H,W)
    dx_bar = dx_bar1 + dx_bar2
    dmu = -np.sum(dx_bar,axis=(2,3,4), keepdims=True) #(N,G,1,1,1)
    dx1 = dx_bar
    dx2 = (dmu * np.ones(x.shape))/(g*H*W)
    dx = (dx1 + dx2).reshape(N,C,H,W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx