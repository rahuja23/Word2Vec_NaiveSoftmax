
import numpy as np

def softmax_backward(softmax_out, Y):
    """

    :param softmax_out: Array of predicted outputs. Shape:(Vocab_size,m)
    :param Y: Array of Actual Outputs. Shape:(1,m)
    :return: derivative of cost function w.r.t softmax_out. Shape: (vocab_size,m)_
    """
    m= Y.shape[1]
    dL_dZ =  (1/m)*(softmax_out-Y)
    assert( dL_dZ.shape == softmax_out.shape)
    return dL_dZ

def dense_backwards(dL_dZ, caches):
    """

    :param dL_dZ: Derivative of cost function w.r.t softmax. Shape:(vocab_size,m)
    :param caches: dictionary results of forwrd propaation
    :return: derivative of cost funciton w.r.t weights and word_vec
    """
    W = caches['W']
    word_vec = caches['word_vec']
    dL_dW = np.dot(dL_dZ, word_vec.T)
    dL_dword_vec = np.dot(W.T, dL_dZ)
    assert( dL_dW.shape == W.shape)
    assert(dL_dword_vec.shape == word_vec.shape)
    return dL_dW, dL_dword_vec

def back_propagation(Y, softmax_out, caches):
    """

    :param Y: Array of actual output
    :param softmax_out: Array of predicted outputs
    :param caches: Dict. of results from forward propagation
    :return: gradients w.r.t parameters
    """

    gradients= dict()
    dL_dZ= softmax_backward(softmax_out,Y)
    dL_dW, dL_dword_vec = dense_backwards(dL_dZ,caches)
    gradients['dL_dZ'] = dL_dZ
    gradients['dL_dW'] = dL_dW
    gradients['dL_dword_vec'] = dL_dword_vec

    return gradients

def update_parameters(parameters, caches, gradients, learning_rate):
    """

    :param parameters: Initialized parameters
    :param caches: Results from forward propagation
    :param gradients: Results from backward propagation
    :param learning_rate: step_size
    :return: Updated parameters
    """
    vocab_size, emb_size = parameters['WRD_EMB'].shape
    inds = caches['inds']
    WRD_EMB = parameters['WRD_EMB']
    dL_dword_vec = gradients['dL_dword_vec']
    m= inds.shape[1]
    WRD_EMB[inds.flatten(),:] -= learning_rate* dL_dword_vec.T
    parameters['W'] -= learning_rate* gradients['dL_dW']
