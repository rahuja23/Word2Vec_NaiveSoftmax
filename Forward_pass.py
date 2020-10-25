from Initialization import initialize_parameters
import numpy as np
def ind_to_word_vecs(inds, parameters):
    """

    :param inds: numpy array. shape:(1,m)
    :param parameters: dict. of intialized weights
    """
    m = inds.shape[1]
    WRD_EMB = parameters['WRD_EMB']
    word_vec = WRD_EMB[inds.flatten(),:].T

    assert( word_vec.shape == (WRD_EMB.shape[1],m))
    return word_vec

def linear_dense(word_vec, parameters):
    """

    :param word_vec: numpy array. Shape: (embedding_size, m)
    :param parameters: Dict. of initialized weights
    """
    m =  word_vec.shape[1]
    W = parameters['W']
    Z = np.dot(W, word_vec)

    assert( Z.shape == (W.shape[0], m))
    return W,Z

def softmax(Z):
    """

    :param Z: Output of the dense layer. Shape: (vocab_size,m)
    """
    softmax_out = np.divide(np.exp(Z), np.sum(np.exp(Z), axis=0, keepdims=True) + 0.001)

    assert(softmax_out.shape == Z.shape)
    return softmax_out

def forwards_propagation(inds, parameters):
    word_vec = ind_to_word_vecs(inds, parameters)
    W, Z = linear_dense(word_vec, parameters)
    softmax_out = softmax(Z)

    caches ={}
    caches['inds'] = inds
    caches['word_vec'] = word_vec
    caches['W'] = W
    caches['Z'] = Z

    return softmax_out, caches
