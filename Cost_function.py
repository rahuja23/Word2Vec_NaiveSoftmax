
import numpy as np


def cross_entropy_loss(Y, softmax_out):
    """

    :param Y: Actual result of shape (1,m)
    :param softmax_out: Result from forward pass. Shape: (vocab_size,m)
    """
    m = softmax_out.shape[1]
    cost = (-1/m) * np.sum(np.sum(Y*np.log(softmax_out + 0.01), axis=0 , keepdims= True), axis=1)
    return cost

