import numpy as np


def initialize_wrd_emb(vocab_size, emb_size):
    '''

    :param vocab_size: int. vocabulary size of the training corpus
    :param emb_size: int. the dimension of word embeddings we want to have.
    '''

    WRD_EMB = np.random.randn(vocab_size, emb_size) * 0.01
    return WRD_EMB

def initialize_dense(input_size, output_size):
    """

    :param input_size: int. size of the input to the dense layer
    :param output_size: int. size of the output from the dense layer
    :return: a matrix of dimensions (output_size,input_size)
    """
    W = np.random.randn(output_size, input_size) * 0.01
    return W

def initialize_parameters(vocab_size, emb_size):
    """
    Initialize all the traning parameters
    """
    WRD_EMB = initialize_wrd_emb(vocab_size,emb_size)
    W = initialize_dense(emb_size, vocab_size)
    parameters = {}
    parameters['WRD_EMB']= WRD_EMB
    parameters['W'] = W

    return parameters


