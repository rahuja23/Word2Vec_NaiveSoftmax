import numpy as np
from Training_Data import X, Y_one_hot, vocab_size, word_to_id, id_to_word
from Initialization import initialize_parameters
from Forward_pass import forwards_propagation
from Backpropagation import back_propagation, update_parameters
from Cost_function import cross_entropy_loss
import matplotlib.pyplot as plt
def skip_gram_model(X,Y, vocab_size, emb_size, learning_rate, epochs, batch_size=256, parameters=None, print_cost=True,
                    plot_cost=True):
    """

    :param X: Array of Input data. Shape:(1,m)
    :param Y: One hot encoding of output. Shape:(vocab-size,m)
    :param vocab_size: Size of the Vocabulary from the corpus
    :param emb_size: Size of the word embeddings.
    :param learning_rate: Step Size
    :param epochs: Traning cycles
    :param parameters: Initialized weight matrices and word vectors.
    :return: Word Embedding
    """
    costs =[]
    m = X.shape[1]
    if parameters == None:
        parameters = initialize_parameters(vocab_size, emb_size)

    for epoch in range(epochs):
        epoch_cost = 0
        batch_inds = list(range(0,m,batch_size))
        np.random.shuffle(batch_inds)
        for i in batch_inds:
            X_batch = X[:,i:i+batch_size]
            Y_batch = Y[:, i:i+batch_size]

            softmax_out, caches = forwards_propagation(X_batch, parameters)
            gradients = back_propagation(Y_batch,softmax_out,caches)
            update_parameters(parameters, caches, gradients,learning_rate)
            cost = cross_entropy_loss(Y_batch, softmax_out)
            epoch_cost += np.squeeze(cost)
        costs.append(epoch_cost)
        if print_cost and epoch % (epochs // 500) == 0:
            print("Cost after epoch {}: {}".format(epoch, epoch_cost))
        if epoch % (epochs // 100) == 0:
            learning_rate *= 0.98

    if plot_cost:
        plt.plot(np.arange(epochs), costs)
        plt.xlabel('#epochs')
        plt.ylabel('Cost')
        plt.show()
    return parameters

paras= skip_gram_model(X,Y_one_hot,vocab_size,50, 0.05, 5000, batch_size=128, parameters=None, print_cost=True, plot_cost=True)
X_test = np.arange(vocab_size)
X_test = np.expand_dims(X_test, axis=0)
softmax_test, _ = forwards_propagation(X_test, paras)
top_sorted_inds = np.argsort(softmax_test, axis=0)[-4:,:]
for input_ind in range(vocab_size):
    input_word = id_to_word[input_ind]
    output_words = [id_to_word[output_ind] for output_ind in top_sorted_inds[::-1, input_ind]]
    print("{}'s neighbor words: {}".format(input_word, output_words))

