import re
import numpy as np

def tokenize(text):
    # Obtain tokens with atleast one alphabet
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+ [\w^\']*')
    return pattern.findall(text.lower())


def mapping(tokens):
    word_to_id = dict()
    id_to_word = dict()

    for i, token in enumerate(set(tokens)):
        # set orders returns an iterable of distinct elements
        word_to_id[token] = i
        id_to_word[i] = token
    return word_to_id, id_to_word


def generate_training_data(tokens, word_to_id, window_size):
    N = len(tokens)
    X, Y= [], []

    for i in range(N):
        nbr_idx= list(range(max(0, i-window_size), i)) + list(range(i+1, min(N, i+window_size+1)))
        for j in nbr_idx:
            X.append(word_to_id[tokens[i]])
            Y.append(word_to_id[tokens[j]])

    X = np.array(X)
    X = np.expand_dims(X, axis=0)
    Y = np.array(Y)
    Y = np.expand_dims(Y, axis=0)

    return X, Y
doc = ["After the deduction of the costs of investing, " \
      "beating the stock market is a loser's game.","I would love to be a part of the project",
       "It is a very important result in the field of NLP."]

tokens_lists=[]
for text in doc:
    token = tokenize(text)
    tokens_lists.append(token)

final_tokens =[]
for var in tokens_lists:
    for item in var:
        final_tokens.append(item)
print(final_tokens)
print(type(final_tokens))
word_to_id, id_to_word = mapping(final_tokens)
X,Y = generate_training_data(final_tokens, word_to_id, 3)
vocab_size= len(id_to_word)
m = Y.shape[1]
Y_one_hot = np.zeros((vocab_size,m))
Y_one_hot[Y.flatten(), np.arange(m)]=1
