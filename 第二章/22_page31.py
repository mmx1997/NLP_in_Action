import numpy as np

sentence = "Thoms Jefferson began building Monticello at the aage of 26."
token_sequence = str.split(sentence)

vocab = sorted(set(token_sequence))
print(','.join(vocab))
# 26.,Jefferson,Monticello,Thoms,aage,at,began,building,of,the

num_tokens = len(token_sequence)
vocab_size = len(vocab)

onehot_vectors = np.zeros((num_tokens, vocab_size), int)
for i, word in enumerate(token_sequence):
    onehot_vectors[i, vocab.index(word)] = 1

print(' '.join(vocab))
print(onehot_vectors)

import pandas as pd

df = pd.DataFrame(onehot_vectors, columns=vocab)
print(df)

df[df == 0] = ''
print(df)