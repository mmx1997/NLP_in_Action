import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
tfidf_model = TfidfVectorizer(min_df=1)

docs = ['NYC is the Big Apple.',
        'NYC is known as the Big Apple.',
        'I love NYC!',
        'I wore a hat to the Big Apple party in NYC.',
        'Come to NYC. See the Big Apple.',
        'Manhattan is called the Big Apple!',
        'New York is a big city for a small cat.',
        'The lion, a big cat, is the king of the jungle.',
        'I love my pet cat.',
        'I love New York City (NYC).',
        'Your dog chased my cat.']

theme_words = ['cat', 'dog', 'apple', 'lion', 'nyc', 'love']

docs_tokens = []
for doc in docs:
    docs_tokens += [tokenizer.tokenize(doc.lower())]

theme_vectors = []

for tw in theme_words:
    zero_vector = []
    for doc_tokens in docs_tokens:
        if tw in doc_tokens:
            zero_vector.append(1)
        else:
            zero_vector.append(0)
    theme_vectors.append(zero_vector)

tdm = pd.DataFrame(theme_vectors, index=theme_words)
print(tdm)
'''
       0   1   2   3   4   5   6   7   8   9   10
cat     0   0   0   0   0   0   1   1   1   0   1
dog     0   0   0   0   0   0   0   0   0   0   1
apple   1   1   0   1   1   1   0   0   0   0   0
lion    0   0   0   0   0   0   0   1   0   0   0
nyc     1   1   1   1   0   0   0   0   0   1   0
love    0   0   1   0   0   0   0   0   1   1   0
'''

U, s, Vt = np.linalg.svd(tdm)
print(pd.DataFrame(U, index=tdm.index).round(2))
'''
          0     1     2     3     4     5
cat   -0.07  0.82 -0.40 -0.00  0.03 -0.40
dog   -0.01  0.21 -0.17 -0.71 -0.31  0.58
apple -0.63 -0.27 -0.57 -0.00  0.42  0.15
lion  -0.01  0.21 -0.17  0.71 -0.31  0.58
nyc   -0.72  0.00  0.31  0.00 -0.58 -0.23
love  -0.28  0.41  0.60 -0.00  0.54  0.32
'''

print(s.round(1))
# [2.9 2.2 1.8 1.  1.  0.6]

S = np.zeros((len(U), len(Vt)))

np.fill_diagonal(S, s)
print(pd.DataFrame(S).round(1))
'''
    0    1    2    3    4    5    6    7    8    9    10
0  2.9  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
1  0.0  2.2  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
2  0.0  0.0  1.8  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
3  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
4  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0
5  0.0  0.0  0.0  0.0  0.0  0.6  0.0  0.0  0.0  0.0  0.0
'''
print(pd.DataFrame(Vt).round(2))
'''
      0     1     2     3     4     5     6     7     8     9     10
0  -0.47 -0.47 -0.34 -0.47 -0.22 -0.22 -0.02 -0.03 -0.12 -0.34 -0.03
1  -0.12 -0.12  0.18 -0.12 -0.12 -0.12  0.37  0.46  0.55  0.18  0.46
2  -0.14 -0.14  0.50 -0.14 -0.31 -0.31 -0.22 -0.31  0.11  0.50 -0.31
3   0.00  0.00  0.00  0.00 -0.00 -0.00  0.00  0.71 -0.00  0.00 -0.71
4  -0.16 -0.16 -0.04 -0.16  0.44  0.44  0.03 -0.30  0.59 -0.04 -0.30
5  -0.15 -0.15  0.16 -0.15  0.27  0.27 -0.72  0.32 -0.14  0.16  0.32
6  -0.66  0.13  0.04  0.25 -0.18  0.47  0.29 -0.00 -0.29  0.25  0.00
7  -0.21  0.58  0.35 -0.53  0.32 -0.17  0.16 -0.00 -0.16 -0.20  0.00
8  -0.45  0.34 -0.17  0.46  0.05 -0.40 -0.35 -0.00  0.35 -0.18  0.00
9  -0.05 -0.05 -0.45 -0.05  0.50 -0.34  0.16 -0.00 -0.16  0.61 -0.00
10 -0.11 -0.48  0.47  0.39  0.43 -0.22  0.21  0.00 -0.21 -0.26 -0.00
'''
