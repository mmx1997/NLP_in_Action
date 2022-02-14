import re

import numpy as np

sentence = "Thoms Jefferson began building Monticello at the aage of 26."

token = re.split(r'[-\s.,!?]+', sentence)  # []表示字符类，字符集。-符号放在第一位，其他位用转义符表示
# token = re.split(r'[\s.,\-!?]+', sentence)  # []表示字符类，字符集。-符号放在第一位，其他位用转义符表示

print(token)
# ['Thoms', 'Jefferson', 'began', 'building', 'Monticello', 'at', 'the', 'aage', 'of', '26', '']

pattern = re.compile(r'[-\s.,!?]+')  # 预编译
print(pattern.split(sentence))
# ['Thoms', 'Jefferson', 'began', 'building', 'Monticello', 'at', 'the', 'aage', 'of', '26', '']
print([x for x in pattern.split(sentence) if x and x not in '- \s,.!?;'])
# ['Thoms', 'Jefferson', 'began', 'building', 'Monticello', 'at', 'the', 'aage', 'of', '26']


from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+|$[0-9.]+|\S+')
print(tokenizer.tokenize(sentence))
# ['Thoms', 'Jefferson', 'began', 'building', 'Monticello', 'at', 'the', 'aage', 'of', '26', '.']

from nltk.tokenize import TreebankWordTokenizer

sentence = "Monticello wasn't designated as UNESCO World Heritage Site until 1987."
tokenizer = TreebankWordTokenizer()  # 能够将wasn't分为was和n't
print(tokenizer.tokenize(sentence))
# ['Monticello', 'was', "n't", 'designated', 'as', 'UNESCO', 'World', 'Heritage', 'Site', 'until', '1987', '.']

from nltk.util import ngrams

print(list(ngrams(['Thoms', 'Jefferson', 'began', 'building', 'Monticello', 'at', 'the', 'aage', 'of', '26'], 2)))
# [('Thoms', 'Jefferson'), ('Jefferson', 'began'), ('began', 'building'), ('building', 'Monticello'), ('Monticello', 'at'), ('at', 'the'), ('the', 'aage'), ('aage', 'of'), ('of', '26')]
two_grams = list(ngrams(['Thoms', 'Jefferson', 'began', 'building', 'Monticello', 'at', 'the', 'aage', 'of', '26'], 2))

print([" ".join(x) for x in two_grams])
# ['Thoms Jefferson', 'Jefferson began', 'began building', 'building Monticello', 'Monticello at', 'at the', 'the aage', 'aage of', 'of 26']

# page 47
import nltk

# nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('english')
print(stopwords[:5])  # ['i', 'me', 'my', 'myself', 'we']

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

print(len(ENGLISH_STOP_WORDS))  # 318
print(len(stopwords))  # 179

print([word for word in stopwords if word not in ENGLISH_STOP_WORDS].__len__())  # 60
print([word for word in ENGLISH_STOP_WORDS if word not in stopwords].__len__())  # 199
print(len(set(ENGLISH_STOP_WORDS).union(stopwords)))  # 378
