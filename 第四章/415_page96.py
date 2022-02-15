# 垃圾短消息数据集
import pandas as pd

sms = open('../data/spam.csv', encoding='latin-1').read().split(',,,\n')
labels = []
texts = []
for i, d in enumerate(sms[1:]):
    temp = d.split(',')
    label = temp[0]
    text = ','.join(temp[1:])
    labels.append(label)
    texts.append(text)

sms = pd.DataFrame(zip(labels, texts), columns=['label_str', 'text'])
sms['label'] = sms.label_str.apply(lambda x: 0 if x == 'ham' else 1)
print(sms.head(3))
'''
  label_str                                               text  label
0       ham  "Go until jurong point, crazy.. Available only...      0
1       ham                      Ok lar... Joking wif u oni...      0
2      spam  Free entry in 2 a wkly comp to win FA Cup fina...      1
'''
print(sms.shape)
# (5522, 3)

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize

tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)

tfidf_docs = tfidf_model.fit_transform(raw_documents=sms.text).toarray()
print(tfidf_docs.shape)
# (5522, 9275)

print(len(sms[sms.label == 1]))
# 743

mask = sms.label.astype(bool).values
print(mask)
# [False False  True ... False False False]
span_centroid = tfidf_docs[mask].mean(axis=0)
han_centroid = tfidf_docs[~mask].mean(axis=0)

print(span_centroid.round(2))
print(han_centroid.round(2))
# [0.06 0.05 0.   ... 0.   0.   0.  ]
# [0.02 0.06 0.   ... 0.   0.   0.  ]


spamminess_score = tfidf_docs.dot(span_centroid - han_centroid)
print(spamminess_score.round(2))
# [-0.01 -0.02  0.04 ... -0.01 -0.    0.  ]

from sklearn.preprocessing import MinMaxScaler

sms['lda_score'] = MinMaxScaler().fit_transform(spamminess_score.reshape(-1, 1))
sms['lda_predict'] = (sms.lda_score > 0.5).astype(int)
print(sms['label lda_predict lda_score'.split()].round(2).head(5))
'''
   label  lda_predict  lda_score
0      0            0       0.19
1      0            0       0.15
2      1            1       0.65
3      0            0       0.15
4      0            0       0.24
'''
print((1 - (sms.label - sms.lda_predict).abs().sum() / len(sms)).round(3))
# 0.976 准确率


