import pandas as pd

sentence = "Thoms Jefferson began building Monticello at the aage of 26."

df = pd.DataFrame(pd.Series(dict([(token, 1) for token in sentence.split()])), columns=['sent']).T

print(df)

sentence = "Thoms Jefferson began building Monticello at the aage of 26.\n"
sentence += "Construction was done mostly by local masons and carpenters.\n"
sentence += "He moved into the South Pavilion in 1770. \n"
sentence += "Turning Monticello into a neoclassical masterpiece was Jefferson's obsession."

corpus = {}
for i, sent in enumerate(sentence.split('\n')):
    corpus[f'sent{i}'] = dict((tok, 1) for tok in sent.split())
df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T
print(df)

df = df.T
print(df.sent0.dot(df.sent1))  # 1
print(df.sent0.dot(df.sent2))  # 1
print(df.sent0.dot(df.sent3))  # 1

print([(k, v) for (k, v) in (df.sent0 & df.sent3).items() if v])  # [('Monticello', 1)]
