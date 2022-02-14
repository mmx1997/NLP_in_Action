# 单词大小写转换
import re

tokens = ['House', 'Visitor', 'Center']
normalized_tokens = [token.lower() for token in tokens]
print(normalized_tokens)


# ['house', 'visitor', 'center'

# 简单词干还原函数
def stem(phrase):
    return ' '.join([re.findall('^(.*ss|.*?)(s)?$', word)[0][0].strip("'") for word in phrase.lower().split()])


print(stem("Houses"))

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
print(' '.join(stemmer.stem(w).strip("'") for w in "dish washer's washed dishes".split()))
