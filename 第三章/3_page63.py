import re

from nltk.tokenize import TreebankWordTokenizer

sentence = "The faster Harry got to the store, the faster Harry, the faster, would get home."
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(sentence.lower())
print(tokens)
# ['The', 'faster', 'Harry', 'got', 'to', 'the', 'store', ',', 'the', 'faster', 'Harry', ',', 'the', 'faster', ',', 'would', 'get', 'home', '.']

from collections import Counter

bag_of_words = Counter(tokens)
print(bag_of_words)
# Counter({'faster': 3, 'the': 3, ',': 3, 'Harry': 2, 'The': 1, 'got': 1, 'to': 1, 'store': 1, 'would': 1, 'get': 1, 'home': 1, '.': 1})

print(bag_of_words.most_common(4))  # 频率最高的4个单词，即按照key对dict排序
# [('faster', 3), ('the', 3), (',', 3), ('Harry', 2)]

time_harry_appears = bag_of_words['harry']
num_unique_words = len(bag_of_words)
tf = time_harry_appears / num_unique_words
print(round(tf, 4))
# 0.1818

kite_from_wiki = "A kite is a tethered heavier-than-air or lighter-than-air craft with wing surfaces that react against the air to create lift and drag forces.[2] A kite consists of wings, tethers and anchors. Kites often have a bridle and tail to guide the face of the kite so the wind can lift it.[3] Some kite designs don’t need a bridle; box kites can have a single attachment point. A kite may have fixed or moving anchors that can balance the kite. One technical definition is that a kite is “a collection of tether-coupled wing sets“.[4] The name derives from its resemblance to a hovering bird." \
                 "The lift that sustains the kite in flight is generated when air moves around the kite's surface, producing low pressure above and high pressure below the wings.[6] The interaction with the wind also generates horizontal drag along the direction of the wind. The resultant force vector from the lift and drag force components is opposed by the tension of one or more of the lines or tethers to which the kite is attached.[7] The anchor point of the kite line may be static or moving (e.g., the towing of a kite by a running person, boat, free-falling anchors as in paragliders and fugitive parakites[8][9] or vehicle).[10][11]" \
                 "The same principles of fluid flow apply in liquids, so kites can be used in underwater currents.[12][13] Paravanes and otter boards operate underwater on an analogous principle." \
                 "Man-lifting kites were made for reconnaissance, entertainment and during development of the first practical aircraft, the biplane." \
                 "Kites have a long and varied history and many different types are flown individually and at festivals worldwide. Kites may be flown for recreation, art or other practical uses. Sport kites can be flown in aerial ballet, sometimes as part of a competition. Power kites are multi-line steerable kites designed to generate large forces which can be used to power activities such as kite surfing, kite landboarding, kite buggying and snow kiting."
kite_from_wiki = re.sub(r'(\[[0-9]+\])', '', kite_from_wiki)

kite_tokens = tokenizer.tokenize(kite_from_wiki.lower())
kite_token_counter = Counter(kite_tokens)
print(kite_token_counter.most_common(4))
# [('the', 24), ('kite', 14), ('a', 13), ('and', 12)]

import nltk

# nltk.download('stopwords')

from nltk.corpus import stopwords

stopwords = stopwords.words('english')
kite_tokens = [x for x in kite_tokens if x not in stopwords]

kite_token_counter = Counter(kite_tokens)
print(kite_token_counter.most_common(4))
# [('kite', 14), (',', 12), ('kites', 8), ('lift', 4)]

document_vector = []
doc_length = len(kite_tokens)

for key, value in kite_token_counter.most_common():
    document_vector.append(value / doc_length)

print(document_vector.__len__())
# 151

docs = [sentence]
docs.append("Harry is hairy and faster than Jill.")
docs.append("Jill is not as hairy as Harry.")

doc_tokens = []
for doc in docs:
    doc_tokens += [sorted(tokenizer.tokenize(doc.lower()))]

print(len(doc_tokens[0]))
# 19
all_doc_tokens = sum(doc_tokens, [])
print(len(all_doc_tokens))
# 35

lexicon = sorted(set(all_doc_tokens))
print(len(lexicon))
# 18
print(lexicon)
# [',', '.', 'and', 'as', 'faster', 'get', 'got', 'hairy', 'harry', 'home', 'is', 'jill', 'not', 'store', 'than', 'the', 'to', 'would']

from collections import OrderedDict

zero_vector = OrderedDict((token, 0) for token in lexicon)
print(zero_vector)
# OrderedDict([(',', 0), ('.', 0), ('and', 0), ('as', 0), ('faster', 0), ('get', 0), ('got', 0), ('hairy', 0), ('harry', 0), ('home', 0), ('is', 0), ('jill', 0), ('not', 0), ('store', 0), ('than', 0), ('the', 0), ('to', 0), ('would', 0)])


import copy

doc_vectors = []
for doc in docs:
    vec = copy.copy(zero_vector)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)
    for k, v in token_counts.items():
        vec[k] = v / len(lexicon)
    doc_vectors.append(vec)

print(doc_vectors.__len__())
# 3


