import nltk
import re

# nltk.download('brown')
from nltk.corpus import brown

print(brown.words()[:10])
# ['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', 'Friday', 'an', 'investigation', 'of']
print(brown.tagged_words()[:5])
# [('The', 'AT'), ('Fulton', 'NP-TL'), ('County', 'NN-TL'), ('Grand', 'JJ-TL'), ('Jury', 'NN-TL')]

print(len(brown.words()))
# 1161192

from collections import Counter

puns = set((',', '.', '--', '-', '!', '?', ':', ';', '``', "''", '(', ')', '[', ']'))
word_list = (x.lower() for x in brown.words() if x not in puns)

token_counts = Counter(word_list)
print(token_counts.most_common(20))
# [('the', 69971), ('of', 36412), ('and', 28853), ('to', 26158),
# ('a', 23195), ('in', 21337), ('that', 10594), ('is', 10109),
# ('was', 9815), ('he', 9548), ('for', 9489), ('it', 8760),
# ('with', 7289), ('as', 7253), ('his', 6996), ('on', 6741),
# ('be', 6377), ('at', 5372), ('by', 5306), ('i', 5164)]

# page 77
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

kite_from_wiki = "A kite is a tethered heavier-than-air or lighter-than-air craft with wing surfaces that react against the air to create lift and drag forces.[2] A kite consists of wings, tethers and anchors. Kites often have a bridle and tail to guide the face of the kite so the wind can lift it.[3] Some kite designs don’t need a bridle; box kites can have a single attachment point. A kite may have fixed or moving anchors that can balance the kite. One technical definition is that a kite is “a collection of tether-coupled wing sets“.[4] The name derives from its resemblance to a hovering bird." \
                 "The lift that sustains the kite in flight is generated when air moves around the kite's surface, producing low pressure above and high pressure below the wings.[6] The interaction with the wind also generates horizontal drag along the direction of the wind. The resultant force vector from the lift and drag force components is opposed by the tension of one or more of the lines or tethers to which the kite is attached.[7] The anchor point of the kite line may be static or moving (e.g., the towing of a kite by a running person, boat, free-falling anchors as in paragliders and fugitive parakites[8][9] or vehicle).[10][11]" \
                 "The same principles of fluid flow apply in liquids, so kites can be used in underwater currents.[12][13] Paravanes and otter boards operate underwater on an analogous principle." \
                 "Man-lifting kites were made for reconnaissance, entertainment and during development of the first practical aircraft, the biplane." \
                 "Kites have a long and varied history and many different types are flown individually and at festivals worldwide. Kites may be flown for recreation, art or other practical uses. Sport kites can be flown in aerial ballet, sometimes as part of a competition. Power kites are multi-line steerable kites designed to generate large forces which can be used to power activities such as kite surfing, kite landboarding, kite buggying and snow kiting."

kite_history = "Kites were invented in Asia, though their exact origin can only be speculated. The oldest depiction of a kite is from a mesolithic period cave painting in Muna island, southeast Sulawesi, Indonesia, which has been dated from 9500–9000 years B.C.[14] It depicts a type of kite called kaghati [id], which are still used by modern Muna people.[15] The kite is made from kolope (forest tuber) leaf for the mainsail, bamboo skin as the frame, and twisted forest pineapple fiber as rope, though modern kites use string.[16]" \
               "In China, the kite has been claimed as the invention of the 5th-century BC Chinese philosophers Mozi (also Mo Di, or Mo Ti) and Lu Ban (also Gongshu Ban, or Kungshu Phan). Materials ideal for kite building were readily available including silk fabric for sail material; fine, high-tensile-strength silk for flying line; and resilient bamboo for a strong, lightweight framework. By 549 AD paper kites were certainly being flown, as it was recorded that in that year a paper kite was used as a message for a rescue mission. Ancient and medieval Chinese sources describe kites being used for measuring distances, testing the wind, lifting men, signaling, and communication for military operations. The earliest known Chinese kites were flat (not bowed) and often rectangular. Later, tailless kites incorporated a stabilizing bowline. Kites were decorated with mythological motifs and legendary figures; some were fitted with strings and whistles to make musical sounds while flying.[17][18][19]" \
               "Kite Flying by Suzuki Harunobu, 1766 (Metropolitan Museum of Art)" \
               "Kite maker from India, image from Travels in India, including Sinde and the Punjab by H. E. Lloyd, 1845" \
               "After its introduction into India, the kite further evolved into the fighter kite, known as the patang in India, where thousands are flown every year on festivals such as Makar Sankranti.[20]" \
               "Kites were known throughout Polynesia, as far as New Zealand, with the assumption being that the knowledge diffused from China along with the people. Anthropomorphic kites made from cloth and wood were used in religious ceremonies to send prayers to the gods.[21] Polynesian kite traditions are used by anthropologists to get an idea of early \"primitive\" Asian traditions that are believed to have at one time existed in Asia.[22]"

kite_from_wiki = re.sub(r'(\[[0-9]+\])', '', kite_from_wiki)
kite_history = re.sub(r'(\[[0-9]+\])', '', kite_history)

kite_intro = kite_from_wiki.lower()
intro_tokens = tokenizer.tokenize(kite_intro)

kite_history = kite_history.lower()
history_tokens = tokenizer.tokenize(kite_history)

intro_total = len(intro_tokens)
history_total = len(history_tokens)

print(intro_total, history_total)
# 339 411

intro_tf = {}
history_tf = {}
intro_counts = Counter(intro_tokens)
intro_tf['kite'] = intro_counts['kite'] / intro_total
history_counts = Counter(history_tokens)
history_tf['kite'] = history_counts['kite'] / history_total

print("Term frequency of 'kite' in inter is: {:.4f}".format(intro_tf['kite']))
print("Term frequency of 'kite' in history is: {:.4f}".format(history_tf['kite']))
# Term frequency of 'kite' in inter is: 0.0413
# Term frequency of 'kite' in history is: 0.0243

intro_tf['and'] = intro_counts['and'] / intro_total
history_tf['and'] = history_counts['and'] / history_total
print("Term frequency of 'and' in inter is: {:.4f}".format(intro_tf['and']))
print("Term frequency of 'and' in history is: {:.4f}".format(history_tf['and']))
# Term frequency of 'and' in inter is: 0.0354
# Term frequency of 'and' in history is: 0.0243

# IDF
num_docs_containing_and = 0
num_docs_containing_kite = 0
num_docs_containing_china = 0
for doc in [intro_tokens, history_tokens]:
    if 'and' in doc:
        num_docs_containing_and += 1
    if 'kite' in doc:
        num_docs_containing_kite += 1
    if 'china' in doc:
        num_docs_containing_china += 1
intro_tf['china'] = intro_counts['china'] / intro_total
history_tf['china'] = history_counts['china'] / history_total

num_docs = 2
intro_idf = {}
history_idf = {}
intro_idf['and'] = num_docs / num_docs_containing_and
history_idf['and'] = num_docs / num_docs_containing_and
intro_idf['kite'] = num_docs / num_docs_containing_kite
history_idf['kite'] = num_docs / num_docs_containing_kite
intro_idf['china'] = num_docs / num_docs_containing_china
history_idf['china'] = num_docs / num_docs_containing_china

intro_tfidf = {}
intro_tfidf['and'] = intro_tf['and'] * intro_idf['and']
intro_tfidf['kite'] = intro_tf['kite'] * intro_idf['kite']
intro_tfidf['china'] = intro_tf['china'] * intro_idf['china']

history_tfidf = {}
history_tfidf['and'] = history_tf['and'] * history_idf['and']
history_tfidf['kite'] = history_tf['kite'] * history_idf['kite']
history_tfidf['china'] = history_tf['china'] * history_idf['china']

