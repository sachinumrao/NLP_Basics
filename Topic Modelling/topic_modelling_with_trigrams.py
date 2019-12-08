# Import dependencies
import sys
import io
import contractions
from gensim import corpora, models
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


# Text input for topic modelling
filename = 'topic_electric_car.txt'
with io.open(filename, encoding='utf-8') as f:
    doc = f.read()

# Function to get wordnet tags to be used in lemmatizer
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Fix contractions in text
doc = contractions.fix(doc)

#  Create lemmatizer and tokenizer
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Get tokens
words = tokenizer.tokenize(doc)

# Convert all tokns to lower case
tokens = [token.lower() for token in words]

# Remove stop words and punctuations
tokens = [token for token in tokens if token not in stop_words]
#tokens = [token for token in tokens if len(token)>2]

# Remove digits
tokens = [token for token in tokens if token.isalpha()]

# Get POS tags
tags = dict(pos_tag(tokens))
wordnet_tags = {token : get_wordnet_pos(tags[token]) for token in tokens}

# Lemmatize tokens
tokens = [lemmatizer.lemmatize(token, wordnet_tags[token]) for token in tokens]

# Create trigrams with these tokens
n_grams = 3
new_tokens = []

for i in range(len(tokens)-(n_grams-1)):
    substr = tokens[i:i+n_grams]
    new_tok = ' '.join(substr)
    new_tokens.append(new_tok)


# Create dictionary from processed tokens
dict_lda = corpora.Dictionary([new_tokens])
#dict_lda.filter_extremes(no_below=3)

# Create corpus
corpus = [dict_lda.doc2bow(new_tokens)]

# Decide number of topics and words in topics
num_topics = 5
num_words = 5

# Create LDA model
model = models.LdaModel(corpus, num_topics=num_topics,
    id2word=dict_lda, passes=10, alpha=[0.05]*num_topics,
    eta=[0.01]*len(dict_lda.keys()))

# Get the output of LDA model
model_output = model.show_topics(formatted=False, num_topics=num_topics,
    num_words=num_words)

# Display the output of model
for i,topic in model_output:
    print(f"Topic : {i}")
    print(topic)
    print()

