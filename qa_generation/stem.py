import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.corpus import alpino as alp
from nltk.tag import UnigramTagger

nltk.download('punkt')
nltk.download('alpino')
dutch_stopwords = set(stopwords.words('dutch'))

stemmer = nltk.stem.snowball.DutchStemmer()

def stem_sentence(sentence):
    # Tokenize the sentence
    words = nltk.word_tokenize(sentence, language="dutch")

    # Lemmatize each word
    stemmed_sentence = []
    for word in words:
        word = ''.join(e for e in word if e.isalnum())  # Remove non-alphanumeric characters
        if word and word not in dutch_stopwords:
            stem = stemmer.stem(word.lower())
            stemmed_sentence.append(stem)
    return " ".join(stemmed_sentence)
