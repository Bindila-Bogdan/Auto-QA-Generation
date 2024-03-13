from typing import Set
from nltk.util import ngrams
from nltk import word_tokenize
from nltk.corpus import stopwords
import string


def generate_ngrams(text: str, n: int) -> Set[str]:
    # Tokenize the text and remove stopwords
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words('dutch') and token not in string.punctuation]

    # Generate n-grams
    n_grams = ngrams(tokens, n)

    # Convert n-grams to a set of strings for easier comparison
    return set([' '.join(gram) for gram in n_grams])


def ngram_similarity(question: str, context: str, n: int) -> float:
    # Generate n-grams for both question and context
    question_ngrams = generate_ngrams(question, n)
    context_ngrams = generate_ngrams(context, n)

    # Calculate Jaccard similarity
    intersection = question_ngrams.intersection(context_ngrams)
    union = question_ngrams.union(context_ngrams)

    # Handle case where both sets are empty
    if not union:
        return 1.0 if not intersection else 0.0

    return len(intersection) / len(union)
