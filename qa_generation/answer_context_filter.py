from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from rouge import Rouge 

def calculate_bleu_score(reference, candidate):
    # Tokenizing the reference and candidate sentences
    reference_tokens = word_tokenize(reference)
    candidate_tokens = word_tokenize(candidate)

    # Calculating BLEU score
    score = sentence_bleu([reference_tokens], candidate_tokens)
    return score


def calculate_rouge_score(reference, candidate):
    rouge = Rouge()

    # Calculating ROUGE scores
    scores = rouge.get_scores(candidate, reference)
    return scores
