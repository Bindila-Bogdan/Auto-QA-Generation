import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def generate_questions(fname):
    sample = pd.read_csv(f"../scripts/folds/data_folds/{fname}.csv")
    sample['Document'] = sample.apply(lambda row: ': '.join(row.astype(str)), axis=1)

    for _ in range(5):
