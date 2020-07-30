import pandas as pd
import spacy 
import re

nlp = spacy.load("en_core_web_sm")
prefixes = ('\\n', ) + nlp.Defaults.prefixes
stops = nlp.Defaults.stop_words


def normalize_text(comment, lowercase, remove_stopwords):
    if lowercase:
        comment = comment.lower()
        comment= re.sub(r'@\w+','',comment)
        comment=re.sub(r'#','',comment)
        comment = re.sub('http.*','',comment)
    comment = nlp(comment)
    lemmatized = list()
    for word in comment:
        lemma = word.lemma_.strip()
        if lemma:
            if not remove_stopwords or (remove_stopwords and lemma not in stops and lemma != '-PRON-'):
                lemmatized.append(lemma)
    return " ".join(lemmatized)